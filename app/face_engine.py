from __future__ import annotations

import gc
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .image_utils import ImageValidationError, align_face, face_thumbnail


logger = logging.getLogger(__name__)


class FaceEngineError(RuntimeError):
    """Base error for face analysis and swapping failures."""


class ModelNotReadyError(FaceEngineError):
    """The engine cannot run because one or more models are unavailable."""


class FaceNotFoundError(FaceEngineError):
    """No usable face was detected in an image."""


class InvalidFaceIndexError(FaceEngineError):
    """A requested target face index does not exist."""


@dataclass(frozen=True, slots=True)
class DetectedFace:
    index: int
    bbox: tuple[int, int, int, int]
    score: float
    thumbnail_bgr: np.ndarray

    @property
    def crop(self) -> np.ndarray:
        return self.thumbnail_bgr


class FaceEngine:
    """Thread-safe, once-loaded InsightFace detection, recognition and swap engine."""

    _REQUIRED_MODEL_PATHS = (
        Path("buffalo_l") / "det_10g.onnx",
        Path("buffalo_l") / "w600k_r50.onnx",
        Path("inswapper_128.onnx"),
    )

    def __init__(
        self,
        model_dir: str | Path,
        detection_size: int = 640,
        detection_threshold: float = 0.5,
    ) -> None:
        if detection_size < 128:
            raise ValueError("detection_size must be at least 128")
        if not 0.0 < detection_threshold < 1.0:
            raise ValueError("detection_threshold must be between 0 and 1")

        self.model_dir = Path(model_dir).expanduser().resolve()
        self.detection_size = int(detection_size)
        self.detection_threshold = float(detection_threshold)

        self._status = "not_initialized"
        self._error: str | None = None
        self._providers: tuple[str, ...] = ()
        self._initialize_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._detector: Any | None = None
        self._recognizer: Any | None = None
        self._swapper: Any | None = None
        self._face_class: type[Any] | None = None

    @property
    def is_ready(self) -> bool:
        return self._status == "ready"

    @property
    def status(self) -> str:
        return self._status

    @property
    def providers(self) -> tuple[str, ...]:
        return self._providers

    @property
    def provider(self) -> str | None:
        return self._providers[0] if self._providers else None

    @property
    def error(self) -> str | None:
        return self._error

    def _missing_model_paths(self) -> list[Path]:
        missing: list[Path] = []
        for relative_path in self._REQUIRED_MODEL_PATHS:
            path = self.model_dir / relative_path
            try:
                if not path.is_file() or path.stat().st_size < 1024:
                    missing.append(path)
            except OSError:
                missing.append(path)
        return missing

    @staticmethod
    def _select_execution_providers(onnxruntime: Any) -> list[str]:
        available = set(onnxruntime.get_available_providers())
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider"]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
        if "CPUExecutionProvider" in available:
            return ["CPUExecutionProvider"]
        raise RuntimeError("ONNX Runtime 沒有可用的 CPU 或 CUDA Execution Provider")

    def _load_with_providers(
        self,
        insightface: Any,
        face_class: type[Any],
        providers: list[str],
    ) -> tuple[Any, Any, Any, tuple[str, ...]]:
        detector_path = self.model_dir / "buffalo_l" / "det_10g.onnx"
        recognizer_path = self.model_dir / "buffalo_l" / "w600k_r50.onnx"
        swapper_path = self.model_dir / "inswapper_128.onnx"

        detector = insightface.model_zoo.get_model(
            str(detector_path), providers=providers
        )
        recognizer = insightface.model_zoo.get_model(
            str(recognizer_path), providers=providers
        )
        swapper = insightface.model_zoo.get_model(
            str(swapper_path), providers=providers
        )
        if detector is None or recognizer is None or swapper is None:
            raise RuntimeError("InsightFace 無法辨識必要的 ONNX 模型")

        ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
        detector.prepare(
            ctx_id=ctx_id,
            input_size=(self.detection_size, self.detection_size),
            det_thresh=self.detection_threshold,
        )
        recognizer.prepare(ctx_id=ctx_id)

        actual_providers: tuple[str, ...] = tuple(providers)
        session = getattr(detector, "session", None)
        if session is not None and hasattr(session, "get_providers"):
            reported = tuple(session.get_providers())
            if reported:
                actual_providers = reported
        self._face_class = face_class
        return detector, recognizer, swapper, actual_providers

    def initialize(self, force: bool = False) -> bool:
        """Load models once; retain a readable error state instead of raising."""

        with self._initialize_lock:
            if self.is_ready and not force:
                return True
            self._status = "loading"
            self._error = None
            self._providers = ()
            self._release_models()

            missing = self._missing_model_paths()
            if missing:
                names = ", ".join(
                    str(path.relative_to(self.model_dir)) for path in missing
                )
                self._set_initialization_error(
                    "模型尚未準備完成，缺少："
                    f"{names}。請執行 python scripts/download_models.py 後再試。"
                )
                return False

            try:
                import insightface
                import onnxruntime
                from insightface.app.common import Face
            except Exception as exc:
                self._set_initialization_error(
                    f"無法載入 InsightFace/ONNX Runtime：{exc}"
                )
                logger.exception("Face engine dependencies could not be imported")
                return False

            try:
                requested_providers = self._select_execution_providers(onnxruntime)
            except Exception as exc:
                self._set_initialization_error(str(exc))
                logger.exception("No usable ONNX Runtime execution provider")
                return False

            try:
                loaded = self._load_with_providers(
                    insightface, Face, requested_providers
                )
            except Exception as first_error:
                if requested_providers[0] != "CUDAExecutionProvider":
                    self._set_initialization_error(
                        f"模型載入失敗：{first_error}"
                    )
                    logger.exception("Face models failed to load on CPU")
                    return False
                logger.warning(
                    "CUDA model initialization failed; retrying on CPU: %s",
                    first_error,
                )
                gc.collect()
                try:
                    loaded = self._load_with_providers(
                        insightface, Face, ["CPUExecutionProvider"]
                    )
                except Exception as cpu_error:
                    self._set_initialization_error(
                        "模型無法使用 CUDA 或 CPU 載入："
                        f"CUDA={first_error}; CPU={cpu_error}"
                    )
                    logger.exception("Face models failed to load after CPU fallback")
                    return False

            (
                self._detector,
                self._recognizer,
                self._swapper,
                self._providers,
            ) = loaded
            self._face_class = Face
            self._status = "ready"
            self._error = None
            logger.info(
                "Face engine initialized with providers: %s",
                ", ".join(self._providers),
            )
            return True

    def _set_initialization_error(self, message: str) -> None:
        self._release_models()
        self._status = "error"
        self._error = message

    def _release_models(self) -> None:
        self._detector = None
        self._recognizer = None
        self._swapper = None
        self._face_class = None
        gc.collect()

    def _require_ready(self) -> None:
        if not self.is_ready:
            detail = self._error or "模型尚未初始化"
            raise ModelNotReadyError(f"人臉模型尚未準備完成：{detail}")

    @staticmethod
    def _validate_bgr(image_bgr: np.ndarray) -> np.ndarray:
        if (
            not isinstance(image_bgr, np.ndarray)
            or image_bgr.dtype != np.uint8
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
            or image_bgr.size == 0
        ):
            raise FaceEngineError("輸入圖片必須是有效的 uint8 BGR 陣列")
        return np.ascontiguousarray(image_bgr)

    @staticmethod
    def _face_sort_key(face: Any) -> tuple[float, float]:
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
        return float(bbox[0]), float(bbox[1])

    @staticmethod
    def _face_area(face: Any) -> float:
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
        return max(0.0, float(bbox[2] - bbox[0])) * max(
            0.0, float(bbox[3] - bbox[1])
        )

    @staticmethod
    def _bounded_bbox(face: Any, image: np.ndarray) -> tuple[int, int, int, int]:
        bbox = np.asarray(face.bbox, dtype=np.float32).reshape(-1)
        height, width = image.shape[:2]
        x1 = int(np.clip(np.floor(bbox[0]), 0, max(0, width - 1)))
        y1 = int(np.clip(np.floor(bbox[1]), 0, max(0, height - 1)))
        x2 = int(np.clip(np.ceil(bbox[2]), x1 + 1, width))
        y2 = int(np.clip(np.ceil(bbox[3]), y1 + 1, height))
        return x1, y1, x2, y2

    def _detect_raw(self, image_bgr: np.ndarray) -> list[Any]:
        if self._detector is None or self._face_class is None:
            raise ModelNotReadyError("人臉偵測模型尚未載入")
        bboxes, landmarks = self._detector.detect(
            image_bgr, max_num=0, metric="default"
        )
        if bboxes is None or len(bboxes) == 0:
            return []
        if landmarks is None or len(landmarks) != len(bboxes):
            raise FaceEngineError("偵測模型未回傳可用的人臉五點座標")

        faces: list[Any] = []
        for bbox_row, face_landmarks in zip(bboxes, landmarks):
            bbox_row = np.asarray(bbox_row, dtype=np.float32).reshape(-1)
            if bbox_row.size < 5:
                continue
            face = self._face_class(
                bbox=bbox_row[:4],
                kps=np.asarray(face_landmarks, dtype=np.float32),
                det_score=float(bbox_row[4]),
            )
            faces.append(face)
        return sorted(faces, key=self._face_sort_key)

    def detect(
        self,
        image_bgr: np.ndarray,
        thumbnail_size: int = 160,
    ) -> list[DetectedFace]:
        self._require_ready()
        image = self._validate_bgr(image_bgr)
        if thumbnail_size < 32 or thumbnail_size > 1024:
            raise ValueError("thumbnail_size must be between 32 and 1024")
        with self._inference_lock:
            try:
                faces = self._detect_raw(image)
            except FaceEngineError:
                raise
            except Exception as exc:
                raise FaceEngineError(f"人臉偵測失敗：{exc}") from exc

        detections: list[DetectedFace] = []
        for index, face in enumerate(faces):
            bbox = self._bounded_bbox(face, image)
            try:
                thumbnail = face_thumbnail(
                    image, bbox, size=thumbnail_size, padding=0.28
                )
            except ImageValidationError as exc:
                raise FaceEngineError(f"無法建立人臉縮圖：{exc}") from exc
            detections.append(
                DetectedFace(
                    index=index,
                    bbox=bbox,
                    score=float(face.det_score),
                    thumbnail_bgr=thumbnail,
                )
            )
        return detections

    def extract_source(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self._require_ready()
        image = self._validate_bgr(image_bgr)
        with self._inference_lock:
            try:
                faces = self._detect_raw(image)
                if not faces:
                    raise FaceNotFoundError("來源照片中沒有偵測到人臉")
                source_face = max(faces, key=self._face_area)
                if self._recognizer is None:
                    raise ModelNotReadyError("人臉辨識模型尚未載入")
                self._recognizer.get(image, source_face)
            except FaceEngineError:
                raise
            except Exception as exc:
                raise FaceEngineError(f"來源人臉分析失敗：{exc}") from exc

        embedding = np.asarray(source_face.embedding, dtype=np.float32).reshape(-1)
        if embedding.size != 512 or not np.all(np.isfinite(embedding)):
            raise FaceEngineError("辨識模型回傳的 embedding 格式無效")
        norm = float(np.linalg.norm(embedding))
        if norm <= 1e-8:
            raise FaceEngineError("辨識模型回傳零向量 embedding")
        normalized_embedding = np.ascontiguousarray(
            embedding / norm, dtype=np.float32
        )
        try:
            aligned = align_face(image, source_face.kps, image_size=112)
        except ImageValidationError as exc:
            raise FaceEngineError(f"來源人臉對齊失敗：{exc}") from exc
        return aligned, normalized_embedding

    @staticmethod
    def _validated_embedding(source_embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
        if vector.size != 512 or not np.all(np.isfinite(vector)):
            raise FaceEngineError("來源 embedding 必須是有效的 512 維向量")
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            raise FaceEngineError("來源 embedding 不可為零向量")
        return np.ascontiguousarray(vector / norm, dtype=np.float32)

    @staticmethod
    def _restore_face_details(
        image_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        strength: float,
    ) -> np.ndarray:
        if strength <= 0.0:
            return image_bgr
        x1, y1, x2, y2 = bbox
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            return image_bgr

        sigma = max(0.6, min(crop.shape[:2]) / 180.0)
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=sigma, sigmaY=sigma)
        amount = 0.35 + 0.65 * strength
        sharpened = cv2.addWeighted(crop, 1.0 + amount, blurred, -amount, 0)

        mask = np.zeros(crop.shape[:2], dtype=np.float32)
        center = (crop.shape[1] // 2, crop.shape[0] // 2)
        axes = (
            max(1, round(crop.shape[1] * 0.43)),
            max(1, round(crop.shape[0] * 0.48)),
        )
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, thickness=-1)
        feather = max(3.0, min(crop.shape[:2]) * 0.08)
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        alpha = np.clip(mask[..., None] * strength, 0.0, 1.0)
        restored_crop = (
            sharpened.astype(np.float32) * alpha
            + crop.astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)
        restored = image_bgr.copy()
        restored[y1:y2, x1:x2] = restored_crop
        return restored

    def swap(
        self,
        image_bgr: np.ndarray,
        source_embedding: np.ndarray,
        target_face_index: int | None = 0,
        restore_face: bool = False,
        restore_strength: float = 0.35,
    ) -> np.ndarray:
        self._require_ready()
        image = self._validate_bgr(image_bgr)
        embedding = self._validated_embedding(source_embedding)
        index: int | None
        if target_face_index is None:
            index = None
        else:
            try:
                index = int(target_face_index)
            except (TypeError, ValueError) as exc:
                raise InvalidFaceIndexError("目標人臉 index 必須是整數") from exc
            if index < 0:
                raise InvalidFaceIndexError("目標人臉 index 不可為負數")
        try:
            strength = float(restore_strength)
        except (TypeError, ValueError) as exc:
            raise FaceEngineError("人臉修復強度必須是數字") from exc
        if not np.isfinite(strength):
            raise FaceEngineError("人臉修復強度無效")
        strength = float(np.clip(strength, 0.0, 1.0))

        with self._inference_lock:
            try:
                faces = self._detect_raw(image)
                if not faces:
                    raise FaceNotFoundError("目標照片中沒有偵測到人臉")
                if index is None:
                    if len(faces) > 1:
                        raise InvalidFaceIndexError(
                            "偵測到多張臉，請先選擇要替換的人臉"
                        )
                    index = 0
                if index >= len(faces):
                    raise InvalidFaceIndexError(
                        f"目標人臉 index {index} 不存在；共偵測到 {len(faces)} 張臉"
                    )
                target_face = faces[index]
                if self._swapper is None or self._face_class is None:
                    raise ModelNotReadyError("換臉模型尚未載入")
                source_face = self._face_class(embedding=embedding)
                result = self._swapper.get(
                    image.copy(), target_face, source_face, paste_back=True
                )
            except FaceEngineError:
                raise
            except Exception as exc:
                raise FaceEngineError(f"換臉推論失敗：{exc}") from exc

        result = self._validate_bgr(result)
        if restore_face and strength > 0.0:
            bbox = self._bounded_bbox(target_face, result)
            result = self._restore_face_details(result, bbox, strength)
        return result


_singleton_lock = threading.Lock()
_singleton: FaceEngine | None = None


def get_face_engine(
    model_dir: str | Path | None = None,
    detection_size: int = 640,
    detection_threshold: float = 0.5,
) -> FaceEngine:
    """Return the process-wide engine; the first call fixes its configuration."""

    global _singleton
    resolved_model_dir = Path(
        model_dir if model_dir is not None else os.getenv("MODEL_DIR", "/models")
    ).expanduser().resolve()
    with _singleton_lock:
        if _singleton is None:
            _singleton = FaceEngine(
                model_dir=resolved_model_dir,
                detection_size=detection_size,
                detection_threshold=detection_threshold,
            )
        elif (
            _singleton.model_dir != resolved_model_dir
            or _singleton.detection_size != int(detection_size)
            or _singleton.detection_threshold != float(detection_threshold)
        ):
            raise FaceEngineError(
                "FaceEngine 單例已使用不同設定建立，無法在同一程序中重新設定"
            )
        return _singleton
