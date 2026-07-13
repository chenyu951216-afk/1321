from __future__ import annotations

import io
import os
import re
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
DEFAULT_MAX_DECODED_PIXELS = 25_000_000
MAX_EMBEDDING_FILE_BYTES = 1_048_576
_SAFE_STEM_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


class ImageValidationError(ValueError):
    """The supplied bytes do not represent an acceptable, safe image."""


class ImageTooLargeError(ImageValidationError):
    """The encoded image or its decoded pixel count exceeds the configured limit."""


class UnsupportedImageError(ImageValidationError):
    """The image's actual file format is not supported."""


class ImageStorageError(RuntimeError):
    """An image or embedding could not be stored safely."""


@dataclass(frozen=True, slots=True)
class DecodedImage:
    bgr: np.ndarray
    format: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    was_resized: bool

    @property
    def mime_type(self) -> str:
        return {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }[self.format]


def _decoded_pixel_limit(max_side: int, explicit_limit: int | None) -> int:
    if explicit_limit is not None:
        if explicit_limit < 1:
            raise ValueError("max_decoded_pixels must be positive")
        return explicit_limit
    scaled_limit = max_side * max_side * 4
    return min(max(DEFAULT_MAX_DECODED_PIXELS, scaled_limit), 40_000_000)


def _validate_dimensions(width: int, height: int, max_pixels: int) -> None:
    if width < 1 or height < 1:
        raise ImageValidationError("圖片尺寸無效")
    if width > 65_535 or height > 65_535 or width * height > max_pixels:
        raise ImageTooLargeError(
            f"圖片解碼尺寸過大；最多允許 {max_pixels:,} 像素"
        )


def _pil_to_rgb(image: Image.Image) -> Image.Image:
    if "A" in image.getbands() or image.mode in {"P", "LA"}:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def decode_image_bytes(
    data: bytes | bytearray | memoryview,
    *,
    max_upload_mb: int = 15,
    max_upload_bytes: int | None = None,
    max_side: int = 2500,
    max_decoded_pixels: int | None = None,
) -> DecodedImage:
    """Validate actual image content, apply EXIF orientation and return BGR pixels."""

    if max_upload_mb < 1 or max_side < 1:
        raise ValueError("Upload and image-side limits must be positive")
    byte_limit = (
        max_upload_mb * 1024 * 1024
        if max_upload_bytes is None
        else max_upload_bytes
    )
    if byte_limit < 1:
        raise ValueError("max_upload_bytes must be positive")

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise ImageValidationError("圖片資料類型無效")
    payload = bytes(data)
    if not payload:
        raise ImageValidationError("上傳的圖片是空檔案")
    if len(payload) > byte_limit:
        raise ImageTooLargeError(
            f"圖片超過上傳限制（最多 {byte_limit // (1024 * 1024)} MB）"
        )

    pixel_limit = _decoded_pixel_limit(max_side, max_decoded_pixels)
    image_format: str | None = None
    original_size: tuple[int, int] | None = None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as probe:
                image_format = (probe.format or "").upper()
                if image_format not in ALLOWED_IMAGE_FORMATS:
                    raise UnsupportedImageError(
                        "僅接受實際內容為 JPG、JPEG、PNG 或 WEBP 的圖片"
                    )
                if getattr(probe, "n_frames", 1) != 1:
                    raise UnsupportedImageError("不接受動態圖片")
                original_size = (int(probe.width), int(probe.height))
                _validate_dimensions(*original_size, pixel_limit)
                probe.verify()

            with Image.open(io.BytesIO(payload)) as opened:
                if (opened.format or "").upper() != image_format:
                    raise ImageValidationError("圖片格式驗證失敗")
                _validate_dimensions(int(opened.width), int(opened.height), pixel_limit)
                oriented = ImageOps.exif_transpose(opened)
                rgb = _pil_to_rgb(oriented)
                rgb.load()

                width, height = rgb.size
                scale = min(1.0, max_side / float(max(width, height)))
                was_resized = scale < 1.0
                if was_resized:
                    resized_size = (
                        max(1, round(width * scale)),
                        max(1, round(height * scale)),
                    )
                    resampling = getattr(Image, "Resampling", Image).LANCZOS
                    rgb = rgb.resize(resized_size, resampling)

                rgb_array = np.asarray(rgb, dtype=np.uint8)
                bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                bgr = np.ascontiguousarray(bgr)
    except (ImageTooLargeError, UnsupportedImageError, ImageValidationError):
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError("圖片解碼尺寸過大") from exc
    except (UnidentifiedImageError, SyntaxError, OSError, ValueError) as exc:
        raise ImageValidationError("圖片內容損壞或格式不受支援") from exc
    except MemoryError as exc:
        raise ImageTooLargeError("圖片解碼需要過多記憶體") from exc

    if image_format is None or original_size is None:
        raise ImageValidationError("無法讀取圖片")
    processed_size = (int(bgr.shape[1]), int(bgr.shape[0]))
    return DecodedImage(
        bgr=bgr,
        format=image_format,
        original_size=original_size,
        processed_size=processed_size,
        was_resized=was_resized,
    )


def _as_bgr_uint8(image_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        raise ImageValidationError("圖片陣列無效")
    image = image_bgr
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ImageValidationError("圖片陣列必須是灰階、BGR 或 BGRA")
    if image.dtype != np.uint8:
        if not np.issubdtype(image.dtype, np.number):
            raise ImageValidationError("圖片陣列資料類型無效")
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def make_thumbnail(
    image_bgr: np.ndarray,
    max_size: int = 320,
    *,
    square: bool = False,
    background_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    image = _as_bgr_uint8(image_bgr)
    if max_size < 1:
        raise ValueError("max_size must be positive")
    height, width = image.shape[:2]
    scale = min(max_size / float(width), max_size / float(height), 1.0)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, size, interpolation=interpolation)
    if not square:
        return resized
    canvas = np.full((max_size, max_size, 3), background_color, dtype=np.uint8)
    x = (max_size - resized.shape[1]) // 2
    y = (max_size - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def face_thumbnail(
    image_bgr: np.ndarray,
    bbox: Iterable[float | int],
    size: int = 160,
    *,
    padding: float = 0.28,
) -> np.ndarray:
    image = _as_bgr_uint8(image_bgr)
    coords = np.asarray(tuple(bbox), dtype=np.float32)
    if coords.shape != (4,) or not np.all(np.isfinite(coords)):
        raise ImageValidationError("人臉位置無效")
    if size < 1 or padding < 0:
        raise ValueError("Thumbnail size and padding must be valid")

    x1, y1, x2, y2 = coords.tolist()
    if x2 <= x1 or y2 <= y1:
        raise ImageValidationError("人臉位置無效")
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + padding * 2.0)
    left = max(0, int(np.floor(center_x - side / 2.0)))
    top = max(0, int(np.floor(center_y - side / 2.0)))
    right = min(image.shape[1], int(np.ceil(center_x + side / 2.0)))
    bottom = min(image.shape[0], int(np.ceil(center_y + side / 2.0)))
    if right <= left or bottom <= top:
        raise ImageValidationError("人臉位置超出圖片範圍")
    return make_thumbnail(image[top:bottom, left:right], size, square=True)


def align_face(
    image_bgr: np.ndarray,
    landmarks: np.ndarray | Iterable[Iterable[float]],
    image_size: int = 112,
) -> np.ndarray:
    """Align five facial landmarks to the ArcFace 112-pixel reference layout."""

    image = _as_bgr_uint8(image_bgr)
    points = np.asarray(landmarks, dtype=np.float32)
    if points.shape != (5, 2) or not np.all(np.isfinite(points)):
        raise ImageValidationError("人臉五點座標無效")
    if image_size < 64 or image_size > 1024:
        raise ValueError("image_size must be between 64 and 1024")

    reference = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    reference *= image_size / 112.0
    transform, _ = cv2.estimateAffinePartial2D(points, reference, method=cv2.LMEDS)
    if transform is None or not np.all(np.isfinite(transform)):
        raise ImageValidationError("無法對齊人臉")
    return cv2.warpAffine(
        image,
        transform,
        (image_size, image_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = None
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise ImageStorageError(f"檔案已存在，拒絕覆寫：{path.name}") from exc
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ImageStorageError(f"無法安全寫入檔案：{path.name}") from exc


def _destination_jpeg_path(
    destination: str | Path,
    stem: str | None,
) -> Path:
    destination_path = Path(destination)
    if stem is not None:
        if not _SAFE_STEM_RE.fullmatch(stem):
            raise ImageStorageError("輸出檔名不安全")
        return destination_path / f"{stem}.jpg"
    if destination_path.suffix:
        if destination_path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ImageStorageError("JPEG 輸出路徑必須使用 .jpg 或 .jpeg")
        return destination_path
    return destination_path / f"{uuid.uuid4().hex}.jpg"


def save_jpeg(
    image_bgr: np.ndarray,
    destination: str | Path,
    quality: int = 95,
    *,
    stem: str | None = None,
) -> Path:
    """Encode without metadata and create a new JPEG without overwriting files."""

    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    image = _as_bgr_uint8(image_bgr)
    ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    )
    if not ok:
        raise ImageStorageError("JPEG 編碼失敗")
    path = _destination_jpeg_path(destination, stem)
    _write_exclusive(path, encoded.tobytes())
    return path


def save_embedding(
    embedding: np.ndarray | Iterable[float],
    path: str | Path,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".npy":
        raise ImageStorageError("Embedding 路徑必須使用 .npy")
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != 512 or not np.all(np.isfinite(vector)):
        raise ImageStorageError("Embedding 必須是有效的 512 維向量")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ImageStorageError("Embedding 不可為零向量")
    vector = np.ascontiguousarray(vector / norm, dtype=np.float32)
    buffer = io.BytesIO()
    np.save(buffer, vector, allow_pickle=False)
    _write_exclusive(destination, buffer.getvalue())
    return destination


def load_embedding(path: str | Path) -> np.ndarray:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ImageStorageError("找不到已儲存的人臉特徵") from exc
    if size < 1 or size > MAX_EMBEDDING_FILE_BYTES:
        raise ImageStorageError("Embedding 檔案大小異常")
    try:
        with source.open("rb") as handle:
            vector = np.load(handle, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ImageStorageError("Embedding 檔案損壞") from exc
    vector = np.asarray(vector, dtype=np.float32)
    if vector.shape != (512,) or not np.all(np.isfinite(vector)):
        raise ImageStorageError("Embedding 格式無效")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        raise ImageStorageError("Embedding 不可為零向量")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)
