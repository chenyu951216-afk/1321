from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
import re
import secrets
import shutil
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .cleanup import FileLeaseRegistry, cleanup_loop
from .config import settings
from .database import Database, FaceRecord
from .face_engine import (
    FaceEngineError,
    FaceNotFoundError,
    InvalidFaceIndexError,
    ModelNotReadyError,
    get_face_engine,
)
from .image_utils import (
    ImageStorageError,
    ImageTooLargeError,
    ImageValidationError,
    decode_image_bytes,
    load_embedding,
    save_embedding,
    save_jpeg,
)
from .schemas import (
    DetectionFaceResponse,
    FaceListResponse,
    FaceResponse,
    HealthResponse,
    RenameFaceRequest,
    SwapResponse,
    TargetDetectionResponse,
)


logger = logging.getLogger(__name__)
APP_DIR = Path(__file__).resolve().parent
TOKEN_PATTERN = re.compile(r"^[0-9a-f]{32}$")
RESULT_PATTERN = re.compile(r"^[0-9a-f]{32}\.jpg$")
TEMP_THUMBNAIL_PATTERN = re.compile(r"^[0-9a-f]{32}_face_[0-9]+\.jpg$")

database = Database(settings.database_path, settings.database_timeout_seconds)
face_engine = get_face_engine(
    model_dir=settings.model_dir,
    detection_size=settings.face_detection_size,
)


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        detail: str,
        code: str,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.extra = extra or {}


class RequestBodyTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    def __init__(self, asgi_app: Any, max_bytes: int) -> None:
        self.asgi_app = asgi_app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") not in {
            "POST",
            "PUT",
            "PATCH",
        }:
            await self.asgi_app(scope, receive, send)
            return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.asgi_app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            response = JSONResponse(
                status_code=413,
                content={
                    "detail": f"圖片過大，單張上限為 {settings.max_upload_mb}MB",
                    "code": "image_too_large",
                },
                headers={"X-Content-Type-Options": "nosniff"},
            )
            await response(scope, receive, send)


class LeasedFileResponse(FileResponse):
    def __init__(
        self,
        leases: FileLeaseRegistry,
        path: Path,
        **kwargs: Any,
    ) -> None:
        self._leases = leases
        self._leased_path = leases.acquire(path)
        try:
            if not self._leased_path.is_file():
                raise ApiError(404, "檔案不存在或已逾期清除", "file_not_found")
            super().__init__(self._leased_path, **kwargs)
        except Exception:
            leases.release(self._leased_path)
            raise

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._leases.release(self._leased_path)


class _FaceLockEntry:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.references = 0


class FaceLockLease:
    def __init__(
        self,
        registry: "FaceLockRegistry",
        face_id: str,
        entry: _FaceLockEntry,
    ) -> None:
        self._registry = registry
        self._face_id = face_id
        self._entry = entry
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._registry._release(self._face_id, self._entry)
        self._released = True


class FaceLockRegistry:
    """Keep one lock per in-use face ID and discard it after the last user."""

    def __init__(self) -> None:
        self._entries: dict[str, _FaceLockEntry] = {}

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    async def acquire(self, face_id: str) -> FaceLockLease:
        entry = self._entries.get(face_id)
        if entry is None:
            entry = _FaceLockEntry()
            self._entries[face_id] = entry
        entry.references += 1
        try:
            await entry.lock.acquire()
        except BaseException:
            self._drop_reference(face_id, entry)
            raise
        return FaceLockLease(self, face_id, entry)

    @asynccontextmanager
    async def hold(self, face_id: str):
        lease = await self.acquire(face_id)
        try:
            yield
        finally:
            lease.release()

    def _release(self, face_id: str, entry: _FaceLockEntry) -> None:
        if self._entries.get(face_id) is not entry or not entry.lock.locked():
            raise RuntimeError("Face lock lease is no longer valid")
        entry.lock.release()
        self._drop_reference(face_id, entry)

    def _drop_reference(self, face_id: str, entry: _FaceLockEntry) -> None:
        if entry.references < 1:
            raise RuntimeError("Face lock reference count is invalid")
        entry.references -= 1
        if entry.references == 0 and self._entries.get(face_id) is entry:
            self._entries.pop(face_id, None)


class LockedFaceFileResponse(FileResponse):
    def __init__(
        self,
        face_lock_lease: FaceLockLease,
        path: Path,
        **kwargs: Any,
    ) -> None:
        self._face_lock_lease = face_lock_lease
        if not path.is_file():
            raise ApiError(404, "人臉縮圖不存在", "file_not_found")
        super().__init__(path, **kwargs)

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._face_lock_lease.release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_directories()
    database.initialize()
    app.state.inference_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
    app.state.cleanup_leases = FileLeaseRegistry()
    app.state.face_locks = FaceLockRegistry()

    await asyncio.to_thread(face_engine.initialize)
    if not face_engine.is_ready:
        logger.error("人臉模型未就緒：%s", face_engine.error or "未知錯誤")

    cleanup_task = asyncio.create_task(
        cleanup_loop(
            settings,
            app.state.cleanup_leases,
            _remove_orphan_face_directories,
        ),
        name="file-cleanup",
    )
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(
    title="照片換臉",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    payload: dict[str, Any] = {"detail": exc.detail, "code": exc.code}
    payload.update(exc.extra)
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.exception_handler(sqlite3.Error)
async def database_error_handler(_: Request, exc: sqlite3.Error) -> JSONResponse:
    logger.exception("SQLite 操作失敗", exc_info=exc)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "人臉資料庫暫時無法使用，請稍後再試",
            "code": "database_error",
        },
    )


@app.exception_handler(ImageStorageError)
async def storage_error_handler(_: Request, exc: ImageStorageError) -> JSONResponse:
    logger.exception("圖片或人臉特徵儲存失敗", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "圖片或人臉資料無法讀寫，請確認檔案與磁碟空間",
            "code": "storage_error",
        },
    )


@app.middleware("http")
async def security_and_size_middleware(request: Request, call_next: Callable[..., Any]):
    if settings.app_username and request.url.path != "/api/health":
        authorization = request.headers.get("authorization", "")
        authenticated = False
        if authorization.lower().startswith("basic "):
            try:
                encoded = authorization.split(None, 1)[1]
                decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
                supplied_username, separator, supplied_password = decoded.partition(":")
                authenticated = bool(separator) and secrets.compare_digest(
                    supplied_username.encode("utf-8"), settings.app_username.encode("utf-8")
                ) and secrets.compare_digest(
                    supplied_password.encode("utf-8"), settings.app_password.encode("utf-8")
                )
            except (binascii.Error, IndexError, UnicodeDecodeError, ValueError):
                authenticated = False

        if not authenticated:
            return JSONResponse(
                status_code=401,
                content={"detail": "需要登入", "code": "authentication_required"},
                headers={"WWW-Authenticate": 'Basic realm="Faceswap", charset="UTF-8"'},
            )

        if (
            request.method not in {"GET", "HEAD", "OPTIONS"}
            and request.headers.get("x-requested-with") != "faceswap-web"
        ):
            return JSONResponse(
                status_code=403,
                content={"detail": "請求來源驗證失敗", "code": "request_verification_failed"},
            )

    if request.method in {"POST", "PUT", "PATCH"}:
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_bytes = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Content-Length 格式錯誤", "code": "invalid_request"},
                )
            multipart_allowance = 1024 * 1024
            if request_bytes > settings.max_upload_bytes + multipart_allowance:
                return JSONResponse(
                    status_code=413,
                    content={
                        "detail": f"圖片過大，單張上限為 {settings.max_upload_mb}MB",
                        "code": "image_too_large",
                    },
                )

    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; "
        "script-src 'self'; object-src 'none'; base-uri 'self'; "
        "frame-ancestors 'none'; form-action 'self'"
    )
    return response


app.add_middleware(
    RequestSizeLimitMiddleware,
    max_bytes=settings.max_upload_bytes + 1024 * 1024,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean_name(name: str) -> str:
    cleaned = " ".join(name.strip().split())
    if not cleaned:
        raise ApiError(422, "請輸入人臉名稱", "invalid_name")
    if len(cleaned) > 80:
        raise ApiError(422, "人臉名稱不可超過 80 個字元", "invalid_name")
    if any(ord(character) < 32 for character in cleaned):
        raise ApiError(422, "人臉名稱含有不允許的控制字元", "invalid_name")
    return cleaned


def _face_response(record: FaceRecord) -> FaceResponse:
    return FaceResponse(
        id=record.id,
        name=record.name,
        thumbnail_url=f"/api/faces/{record.id}/thumbnail",
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _face_locks(request: Request) -> FaceLockRegistry:
    return request.app.state.face_locks


def _ensure_model_ready() -> None:
    if not face_engine.is_ready:
        detail = face_engine.error or "模型尚未準備完成"
        raise ApiError(503, f"模型尚未準備完成：{detail}", "model_not_ready")


def _engine_api_error(exc: FaceEngineError, action: str) -> ApiError:
    if isinstance(exc, ModelNotReadyError):
        return ApiError(503, str(exc), "model_not_ready")
    if isinstance(exc, FaceNotFoundError):
        return ApiError(422, "沒有偵測到臉，請換一張清楚的人像照片", "face_not_found")
    if isinstance(exc, InvalidFaceIndexError):
        return ApiError(422, str(exc), "invalid_face_index")
    logger.exception("%s失敗", action, exc_info=exc)
    return ApiError(500, f"{action}失敗，請稍後再試", "processing_failed")


async def _read_upload(upload: UploadFile) -> bytes:
    data = bytearray()
    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > settings.max_upload_bytes:
                raise ApiError(
                    413,
                    f"圖片過大，單張上限為 {settings.max_upload_mb}MB",
                    "image_too_large",
                )
    finally:
        await upload.close()

    if not data:
        raise ApiError(400, "上傳的圖片是空檔案", "empty_image")
    return bytes(data)


async def _run_inference(
    request: Request,
    function: Callable[..., Any],
    *args: Any,
) -> Any:
    semaphore: asyncio.Semaphore = request.app.state.inference_semaphore
    try:
        await asyncio.wait_for(
            semaphore.acquire(), timeout=settings.job_queue_timeout_seconds
        )
    except TimeoutError as exc:
        raise ApiError(429, "目前處理工作較多，請稍後再試", "server_busy") from exc

    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(worker)
        raise
    finally:
        semaphore.release()


async def _run_thread_to_completion(
    function: Callable[..., Any],
    *args: Any,
) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        with suppress(Exception):
            await asyncio.shield(worker)
        raise


def _safe_path(root: Path, path: str | Path, must_exist: bool = True) -> Path:
    root_resolved = root.resolve()
    candidate = Path(path).resolve()
    if not candidate.is_relative_to(root_resolved):
        raise ApiError(500, "儲存路徑驗證失敗", "unsafe_storage_path")
    if must_exist and not candidate.is_file():
        raise ApiError(404, "檔案不存在或已逾期清除", "file_not_found")
    return candidate


def _remove_temp_token(token: str) -> None:
    if not TOKEN_PATTERN.fullmatch(token):
        return
    candidates = [settings.temp_dir / f"{token}.jpg"]
    candidates.extend(settings.temp_dir.glob(f"{token}_face_*.jpg"))
    for candidate in candidates:
        try:
            safe_candidate = _safe_path(settings.temp_dir, candidate, must_exist=False)
            safe_candidate.unlink(missing_ok=True)
        except (ApiError, OSError):
            logger.warning("無法移除暫存檔：%s", candidate, exc_info=True)


def _protect_cleanup_path(request: Request, path: Path) -> Path:
    leases: FileLeaseRegistry = request.app.state.cleanup_leases
    return leases.acquire(path)


def _release_cleanup_path(app_instance: FastAPI, path: Path) -> None:
    leases: FileLeaseRegistry = app_instance.state.cleanup_leases
    leases.release(path)


def _create_face_assets(data: bytes, face_id: str) -> tuple[Path, Path, Path]:
    decoded = decode_image_bytes(
        data,
        max_upload_mb=settings.max_upload_mb,
        max_side=settings.max_image_side,
    )
    aligned_bgr, embedding = face_engine.extract_source(decoded.bgr)
    face_directory = settings.faces_dir / face_id
    face_directory.mkdir(parents=False, exist_ok=False)
    try:
        original_path = save_jpeg(
            decoded.bgr,
            face_directory,
            quality=settings.jpeg_quality,
            stem="original",
        )
        aligned_path = save_jpeg(
            aligned_bgr,
            face_directory,
            quality=settings.jpeg_quality,
            stem="aligned",
        )
        embedding_path = face_directory / "embedding.npy"
        save_embedding(embedding, embedding_path)
        return original_path, aligned_path, embedding_path
    except Exception:
        shutil.rmtree(face_directory, ignore_errors=True)
        raise


def _prepare_target(data: bytes, token: str, include_thumbnails: bool) -> tuple[Path, list[Any]]:
    decoded = decode_image_bytes(
        data,
        max_upload_mb=settings.max_upload_mb,
        max_side=settings.max_image_side,
    )
    try:
        target_path = save_jpeg(
            decoded.bgr,
            settings.temp_dir,
            quality=settings.jpeg_quality,
            stem=token,
        )
        stored_bgr = _load_target_image(target_path)
        detections = face_engine.detect(stored_bgr)
        if not detections:
            raise FaceNotFoundError("沒有偵測到臉")
        if include_thumbnails:
            for detection in detections:
                save_jpeg(
                    detection.thumbnail_bgr,
                    settings.temp_dir,
                    quality=90,
                    stem=f"{token}_face_{detection.index}",
                )
        return target_path, detections
    except Exception:
        _remove_temp_token(token)
        raise


def _load_target_image(target_path: Path) -> Any:
    image = cv2.imread(str(target_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageValidationError("目標圖片已損毀或無法讀取")
    return image


def _perform_swap(
    target_path: Path,
    source_embedding: Any,
    target_face_index: int | None,
    restore_face: bool,
    restore_strength: float,
) -> tuple[Path, float]:
    started_at = time.perf_counter()
    target_bgr = _load_target_image(target_path)
    result_bgr = face_engine.swap(
        target_bgr,
        source_embedding,
        target_face_index=target_face_index,
        restore_face=restore_face,
        restore_strength=restore_strength,
    )
    result_path = save_jpeg(
        result_bgr,
        settings.results_dir,
        quality=settings.jpeg_quality,
        stem=uuid.uuid4().hex,
    )
    return result_path, time.perf_counter() - started_at


def _delete_face_assets(record: FaceRecord) -> None:
    parents: set[Path] = set()
    for stored_path in (
        record.original_image_path,
        record.aligned_image_path,
        record.embedding_path,
    ):
        try:
            path = _safe_path(settings.faces_dir, stored_path, must_exist=False)
            parents.add(path.parent)
            path.unlink(missing_ok=True)
        except (ApiError, OSError):
            logger.warning("無法刪除人臉檔案：%s", stored_path, exc_info=True)
    for parent in parents:
        try:
            parent.rmdir()
        except OSError:
            logger.warning("無法刪除人臉資料夾：%s", parent, exc_info=True)


def _remove_orphan_face_directories() -> int:
    valid_ids = {record.id for record in database.list_faces()}
    cutoff = time.time() - 3600
    removed = 0
    if not settings.faces_dir.exists():
        return removed
    for path in settings.faces_dir.iterdir():
        try:
            if (
                path.is_dir()
                and TOKEN_PATTERN.fullmatch(path.name)
                and path.name not in valid_ids
                and path.stat().st_mtime < cutoff
            ):
                shutil.rmtree(path)
                removed += 1
        except OSError:
            logger.warning("無法清理孤立的人臉資料夾：%s", path, exc_info=True)
    return removed


def _create_face_transaction(data: bytes, face_id: str, name: str) -> FaceRecord:
    original_path, aligned_path, embedding_path = _create_face_assets(data, face_id)
    timestamp = _utc_now()
    record = FaceRecord(
        id=face_id,
        name=name,
        original_image_path=str(original_path),
        aligned_image_path=str(aligned_path),
        embedding_path=str(embedding_path),
        created_at=timestamp,
        updated_at=timestamp,
    )
    try:
        database.create_face(record)
    except Exception:
        _delete_face_assets(record)
        raise
    return record


def _delete_face_transaction(face_id: str) -> FaceRecord | None:
    record = database.get_face(face_id)
    if record is None:
        return None
    if not TOKEN_PATTERN.fullmatch(record.id):
        raise ImageStorageError("人臉識別碼格式無效，已拒絕刪除檔案")

    paths = [
        _safe_path(settings.faces_dir, stored_path, must_exist=False)
        for stored_path in (
            record.original_image_path,
            record.aligned_image_path,
            record.embedding_path,
        )
    ]
    parents = {path.parent for path in paths}
    if len(parents) != 1:
        raise ImageStorageError("人臉資產路徑不一致，已拒絕刪除")
    face_directory = parents.pop()
    if face_directory.parent != settings.faces_dir.resolve() or face_directory.name != record.id:
        raise ImageStorageError("人臉資料夾路徑驗證失敗，已拒絕刪除")
    if not face_directory.is_dir():
        raise ImageStorageError("找不到完整的人臉資產，資料庫記錄未刪除")

    tombstone = settings.faces_dir / f".deleting-{record.id}-{uuid.uuid4().hex}"
    try:
        os.replace(face_directory, tombstone)
    except OSError as exc:
        raise ImageStorageError("無法鎖定待刪除的人臉資產") from exc

    try:
        deleted = database.delete_face(face_id)
    except Exception:
        try:
            os.replace(tombstone, face_directory)
        except OSError as restore_error:
            raise ImageStorageError("資料庫刪除失敗，且無法還原人臉資產") from restore_error
        raise

    if deleted is None:
        os.replace(tombstone, face_directory)
        return None

    try:
        shutil.rmtree(tombstone)
    except OSError as exc:
        raise ImageStorageError(
            "人臉已從資料庫移除，但檔案清理尚未完成；背景工作將重試"
        ) from exc
    return deleted


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "max_upload_mb": settings.max_upload_mb,
            "max_image_side": settings.max_image_side,
            "result_retention_hours": settings.result_retention_hours,
        },
    )


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if face_engine.is_ready else "degraded",
        model_ready=face_engine.is_ready,
        model_status=face_engine.status,
        provider=face_engine.provider,
        model_error=face_engine.error,
    )


@app.get("/api/faces", response_model=FaceListResponse)
async def list_faces() -> FaceListResponse:
    records = await asyncio.to_thread(database.list_faces)
    return FaceListResponse(faces=[_face_response(item) for item in records])


@app.post("/api/faces", response_model=FaceResponse, status_code=201)
async def create_face(
    request: Request,
    name: str = Form(...),
    image: UploadFile = File(...),
) -> FaceResponse:
    _ensure_model_ready()
    clean_name = _clean_name(name)
    data = await _read_upload(image)
    face_id = uuid.uuid4().hex
    try:
        record = await _run_inference(
            request, _create_face_transaction, data, face_id, clean_name
        )
    except ImageTooLargeError as exc:
        raise ApiError(413, str(exc), "image_too_large") from exc
    except ImageValidationError as exc:
        raise ApiError(415, str(exc), "invalid_image") from exc
    except FaceEngineError as exc:
        raise _engine_api_error(exc, "建立人臉") from exc
    except FileExistsError as exc:
        raise ApiError(409, "人臉識別碼衝突，請重試", "storage_conflict") from exc
    except (ImageStorageError, OSError) as exc:
        logger.exception("儲存人臉檔案失敗")
        raise ApiError(500, "無法儲存人臉資料，請確認磁碟空間", "storage_error") from exc

    except sqlite3.Error as exc:
        logger.exception("寫入人臉資料庫失敗")
        raise ApiError(503, "人臉資料庫暫時無法寫入，請稍後再試", "database_error") from exc
    return _face_response(record)


@app.patch("/api/faces/{face_id}", response_model=FaceResponse)
async def rename_face(
    request: Request,
    face_id: str,
    payload: RenameFaceRequest,
) -> FaceResponse:
    clean_name = _clean_name(payload.name)
    async with _face_locks(request).hold(face_id):
        try:
            record = await _run_thread_to_completion(
                database.rename_face, face_id, clean_name, _utc_now()
            )
        except sqlite3.Error as exc:
            raise ApiError(503, "人臉資料庫暫時無法寫入，請稍後再試", "database_error") from exc
    if record is None:
        raise ApiError(404, "找不到指定的人臉", "face_record_not_found")
    return _face_response(record)


@app.delete("/api/faces/{face_id}", status_code=204)
async def delete_face(request: Request, face_id: str) -> Response:
    async with _face_locks(request).hold(face_id):
        try:
            record = await _run_thread_to_completion(_delete_face_transaction, face_id)
        except sqlite3.Error as exc:
            raise ApiError(503, "人臉資料庫暫時無法寫入，請稍後再試", "database_error") from exc
    if record is None:
        raise ApiError(404, "找不到指定的人臉", "face_record_not_found")
    return Response(status_code=204)


@app.get("/api/faces/{face_id}/thumbnail", include_in_schema=False)
async def get_face_thumbnail(request: Request, face_id: str) -> FileResponse:
    face_lock_lease = await _face_locks(request).acquire(face_id)
    try:
        record = await asyncio.to_thread(database.get_face, face_id)
        if record is None:
            raise ApiError(404, "找不到指定的人臉", "face_record_not_found")
        path = _safe_path(settings.faces_dir, record.aligned_image_path)
        return LockedFaceFileResponse(
            face_lock_lease,
            path,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=3600"},
        )
    except Exception:
        face_lock_lease.release()
        raise


@app.post("/api/detect-target-faces", response_model=TargetDetectionResponse)
async def detect_target_faces(
    request: Request,
    image: UploadFile = File(...),
) -> TargetDetectionResponse:
    _ensure_model_ready()
    data = await _read_upload(image)
    token = uuid.uuid4().hex
    try:
        _, detections = await _run_inference(
            request, _prepare_target, data, token, True
        )
    except ImageTooLargeError as exc:
        raise ApiError(413, str(exc), "image_too_large") from exc
    except ImageValidationError as exc:
        raise ApiError(415, str(exc), "invalid_image") from exc
    except FaceEngineError as exc:
        raise _engine_api_error(exc, "偵測人臉") from exc
    except (ImageStorageError, OSError) as exc:
        logger.exception("儲存目標圖片失敗")
        raise ApiError(500, "無法暫存目標圖片，請確認磁碟空間", "storage_error") from exc

    faces = [
        DetectionFaceResponse(
            index=item.index,
            bbox=list(item.bbox),
            thumbnail_url=f"/api/temp/{token}_face_{item.index}.jpg",
        )
        for item in detections
    ]
    return TargetDetectionResponse(token=token, face_count=len(faces), faces=faces)


@app.get("/api/temp/{filename}", include_in_schema=False)
async def get_temp_thumbnail(request: Request, filename: str) -> FileResponse:
    if not TEMP_THUMBNAIL_PATTERN.fullmatch(filename):
        raise ApiError(404, "找不到暫存縮圖", "file_not_found")
    path = _safe_path(
        settings.temp_dir, settings.temp_dir / filename, must_exist=False
    )
    return LeasedFileResponse(
        request.app.state.cleanup_leases,
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, no-store, max-age=0"},
    )


@app.post("/api/swap", response_model=SwapResponse)
async def swap_face(
    request: Request,
    face_id: str = Form(...),
    target_token: str | None = Form(None),
    target_image: UploadFile | None = File(None),
    target_face_index: int | None = Form(None),
    restore_face: bool = Form(False),
    restore_strength: float = Form(0.35),
) -> SwapResponse:
    _ensure_model_ready()
    if target_token and target_image is not None:
        await target_image.close()
        raise ApiError(400, "target_token 與 target_image 只能擇一提供", "invalid_request")
    if not target_token and target_image is None:
        raise ApiError(400, "請提供目標圖片或暫存圖片 token", "missing_target")
    if not 0.0 <= restore_strength <= 1.0:
        if target_image is not None:
            await target_image.close()
        raise ApiError(422, "人臉修復強度必須介於 0 與 1", "invalid_restore_strength")

    async with _face_locks(request).hold(face_id):
        record = await asyncio.to_thread(database.get_face, face_id)
        if record is None:
            if target_image is not None:
                await target_image.close()
            raise ApiError(404, "找不到指定的來源人臉", "face_record_not_found")
        embedding_path = _safe_path(settings.faces_dir, record.embedding_path)
        source_embedding = await _run_thread_to_completion(
            load_embedding, embedding_path
        )

    created_target = False
    token = target_token or uuid.uuid4().hex
    if not TOKEN_PATTERN.fullmatch(token):
        if target_image is not None:
            await target_image.close()
        raise ApiError(400, "目標圖片 token 格式錯誤", "invalid_target_token")

    if target_image is not None:
        data = await _read_upload(target_image)
        created_target = True
        try:
            target_path, _ = await _run_inference(
                request, _prepare_target, data, token, False
            )
        except ImageTooLargeError as exc:
            raise ApiError(413, str(exc), "image_too_large") from exc
        except ImageValidationError as exc:
            raise ApiError(415, str(exc), "invalid_image") from exc
        except FaceEngineError as exc:
            raise _engine_api_error(exc, "偵測人臉") from exc
        except (ImageStorageError, OSError) as exc:
            raise ApiError(
                500,
                "無法暫存目標圖片，請確認磁碟空間",
                "storage_error",
            ) from exc
        except (ApiError, asyncio.CancelledError):
            _remove_temp_token(token)
            raise
        protected_target = _protect_cleanup_path(request, target_path)
    else:
        target_path = _safe_path(
            settings.temp_dir,
            settings.temp_dir / f"{token}.jpg",
            must_exist=False,
        )
        protected_target = _protect_cleanup_path(request, target_path)
        try:
            os.utime(target_path, None)
        except FileNotFoundError as exc:
            _release_cleanup_path(request.app, protected_target)
            raise ApiError(404, "暫存圖片不存在或已逾期清除", "file_not_found") from exc
        except OSError as exc:
            _release_cleanup_path(request.app, protected_target)
            raise ApiError(500, "無法讀取暫存圖片", "storage_error") from exc

    swap_completed = False
    try:
        try:
            result_path, elapsed = await _run_inference(
                request,
                _perform_swap,
                target_path,
                source_embedding,
                target_face_index,
                restore_face,
                restore_strength,
            )
            swap_completed = True
        except ImageValidationError as exc:
            raise ApiError(415, str(exc), "invalid_image") from exc
        except FaceEngineError as exc:
            raise _engine_api_error(exc, "換臉") from exc
        except (ImageStorageError, OSError) as exc:
            logger.exception("讀寫換臉檔案失敗")
            raise ApiError(500, "無法讀寫換臉檔案，請確認磁碟空間", "storage_error") from exc
    finally:
        _release_cleanup_path(request.app, protected_target)
        if created_target and not swap_completed:
            _remove_temp_token(token)

    filename = result_path.name
    elapsed_rounded = round(elapsed, 3)
    return SwapResponse(
        result_url=f"/api/results/{filename}",
        download_url=f"/api/results/{filename}?download=true",
        processing_time=elapsed_rounded,
        processing_time_ms=round(elapsed * 1000),
    )


@app.get("/api/results/{filename}")
async def get_result(request: Request, filename: str, download: bool = False) -> FileResponse:
    if not RESULT_PATTERN.fullmatch(filename):
        raise ApiError(404, "找不到結果圖片", "file_not_found")
    path = _safe_path(
        settings.results_dir, settings.results_dir / filename, must_exist=False
    )
    return LeasedFileResponse(
        request.app.state.cleanup_leases,
        path,
        media_type="image/jpeg",
        filename=f"換臉結果-{filename}" if download else None,
        headers={
            "Cache-Control": "private, no-store, max-age=0",
            "Pragma": "no-cache",
        },
    )
