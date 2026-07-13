from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _read_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"環境變數 {name} 必須是整數") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"環境變數 {name} 必須介於 {minimum} 與 {maximum} 之間")
    return value


def _read_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"環境變數 {name} 必須是數字") from exc
    if not minimum <= value <= maximum:
        raise RuntimeError(f"環境變數 {name} 必須介於 {minimum} 與 {maximum} 之間")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    port: int
    data_dir: Path
    model_dir: Path
    max_upload_mb: int
    max_image_side: int
    temp_retention_hours: float
    result_retention_hours: float
    max_concurrent_jobs: int
    face_detection_size: int
    jpeg_quality: int
    database_timeout_seconds: float
    cleanup_interval_seconds: int
    job_queue_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            port=_read_int("PORT", 8080, 1, 65535),
            data_dir=Path(os.getenv("DATA_DIR", "/data")).expanduser().resolve(),
            model_dir=Path(os.getenv("MODEL_DIR", "/models")).expanduser().resolve(),
            max_upload_mb=_read_int("MAX_UPLOAD_MB", 15, 1, 100),
            max_image_side=_read_int("MAX_IMAGE_SIDE", 2500, 256, 10000),
            temp_retention_hours=_read_float("TEMP_RETENTION_HOURS", 24.0, 0.1, 720.0),
            result_retention_hours=_read_float("RESULT_RETENTION_HOURS", 24.0, 0.1, 720.0),
            max_concurrent_jobs=_read_int("MAX_CONCURRENT_JOBS", 1, 1, 2),
            face_detection_size=_read_int("FACE_DETECTION_SIZE", 640, 320, 1280),
            jpeg_quality=_read_int("JPEG_QUALITY", 95, 70, 100),
            database_timeout_seconds=_read_float("DATABASE_TIMEOUT_SECONDS", 30.0, 1.0, 120.0),
            cleanup_interval_seconds=_read_int("CLEANUP_INTERVAL_SECONDS", 3600, 60, 86400),
            job_queue_timeout_seconds=_read_float("JOB_QUEUE_TIMEOUT_SECONDS", 30.0, 1.0, 600.0),
        )

    @property
    def faces_dir(self) -> Path:
        return self.data_dir / "faces"

    @property
    def temp_dir(self) -> Path:
        return self.data_dir / "temp"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def swapper_model_path(self) -> Path:
        return self.model_dir / "inswapper_128.onnx"

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.faces_dir,
            self.temp_dir,
            self.results_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings.from_env()

