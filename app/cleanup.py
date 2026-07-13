from __future__ import annotations

import asyncio
import logging
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import Settings


logger = logging.getLogger(__name__)


class FileLeaseRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counts: dict[Path, int] = {}

    def acquire(self, path: Path) -> Path:
        resolved = path.resolve()
        with self._lock:
            self._counts[resolved] = self._counts.get(resolved, 0) + 1
        return resolved

    def release(self, path: Path) -> None:
        resolved = path.resolve()
        with self._lock:
            count = self._counts.get(resolved, 0)
            if count <= 1:
                self._counts.pop(resolved, None)
            else:
                self._counts[resolved] = count - 1

    @contextmanager
    def cleanup_path(self, path: Path) -> Iterator[bool]:
        resolved = path.resolve()
        with self._lock:
            yield self._counts.get(resolved, 0) > 0


def remove_expired_files(
    directory: Path,
    retention_hours: float,
    leases: FileLeaseRegistry | None = None,
) -> int:
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    if not directory.exists():
        return removed
    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            if leases is None:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    removed += 1
            else:
                with leases.cleanup_path(path) as protected:
                    if not protected and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                        removed += 1
        except OSError:
            logger.warning("無法清理逾期檔案：%s", path, exc_info=True)
    return removed


def run_cleanup(
    settings: Settings,
    leases: FileLeaseRegistry | None = None,
) -> int:
    return remove_expired_files(
        settings.temp_dir, settings.temp_retention_hours, leases
    ) + remove_expired_files(
        settings.results_dir, settings.result_retention_hours, leases
    )


def remove_stale_face_tombstones(faces_dir: Path, minimum_age_seconds: int = 3600) -> int:
    cutoff = time.time() - minimum_age_seconds
    removed = 0
    if not faces_dir.exists():
        return removed
    for path in faces_dir.glob(".deleting-*"):
        try:
            if path.is_dir() and path.stat().st_mtime < cutoff:
                shutil.rmtree(path)
                removed += 1
        except OSError:
            logger.warning("無法清理待刪除的人臉資料：%s", path, exc_info=True)
    return removed


async def cleanup_loop(
    settings: Settings,
    leases: FileLeaseRegistry | None = None,
    extra_cleanup: Callable[[], int] | None = None,
) -> None:
    while True:
        try:
            removed = await asyncio.to_thread(run_cleanup, settings, leases)
            removed += await asyncio.to_thread(
                remove_stale_face_tombstones, settings.faces_dir
            )
            if extra_cleanup is not None:
                removed += await asyncio.to_thread(extra_cleanup)
            if removed:
                logger.info("已清理 %d 個逾期暫存或結果檔案", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("背景清理工作失敗")
        await asyncio.sleep(settings.cleanup_interval_seconds)
