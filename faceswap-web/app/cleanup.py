from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from .config import Settings


logger = logging.getLogger(__name__)


def remove_expired_files(directory: Path, retention_hours: float) -> int:
    cutoff = time.time() - retention_hours * 3600
    removed = 0
    if not directory.exists():
        return removed

    for path in directory.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
        except OSError:
            logger.warning("無法清理逾期檔案：%s", path, exc_info=True)
    return removed


def run_cleanup(settings: Settings) -> int:
    return remove_expired_files(
        settings.temp_dir, settings.temp_retention_hours
    ) + remove_expired_files(
        settings.results_dir, settings.result_retention_hours
    )


async def cleanup_loop(settings: Settings) -> None:
    while True:
        try:
            removed = await asyncio.to_thread(run_cleanup, settings)
            if removed:
                logger.info("已清理 %d 個逾期暫存或結果檔案", removed)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("背景清理工作失敗")
        await asyncio.sleep(settings.cleanup_interval_seconds)
