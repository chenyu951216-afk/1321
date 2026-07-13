from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


OFFICIAL_RELEASE_BASE = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7"
)


@dataclass(frozen=True, slots=True)
class DownloadSpec:
    name: str
    url: str
    size: int
    sha256: str


OFFICIAL_INSWAPPER = DownloadSpec(
    name="inswapper_128.onnx",
    url=f"{OFFICIAL_RELEASE_BASE}/inswapper_128.onnx",
    size=554_253_681,
    sha256="e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af",
)
OFFICIAL_BUFFALO = DownloadSpec(
    name="buffalo_l.zip",
    url=f"{OFFICIAL_RELEASE_BASE}/buffalo_l.zip",
    size=288_621_354,
    sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
)
BUFFALO_FILES = {
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
}
MAX_ARCHIVE_MEMBERS = 100
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 700 * 1024 * 1024


class ModelDownloadError(RuntimeError):
    pass


def _environment_spec(default: DownloadSpec, prefix: str) -> DownloadSpec:
    url = (os.getenv(f"{prefix}_MODEL_URL") or default.url).strip()
    sha256 = (
        os.getenv(f"{prefix}_MODEL_SHA256") or default.sha256
    ).strip().lower()
    raw_size = (os.getenv(f"{prefix}_MODEL_SIZE") or str(default.size)).strip()
    if not url.startswith("https://"):
        raise ModelDownloadError(f"{prefix}_MODEL_URL 必須使用 HTTPS")
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ModelDownloadError(f"{prefix}_MODEL_SHA256 必須是 64 位十六進位 SHA-256")
    try:
        size = int(raw_size)
    except ValueError as exc:
        raise ModelDownloadError(f"{prefix}_MODEL_SIZE 必須是整數") from exc
    if size < 1024:
        raise ModelDownloadError(f"{prefix}_MODEL_SIZE 太小")
    return DownloadSpec(default.name, url, size, sha256)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(path: Path, expected_size: int, expected_sha256: str) -> tuple[bool, str]:
    try:
        actual_size = path.stat().st_size
    except OSError as exc:
        return False, str(exc)
    if actual_size != expected_size:
        return False, f"大小 {actual_size:,}，預期 {expected_size:,} bytes"
    try:
        actual_sha256 = _sha256(path)
    except OSError as exc:
        return False, str(exc)
    if actual_sha256 != expected_sha256:
        return False, f"SHA-256 不符（取得 {actual_sha256}）"
    return True, "ok"


def _download_once(spec: DownloadSpec, destination: Path, timeout: int) -> None:
    request = urllib.request.Request(
        spec.url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "faceswap-web-model-downloader/1.0",
        },
    )
    digest = hashlib.sha256()
    downloaded = 0
    next_progress = 10
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("xb") as output:
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None and int(raw_length) != spec.size:
                raise ModelDownloadError(
                    f"伺服器回報大小 {raw_length}，預期 {spec.size} bytes"
                )
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                downloaded += len(chunk)
                if downloaded > spec.size:
                    raise ModelDownloadError("下載內容超過預期大小，已中止")
                digest.update(chunk)
                output.write(chunk)
                progress = downloaded * 100 // spec.size
                if progress >= next_progress:
                    print(f"  {spec.name}: {progress}%", flush=True)
                    next_progress += 10
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if downloaded != spec.size:
        destination.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"下載不完整：取得 {downloaded:,}，預期 {spec.size:,} bytes"
        )
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != spec.sha256:
        destination.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"SHA-256 驗證失敗：取得 {actual_sha256}，預期 {spec.sha256}"
        )


def _download_verified(spec: DownloadSpec, directory: Path, retries: int, timeout: int) -> Path:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary = directory / f".{spec.name}.{uuid.uuid4().hex}.part"
        print(f"下載 {spec.name}（第 {attempt}/{retries} 次）…", flush=True)
        try:
            _download_once(spec, temporary, timeout)
            return temporary
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            last_error = exc
            if attempt < retries:
                print(f"  下載失敗：{exc}；稍後重試。", file=sys.stderr, flush=True)
                time.sleep(min(2**attempt, 8))
    raise ModelDownloadError(f"{spec.name} 下載失敗：{last_error}")


def _validate_buffalo_directory(directory: Path) -> tuple[bool, str]:
    for filename, expected_hash in BUFFALO_FILES.items():
        path = directory / filename
        try:
            if not path.is_file() or path.stat().st_size < 1024:
                return False, f"缺少 {filename}"
            actual_hash = _sha256(path)
        except OSError as exc:
            return False, str(exc)
        if actual_hash != expected_hash:
            return False, f"{filename} SHA-256 不符"
    return True, "ok"


def _safe_archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = archive.infolist()
    if len(members) > MAX_ARCHIVE_MEMBERS:
        raise ModelDownloadError("buffalo_l 壓縮檔包含過多項目")
    total_size = 0
    selected: dict[str, zipfile.ZipInfo] = {}
    for member in members:
        normalized_name = member.filename.replace("\\", "/")
        pure_path = PurePosixPath(normalized_name)
        if (
            pure_path.is_absolute()
            or ".." in pure_path.parts
            or any(":" in part for part in pure_path.parts)
        ):
            raise ModelDownloadError("buffalo_l 壓縮檔包含不安全路徑")
        unix_mode = member.external_attr >> 16
        if stat.S_ISLNK(unix_mode):
            raise ModelDownloadError("buffalo_l 壓縮檔不可包含符號連結")
        total_size += member.file_size
        if total_size > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
            raise ModelDownloadError("buffalo_l 解壓後大小異常")
        basename = pure_path.name
        if basename in BUFFALO_FILES:
            if basename in selected:
                raise ModelDownloadError(f"buffalo_l 壓縮檔重複包含 {basename}")
            selected[basename] = member
    missing = set(BUFFALO_FILES) - set(selected)
    if missing:
        raise ModelDownloadError(f"buffalo_l 壓縮檔缺少：{', '.join(sorted(missing))}")
    return selected


def _extract_buffalo(archive_path: Path, model_dir: Path) -> Path:
    temporary_dir = Path(tempfile.mkdtemp(prefix=".buffalo_l-", dir=model_dir))
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            selected = _safe_archive_members(archive)
            for filename, member in selected.items():
                output_path = temporary_dir / filename
                digest = hashlib.sha256()
                with archive.open(member, "r") as source, output_path.open("xb") as output:
                    copied = 0
                    while chunk := source.read(1024 * 1024):
                        copied += len(chunk)
                        if copied > member.file_size:
                            raise ModelDownloadError(f"{filename} 解壓大小異常")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if copied != member.file_size or digest.hexdigest() != BUFFALO_FILES[filename]:
                    raise ModelDownloadError(f"{filename} 解壓後校驗失敗")
        valid, reason = _validate_buffalo_directory(temporary_dir)
        if not valid:
            raise ModelDownloadError(f"buffalo_l 模型驗證失敗：{reason}")
        return temporary_dir
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _promote_directory(temporary: Path, target: Path) -> None:
    backup: Path | None = None
    if target.exists():
        backup = target.parent / f".{target.name}.old-{uuid.uuid4().hex}"
        os.replace(target, backup)
    try:
        os.replace(temporary, target)
    except BaseException:
        if backup is not None and backup.exists() and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        if backup.is_dir():
            shutil.rmtree(backup, ignore_errors=True)
        else:
            backup.unlink(missing_ok=True)


def _ensure_inswapper(spec: DownloadSpec, model_dir: Path, force: bool, retries: int, timeout: int) -> None:
    target = model_dir / "inswapper_128.onnx"
    if target.exists() and not force:
        valid, reason = _validate_file(target, spec.size, spec.sha256)
        if valid:
            print("inswapper_128.onnx 已存在且校驗通過，跳過下載。")
            return
        print(f"現有 inswapper_128.onnx 無效（{reason}），將重新下載。", file=sys.stderr)
    temporary = _download_verified(spec, model_dir, retries, timeout)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print("inswapper_128.onnx 安裝完成。")


def _ensure_buffalo(spec: DownloadSpec, model_dir: Path, force: bool, retries: int, timeout: int) -> None:
    target = model_dir / "buffalo_l"
    if target.exists() and not force:
        valid, reason = _validate_buffalo_directory(target)
        if valid:
            print("buffalo_l 已存在且校驗通過，跳過下載。")
            return
        print(f"現有 buffalo_l 無效（{reason}），將重新下載。", file=sys.stderr)
    archive_path = _download_verified(spec, model_dir, retries, timeout)
    temporary_dir: Path | None = None
    try:
        temporary_dir = _extract_buffalo(archive_path, model_dir)
        _promote_directory(temporary_dir, target)
        temporary_dir = None
    finally:
        if temporary_dir is not None and temporary_dir.exists():
            shutil.rmtree(temporary_dir, ignore_errors=True)
        archive_path.unlink(missing_ok=True)
    print("buffalo_l 安裝完成（僅保留偵測與辨識所需模型）。")


class _DownloadLock:
    def __init__(self, model_dir: Path) -> None:
        self.path = model_dir / ".download_models.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "_DownloadLock":
        try:
            self.descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            try:
                stale = time.time() - self.path.stat().st_mtime > 3600
            except OSError:
                stale = False
            if stale:
                self.path.unlink(missing_ok=True)
                self.descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            else:
                raise ModelDownloadError("另一個模型下載程序正在執行") from exc
        os.write(self.descriptor, str(os.getpid()).encode("ascii"))
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        self.path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="下載並校驗 InsightFace 必要模型")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(os.getenv("MODEL_DIR", "/models")),
        help="模型目錄（預設讀取 MODEL_DIR，否則 /models）",
    )
    parser.add_argument("--force", action="store_true", help="即使模型有效也重新下載")
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="下載失敗時顯示錯誤但回傳成功退出碼，供容器建置使用",
    )
    parser.add_argument("--allow-missing", dest="allow_failure", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--accept-license",
        action="store_true",
        help="確認已閱讀模型授權提醒；此旗標本身不授予任何權利",
    )
    parser.add_argument("--retries", type=int, default=3, help="每個模型的下載嘗試次數")
    parser.add_argument("--timeout", type=int, default=120, help="單次網路操作逾時秒數")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.retries < 1 or args.retries > 10 or args.timeout < 10 or args.timeout > 1800:
        print("錯誤：--retries 或 --timeout 超出合理範圍。", file=sys.stderr)
        return 2

    print(
        "授權提醒：InsightFace 程式碼採 MIT；官方預訓練模型僅供非商業研究。"
        "inswapper 系列與其他用途請向 InsightFace 取得相應授權。",
        file=sys.stderr,
    )
    if args.accept_license:
        print("已記錄授權提醒確認；此確認不會擴張模型原有授權。", file=sys.stderr)

    try:
        inswapper = _environment_spec(OFFICIAL_INSWAPPER, "INSWAPPER")
        buffalo = _environment_spec(OFFICIAL_BUFFALO, "BUFFALO_L")
        model_dir = args.model_dir.expanduser().resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        with _DownloadLock(model_dir):
            failures: list[str] = []
            for label, operation in (
                (
                    "inswapper_128.onnx",
                    lambda: _ensure_inswapper(inswapper, model_dir, args.force, args.retries, args.timeout),
                ),
                (
                    "buffalo_l",
                    lambda: _ensure_buffalo(buffalo, model_dir, args.force, args.retries, args.timeout),
                ),
            ):
                try:
                    operation()
                except Exception as exc:
                    failures.append(f"{label}: {exc}")
                    print(f"模型處理失敗 [{label}]：{exc}", file=sys.stderr, flush=True)
            if failures:
                raise ModelDownloadError("；".join(failures))
    except Exception as exc:
        print(f"模型尚未準備完成：{exc}", file=sys.stderr)
        if args.allow_failure:
            print("已啟用 --allow-failure；容器可繼續建置，應用程式啟動時可重試。", file=sys.stderr)
            return 0
        return 1

    print(f"所有必要模型已準備於：{model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
