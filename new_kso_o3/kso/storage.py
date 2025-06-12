"""Storage abstraction layer supporting local filesystem & S3-compatible stores."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable, BinaryIO

import boto3
from botocore.client import Config as BotoConfig

from .config import load_project_config, StorageConfig


@runtime_checkable
class BaseStorage(Protocol):
    def exists(self, path: str | Path) -> bool: ...  # noqa: D401,Ellipsis

    def open(self, path: str | Path, mode: str = "rb") -> BinaryIO: ...

    def upload_file(self, src: Path, dst: str): ...

    def download_file(self, src: str, dst: Path): ...


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _full_path(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else self.root / p

    # --- BaseStorage API ----------------------------------------------------
    def exists(self, path: str | Path) -> bool:
        return self._full_path(path).exists()

    def open(self, path: str | Path, mode: str = "rb"):
        return self._full_path(path).open(mode)

    def upload_file(self, src: Path, dst: str):
        dst_path = self._full_path(dst)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        dst_path.write_bytes(src.read_bytes())

    def download_file(self, src: str, dst: Path):
        src_path = self._full_path(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src_path.read_bytes())


class S3Storage:
    def __init__(self, cfg: StorageConfig):
        remote = cfg.remote  # type: ignore[assignment]
        assert remote, "Remote config required"
        session = boto3.session.Session()
        self.client = session.client(
            "s3",
            endpoint_url=remote.endpoint,
            aws_access_key_id=remote.aws_access_key_id,
            aws_secret_access_key=remote.aws_secret_access_key,
            region_name=remote.region,
            config=BotoConfig(signature_version="s3v4"),
        )
        self.bucket = remote.bucket
        self.local_root = cfg.local_root

    # --- helpers -----------------------------------------------------------
    def _key(self, path: str | Path) -> str:
        return str(path).lstrip("/")

    # --- BaseStorage API ---------------------------------------------------
    def exists(self, path: str | Path) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(path))
            return True
        except self.client.exceptions.ClientError:  # type: ignore[attr-defined]
            return False

    def open(self, path: str | Path, mode: str = "rb"):
        # For simplicity, download to tmp then open. Could stream.
        import tempfile

        tmp = Path(tempfile.mktemp())
        self.download_file(path, tmp)
        return tmp.open(mode)

    def upload_file(self, src: Path, dst: str):
        self.client.upload_file(str(src), self.bucket, self._key(dst))

    def download_file(self, src: str, dst: Path):
        dst.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(src), str(dst))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_storage() -> BaseStorage:
    cfg = load_project_config().storage
    if cfg.remote and cfg.remote.provider.lower() == "s3":
        return S3Storage(cfg)
    return LocalStorage(cfg.local_root)
