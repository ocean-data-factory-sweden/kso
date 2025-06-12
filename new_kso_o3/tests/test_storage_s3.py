"""S3Storage integration tests using moto mock.
Skip automatically if moto isn't installed (dev optional dependency).
"""
from __future__ import annotations

import importlib
import os
from typing import TYPE_CHECKING, Any  # noqa: F401  # silences unused import lint for TYPE_CHECKING scope

import pytest

moto = pytest.importorskip("moto")  # skip suite if package missing
from moto import mock_s3  # type: ignore  # noqa: E402


@pytest.mark.usefixtures("monkeypatch")
@mock_s3
def test_s3_storage_roundtrip(tmp_path, monkeypatch):
    """Upload and download via mocked S3 backend."""
    # ---- Arrange mocked AWS env ----
    bucket_name = "kso-test-bucket"
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")

    import boto3

    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket=bucket_name)

    # ---- Build temporary config ----
    cfg_dir = tmp_path / "cfg"
    (cfg_dir / "models").mkdir(parents=True)

    # minimal configs
    (cfg_dir / "project.yaml").write_text("""\nproject_name: s3_test\nmodel: pytorch\noutput_dir: outputs\n""")
    (cfg_dir / "storage.yaml").write_text(
        f"""\nlocal_root: ./data\nremote:\n  provider: s3\n  endpoint: http://s3.amazonaws.com\n  bucket: {bucket_name}\n"""
    )
    (cfg_dir / "models" / "pytorch.yaml").write_text("arch: resnet18\nnum_classes: 2\n")

    # Override CONFIG_DIR
    import kso.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", cfg_dir)

    # Reload storage module to pick up new config
    import kso.storage as storage_mod

    importlib.reload(storage_mod)

    store = storage_mod.get_storage()
    assert store.__class__.__name__ == "S3Storage"

    # ---- Roundtrip ----
    src_file = tmp_path / "src.txt"
    content = b"hello s3"
    src_file.write_bytes(content)

    store.upload_file(src_file, "artifacts/hello.txt")
    assert store.exists("artifacts/hello.txt") is True

    dst_file = tmp_path / "dst.txt"
    store.download_file("artifacts/hello.txt", dst_file)

    assert dst_file.read_bytes() == content
