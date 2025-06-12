"""Unit tests for storage abstraction layer."""
import importlib

import kso.storage as storage_mod


def test_local_storage_roundtrip(tmp_path):
    # Arrange
    root = tmp_path / "data"
    store = storage_mod.LocalStorage(root)

    src_file = tmp_path / "src.txt"
    src_file.write_text("hello")

    # Act: upload -> download
    store.upload_file(src_file, "dest/hello.txt")
    assert store.exists("dest/hello.txt")

    dst_file = tmp_path / "download.txt"
    store.download_file("dest/hello.txt", dst_file)

    # Assert
    assert dst_file.read_text() == "hello"


def test_get_storage_defaults_to_local(tmp_path, monkeypatch):
    """get_storage() should return LocalStorage when no remote is configured."""

    # Build minimal fake config dir
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()

    (cfg_dir / "project.yaml").write_text("""\nproject_name: test\nmodel: pytorch\noutput_dir: outputs\n""")
    (cfg_dir / "storage.yaml").write_text("""\nlocal_root: ./data\n""")
    model_dir = cfg_dir / "models"
    model_dir.mkdir()
    (model_dir / "pytorch.yaml").write_text("arch: resnet18\nnum_classes: 2\n")

    # Monkeypatch CONFIG_DIR used in kso.config
    import kso.config as config_mod

    monkeypatch.setattr(config_mod, "CONFIG_DIR", cfg_dir)

    # Reload storage module to ensure new config is picked up
    importlib.reload(storage_mod)

    store = storage_mod.get_storage()
    assert isinstance(store, storage_mod.LocalStorage)
