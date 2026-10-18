"""Model cache reuse refuses modified bytes and unsafe source archives."""

import hashlib
import io
from pathlib import Path
import sys
import tarfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from download_sam3d_models import source_tree, verify


def test_pinned_weight_reuse_detects_modification(tmp_path):
    file = tmp_path / "weights"
    file.write_bytes(b"original")
    pin = dict(size=8, sha256=hashlib.sha256(b"original").hexdigest())
    verify(file, pin)
    file.write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed"):
        verify(file, pin)


def archive(path, name, raw=b"pinned source"):
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(raw)
        tar.addfile(info, io.BytesIO(raw))


def test_source_cache_is_verified_when_reused(tmp_path):
    file = tmp_path / "source.tar.gz"
    archive(file, "dinov2-abc/hubconf.py")
    out = tmp_path / "source"
    source_tree(file, out, "abc", False)
    source_tree(file, out, "abc", True)
    (out / "hubconf.py").write_bytes(b"changed source")
    with pytest.raises(ValueError, match="source changed"):
        source_tree(file, out, "abc", True)


def test_unsafe_source_member_never_extracts(tmp_path):
    file = tmp_path / "source.tar.gz"
    archive(file, "dinov2-abc/../../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        source_tree(file, tmp_path / "source", "abc", False)
    assert not (tmp_path / "outside").exists()
