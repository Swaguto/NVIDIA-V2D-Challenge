import hashlib
import os
import time
from pathlib import Path

import pytest

from issue6 import manifest


def test_unchanged_file_is_hashed_once_and_same_size_edit_is_detected(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    path.write_bytes(b"first")
    if manifest._USE_FILE_HASH_CACHE:
        time.sleep(2.05)
    before = path.stat()
    initially_cacheable = manifest._cacheable_file(before)
    original_open = Path.open
    reads = []

    def counted_open(self, *args, **kwargs):
        if self == path.resolve() and args == ("rb",):
            reads.append(self)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)
    assert manifest.file_hash(path) == hashlib.sha256(b"first").hexdigest()
    assert manifest.file_hash(path) == hashlib.sha256(b"first").hexdigest()
    reads_before_edit = 1 if initially_cacheable else 2
    assert len(reads) == reads_before_edit
    path.write_bytes(b"other")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert manifest.file_hash(path) == hashlib.sha256(b"other").hexdigest()
    assert len(reads) == reads_before_edit + 1


def test_coarse_change_time_requires_two_second_age_before_cache(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(manifest, "_USE_FILE_HASH_CACHE", True)
    monkeypatch.setattr(manifest.time, "time_ns", lambda: 10_000_000_000)
    assert not manifest._cacheable_file(SimpleNamespace(st_ino=1, st_ctime_ns=9_000_000_000))
    assert not manifest._cacheable_file(SimpleNamespace(st_ino=1, st_ctime_ns=8_000_000_000))
    assert manifest._cacheable_file(SimpleNamespace(st_ino=1, st_ctime_ns=7_000_000_000))
    assert not manifest._cacheable_file(SimpleNamespace(st_ino=0, st_ctime_ns=7_000_000_000))


def test_same_path_replacement_invalidates_hash_cache(tmp_path):
    path = tmp_path / "input.bin"
    replacement = tmp_path / "replacement.bin"
    path.write_bytes(b"first")
    before = path.stat()
    manifest.file_hash(path)
    replacement.write_bytes(b"other")
    os.utime(replacement, ns=(before.st_atime_ns, before.st_mtime_ns))
    replacement.replace(path)
    assert manifest.file_hash(path) == hashlib.sha256(b"other").hexdigest()


def test_hash_rejects_file_changed_while_reading(tmp_path, monkeypatch):
    path = tmp_path / "input.bin"
    path.write_bytes(b"first")
    actual_sha256 = hashlib.sha256

    class MutatingDigest:
        def __init__(self):
            self.digest = actual_sha256()

        def update(self, chunk):
            self.digest.update(chunk)
            path.write_bytes(b"changed during hashing")

        def hexdigest(self):
            return self.digest.hexdigest()

    monkeypatch.setattr(manifest.hashlib, "sha256", MutatingDigest)
    with pytest.raises(ValueError, match="changed during hashing"):
        manifest.file_hash(path)
