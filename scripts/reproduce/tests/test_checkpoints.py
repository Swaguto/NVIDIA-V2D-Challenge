"""Fail closed on changed evidence, unsafe archives and altered dispositions."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checkpoint import compare, restore, sha


def archive(
    tmp_path,
    name="data.txt",
    content=b"evidence",
    recorded=b"evidence",
    duplicate=False,
    link=False,
):
    path = tmp_path / "checkpoint.tar.gz"
    manifest = {
        "files": {
            name: {
                "bytes": len(recorded),
                "sha256": hashlib.sha256(recorded).hexdigest(),
            }
        }
    }
    with tarfile.open(path, "w:gz") as tar:
        for filename, payload in [
            ("artifact-manifest.json", json.dumps(manifest).encode()),
            (name, content),
        ]:
            info = tarfile.TarInfo(filename)
            info.size = len(payload)
            if filename == name and link:
                info.type = tarfile.SYMTYPE
                info.linkname = "../escape"
                info.size = 0
                tar.addfile(info)
            else:
                tar.addfile(info, io.BytesIO(payload))
        if duplicate:
            info = tarfile.TarInfo("./" + name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


def test_verified_restore_and_existing_destination(tmp_path):
    path = archive(tmp_path)
    out = tmp_path / "fresh"
    restore(path, sha(path), out)
    assert (out / "data.txt").read_bytes() == b"evidence"
    with pytest.raises(FileExistsError):
        restore(path, sha(path), out)


@pytest.mark.parametrize(
    "options",
    [
        {"name": "../escape"},
        {"name": "/absolute"},
        {"name": "C:/absolute"},
        {"name": "folder\\escape"},
        {"link": True},
        {"duplicate": True},
        {"recorded": b"changed"},
    ],
)
def test_unsafe_or_changed_members_write_nothing(tmp_path, options):
    path = archive(tmp_path, **options)
    out = tmp_path / "fresh"
    with pytest.raises(ValueError):
        restore(path, sha(path), out)
    assert not out.exists()


def test_wrong_archive_digest_writes_nothing(tmp_path):
    path = archive(tmp_path)
    out = tmp_path / "fresh"
    with pytest.raises(ValueError):
        restore(path, "0" * 64, out)
    assert not out.exists()


@pytest.mark.parametrize(
    "old,new",
    [
        ([None], [False]),
        ([True], [1]),
        ({"count": 72}, {"count": 73}),
        ({"value": 1.0}, {"value": float("nan")}),
        ([1], [1, 2]),
    ],
)
def test_changed_unknowns_counts_and_masks_fail(old, new):
    with pytest.raises(ValueError):
        compare(old, new)


def test_only_float_roundoff_is_tolerated():
    compare(
        {"x": 0.1, "mask": [True, None], "count": 72},
        {"x": 0.1 + 1e-10, "mask": [True, None], "count": 72},
    )
    with pytest.raises(ValueError):
        compare({"x": 0.1}, {"x": 0.1001})
