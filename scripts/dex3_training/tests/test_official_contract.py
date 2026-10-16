"""The preflight must reject corrupt upstream bytes and oversized headers."""

import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_official_contract import metadata, verify_kit


def test_kit_detects_changed_evaluator(tmp_path):
    script = b"official evaluator"
    manifest = {"files": {"evaluate.py": hashlib.sha256(script).hexdigest()}}
    for changed in (False, True):
        kit = tmp_path / f"kit-{changed}.zip"
        with zipfile.ZipFile(kit, "w") as z:
            z.writestr("kit/KIT_MANIFEST.json", json.dumps(manifest))
            z.writestr("kit/evaluate.py", b"changed" if changed else script)
            z.writestr(
                "kit/data/track_3_evaluation_objects.json", '{"episodes":{"1":["pot"]}}'
            )
        if changed:
            with pytest.raises(ValueError, match="checksum mismatch"):
                verify_kit(kit)
        else:
            assert verify_kit(kit)[1] == {"1": ["pot"]}


def test_checkpoint_header_rejects_unbounded_length(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 2**60) + b"{}")
    with pytest.raises(ValueError, match="header length"):
        metadata(path)
