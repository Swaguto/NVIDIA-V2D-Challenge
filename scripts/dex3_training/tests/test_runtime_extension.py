"""Extending a verified runtime must not accept changed patched source."""

import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import install_extension as installer


def test_extension_can_extend_only_matching_installed_patch_receipts(
    tmp_path, monkeypatch
):
    base = tmp_path / "src/flash_chord"
    base.mkdir(parents=True)
    p = base / "a.py"
    p.write_text("original")
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(installer, "PATCHES", {"a.py": (before, "original", "changed")})
    installer.install(tmp_path)
    with pytest.raises(ValueError):
        installer.install(tmp_path)
    installer.install(tmp_path, extend_existing=True)
    p.write_text("changed by someone else")
    with pytest.raises(ValueError):
        installer.install(tmp_path, extend_existing=True)


def test_extending_adds_a_new_hook_without_rewriting_existing_hooks(
    tmp_path, monkeypatch
):
    base = tmp_path / "src/flash_chord"
    base.mkdir(parents=True)
    for name in ["a.py", "b.py"]:
        (base / name).write_text("original")
    before = hashlib.sha256(b"original").hexdigest()
    monkeypatch.setattr(installer, "PATCHES", {"a.py": (before, "original", "changed")})
    installer.install(tmp_path)
    monkeypatch.setattr(
        installer,
        "PATCHES",
        {n: (before, "original", "changed") for n in ["a.py", "b.py"]},
    )
    receipt = installer.install(tmp_path, extend_existing=True)
    assert set(receipt) == {"a.py", "b.py"}
    assert (base / "a.py").read_text() == (base / "b.py").read_text() == "changed"
    assert json.loads((tmp_path / "dex3-extension-patches.json").read_text()) == receipt
