"""Exercise the real Bash launcher with a failing CPU interpreter stub."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
pytestmark = pytest.mark.skipif(os.name == "nt" or shutil.which("bash") is None,
                                reason="Bash launcher controls run on Linux CI")


def launch(out, tmp_path, monkeypatch, **overrides):
    calls = tmp_path / "calls.txt"
    interpreter = tmp_path / "python-stub"
    interpreter.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$CALLS"\nexit 23\n')
    interpreter.chmod(0o755)
    for key, value in {"EPISODE": "2", "DATA": "/unused/data", "RESULTS": "/unused/results",
                       "BUNDLE": "/unused/bundle", "OUT": str(out), "PYTHON": str(interpreter),
                       "CALLS": str(calls), **overrides}.items():
        monkeypatch.setenv(key, value)
    result = subprocess.run(["bash", str(ROOT / "scripts/track3/refine_episode.sh")],
                            capture_output=True, text=True, timeout=10, check=False)
    return result, calls


@pytest.mark.parametrize("artifact", ["inputs", "refined", "report", "slam", "slam-inputs"])
def test_existing_pipeline_stage_is_preserved_before_any_command(tmp_path, monkeypatch, artifact):
    out = tmp_path / "experiment"
    stage = out / artifact
    stage.mkdir(parents=True)
    marker = stage / "inputs.json"
    marker.write_text('"preserve these exact bytes"')
    result, calls = launch(out, tmp_path, monkeypatch, RUN_SLAM="1",
                           DROID_WEIGHTS="/unused/droid.pth", DROID_REPO="/unused/droid")
    assert result.returncode == 1
    assert "choose a new OUT" in result.stderr
    assert not calls.exists()
    assert marker.read_text() == '"preserve these exact bytes"'


def test_input_assembly_failure_requires_a_new_output_directory(tmp_path, monkeypatch):
    out = tmp_path / "experiment"
    first, calls = launch(out, tmp_path, monkeypatch)
    assert first.returncode == 23
    assert "gsplat_inputs.py" in calls.read_text()
    assert out.is_dir()
    recorded = calls.read_bytes()
    second, _ = launch(out, tmp_path, monkeypatch, EPISODE="39", START="10", FRAMES="3")
    assert second.returncode == 1
    assert "choose a new OUT" in second.stderr
    assert calls.read_bytes() == recorded


def test_dangling_output_symlink_is_preserved(tmp_path, monkeypatch):
    out = tmp_path / "experiment"
    out.symlink_to(tmp_path / "missing", target_is_directory=True)
    result, calls = launch(out, tmp_path, monkeypatch)
    assert result.returncode == 1
    assert "choose a new OUT" in result.stderr
    assert out.is_symlink()
    assert not calls.exists()
