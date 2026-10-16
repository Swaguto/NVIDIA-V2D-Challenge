"""A half-speed inertial plan cannot silently become a new 1x training pack."""

import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from prepare_task_reference import prepare


def test_mismatched_force_clock_fails_before_writing_inputs(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "manifest.json").write_text(json.dumps({}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"source_motion_speed": 0.5}))
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="force plan computed at source speed 1.0"):
        prepare(base, tmp_path / "missing-case", spec, out)
    assert not out.exists()
