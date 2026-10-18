import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2d_dex3_flashchord.startup_trace import StartupTrace


class ReadOnlyBuffer:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def numpy(self):
        return self.values

    def assign(self, *_):
        raise AssertionError("Diagnostic wrote to a simulation buffer")

    def zero_(self):
        raise AssertionError("Diagnostic cleared a simulation buffer")


def test_captures_independent_reset_and_warmstart_evidence_without_writes(tmp_path):
    q = ReadOnlyBuffer([[1, 2, 3]])
    warmstart = ReadOnlyBuffer([[7, 8, 9]])
    env = SimpleNamespace(
        state_0=SimpleNamespace(joint_q=q),
        solver=SimpleNamespace(mjw_data=SimpleNamespace(qacc_warmstart=warmstart)),
    )
    trace = StartupTrace()
    trace.capture("before-reset", env, pending_action=np.array([[0.1]]))
    np.testing.assert_array_equal(q.values, [[1, 2, 3]])
    np.testing.assert_array_equal(warmstart.values, [[7, 8, 9]])
    # Model an existing runtime reset after the passive snapshot.  Its new
    # values must not overwrite evidence about the earlier state.
    q.values[:] = 4
    warmstart.values[:] = 0
    trace.capture("after-reset", env)
    report = trace.save(tmp_path)
    with np.load(tmp_path / "startup-state-trace.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["snapshot_000__state_0__joint_q"], [[1, 2, 3]])
        np.testing.assert_array_equal(z["snapshot_000__native__qacc_warmstart"], [[7, 8, 9]])
        np.testing.assert_array_equal(z["snapshot_001__native__qacc_warmstart"], [[0, 0, 0]])
        for row in report["snapshots"]:
            for field in row["fields"].values():
                assert hashlib.sha256(z[field["key"]].tobytes(order="C")).hexdigest() == field["sha256"]
    assert report["simulation_buffers_written"] is False
    assert report["extra_physics_steps"] == 0
    assert report["replay_variability_cause_proven"] is False
    assert report["arrays_sha256"] == hashlib.sha256((tmp_path / "startup-state-trace.npz").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        trace.save(tmp_path)


def test_missing_and_nonfinite_buffers_remain_visible(tmp_path):
    env = SimpleNamespace(observation=ReadOnlyBuffer([np.nan, np.inf]))
    trace = StartupTrace()
    trace.capture("construction", env)
    trace.save(tmp_path)
    report = json.loads((tmp_path / "startup-state-trace.json").read_text())
    row = report["snapshots"][0]
    assert "native.qacc_warmstart" in row["missing_fields"]
    assert row["fields"]["env.observation"]["finite"] is False
    with np.load(tmp_path / "startup-state-trace.npz", allow_pickle=False) as z:
        assert np.isnan(z["snapshot_000__env__observation"][0])
        assert np.isposinf(z["snapshot_000__env__observation"][1])
