"""Review bounds follow the input contract, never a falling simulated object."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from render_native_readback import reference_view


def test_view_is_bound_to_reference_and_rejects_tampering(tmp_path):
    poses = np.zeros((2, 1, 7))
    poses[:, 0, 0] = [0, 0.5]
    target = tmp_path / "targets.npz"
    np.savez(target, object_poses=poses)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                files={"targets.npz": hashlib.sha256(target.read_bytes()).hexdigest()},
                objects=[dict(radius_m=0.2)],
            )
        )
    )
    report = dict(
        source_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    bounds = reference_view(tmp_path, report)
    np.testing.assert_allclose(bounds[0], [-0.2, 0.7])
    np.testing.assert_allclose(bounds[1:], [[-0.3, 0.3], [-0.3, 0.3]])
    np.savez(target, object_poses=poses + 1)
    with pytest.raises(ValueError, match="reference changed"):
        reference_view(tmp_path, report)
