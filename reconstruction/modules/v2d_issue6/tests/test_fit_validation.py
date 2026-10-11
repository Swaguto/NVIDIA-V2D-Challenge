"""The actual native-fit evidence check must remain active under Python -O."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HELPERS = Path(__file__).resolve().parents[3] / "validation/issue6"
KEYS = ("trans", "root_orient", "pose_body", "betas", "cam_R", "cam_t")


@pytest.mark.parametrize("corruption", [None, "changed_shape", *KEYS])
def test_optimized_fit_validation(tmp_path, corruption):
    initial = tmp_path / "initial.npz"
    final = tmp_path / "final.npz"
    values = {key: np.zeros((2, 3)) for key in KEYS}
    np.savez(initial, **values)
    if corruption == "changed_shape":
        values["betas"][0, 0] = 1
    elif corruption is not None:
        values[corruption][0, 0] = np.nan
    np.savez(final, **values)
    result = subprocess.run(
        [
            sys.executable,
            "-O",
            "-c",
            "import sys; sys.path.insert(0, sys.argv[1]); from fit_validation import verify_fit; "
            "verify_fit(sys.argv[2], sys.argv[3], True)",
            str(HELPERS),
            str(initial),
            str(final),
        ],
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (corruption is None), result.stderr
