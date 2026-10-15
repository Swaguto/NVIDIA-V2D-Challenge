"""Check the real installed loader and reject unsupported training contracts."""

import argparse
import hashlib
import json
from pathlib import Path

from flash_chord.data import load_reference
from flash_chord.data.reference import HandPoseReference
from v2d_dex3_flashchord.embodiment import Dex3Hands, Dex3HandsConfig
from v2d_dex3_flashchord.objective import Dex3ObjectiveConfig


def must_reject(call, phrase):
    try:
        call()
    except ValueError as error:
        if phrase not in str(error):
            raise AssertionError(str(error)) from error
        return True
    raise AssertionError("Invalid contract was accepted")


def check(path):
    reference = load_reference(path, control_fps=20.0)
    if not isinstance(reference, HandPoseReference) or reference.sides != (
        "left",
        "right",
    ):
        raise AssertionError("Dex3 reference capability differs")
    result = dict(
        installed_dispatch_loads_dual_hand=True,
        unknown_contact_reward_rejected=must_reject(
            lambda: Dex3ObjectiveConfig(), "requires measured"
        ),
        implicit_resampling_rejected=must_reject(
            lambda: load_reference(path, control_fps=21.0), "source-rate"
        ),
        changed_manifest_rejected=must_reject(
            lambda: Dex3Hands(
                Dex3HandsConfig(
                    left_urdf_path=path.parent / "robot/urdfs/g1/dex3_left.urdf",
                    right_urdf_path=path.parent / "robot/urdfs/g1/dex3_right.urdf",
                    reference_manifest_sha256="0" * 64,
                )
            ).bind_reference(None, None, reference),
            "reference manifest differs",
        ),
        frame_window_retains_shape=reference.frame_window(1, 3).num_frames == 2,
        frame_count=reference.num_frames,
        input_manifest_sha256=hashlib.sha256(
            (path.parent / "manifest.json").read_bytes()
        ).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        physical_task_accepted=False,
    )
    if not result["frame_window_retains_shape"]:
        raise AssertionError("Window shape differs")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = check(args.reference)
    with args.out.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
