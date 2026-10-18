import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_hand_reference_contacts import canonical_id, summarize


def test_joint_lookup_accepts_native_prefixes_and_rejects_ambiguous_bindings():
    names = ["other", "dex3_left_left_hand_index_0_joint", "robot/right_wrist_rotvec"]
    assert canonical_id(names, "left_hand_index_0_joint") == 1
    assert canonical_id(names, "right_wrist_rotvec") == 2
    with pytest.raises(ValueError, match="ambiguous"):
        canonical_id(names + ["left_hand_index_0_joint"], "left_hand_index_0_joint")
    with pytest.raises(ValueError, match="missing"):
        canonical_id(names, "thumb_joint")


def test_hand_contact_summary_counts_frames_not_repeated_manifold_points():
    def c(side, target, gap):
        return dict(hand=side, target=target, signed_gap_m=gap)

    frames = [
        dict(
            reference_frame=0,
            contacts=[c("left", "cup", -0.012), c("left", "cup", -0.020)],
        ),
        dict(reference_frame=1, contacts=[]),
        dict(
            reference_frame=2,
            contacts=[c("left", "cup", -0.0005), c("right", "support", -0.002)],
        ),
    ]
    rows = summarize(frames)
    assert rows[0]["frames_over_tolerance"] == 1
    assert rows[0]["maximum_penetration_m"] == 0.020
    assert rows[0]["worst_reference_frame"] == 0
    assert rows[1]["target"] == "support" and rows[1]["frames_over_tolerance"] == 1


def test_native_contact_summary_requires_full_clock_and_finite_gaps():
    with pytest.raises(ValueError, match="ordered"):
        summarize([dict(reference_frame=1, contacts=[])])
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize(
            [
                dict(
                    reference_frame=0,
                    contacts=[dict(hand="left", target="cup", signed_gap_m=np.nan)],
                )
            ]
        )
