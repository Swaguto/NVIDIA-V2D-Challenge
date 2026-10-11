"""Prevent a handedness flip from stealing the other wearer's observed track."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "validation/issue6"))
from native_identity_guard import identity_conflicts


def candidate(x, depth=0.4):
    return {"box_xyxy": [x - 5, 60, x + 5, 80], "depth_m": depth}


def row(frame, left=None, right=None, bootstrap=False):
    candidates = []
    selected = {}
    for side, value in [("left", left), ("right", right)]:
        index = len(candidates) if value is not None else None
        if value is not None:
            candidates.append(value)
        selected[side] = {
            "candidate_index": index,
            "reason": "nearby_lower_image_hand_bootstrap" if bootstrap else "stereo_depth_and_temporal_continuity",
        }
    return {"frame_index": frame, "candidates": candidates, "selected": selected}


def sequence():
    return {
        "settings": {"maximum_gap_frames": 12},
        "frames": [
            row(0, candidate(20), candidate(80), True),
            row(13, candidate(30)),
            row(14, candidate(32)),
            row(15, right=candidate(34), bootstrap=True),
            row(16, right=candidate(36)),
            row(17, right=candidate(38)),
            row(18, candidate(40), candidate(80)),
        ],
    }


class NativeIdentityGuardTests(unittest.TestCase):
    def test_transient_label_flip_stays_unknown_until_identity_recovers(self):
        audit = sequence()
        original = copy.deepcopy(audit)
        conflicts = identity_conflicts(audit, 100, 100)
        self.assertEqual([(c["frame_id"], c["hand"]) for c in conflicts], [(15, "right"), (16, "right"), (17, "right")])
        self.assertEqual(audit, original)

    def test_separate_reappearing_hand_is_not_the_other_track(self):
        audit = sequence()
        audit["frames"][3:] = [row(15, right=candidate(80), bootstrap=True)]
        self.assertEqual(identity_conflicts(audit, 100, 100), [])

    def test_two_distinct_observations_do_not_trigger_single_hand_quarantine(self):
        audit = sequence()
        audit["frames"][3:] = [row(15, candidate(34), candidate(38), True)]
        self.assertEqual(identity_conflicts(audit, 100, 100), [])

    def test_different_measured_depth_cannot_extend_the_other_hand_track(self):
        audit = sequence()
        audit["frames"][3:] = [row(15, right=candidate(34, 0.8), bootstrap=True)]
        self.assertEqual(identity_conflicts(audit, 100, 100), [])

    def test_expired_opposite_track_does_not_establish_identity(self):
        audit = sequence()
        audit["frames"][3:] = [row(27, right=candidate(58), bootstrap=True)]
        self.assertEqual(identity_conflicts(audit, 100, 100), [])


if __name__ == "__main__":
    unittest.main()
