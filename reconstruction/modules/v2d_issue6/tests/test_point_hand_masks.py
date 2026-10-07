"""Image mask repair cannot bypass frozen observation or identity evidence."""

import copy
import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[3] / "validation/issue6/point_hand_masks.py"
spec = importlib.util.spec_from_file_location("point_hand_masks", path)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def evidence():
    c = {"fx": 100, "fy": 100, "cx": 100, "cy": 100, "width": 200, "height": 200}
    joints = [[(j % 5) / 100, (j // 5) / 100, 1] for j in range(21)]
    hand = {
        "frame_id": 5,
        "track_id": 2,
        "is_right": False,
        "mano": {"betas": [0]},
        "metric_valid": False,
        "joints_camera_m": joints,
    }
    raw = {"frame_idx": 5, "track_id": 2, "is_right": False, "mano": {"betas": [0]}, "directly_observed": True}
    return hand, c, [90, 90, 120, 120], True, raw


class PointPromptTests(unittest.TestCase):
    def test_unsupported_depth_fit_can_repair_mask_without_becoming_metric_valid(self):
        values = evidence()
        before = copy.deepcopy(values)
        points, reason = repair.projected_prompt(*values)
        self.assertEqual(len(points), 6)
        self.assertIsNone(reason)
        self.assertEqual(values, before)
        self.assertFalse(values[0]["metric_valid"])

    def test_no_prompt_for_unobserved_or_ambiguous_fit(self):
        values = list(evidence())
        values[3] = False
        self.assertIsNone(repair.projected_prompt(*values)[0])
        values[3] = True
        values[0] = None
        self.assertIsNone(repair.projected_prompt(*values)[0])

    def test_other_hand_or_changed_fit_is_rejected(self):
        for key, value in (
            ("is_right", True),
            ("frame_idx", 6),
            ("track_id", 3),
            ("mano", {"betas": [1]}),
            ("directly_observed", False),
        ):
            values = evidence()
            values[-1][key] = value
            self.assertEqual(repair.projected_prompt(*values)[1], "frozen_fit_observation_or_identity_mismatch")

    def test_points_outside_selected_detection_or_behind_camera_do_not_prompt(self):
        values = list(evidence())
        values[2] = [0, 0, 20, 20]
        self.assertIsNone(repair.projected_prompt(*values)[0])
        values = evidence()
        for joint in values[0]["joints_camera_m"]:
            joint[2] = -1
        self.assertIsNone(repair.projected_prompt(*values)[0])

    def test_nonfinite_or_duplicate_points_cannot_supply_support(self):
        values = evidence()
        values[0]["joints_camera_m"][0][0] = float("nan")
        self.assertIsNone(repair.projected_prompt(*values)[0])
        values = evidence()
        values[0]["joints_camera_m"] = [[0, 0, 1]] * 21
        self.assertIsNone(repair.projected_prompt(*values)[0])


if __name__ == "__main__":
    unittest.main()
