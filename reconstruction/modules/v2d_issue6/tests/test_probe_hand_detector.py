"""Crop coordinates, hypothesis isolation and holdout protection are essential."""

import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[3] / "validation/issue6/probe_hand_detector.py"
spec = importlib.util.spec_from_file_location("probe_hand_detector", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def candidate(view, confidence, box=(10, 10, 30, 30), side=0, index=0):
    return {
        "source_view": view,
        "confidence": confidence,
        "box_xyxy": list(box),
        "class": side,
        "candidate_id": f"{view}:{index}",
    }


class DetectorProbeTests(unittest.TestCase):
    def test_crop_coordinates_return_to_the_original_camera_axis(self):
        for crop, _ in probe.source_views(1280, 800).values():
            x0, y0, x1, y1 = crop
            self.assertEqual(probe.source_box([0, 0, x1 - x0, y1 - y0], crop), list(crop))
        self.assertEqual(probe.source_box([20, 10, 80, 50], (512, 240, 1280, 800)), [532, 250, 592, 290])

    def test_outside_pixels_do_not_create_support_in_the_source_image(self):
        crop = (512, 240, 1280, 800)
        self.assertEqual(probe.source_box([-50, -30, 800, 600], crop), list(crop))
        for box in ([800, 0, 900, 20], [0, 0, 0, 20], [20, 20, 10, 30], [float("nan"), 0, 10, 20], [0, 0, 10]):
            self.assertIsNone(probe.source_box(box, crop))

    def test_baseline_cannot_inherit_low_confidence_or_crop_candidates(self):
        rows = [candidate("full640", 0.5), candidate("full640", 0.49, index=1), candidate("tile00", 0.9)]
        baseline = probe.variant_candidates(rows, ("full640",), 0.5)
        self.assertEqual([r["candidate_id"] for r in baseline], ["full640:0"])
        baseline[0]["box_xyxy"][0] = 999
        baseline[0]["owner_rejection"] = "changed by association"
        self.assertEqual(rows[0]["box_xyxy"][0], 10)
        self.assertNotIn("owner_rejection", rows[0])

    def test_tile_merge_does_not_merge_opposite_hands_or_other_objects(self):
        rows = [
            candidate("full640", 0.9),
            candidate("tile00", 0.8),
            candidate("tile00", 0.7, side=1, index=1),
            candidate("tile00", 0.6, box=(80, 80, 100, 100), index=2),
        ]
        merged = probe.variant_candidates(rows, ("full640", "tile00"), 0.1)
        self.assertEqual([r["candidate_id"] for r in merged], ["full640:0", "tile00:1", "tile00:2"])

    def test_development_sweeps_cannot_tune_the_five_holdouts(self):
        for episode in (0, 11, 23, 31, 39):
            with self.assertRaises(ValueError):
                probe.development_episode(episode)
        for episode in (12, 22):
            probe.development_episode(episode)


if __name__ == "__main__":
    unittest.main()
