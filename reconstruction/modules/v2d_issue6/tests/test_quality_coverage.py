"""Acceptance must measure source-visible grasps and actual raw contact evidence."""

import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[3] / "validation/issue6/audit_quality_coverage.py"
spec = importlib.util.spec_from_file_location("quality_coverage", path)
quality = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quality)


def evidence(count=10, invalid=(), raw=()):
    frames = [{"frame_id": i, "timestamp_s": i / 10} for i in range(count)]
    rows = []
    for f in frames:
        for hand in ("left", "right"):
            for tip in sorted(quality.TIPS):
                valid = f["frame_id"] not in invalid
                contact = f["frame_id"] in raw
                rows.append(
                    {
                        "frame_id": f["frame_id"],
                        "timestamp_s": f["timestamp_s"],
                        "hand": hand,
                        "fingertip": tip,
                        "object_id": "cup",
                        "valid": valid,
                        "raw_contact": contact if valid else None,
                        "window_contact": True if valid else None,
                        "bridged": valid and not contact,
                        "reason": None if valid else "unanchored_hand",
                    }
                )
    return frames, rows


class QualityCoverageTests(unittest.TestCase):
    def interval(self, start=0, end=9):
        return {
            "hand": "left",
            "object_id": "cup",
            "start_frame": start,
            "end_frame": end,
            "visibility": "visible_hold",
            "source_evidence": "independent source-frame review",
        }

    def test_bridges_cannot_replace_raw_contact_evidence(self):
        frames, rows = evidence()
        result = quality.evaluate_visible_interval(rows, frames, self.interval())
        self.assertEqual(result["usable_fraction"], 1)
        self.assertEqual(result["raw_contact_fraction"], 0)
        self.assertEqual(result["status"], "failed_metrics")

    def test_interval_denominator_excludes_unrelated_source_tail(self):
        frames, rows = evidence(invalid=(8, 9), raw=range(8))
        result = quality.evaluate_visible_interval(rows, frames, self.interval(end=7))
        self.assertEqual(result["eligible_frames"], 8)
        self.assertEqual(result["usable_fraction"], 1)
        self.assertFalse(result["quality_acceptance"])

    def test_small_total_dropout_can_fail_longest_gap(self):
        frames, rows = evidence(count=100, invalid=range(10, 16), raw=range(100))
        result = quality.evaluate_visible_interval(rows, frames, self.interval(end=99))
        self.assertAlmostEqual(result["usable_fraction"], 0.94)
        self.assertAlmostEqual(result["longest_unsupported_gap"]["seconds"], 0.6)
        self.assertEqual(result["status"], "failed_metrics")

    def test_unknown_tail_gap_is_counted(self):
        frames, _ = evidence()
        gap = quality.longest_gap(frames, {i: i < 7 for i in range(10)})
        self.assertEqual((gap["start_frame"], gap["end_frame"]), (7, 9))
        self.assertAlmostEqual(gap["seconds"], 0.3)

    def test_unknown_cannot_be_mislabeled_as_noncontact(self):
        frames, rows = evidence(invalid=(4,))
        next(row for row in rows if not row["valid"])["raw_contact"] = False
        with self.assertRaisesRegex(ValueError, "Unknown contact"):
            quality.measure(rows, frames)

    def test_annotations_and_geometry_review_are_required(self):
        frames, rows = evidence(raw=range(10))
        interval = self.interval()
        del interval["source_evidence"]
        with self.assertRaisesRegex(ValueError, "source-video evidence"):
            quality.evaluate_visible_interval(rows, frames, interval)
        result = quality.evaluate_visible_interval(rows, frames, self.interval())
        self.assertEqual(result["status"], "metrics_passed_requires_geometry_and_event_review")
        self.assertFalse(result["quality_acceptance"])

    def test_incomplete_finger_rows_cannot_accept(self):
        frames, rows = evidence(raw=range(10))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            quality.evaluate_visible_interval(rows[1:], frames, self.interval())


if __name__ == "__main__":
    unittest.main()
