"""Audit missing evidence and distinguish strict contacts from the legacy ablation."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[3] / "validation/issue6/audit_completed_contacts.py"


def write_run(tmp_path, maximum):
    run = tmp_path / "e002"
    (run / "contacts").mkdir(parents=True)
    (run / "report").mkdir()
    frames = [{"id": frame, "timestamp_s": frame / 20} for frame in range(3)]
    manifest = {
        "frames": frames,
        "objects": [{"id": "cup"}],
        "settings": {"bridge_frames": maximum, "contact_threshold_m": 0.02},
    }
    (run / "inputs.lock.json").write_text(json.dumps({"manifest": manifest}))
    rows = []
    for frame in frames:
        for side in ("left", "right"):
            for tip in ("thumb", "index", "middle", "ring", "little"):
                raw = side == "right" and tip == "thumb" and frame["id"] != 1
                bridged = side == "right" and tip == "thumb" and frame["id"] == 1 and maximum > 0
                rows.append(
                    {
                        "frame_id": frame["id"],
                        "timestamp_s": frame["timestamp_s"],
                        "hand": side,
                        "fingertip": tip,
                        "object_id": "cup",
                        "hand_track_id": 3 if side == "right" else 2,
                        "hand_provenance": "predicted",
                        "object_provenance": "predicted",
                        "camera_provenance": "predicted",
                        "valid": True,
                        "surface_distance_m": 0.01 if raw else 0.04,
                        "raw_contact": raw,
                        "window_contact": raw or bridged,
                        "bridged": bridged,
                        "reason": None,
                    }
                )
    (run / "contacts/contacts.json").write_text(json.dumps(rows))
    (run / "contacts/settings.json").write_text(json.dumps({"threshold_m": 0.02, "bridge_frames": maximum}))
    report = {
        "frame_axis": frames,
        "contact_valid_count": len(rows),
        "contact_unknown_count": 0,
        "raw_contact_count": 2,
        "bridged_count": int(maximum > 0),
    }
    (run / "report/report.json").write_text(json.dumps(report))
    return run, rows, report


def audit(tmp_path, optimize=False):
    return subprocess.run(
        [
            sys.executable,
            *(["-O"] if optimize else []),
            str(AUDIT),
            "--root",
            str(tmp_path),
            "--episodes",
            "2",
            "--direct-run-root",
            "--require-complete",
        ],
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("corruption", [None, "provenance", "missing_tip", "bridge"])
def test_optimized_audit_retains_integrity_checks(tmp_path, corruption):
    run, rows, report = write_run(tmp_path, 0)
    if corruption == "provenance":
        rows[0]["hand_provenance"] = "ground_truth_assisted"
    elif corruption == "missing_tip":
        rows = [row for row in rows if not (row["hand"] == "left" and row["fingertip"] == "little")]
        report["contact_valid_count"] = len(rows)
    elif corruption == "bridge":
        row = next(
            row for row in rows if row["hand"] == "right" and row["fingertip"] == "thumb" and row["frame_id"] == 1
        )
        row.update(bridged=True, window_contact=True)
        report["bridged_count"] = 1
    (run / "contacts/contacts.json").write_text(json.dumps(rows))
    (run / "report/report.json").write_text(json.dumps(report))
    result = audit(tmp_path, optimize=True)
    assert (result.returncode == 0) == (corruption is None), result.stderr
    if corruption is not None:
        assert not (tmp_path / "contact-artifact-audit.json").exists()


class ContactArtifactAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_declared_strict_and_legacy_policies_pass(self):
        for maximum in (0, 15):
            with self.subTest(maximum=maximum):
                root = self.root / str(maximum)
                write_run(root, maximum)
                result = audit(root)
                self.assertEqual(result.returncode, 0, result.stderr)
                receipt = json.loads((root / "contact-artifact-audit.json").read_text())[0]
                self.assertEqual(receipt["bridge_frames"], maximum)
                self.assertEqual(receipt["bridged"], int(maximum > 0))

    def test_missing_fingertip_fails_even_when_time_axis_and_counts_match(self):
        run, rows, report = write_run(self.root, 0)
        rows = [row for row in rows if not (row["hand"] == "left" and row["fingertip"] == "little")]
        report["contact_valid_count"] = len(rows)
        (run / "contacts/contacts.json").write_text(json.dumps(rows))
        (run / "report/report.json").write_text(json.dumps(report))
        self.assertNotEqual(audit(self.root).returncode, 0)

    def test_settings_cannot_hide_a_bridge_behind_strict_manifest(self):
        run, _, _ = write_run(self.root, 15)
        lock = json.loads((run / "inputs.lock.json").read_text())
        lock["manifest"]["settings"]["bridge_frames"] = 0
        (run / "inputs.lock.json").write_text(json.dumps(lock))
        self.assertNotEqual(audit(self.root).returncode, 0)

    def test_manifest_timestamp_mismatch_fails(self):
        run, _, _ = write_run(self.root, 0)
        lock = json.loads((run / "inputs.lock.json").read_text())
        lock["manifest"]["frames"][1]["timestamp_s"] += 0.01
        (run / "inputs.lock.json").write_text(json.dumps(lock))
        self.assertNotEqual(audit(self.root).returncode, 0)

    def test_strict_run_rejects_fabricated_bridge_with_matching_counts(self):
        run, rows, report = write_run(self.root, 0)
        row = next(
            row for row in rows if row["hand"] == "right" and row["fingertip"] == "thumb" and row["frame_id"] == 1
        )
        row.update(bridged=True, window_contact=True)
        report["bridged_count"] = 1
        (run / "contacts/contacts.json").write_text(json.dumps(rows))
        (run / "report/report.json").write_text(json.dumps(report))
        self.assertNotEqual(audit(self.root).returncode, 0)


if __name__ == "__main__":
    unittest.main()
