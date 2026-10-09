"""Portable public evidence locations must agree with the accepted run receipt."""
import json
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parents[3] / "docs/team/issue7"


def test_public_receipts_do_not_contain_personal_workspace_paths():
    pattern = re.compile(r"[A-Za-z]:[/\\]+Users[/\\]+", re.IGNORECASE)
    for name in ("RESULTS.json", "RESULTS.md", "REPRODUCE.md", "REVIEW.md"):
        assert pattern.search((DOCS / name).read_text(encoding="utf-8")) is None, name


def test_reproduction_names_the_accepted_slam_run():
    receipt = json.loads((DOCS / "RESULTS.json").read_text(encoding="utf-8"))
    assert receipt["history"]["slam_trial2"].startswith("Accepted")
    reproduction = (DOCS / "REPRODUCE.md").read_text(encoding="utf-8")
    assert "runs/issue7-validation/slam-trial2/" in reproduction
    assert "slam-trial4" not in reproduction
