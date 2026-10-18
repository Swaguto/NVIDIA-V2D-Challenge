import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_reference_contacts import summarize_trajectory


def test_empty_contact_knots_are_retained_and_pairs_summarized_once_per_frame():
    def pair(depth):
        return {"bodies": ["pot", "support"], "maximum_penetration_m": depth}

    frames = [
        {"reference_frame": 0, "contact_pairs": []},
        {"reference_frame": 1, "contact_pairs": [pair(0.0005)]},
        {"reference_frame": 2, "contact_pairs": [pair(0.02)]},
    ]
    assert summarize_trajectory(frames) == [
        {
            "bodies": ["pot", "support"],
            "frames_over_tolerance": 1,
            "maximum_penetration_m": 0.02,
            "worst_reference_frame": 2,
        }
    ]
    with pytest.raises(ValueError, match="every ordered"):
        summarize_trajectory(frames[1:])
    frames[2]["contact_pairs"].append(pair(0.03))
    with pytest.raises(ValueError, match="repeated"):
        summarize_trajectory(frames)
