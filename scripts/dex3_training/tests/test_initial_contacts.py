import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from audit_initial_contacts import summarize_contacts


def test_only_object_support_and_object_object_contacts_are_reported():
    rows = [(2, 8, -0.03), (8, 2, -0.01), (3, 2, 0.001), (5, 2, -0.1), (5, 8, -0.2)]
    result = summarize_contacts(rows, {2: "pot", 3: "spoon"}, 8)
    assert result == [
        {
            "bodies": ["pot", "spoon"],
            "contact_samples": 1,
            "minimum_signed_gap_m": 0.001,
            "maximum_penetration_m": 0.0,
        },
        {
            "bodies": ["pot", "support"],
            "contact_samples": 2,
            "minimum_signed_gap_m": -0.03,
            "maximum_penetration_m": 0.03,
        },
    ]
    with pytest.raises(ValueError, match="Nonfinite"):
        summarize_contacts([(2, 8, float("nan"))], {2: "pot"}, 8)
