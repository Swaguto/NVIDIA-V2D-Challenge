from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.training_metrics import nonfinite_fields


def test_bad_metrics_are_named_and_not_replaced_with_plausible_values():
    metrics = {"reward": float("nan"), "kl": float("inf"), "steps": 10, "loss": 0.1}
    assert nonfinite_fields(metrics) == {"reward": "nan", "kl": "inf"}
    assert metrics["kl"] == float("inf")
