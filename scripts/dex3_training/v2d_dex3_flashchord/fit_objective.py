"""Rank native grasp candidates by violations of the fixed engineering gates."""

import numpy as np


def constraint_cost(metrics, joint_margin_rad, penetration_m):
    """A better safe rotation cannot compensate for a worse slipping grasp."""
    m = metrics
    ratios = np.array([
        m["maximum_relative_position_drift_m"] / 0.01,
        m["maximum_relative_rotation_drift_deg"] / 5,
        (1 - m["all_required_pad_sampled_support_fraction"]) / 0.05,
        1 + (m["maximum_reference_lift_m"] - m["maximum_actual_lift_m"] - 0.01) / 0.01,
        m["maximum_source_position_error_m"] / 0.03,
        m["maximum_source_orientation_error_deg"] / 15,
        1 + (0.003 - joint_margin_rad) / 0.003,
        penetration_m / 0.002,
    ], dtype=float)
    if not np.isfinite(ratios).all():
        raise ValueError("Candidate constraint metrics must be finite")
    ratios = np.maximum(ratios, 0)
    violations = np.maximum(ratios - 1, 0)
    # Violated constraints dominate. Already-safe quantities provide only a
    # small tie-break; the acceptance predicates remain in the native auditor.
    return float(1000 * violations.max() + 100 * (violations ** 2).sum()
                 + 0.01 * (np.minimum(ratios, 1) ** 2).sum())
