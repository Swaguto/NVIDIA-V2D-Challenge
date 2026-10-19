"""Necessary next-pose boxes for a CORRELATED pair, without a graph recurrence.

The caller supplies the original aligned reference, actual URDF limits and
explicit original tolerances. These boxes are local outer restrictions only.
They do not establish that the supplied pair has a feasible prefix or future.
"""
import numpy as np


def next_correction_box(pair, original, dt, lower, upper, velocity, *,
                        component=.25, adjacent=.05, second=.05,
                        motion_tolerance=1e-7, position_tolerance=1e-8):
    pair = np.asarray(pair, float)
    original = np.asarray(original, float)
    lower, upper, velocity = [np.asarray(v, float) for v in [lower, upper, velocity]]
    dt = np.asarray(dt, float)
    if (pair.ndim != 2 or pair.shape[0] != 2 or original.shape != (3, pair.shape[1])
            or any(v.shape != (pair.shape[1],) for v in [lower, upper, velocity])
            or dt.shape != (2,)
            or not np.isfinite(np.r_[pair.ravel(), original.ravel(), lower, upper,
                                     velocity, dt, component, adjacent, second,
                                     motion_tolerance, position_tolerance]).all()
            or (lower > upper).any() or (velocity <= 0).any() or (dt <= 0).any()
            or min(component, adjacent, second, motion_tolerance, position_tolerance) < 0):
        raise ValueError('Finite correlated pair, three references and actual limits required')
    d = pair - original[:2]
    pair_checks = dict(
        actual_position=bool(((pair >= lower-position_tolerance) & (pair <= upper+position_tolerance)).all()),
        component=bool((abs(d) <= component+motion_tolerance).all()),
        actual_velocity=bool((abs(pair[1]-pair[0]) <= velocity*dt[0]+motion_tolerance).all()),
        adjacent_added=bool((abs(d[1]-d[0]) <= adjacent+motion_tolerance).all()),
    )
    centers = np.array([np.zeros_like(lower), d[1], 2*d[1]-d[0],
                        lower-original[2], pair[1]-velocity*dt[1]-original[2]])
    lower_terms = np.array([
        -np.ones_like(lower)*(component+motion_tolerance),
        d[1]-adjacent-motion_tolerance,
        2*d[1]-d[0]-second-motion_tolerance,
        lower-original[2]-position_tolerance,
        pair[1]-velocity*dt[1]-original[2]-motion_tolerance,
    ])
    upper_terms = np.array([
        np.ones_like(lower)*(component+motion_tolerance),
        d[1]+adjacent+motion_tolerance,
        2*d[1]-d[0]+second+motion_tolerance,
        upper-original[2]+position_tolerance,
        pair[1]+velocity*dt[1]-original[2]+motion_tolerance,
    ])
    lo, hi = lower_terms.max(axis=0), upper_terms.min(axis=0)
    return dict(pair_passed_local_checks=all(pair_checks.values()), pair_checks=pair_checks,
                pair_corrections=d, lower_terms=lower_terms, upper_terms=upper_terms,
                term_names=['component', 'adjacent_added', 'second_added', 'actual_position', 'actual_velocity'],
                correction_lower=lo, correction_upper=hi,
                joint_lower=original[2]+lo, joint_upper=original[2]+hi,
                empty=bool((lo > hi).any()), lower_active=lower_terms.argmax(axis=0),
                upper_active=upper_terms.argmin(axis=0),
                prefix_and_future_viability_unassessed=True)
