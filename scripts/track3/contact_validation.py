"""Contact precision against explicit independent labels, with unknowns retained.

Grasp timings alone are not fingertip truth. Benchmarks must declare their
granularity; sparse labels produce conditional precision and coverage, never a
claim that unlabelled predictions have been validated.
"""
from __future__ import annotations

import math


def validate_reference(reference, frames, objects, hands, tips, episode=None):
    if reference.get('schema_version') != 1:
        raise ValueError('Expected contact-truth schema_version 1')
    if reference.get('granularity') not in ('fingertip', 'hand_object'):
        raise ValueError('Contact truth must declare fingertip or hand_object granularity')
    if reference.get('independent_of_reconstruction') is not True:
        raise ValueError('Predicted contacts cannot serve as independent contact truth')
    evidence = reference.get('evidence', {}).get('source_sha256', '')
    if not isinstance(evidence, str) or len(evidence) != 64 or any(c not in '0123456789abcdef' for c in evidence):
        raise ValueError('Independent labels require a source evidence SHA256')
    if not isinstance(reference.get('source'), str) or not reference['source'].strip():
        raise ValueError('Independent labels require a documented source')
    if type(reference.get('episode')) is not int or (episode is not None and reference['episode'] != episode):
        raise ValueError('Contact truth is for a different or missing episode')
    rows = reference.get('labels')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Independent contact labels are empty')
    labels = {}
    for row in rows:
        frame, hand, obj = row.get('frame'), row.get('hand'), row.get('object_id')
        if type(frame) is not int or frame not in frames or hand not in hands or obj not in objects:
            raise ValueError('Contact label uses an absent frame, hand or object')
        tip = row.get('fingertip') if reference['granularity'] == 'fingertip' else None
        if reference['granularity'] == 'hand_object' and row.get('fingertip') is not None:
            raise ValueError('Fingertip truth cannot silently become a hand-object label')
        if (tip is not None and tip not in tips) or (reference['granularity'] == 'fingertip' and tip is None):
            raise ValueError('Invalid fingertip in independent contact label')
        if 'contact' not in row:
            raise ValueError('Contact label must explicitly record truth or unknown')
        truth = row['contact']
        if truth is not None and type(truth) is not bool:
            raise ValueError('Contact truth must be true, false or null (unknown)')
        key = (frame, hand, obj, tip)
        if key in labels:
            raise ValueError('Duplicate independent contact label')
        labels[key] = truth
    return labels


def contact_prediction(table, frame, hand, obj, tip, tips):
    distances = []
    for finger in ([tip] if tip is not None else tips):
        row = table.get((hand, finger, obj), {}).get(frame)
        distance = float(row[2]) if row is not None else float('nan')
        distances.append(distance)
    observed = [math.isfinite(d) and d >= 0 for d in distances]
    if any(ok and d < 0.02 for ok, d in zip(observed, distances)):
        return True
    if all(observed):
        return False
    return None


def score_contact_reference(table, reference, frames, objects, hands, tips, episode=None):
    frames = list(frames)
    labels = validate_reference(reference, set(frames), objects, hands, tips, episode)
    counts = {'true_positive': 0, 'false_positive': 0, 'true_negative': 0, 'false_negative': 0,
              'unknown_geometry': 0, 'unknown_geometry_positive_labels': 0}
    positive_labels = sum(truth is True for truth in labels.values())
    known_labels = sum(truth is not None for truth in labels.values())
    for (frame, hand, obj, tip), truth in labels.items():
        if truth is None:
            continue
        prediction = contact_prediction(table, frame, hand, obj, tip, tips)
        if prediction is None:
            counts['unknown_geometry'] += 1
            counts['unknown_geometry_positive_labels'] += int(truth)
        else:
            name = ('true_' if prediction == truth else 'false_') + ('positive' if prediction else 'negative')
            counts[name] += 1
    all_positive = sum(
        contact_prediction(table, frame, hand, obj, tip, tips) is True
        for frame in frames for hand in hands for obj in objects
        for tip in (tips if reference['granularity'] == 'fingertip' else [None])
    )
    evaluated_positive = counts['true_positive'] + counts['false_positive']
    return {**counts, 'granularity': reference['granularity'], 'source': reference['source'],
            'evidence': reference['evidence'], 'labels': len(labels), 'known_labels': known_labels,
            'positive_labels': positive_labels, 'unknown_labels': len(labels) - known_labels,
            'all_positive_predictions': all_positive,
            'positive_predictions_without_truth': all_positive - evaluated_positive,
            'precision_on_labelled_predictions': counts['true_positive'] / evaluated_positive if evaluated_positive else None,
            'positive_prediction_label_coverage': evaluated_positive / all_positive if all_positive else None,
            'recall_on_labelled_positives': counts['true_positive'] / positive_labels if positive_labels else None,
            'geometry_coverage_on_known_labels': (known_labels - counts['unknown_geometry']) / known_labels if known_labels else None,
            'threshold_m': 0.02,
            'interpretation': 'Precision is conditional on independently labelled predictions at the declared granularity. Unknown labels and missing geometry are reported; missing positive geometry stays in the recall denominator. Hand-object labels do not establish fingertip or metric 3D contact truth.'}
