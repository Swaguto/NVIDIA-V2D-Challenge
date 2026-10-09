"""Independent labels, unknown geometry and benchmark coverage controls."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contact_validation as cv


def reference():
    return {'schema_version': 1, 'episode': 2, 'granularity': 'fingertip',
            'independent_of_reconstruction': True, 'source': 'Independent source image annotations',
            'evidence': {'source_sha256': 'a' * 64},
            'labels': [{'frame': f, 'hand': 'left', 'object_id': 'pot', 'fingertip': 'index', 'contact': truth}
                       for f, truth in enumerate([True, False, True, None])]}


def test_precision_does_not_replace_unknown_labels_or_missing_geometry():
    table = {('left', 'index', 'pot'): {f: (None, None, d, None) for f, d in ((0, .01), (1, .01), (3, .01))}}
    result = cv.score_contact_reference(table, reference(), range(4), ['pot'], ['left'], ['index'], 2)
    assert result['precision_on_labelled_predictions'] == .5
    assert result['positive_prediction_label_coverage'] == pytest.approx(2 / 3)
    assert result['recall_on_labelled_positives'] == .5
    assert result['unknown_geometry_positive_labels'] == 1
    assert result['positive_predictions_without_truth'] == 1


def test_partial_hand_geometry_cannot_be_called_no_contact():
    table = {('left', 'index', 'pot'): {0: (None, None, .1, None)}}
    assert cv.contact_prediction(table, 0, 'left', 'pot', None, ['index', 'thumb']) is None
    table[('left', 'index', 'pot')][0] = (None, None, .01, None)
    assert cv.contact_prediction(table, 0, 'left', 'pot', None, ['index', 'thumb']) is True


@pytest.mark.parametrize('change', ['predicted', 'episode', 'duplicate', 'absent', 'truth_type'])
def test_invalid_or_self_referential_benchmark_is_rejected(change):
    truth = reference()
    if change == 'predicted':
        truth['independent_of_reconstruction'] = False
    elif change == 'episode':
        truth['episode'] = 0
    elif change == 'duplicate':
        truth['labels'].append(truth['labels'][0])
    elif change == 'absent':
        truth['labels'][0]['frame'] = 100
    else:
        truth['labels'][0]['contact'] = 1
    with pytest.raises(ValueError):
        cv.score_contact_reference({}, truth, range(4), ['pot'], ['left'], ['index'], 2)
