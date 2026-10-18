import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from instance_mask_quality import numbered_mask_conflicts


def test_fused_numbered_masks_are_unknown_without_guessing_the_front_object():
    a = np.zeros((20, 30), bool)
    b = a.copy()
    a[2:14, 2:10] = True
    b[2:14, 18:26] = True
    names = ["wooden_piece_1", "wooden_piece_2"]
    assert numbered_mask_conflicts(names, dict(zip(names, [a, b]))) == set()
    assert numbered_mask_conflicts(names, dict(zip(names, [a | b, b]))) == {0, 1}


def test_partial_separate_visibility_and_missing_masks_do_not_create_conflicts():
    a = np.zeros((20, 30), bool)
    b = a.copy()
    a[2:14, 2:12] = True
    b[2:14, 10:20] = True
    names = ["wooden_piece_1", "wooden_piece_2"]
    assert numbered_mask_conflicts(names, dict(zip(names, [a, b]))) == set()
    assert numbered_mask_conflicts(names, {names[0]: a}) == set()
    assert (
        numbered_mask_conflicts(
            ["blue_cup", "beige_cup"], {"blue_cup": a, "beige_cup": a}
        )
        == set()
    )


def test_numbered_conflict_check_rejects_mixed_camera_grids():
    with pytest.raises(ValueError, match="pixel grid"):
        numbered_mask_conflicts(
            ["piece_1", "piece_2"],
            {"piece_1": np.ones((3, 4)), "piece_2": np.ones((4, 3))},
        )


def test_old_numbered_mask_cache_cannot_silently_skip_the_new_source_gate():
    from batch_public import require_automatic_mask_prompts

    old = {
        "objects": ["wooden_piece_1", "wooden_piece_2"],
        "manual_annotations_used": False,
    }
    with pytest.raises(ValueError, match="predates ambiguity"):
        require_automatic_mask_prompts(old)
    require_automatic_mask_prompts(
        dict(
            old,
            numbered_mask_quality_policy="shared-foreground-overlap-unknown-v1",
            numbered_overlap_intersection_over_smaller_threshold=0.70,
        )
    )
