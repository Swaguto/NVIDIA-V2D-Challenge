import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from object_prompts import assign_label, numbered_groups, numbered_instance_seeds


def test_natural_detector_aliases_and_specific_lids_map_without_guessing_colors():
    assert assign_label("watering can", ["blue_cup", "water_pitcher"]) == 1
    assert assign_label("pitcher", ["blue_cup", "water_pitcher"]) == 1
    assert assign_label("cup", ["blue_cup", "water_pitcher"]) == 0
    assert assign_label("cup", ["blue_cup", "beige_cup"]) is None
    assert assign_label("white pot lid", ["white_pot", "white_pot_lid"]) == 1


def test_combined_numbered_label_does_not_arbitrarily_select_the_last_slot():
    names = ["wooden_piece_1", "wooden_piece_2"]
    assert assign_label("wooden piece 1 wooden piece 2", names) is None
    assert numbered_groups(names) == [(0, 1)]
    one = {"label": "wooden board", "score": 0.8, "box": [10, 10, 30, 50]}
    assert numbered_instance_seeds([one], names, (0, 1)) == {}


def test_distinct_boards_keep_inferred_identity_and_reject_a_fused_parent_box():
    names = ["wooden_piece_1", "wooden_piece_2"]
    ds = [
        {"label": "wooden board", "score": 0.7, "box": [70, 10, 90, 50]},
        {"label": "wooden board", "score": 0.8, "box": [10, 10, 30, 50]},
        {"label": "wooden board", "score": 0.99, "box": [0, 0, 100, 60]},
    ]
    seeds = numbered_instance_seeds(ds, names, (0, 1))
    assert seeds[0]["box"] == [10, 10, 30, 50] and seeds[1]["box"] == [70, 10, 90, 50]
    assert all(
        s["identity_policy"] == "inferred_initial_left_to_right" for s in seeds.values()
    )


def test_specific_numbered_labels_override_the_spatial_convention():
    names = ["wooden_piece_1", "wooden_piece_2"]
    ds = [
        {"label": "wooden piece 1", "score": 0.7, "box": [70, 10, 90, 50]},
        {"label": "wooden piece 2", "score": 0.8, "box": [10, 10, 30, 50]},
    ]
    seeds = numbered_instance_seeds(ds, names, (0, 1))
    assert seeds[0]["box"][0] == 70
    assert all(
        s["identity_policy"] == "automatic_specific_label" for s in seeds.values()
    )
