import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from scored_targets import load_world_targets


def fixture(root):
    for slot, name in enumerate(["pot", "lid"]):
        folder = root / f"e001/ego_object_poses/e001/{name}/object_to_world"
        folder.mkdir(parents=True)
        for frame in [0, 1, 2]:
            record = dict(
                episode=1,
                frame_index=frame,
                object=name,
                mesh_sha256=name + "-digest",
                coordinate_frame="fixed_world",
                gt_use="independent",
                status="inferred" if frame == 1 else "observed",
                inference_method="short_gap_interpolation",
                scale=[1, 1, 1],
                translation=[slot + 1, frame * 0.1, 0.5],
                rotation=[1, 0, 0, 0],
            )
            (folder / f"{frame:06d}.json").write_text(json.dumps(record))
    return root / "e001/ego_object_poses/e001/pot/object_to_world/000001.json"


def load(root):
    return load_world_targets(
        root,
        1,
        ["pot", "lid"],
        np.arange(3),
        {"pot": "pot-digest", "lid": "lid-digest"},
    )


def test_slot_order_metric_coordinates_and_unknown_observation_survive(tmp_path):
    fixture(tmp_path)
    poses, valid, provenance, files = load(tmp_path)
    np.testing.assert_array_equal(poses[:, :, 0, 3], [[1, 2]] * 3)
    np.testing.assert_array_equal(valid[:, 0], [True, False, True])
    assert provenance["pot"]["1"] == "inferred:short_gap_interpolation"
    assert len(files) == 6


@pytest.mark.parametrize(
    "change",
    [
        {"mesh_sha256": "different"},
        {"gt_use": "gt_assisted"},
        {"coordinate_frame": "rectified_camera"},
        {"frame_index": 2},
        {"inference_method": None},
        {"rotation": [0, 0, 0, 0]},
    ],
)
def test_scored_targets_reject_changed_geometry_labels_or_frames(tmp_path, change):
    path = fixture(tmp_path)
    record = json.loads(path.read_text())
    record.update(change)
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        load(tmp_path)


def test_missing_target_never_becomes_identity(tmp_path):
    fixture(tmp_path).unlink()
    with pytest.raises(FileNotFoundError):
        load(tmp_path)
