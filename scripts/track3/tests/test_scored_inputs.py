import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from acquire_scored_inputs import select, verify
from batch_public import roster
from prepare_scored_view import prepare


def inventory():
    files = [
        {
            "type": "file",
            "path": f"track_3/evaluation/videos/chunk-000/observation.images.ego_cam_{cam}/episode_000003.mp4",
        }
        for cam in "abc"
    ]
    files.append(
        {"type": "file", "path": "track_3/public/data/chunk-000/episode_000002.parquet"}
    )
    return {"revision": "a" * 40, "pagination_link": None, "items": files}


def test_evaluation_input_selection_excludes_public_pose_labels():
    selected = select(inventory(), {"3": ["white_pot", "wooden_spoon"]})
    assert len(selected) == 3
    assert not any(item["path"].endswith("parquet") for item in selected)


def test_inference_acquisition_excludes_cad_and_unneeded_public_metadata():
    data = inventory()
    data["items"] += [
        {"type": "file", "path": p}
        for p in [
            "track_3/public/meta/camera_calibration.json",
            "track_3/public/meta/episodes_metadata.jsonl",
            "track_3/public/mesh/cup/cup.glb",
            "track_3/public/urdf/cup.urdf",
        ]
    ]
    selected = select(data, {"3": ["white_pot", "wooden_spoon"]}, videos_only=True)
    assert len(selected) == 4
    assert {Path(item["path"]).suffix for item in selected} == {".mp4", ".json"}


def test_video_view_cannot_expose_old_diagnostic_assets(tmp_path):
    from run_scored_reconstruction import validate_video_mount

    root = tmp_path / "dataset"
    (root / "track_3/evaluation/videos").mkdir(parents=True)
    camera = root / "track_3/public/meta/camera_calibration.json"
    camera.parent.mkdir(parents=True)
    camera.write_text("{}")
    receipt = tmp_path / "acquisition.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "v2d.scored-input-acquisition.v1",
                "revision": "a" * 40,
                "pose_labels_downloaded": False,
                "videos_only": True,
                "files": [
                    {
                        "path": str(camera.relative_to(root)),
                        "size": 2,
                        "sha256": hashlib.sha256(b"{}").hexdigest(),
                    }
                ],
            }
        )
    )
    view = tmp_path / "raw"
    prepare(root, receipt, view, videos_only=True)
    assert sorted(p.name for p in view.iterdir()) == [
        "input-view.json",
        "meta",
        "videos",
    ]
    validate_video_mount(tmp_path, view)
    (root / "track_3/public/mesh").mkdir()
    with pytest.raises(ValueError, match="only videos"):
        prepare(root, receipt, tmp_path / "rejected", videos_only=True)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_camera",
        "pagination",
        "mutable_revision",
        "unexpected_episode",
        "escape",
    ],
)
def test_incomplete_or_unexpected_inventory_fails_closed(bad):
    data = inventory()
    if bad == "missing_camera":
        data["items"].pop(0)
    if bad == "pagination":
        data["pagination_link"] = "next page"
    if bad == "mutable_revision":
        data["revision"] = "main"
    if bad == "unexpected_episode":
        data["items"][0]["path"] = data["items"][0]["path"].replace("000003", "000004")
    if bad == "escape":
        data["items"][0]["path"] = "../private"
    with pytest.raises(ValueError):
        select(data, {"3": ["white_pot", "wooden_spoon"]})


def test_reused_file_must_match_upstream_identity(tmp_path):
    path = tmp_path / "video"
    path.write_bytes(b"verified")
    item = {
        "path": "video",
        "size": 8,
        "lfs": {"oid": hashlib.sha256(b"verified").hexdigest()},
    }
    verify(path, item)
    path.write_bytes(b"modified")
    with pytest.raises(ValueError, match="identity"):
        verify(path, item)


def test_scored_roster_preserves_unsorted_object_slots_without_pose_tables(tmp_path):
    path = tmp_path / "roster.json"
    path.write_text(
        json.dumps(
            {
                "episodes": {
                    "3": ["white_pot", "wooden_spoon"],
                    "1": ["white_pot", "white_pot_lid"],
                }
            }
        )
    )
    assert list(roster(tmp_path / "no-pose-tables", path)) == [
        (1, ["white_pot", "white_pot_lid"]),
        (3, ["white_pot", "wooden_spoon"]),
    ]


def test_scored_view_rejects_corrupted_acquired_sources_before_creating_output(
    tmp_path,
):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "video").write_bytes(b"corrupt")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "v2d.scored-input-acquisition.v1",
                "pose_labels_downloaded": False,
                "files": [{"path": "video", "size": 7, "sha256": "a" * 64}],
            }
        )
    )
    output = tmp_path / "view"
    with pytest.raises(ValueError):
        prepare(root, receipt, output)
    assert not output.exists()
