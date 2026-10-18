"""A real tiny video fixture exercises scored ingestion without pose tables."""

import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "reconstruction/modules/v2d_issue6"))
from issue6.track3_adapter import export


def fixture(tmp_path, monkeypatch, right_fps=20):
    kit = tmp_path / "kit"
    (kit / "data").mkdir(parents=True)
    (kit / "data/track_3_evaluation_objects.json").write_text(
        json.dumps({"episodes": {"1": ["z_object", "a_object"]}})
    )
    pq.write_table(
        pa.table(
            {
                "row_id": [
                    f"track_3/e000001/w000/f{frame:06d}/o{slot}"
                    for frame in range(3)
                    for slot in range(2)
                ]
            }
        ),
        kit / "data/track_3_sample_submission.parquet",
    )
    data = tmp_path / "data"
    (data / "meta").mkdir(parents=True)
    (data / "meta/camera_calibration.json").write_text("{}")
    for cam, rate in [("c", 20), ("b", right_fps)]:
        path = (
            data
            / f"videos/chunk-000/observation.images.ego_cam_{cam}/episode_000001.mp4"
        )
        path.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), rate, (16, 16)
        )
        assert writer.isOpened()
        for value in (10, 20, 30):
            writer.write(np.full((16, 16, 3), value, np.uint8))
        writer.release()
    for name in ("z_object", "a_object"):
        folder = data / "mesh" / name
        folder.mkdir(parents=True)
        (folder / (name + ".glb")).write_bytes(
            b"mesh placeholder not consumed by this input adapter"
        )
    poison = data / "data/chunk-000/episode_000001.parquet"
    poison.parent.mkdir(parents=True)
    poison.write_bytes(b"Pose-label tables must never be opened in scored mode")
    rig = SimpleNamespace(
        left="ego_cam_c",
        right="ego_cam_b",
        size=(16, 16),
        P1=np.c_[np.eye(3), np.zeros(3)],
        rectify_pair=lambda left, right: (left, right),
    )
    monkeypatch.setitem(
        sys.modules,
        "stereo_depth",
        SimpleNamespace(
            load_rig=lambda path: rig,
            stereo_depth=lambda *args: (np.ones((16, 16)), np.ones((16, 16), bool)),
        ),
    )
    return kit, data


def test_scored_video_clock_and_slot_order_need_no_pose_tables(tmp_path, monkeypatch):
    kit, data = fixture(tmp_path, monkeypatch)
    out = tmp_path / "output"
    path = export(
        data,
        tmp_path / "predictions",
        1,
        out,
        0,
        0,
        {"backend": "hamer"},
        scored_kit=kit,
    )
    manifest = json.loads(path.read_text())
    assert [o["id"] for o in manifest["objects"]] == ["z_object", "a_object"]
    assert [f["timestamp_s"] for f in manifest["frames"]] == [0, 0.05, 0.1]
    assert manifest["input_contract"]["reference_pose_tables_read"] is False
    assert all(not Path(o["mesh"]).is_absolute() for o in manifest["objects"])
    assert all(
        not state["valid"]
        for frame in manifest["frames"]
        for state in frame["objects"].values()
    )


def test_partial_scored_source_is_rejected_before_outputs(tmp_path, monkeypatch):
    kit, data = fixture(tmp_path, monkeypatch)
    out = tmp_path / "output"
    with pytest.raises(ValueError, match="complete original"):
        export(
            data,
            tmp_path / "predictions",
            1,
            out,
            0,
            2,
            {"backend": "hamer"},
            scored_kit=kit,
        )
    assert not out.exists()


def test_stereo_timestamps_cannot_silently_use_different_clocks(tmp_path, monkeypatch):
    kit, data = fixture(tmp_path, monkeypatch, right_fps=25)
    with pytest.raises(ValueError, match="timestamps"):
        export(
            data,
            tmp_path / "predictions",
            1,
            tmp_path / "output",
            0,
            0,
            {"backend": "hamer"},
            scored_kit=kit,
        )


def test_human_input_preparation_never_opens_provided_object_geometry(
    tmp_path, monkeypatch
):
    kit, data = fixture(tmp_path, monkeypatch)
    import issue6.track3_adapter as adapter

    def reject_geometry(*args):
        raise AssertionError("Human observation preparation opened object geometry")

    monkeypatch.setattr(adapter, "mesh_for", reject_geometry)
    path = export(
        data,
        tmp_path / "predictions",
        1,
        tmp_path / "output",
        0,
        0,
        {"backend": "hamer"},
        scored_kit=kit,
        hands_only=True,
    )
    manifest = json.loads(path.read_text())
    assert manifest["objects"] == []
    assert manifest["input_contract"]["official_object_slot_names"] == [
        "z_object",
        "a_object",
    ]
    assert manifest["input_contract"]["object_geometry_read"] is False
    assert all(row["objects"] == {} for row in manifest["frames"])


@pytest.mark.parametrize("wrong_pose_mesh", [False, True])
def test_binding_own_objects_preserves_human_bytes_and_rejects_different_pose_geometry(
    tmp_path, monkeypatch, wrong_pose_mesh
):
    kit, data = fixture(tmp_path, monkeypatch)
    human = export(
        data,
        tmp_path / "predictions",
        1,
        tmp_path / "human",
        0,
        0,
        {"backend": "hamer"},
        scored_kit=kit,
        hands_only=True,
    )
    sys.path.insert(0, str(ROOT / "scripts/track3"))
    from bind_scored_objects import bind
    from reconstructed_view import prepare
    from issue6.manifest import load

    before = load(human)
    image_path = Path(before["frames"][0]["color"])
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    names = before["input_contract"]["official_object_slot_names"]
    for name in names:
        folder = tmp_path / "geometry" / name
        folder.mkdir(parents=True)
        payload = ("own video-derived geometry " + name).encode()
        (folder / "mesh.glb").write_bytes(payload)
        proof = dict(
            schema="v2d.video-object-reconstruction.v1",
            stage="multi_view_tsdf",
            episode=1,
            object=name,
            provided_meshes_read=False,
            reference_pose_labels_read=False,
            units="metres",
            body_origin_frozen=True,
            mesh_sha256=hashlib.sha256(payload).hexdigest(),
        )
        (folder / "manifest.json").write_text(json.dumps(proof))
    view = tmp_path / "view"
    prepare(data, tmp_path / "geometry", 1, names, view)
    if wrong_pose_mesh:
        folder = (
            tmp_path
            / "predictions/e001/poses/e001/ego_object_poses/e001/z_object/object_to_cam/ego_cam_c"
        )
        folder.mkdir(parents=True)
        (folder / "000000.json").write_text(
            json.dumps(
                dict(
                    episode=1,
                    frame_index=0,
                    object="z_object",
                    status="observed",
                    gt_use="none",
                    camera_frame="ego_cam_c",
                    coordinate_frame="rectified_camera",
                    mesh_sha256="different",
                    rotation=[1, 0, 0, 0],
                    translation=[0, 0, 0.5],
                    scale=1.0,
                )
            )
        )
        with pytest.raises(ValueError, match="different geometry"):
            bind(
                human, view, tmp_path / "predictions", tmp_path / "final/manifest.json"
            )
    else:
        # Foreground masks still constrain hand alignment when no 3-D object
        # pose was accepted. This must not fabricate object observations.
        mask = tmp_path / "predictions/e001/masks/z_object/000000.png"
        mask.parent.mkdir(parents=True)
        cv2.imwrite(str(mask), np.ones((8, 10), np.uint8) * 255)
        receipt = bind(
            human, view, tmp_path / "predictions", tmp_path / "final/manifest.json"
        )
        after = load(tmp_path / "final/manifest.json")
        assert [o["id"] for o in after["objects"]] == names
        assert after["frames"][0]["color"] == str(image_path)
        assert hashlib.sha256(image_path.read_bytes()).hexdigest() == image_hash
        assert before["objects"] == [] and not receipt["human_observations_changed"]
        state = after["frames"][0]["objects"]["z_object"]
        assert state["mask"] == str(mask) and state["valid"] is False
        assert "pose_path" not in state
        assert (
            state["mask_provenance"] == "predicted_image_foreground_independent_of_pose"
        )
