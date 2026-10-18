"""A real tiny video fixture exercises scored ingestion without pose tables."""

import json
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
