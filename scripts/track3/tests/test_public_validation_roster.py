"""Held-out inference receives names/clock only, without pose values or CAD."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from make_public_validation_roster import make


def inputs(tmp_path, right_fps=20):
    source = tmp_path / "inference"
    (source / "meta").mkdir(parents=True)
    (source / "meta/camera_calibration.json").write_text("{}")
    metadata = tmp_path / "reference-only"
    metadata.mkdir()
    table = pa.table({"secret_pose_values": [13, 17, 23]}).replace_schema_metadata(
        {b"objects": json.dumps(["z_object", "a_object"]).encode()}
    )
    pq.write_table(table, metadata / "episode_000011.parquet")
    for cam in ["a", "b", "c"]:
        p = (
            source
            / f"videos/chunk-000/observation.images.ego_cam_{cam}/episode_000011.mp4"
        )
        p.parent.mkdir(parents=True)
        writer = cv2.VideoWriter(
            str(p),
            cv2.VideoWriter_fourcc(*"mp4v"),
            right_fps if cam == "b" else 20,
            (16, 16),
        )
        assert writer.isOpened()
        for v in [0, 40, 80]:
            writer.write(np.full((16, 16, 3), v, np.uint8))
        writer.release()
    return source, metadata


def test_names_and_clock_are_exported_but_reference_values_are_never_decoded(
    tmp_path, monkeypatch
):
    source, metadata = inputs(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("Read reference pose columns")

    monkeypatch.setattr(pq, "read_table", forbidden)
    out = tmp_path / "roster.json"
    r = make(source, metadata, [11], out)
    assert r["episodes"] == {"11": ["z_object", "a_object"]}
    assert r["video_axes"]["11"]["frames"] == 3
    assert r["reference_pose_values_read"] is False
    assert r["method_frozen"] is False
    assert "secret_pose_values" not in out.read_text()


def test_reference_tables_inside_inference_view_and_mismatched_clocks_fail(tmp_path):
    source, metadata = inputs(tmp_path, right_fps=25)
    with pytest.raises(ValueError, match="clocks differ"):
        make(source, metadata, [11], tmp_path / "out.json")
    with pytest.raises(ValueError, match="outside"):
        make(source, source / "metadata", [11], tmp_path / "out.json")


def test_annotated_mask_cache_cannot_be_reused_for_automatic_video_validation():
    from batch_public import require_automatic_mask_prompts

    require_automatic_mask_prompts({"seeds": {"cup": {"label": "cup", "score": 0.9}}})
    for report in [
        {"seeds": {"water_pitcher": {"label": "manual:water_pitcher"}}},
        {"manual_annotations_used": True, "seeds": {}},
    ]:
        with pytest.raises(ValueError, match="automatic prompts"):
            require_automatic_mask_prompts(report)
