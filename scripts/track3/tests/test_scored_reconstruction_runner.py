"""Host runner must reject mutable images, leaked evaluation assets and changed code."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_scored_reconstruction import PERCEPTION, SAM3D, image_digest, validate


@pytest.fixture
def config(tmp_path):
    c = {
        "episode": 3,
        "data_subdir": "raw",
        "perception_image": PERCEPTION,
        "sam3d_image": SAM3D,
        "max_seconds": 1800,
    }
    for name in [
        "source_root",
        "models",
        "observations",
        "temporal_fits",
        "cameras",
        "overlay",
    ]:
        p = tmp_path / name
        p.mkdir()
        c[name] = str(p)
    raw = Path(c["source_root"]) / "raw"
    (raw / "videos").mkdir(parents=True)
    (raw / "meta").mkdir()
    (raw / "meta/camera_calibration.json").write_text("{}")
    roster = tmp_path / "roster.json"
    roster.write_text(json.dumps({"episodes": {"3": ["white_pot", "wooden_spoon"]}}))
    c["roster"] = str(roster)
    (Path(c["overlay"]) / "context-manifest.json").write_text(
        json.dumps(
            {
                "schema": "v2d.scored-perception-context.v1",
                "source_commit": "test",
                "files": {},
                "private_footage_included": False,
                "model_weights_included": False,
                "credentials_included": False,
            }
        )
    )
    return c


def test_pinned_images_and_distinct_official_slot_order(config):
    assert image_digest(PERCEPTION) == PERCEPTION
    assert validate(config) == (["white_pot", "wooden_spoon"], "test")
    for mutable in ["image:latest", "image@sha256:123", "https://example.com/image"]:
        with pytest.raises(ValueError, match="immutable"):
            image_digest(mutable)


@pytest.mark.parametrize("folder", ["mesh", "urdf", "data"])
def test_inference_cannot_mount_reference_geometry_or_pose_tables(config, folder):
    (Path(config["source_root"]) / "raw" / folder).mkdir()
    with pytest.raises(ValueError, match="only videos"):
        validate(config)


def test_symlink_escape_is_not_a_portable_source_view(config, tmp_path):
    raw = Path(config["source_root"]) / "raw"
    (raw / "videos").rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (raw / "videos").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escape"):
        validate(config)


def test_changed_overlay_and_unbounded_run_fail_before_launch(config):
    c = dict(config, max_seconds=0)
    with pytest.raises(ValueError, match="deadline"):
        validate(c)
    p = Path(config["overlay"]) / "context-manifest.json"
    m = json.loads(p.read_text())
    m["files"]["missing.py"] = "f" * 64
    p.write_text(json.dumps(m))
    with pytest.raises(FileNotFoundError):
        validate(config)
