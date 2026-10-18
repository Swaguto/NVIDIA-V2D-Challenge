"""Host runner must reject mutable images, leaked evaluation assets and changed code."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from run_scored_reconstruction import (
    PERCEPTION,
    SAM3D,
    image_digest,
    relocate_camera_paths,
    run,
    sha,
    validate,
)


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


@pytest.mark.parametrize("relative", ["other/labels.parquet", "hidden/object.glb"])
def test_entire_parent_mount_is_checked_for_reference_assets(config, relative):
    path = Path(config["source_root"]) / relative
    path.parent.mkdir()
    path.touch()
    with pytest.raises(ValueError, match="Common inference mount"):
        validate(config)


def hand_config(config):
    config["split"] = "public"
    roster = Path(config["roster"])
    metadata = json.loads(roster.read_text())
    metadata.update(schema="v2d.public-video-validation-roster.v1", split="public")
    roster.write_text(json.dumps(metadata))
    for name in ["hand-runtime-v1/hamer/_DATA/data/mano", "hand-runtime-v1/public"]:
        (Path(config["models"]) / name).mkdir(parents=True)
    return config


def test_public_hand_driver_never_requires_reference_parquets_or_a_scored_kit(config):
    from run_scored_hands import validate as validate_hands

    c = hand_config(config)
    assert validate_hands(c) == "test"
    c["split"] = "evaluation"
    c["kit"] = "/does-not-exist"
    with pytest.raises(ValueError, match="official kit"):
        validate_hands(c)


def test_public_hand_driver_requires_its_explicit_split_contract(config):
    from run_scored_hands import validate as validate_hands

    c = hand_config(config)
    path = Path(c["roster"])
    roster = json.loads(path.read_text())
    roster.pop("schema")
    path.write_text(json.dumps(roster))
    with pytest.raises(ValueError, match="explicit video-only roster"):
        validate_hands(c)


def test_camera_relocation_keeps_original_bytes_and_relative_image_paths(tmp_path):
    source = tmp_path / "observations/e003/manifest.json"
    source.parent.mkdir(parents=True)
    camera = tmp_path / "camera/ego_cam_c"
    camera.mkdir(parents=True)
    record = {
        "episode": 3,
        "frame_index": 0,
        "gt_use": "independent",
        "camera_frame": "ego_cam_c",
        "translation": [0.1, 0.2, 0.3],
    }
    (camera / "000000.json").write_text(json.dumps(record))
    m = {
        "episode": 3,
        "color_camera": {"id": "ego_cam_c"},
        "frames": [
            {
                "id": 0,
                "color": "color/000000.png",
                "depth_path": "depth/000000.npy",
                "camera_pose_path": "../../camera/ego_cam_c/000000.json",
            }
        ],
        "input_contract": {},
    }
    source.write_text(json.dumps(m))
    original = source.read_bytes()
    output = tmp_path / "relocated.json"
    proof = relocate_camera_paths(source, camera.parent, output, "e003")
    actual = json.loads(output.read_text())
    assert source.read_bytes() == original
    assert actual["frames"][0]["color"] == m["frames"][0]["color"]
    assert actual["frames"][0]["depth_path"] == m["frames"][0]["depth_path"]
    assert (
        actual["frames"][0]["camera_pose_path"]
        == "/stage/source/e003/camera/ego_cam_c/000000.json"
    )
    assert proof["camera_values_changed"] is False
    assert proof["camera_files_sha256"]["000000.json"] == sha(camera / "000000.json")
    changed = tmp_path / "changed/ego_cam_c"
    changed.mkdir(parents=True)
    (changed / "000000.json").write_text(
        json.dumps(dict(record, translation=[1.0, 2.0, 3.0]))
    )
    with pytest.raises(ValueError, match="numerical source data"):
        relocate_camera_paths(source, changed.parent, tmp_path / "bad.json", "e003")


def test_bind_resume_does_not_repeat_neural_inference_and_preserves_failure(
    config, tmp_path, monkeypatch
):
    import os
    import shutil
    import subprocess

    out = tmp_path / "run"
    (out / "inputs").mkdir(parents=True)
    config.update(out=str(out), resume_from_bind=True)
    shutil.copyfile(config["roster"], out / "inputs/roster.json")
    camera = Path(config["cameras"]) / "e003/camera"
    (camera / "ego_cam_c").mkdir(parents=True)
    (camera / "report.json").write_text("{}")
    (camera / "ego_cam_c/000000.json").write_text(
        json.dumps(
            {
                "episode": 3,
                "frame_index": 0,
                "gt_use": "independent",
                "camera_frame": "ego_cam_c",
            }
        )
    )
    shutil.copytree(camera, out / "source/e003/camera")
    (out / "source/e003/masks").mkdir()
    (out / "source/e003/masks/report.json").write_text(
        json.dumps({"mask_counts": {"white_pot": 1, "wooden_spoon": 1}})
    )
    obs = Path(config["observations"]) / "e003/manifest.json"
    obs.parent.mkdir()
    obs.write_text(
        json.dumps(
            {
                "episode": 3,
                "color_camera": {"id": "ego_cam_c"},
                "frames": [
                    {
                        "id": 0,
                        "camera_pose_path": os.path.relpath(
                            camera / "ego_cam_c/000000.json", obs.parent
                        ),
                    }
                ],
                "input_contract": {},
            }
        )
    )
    fit = Path(config["temporal_fits"]) / "e003"
    fit.mkdir()
    for name in ["fit_receipt.json", "observation_gate.json"]:
        (fit / name).write_text("{}")
    model = Path(config["models"]) / "sam3d/download-receipt.json"
    model.parent.mkdir()
    model.write_text("{}")
    (out / "selections").mkdir()
    for obj in ["white_pot", "wooden_spoon"]:
        (out / "selections" / (obj + ".json")).write_text('{"selected":{"frame":0}}')
    names = (
        ["masks"]
        + [
            obj + "-" + stage
            for obj in ["white_pot", "wooden_spoon"]
            for stage in ["selection", "sam3d", "review"]
        ]
        + ["view", "poses", "world"]
    )
    previous = {
        "source_commit": "original",
        "episode": 3,
        "object_slot_names": ["white_pot", "wooden_spoon"],
        "images": {"perception": PERCEPTION, "sam3d": SAM3D},
        "config_sha256": "original-config",
        "stages": [{"name": n, "returncode": 0} for n in names]
        + [{"name": "bind", "returncode": 1}],
        "error": "missing camera namespace",
        "sources_sha256": {
            "roster": sha(Path(config["roster"])),
            "hand_observations": sha(obs),
            "temporal_fit": sha(fit / "fit_receipt.json"),
            "observation_gate": sha(fit / "observation_gate.json"),
            "camera_report": sha(camera / "report.json"),
            "model_download_receipt": sha(model),
        },
    }
    (out / "receipt.json").write_text(json.dumps(previous))
    (out / "bind.log").write_text("original missing-camera stack")
    calls = []

    def fail_bind(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(subprocess, "run", fail_bind)
    with pytest.raises(subprocess.CalledProcessError):
        run(config)
    assert len(calls) == 1 and any(
        str(a).endswith("/bind_scored_objects.py") for a in calls[0]
    )
    assert json.loads((out / "receipt-before-bind-repair.json").read_text()) == previous
    assert (
        out / "bind-before-repair.log"
    ).read_text() == "original missing-camera stack"
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["resume_history"]["upstream_inference_repeated"] is False
    assert receipt["sources_sha256"] == previous["sources_sha256"]
