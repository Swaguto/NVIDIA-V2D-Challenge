from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from issue6 import backend
from issue6.manifest import (
    camera_from_object,
    inspect_inputs,
    load,
    pose,
    read_json,
    real_file,
    validate,
    write_json,
)
from issue6.pipeline import compare_reports, contacts_stage, execute


@pytest.mark.parametrize(
    "settings",
    [
        {"contact_threshold_m": float("nan")},
        {"contact_threshold_m": 0},
        {"contact_threshold_m": True},
        {"bridge_frames": -1},
        {"bridge_frames": 1.5},
        {"bridge_frames": True},
    ],
)
def test_invalid_contact_settings_fail_before_inference(bundle, settings):
    bundle["settings"] = settings
    with pytest.raises(ValueError, match="Invalid contact settings"):
        validate(bundle)


def test_conflicting_camera_pose_rejected(bundle):
    bundle["frames"][0].update(camera_from_world=np.eye(4).tolist(), camera_pose_path="000010.json")
    with pytest.raises(ValueError, match="exactly one camera"):
        validate(bundle)


def test_inward_normals_marked_ambiguous(bundle, tmp_path):
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    mesh.invert()
    mesh.export(bundle["objects"][0]["mesh"])
    root = tmp_path / "inward"
    (root / "align").mkdir(parents=True)
    synthetic_worker(bundle, root, root / "align")
    out = root / "contacts"
    out.mkdir()
    contacts_stage(bundle, root, out)
    rows = read_json(out / "contacts.json")
    assert all(r["geometry_ambiguous"] for r in rows)
    assert any(r["valid"] for r in rows)


@pytest.fixture
def bundle(tmp_path):
    c = dict(
        id="ego_cam_a",
        width=64,
        height=48,
        fx=40.0,
        fy=40.0,
        cx=32.0,
        cy=24.0,
        convention="rectified_pinhole",
    )
    trimesh.creation.box(extents=[0.2, 0.2, 0.2]).export(tmp_path / "box.ply")
    t = np.eye(4)
    t[2, 3] = 1
    frames = []
    raw = tmp_path / "raw"
    for n in range(5):
        idx = n + 10
        Image.new("RGB", (64, 48), (40, 50, 60)).save(tmp_path / f"{idx}.png")
        np.save(tmp_path / f"{idx}.npy", np.ones((48, 64), np.float32))
        frame = dict(
            id=idx,
            timestamp_s=n * 0.05,
            color=f"{idx}.png",
            depth_path=f"{idx}.npy",
            objects={
                "box": dict(
                    valid=n != 3,
                    direction="camera_from_object",
                    matrix=t.tolist(),
                    occlusion_clear=True,
                )
            },
        )
        frames.append(frame)
        write_json(
            raw / "1" / f"{idx:06d}.json",
            dict(
                frame_idx=idx,
                track_id=1,
                is_right=True,
                image_size=[64, 48],
                camera=dict(pred_cam_t_full=[0, 0, 1], scaled_focal_length=40),
                mano=dict(betas=[0.0] * 10, global_orient=[0.0] * 3, hand_pose=[0.0] * 45),
            ),
        )
    m = dict(
        schema_version=1,
        episode=2,
        mode="synthetic",
        units="metres",
        camera_provenance="synthetic",
        hand_provenance="synthetic",
        color_camera=c,
        depth=dict(
            camera=c,
            color_from_depth=np.eye(4).tolist(),
            encoding="metres",
            provenance="synthetic",
        ),
        objects=[dict(id="box", mesh="box.ply", provenance="synthetic")],
        frames=frames,
        hand_source=dict(backend="import", records_dir="raw"),
    )
    write_json(tmp_path / "manifest.json", m)
    return load(tmp_path / "manifest.json")


def synthetic_worker(m, root, out):
    """Test-only stand-in: never shipped as a real reconstruction backend."""
    hands = []
    for n, f in enumerate(m["frames"]):
        x = 0.2 if n == 1 else 0.105
        hands.append(
            dict(
                frame_id=f["id"],
                timestamp_s=f["timestamp_s"],
                track_id=1,
                is_right=True,
                metric_valid=True,
                joints_camera_m=np.tile([x, 0, 1], (21, 1)).tolist(),
                diagnostics={"median_abs_residual_m": 0.001},
            )
        )
    write_json(out / "hands.json", hands)
    np.savez_compressed(
        out / "hands.npz",
        joints_camera_m=np.array([h["joints_camera_m"] for h in hands]),
    )


def test_cpu_pipeline_synthetic_only(bundle, tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "align_worker", synthetic_worker)
    root = tmp_path / "result"
    for stage in ("validate", "hands", "depth", "align", "contacts", "report"):
        execute(bundle, root, stage)
    rows = read_json(root / "contacts" / "contacts.json")
    right = [r for r in rows if r["hand"] == "right" and r["fingertip"] == "index"]
    assert [r["raw_contact"] for r in right] == [True, False, True, None, True]
    assert [r["window_contact"] for r in right] == [True, True, True, None, True]
    assert all(r["raw_contact"] is None for r in rows if r["hand"] == "left")
    report = read_json(root / "report" / "report.json")
    assert report["mode"] == "synthetic" and not report["issue6_complete"]
    assert report["contact_precision"] is None and report["bridged_count"] == 5
    assert len(list((root / "report" / "overlays").glob("*.png"))) == 5
    assert (root / "report" / "review.html").is_file()
    with pytest.raises(ValueError, match="already exists"):
        execute(bundle, root, "report")


def test_stale_upstream_output_rejected(bundle, tmp_path, monkeypatch):
    root = tmp_path / "stale"
    execute(bundle, root, "hands")
    execute(bundle, root, "depth")
    (root / "hands" / "records" / "1" / "000010.json").write_text("{}")
    monkeypatch.setattr(backend, "align_worker", synthetic_worker)
    with pytest.raises(ValueError, match="Stale"):
        execute(bundle, root, "align")


def test_changed_source_input_rejected(bundle, tmp_path):
    root = tmp_path / "changed"
    execute(bundle, root, "hands")
    Image.new("RGB", (64, 48), "red").save(bundle["frames"][0]["color"])
    with pytest.raises(ValueError, match="Inputs/settings/code changed"):
        execute(bundle, root, "depth")


def test_failure_receipt_is_not_success(bundle, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(backend, "reconstruct", fail)
    root = tmp_path / "failed"
    with pytest.raises(RuntimeError):
        execute(bundle, root, "hands")
    assert read_json(root / "hands" / "receipt.json")["status"] == "failed"


def test_missing_depth_remains_nan(bundle, tmp_path):
    for f in bundle["frames"]:
        f["depth_path"] = None
    root = tmp_path / "no_depth"
    execute(bundle, root, "depth")
    assert np.isnan(np.load(root / "depth" / "000010.npy")).all()
    assert not inspect_inputs(bundle)["complete_inputs"]


@pytest.mark.parametrize("change", ["ids", "time", "duplicate_objects", "units", "synthetic", "holdout"])
def test_manifest_rejects_contract_mismatch(bundle, change):
    if change == "ids":
        bundle["frames"][1]["id"] = 20
    elif change == "time":
        bundle["frames"][1]["timestamp_s"] = 0
    elif change == "duplicate_objects":
        bundle["objects"] *= 2
    elif change == "units":
        bundle["units"] = "millimetres"
    elif change == "synthetic":
        bundle["mode"] = "gt_assisted_development"
    else:
        bundle["mode"] = "gt_assisted_development"
        bundle["episode"] = 11
    with pytest.raises(ValueError):
        validate(bundle)


def test_gt_camera_provenance_retained_with_predicted_objects(bundle):
    bundle["mode"] = "predicted_object_integration"
    bundle["camera_provenance"] = "ground_truth_assisted"
    bundle["hand_provenance"] = "predicted"
    bundle["depth"]["provenance"] = "predicted"
    bundle["objects"][0]["provenance"] = "predicted"
    validate(bundle)
    assert bundle["camera_provenance"] == "ground_truth_assisted"


def test_transform3d_wxyz_and_object_world_composition(tmp_path):
    p = tmp_path / "000010.json"
    write_json(p, dict(rotation=[1, 0, 0, 0], translation=[1, 2, 3]))
    t = pose({"pose_path": str(p)})
    np.testing.assert_allclose(t[:3, 3], [1, 2, 3])
    c = np.eye(4)
    c[0, 3] = 5
    f = dict(
        camera_valid=True,
        camera_from_world=c.tolist(),
        objects={"box": dict(valid=True, direction="world_from_object", pose_path=str(p))},
    )
    np.testing.assert_allclose(camera_from_object(f, {"id": "box"})[:3, 3], [6, 2, 3])
    f["camera_valid"] = False
    assert camera_from_object(f, {"id": "box"}) is None


def test_lfs_pointer_is_not_an_asset(tmp_path):
    p = tmp_path / "MANO_RIGHT.pkl"
    p.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1234\n")
    with pytest.raises(ValueError, match="LFS pointer"):
        real_file(p)


def test_hand_file_identity_mismatch_rejected(bundle, tmp_path):
    p = Path(bundle["hand_source"]["records_dir"]) / "1" / "000010.json"
    r = read_json(p)
    r["frame_idx"] = 11
    write_json(p, r)
    with pytest.raises(ValueError, match="identifiers"):
        execute(bundle, tmp_path / "bad_hand", "hands")


def test_ambiguous_duplicate_hands_are_unknown(bundle, tmp_path):
    root = tmp_path / "duplicates"
    out = root / "contacts"
    out.mkdir(parents=True)
    synthetic_worker(bundle, root, root / "align")
    p = root / "align" / "hands.json"
    rows = read_json(p)
    rows.append({**rows[0], "track_id": 2})
    write_json(p, rows)
    contacts_stage(bundle, root, out)
    rows = read_json(out / "contacts.json")
    assert all(r["reason"] == "ambiguous_hand" for r in rows if r["frame_id"] == 10 and r["hand"] == "right")


def test_comparison_rejects_different_frame_axes(tmp_path):
    a = dict(comparison_sha256="same", frame_axis=[dict(id=0, timestamp_s=0)])
    b = dict(comparison_sha256="same", frame_axis=[dict(id=1, timestamp_s=0.05)])
    write_json(tmp_path / "a.json", a)
    write_json(tmp_path / "b.json", b)
    with pytest.raises(ValueError, match="matching frames"):
        compare_reports(tmp_path / "a.json", tmp_path / "b.json")


def test_renderer_reindexes_nonzero_original_ids_without_mutating_records(bundle, tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    for f in bundle["frames"]:
        Image.open(f["color"]).save(images / f"{f['id']:06d}.png")
    original = Path(bundle["hand_source"]["records_dir"])
    image_dir, hand_dir = backend.prepare_render_inputs(images, original, bundle["frames"], tmp_path / "render")
    assert (image_dir / "000000.png").exists()
    assert read_json(hand_dir / "1" / "000000.json")["frame_idx"] == 0
    assert read_json(hand_dir / "1" / "000000.json")["original_frame_idx"] == 10
    assert read_json(original / "1" / "000010.json")["frame_idx"] == 10


def test_cad_occlusion_adapter_prepares_color_pose(bundle, tmp_path, monkeypatch):
    # #5 may have valid poses but only stereo-view masks. Render the CAD in
    # color coordinates rather than treating those pixels as automatically clear.
    for f in bundle["frames"]:
        f["objects"]["box"].pop("occlusion_clear")
    monkeypatch.setattr(backend, "align_worker", synthetic_worker)
    root = tmp_path / "cad"
    for stage in ("hands", "depth", "align"):
        execute(bundle, root, stage)
    job = read_json(root / "align" / "job.json")
    assert job["frames"][0]["occlusion_known"]
    np.testing.assert_allclose(
        np.array(job["frames"][0]["projected_objects"][0]["camera_from_object"])[:3, 3],
        [0, 0, 1],
    )
    assert not job["frames"][3]["occlusion_known"]
    assert (root / "align" / "object_0.npz").exists()


def test_invalid_object_masks_stop_alignment(bundle, tmp_path, monkeypatch):
    p = tmp_path / "bad_mask.png"
    Image.new("L", (1, 1), 255).save(p)
    bundle["frames"][0]["objects"]["box"]["mask"] = str(p)
    root = tmp_path / "bad_masks"
    for stage in ("hands", "depth"):
        execute(bundle, root, stage)
    monkeypatch.setattr(backend, "align_worker", synthetic_worker)
    with pytest.raises(ValueError, match="pixel grid"):
        execute(bundle, root, "align")


def test_pose_filename_mismatch_fails(bundle, tmp_path):
    p = tmp_path / "000020.json"
    write_json(p, dict(rotation=[1, 0, 0, 0], translation=[0, 0, 1]))
    state = bundle["frames"][0]["objects"]["box"]
    state.pop("matrix")
    state["pose_path"] = str(p)
    with pytest.raises(ValueError, match="original frame ID"):
        validate(bundle)


def test_unanchored_hands_do_not_produce_contacts(bundle, tmp_path):
    root = tmp_path / "unanchored"
    (root / "contacts").mkdir(parents=True)
    synthetic_worker(bundle, root, root / "align")
    rows = read_json(root / "align" / "hands.json")
    for r in rows:
        r["metric_valid"] = False
    write_json(root / "align" / "hands.json", rows)
    contacts_stage(bundle, root, root / "contacts")
    contacts = read_json(root / "contacts" / "contacts.json")
    assert all(r["raw_contact"] is None and not r["bridged"] for r in contacts)


def test_scaled_pose_rejected_instead_of_silently_changing_mesh_units(tmp_path):
    p = tmp_path / "000000.json"
    write_json(p, dict(rotation=[1, 0, 0, 0], translation=[0, 0, 1], scale=[1000, 1000, 1000]))
    with pytest.raises(ValueError, match="scale"):
        pose({"pose_path": str(p)})


def test_detection_reference_must_be_inside_smoke_clip(bundle):
    bundle["settings"] = {"reference_frame_id": 100}
    with pytest.raises(ValueError, match="reference_frame_id"):
        validate(bundle)
