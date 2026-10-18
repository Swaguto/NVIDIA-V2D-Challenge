"""Check all-stage roster wiring and fail before writing unsafe or stale plans."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from make_roster_configs import create


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def settings(tmp_path):
    c = {
        key: str(tmp_path / key)
        for key in (
            "source_root",
            "models",
            "perception_context",
            "policy_context",
            "robot_assets",
            "robot_asset_lock",
            "kit",
            "work_root",
        )
    }
    c["data_subdir"] = "evaluation"
    data = Path(c["source_root"]) / c["data_subdir"]
    put(data / "meta/camera_calibration.json", "{}")
    for ep in (3, 10):
        for camera in "abc":
            put(
                data
                / f"videos/chunk-000/observation.images.ego_cam_{camera}/episode_{ep:06d}.mp4",
                "fixture",
            )
    for relative in ("hand-runtime-v1/hamer/_DATA/data/mano", "hand-runtime-v1/public"):
        (Path(c["models"]) / relative).mkdir(parents=True)
    for key, schema, stages, flags in (
        (
            "perception_context",
            "v2d.scored-perception-context.v1",
            ["hands", "reconstruction"],
            [
                "private_footage_included",
                "model_weights_included",
                "credentials_included",
            ],
        ),
        (
            "policy_context",
            "v2d.scored-policy-context.v1",
            ["policy"],
            [
                "private_inputs_included",
                "robot_assets_included",
                "model_weights_included",
                "credentials_included",
            ],
        ),
    ):
        root = Path(c[key])
        files = {
            f"code/scripts/track3/run_scored_{s}.py": put(
                root / f"code/scripts/track3/run_scored_{s}.py", "# fixture\n"
            )
            for s in stages
        }
        if key == "policy_context":
            name = "code/scripts/track3/reference_contact_gate.py"
            files[name] = put(root / name, "# fixture contact gate\n")
        put(
            root / "context-manifest.json",
            json.dumps(
                dict(
                    schema=schema,
                    source_commit="a" * 40,
                    files=files,
                    **{f: False for f in flags},
                )
            ),
        )
    assets = {
        f"urdfs/g1/dex3_{side}.urdf": put(
            Path(c["robot_assets"]) / f"urdfs/g1/dex3_{side}.urdf", "fixture URDF"
        )
        for side in ("left", "right")
    }
    put(
        Path(c["robot_asset_lock"]),
        json.dumps(dict(schema="v2d.dex3.robot-assets.v1", files_sha256=assets)),
    )
    kit = Path(c["kit"])
    put(
        kit / "data/track_3_evaluation_objects.json",
        json.dumps(dict(episodes={"3": ["pot", "spoon"], "10": ["cup"]})),
    )
    ids = [
        f"track_3/e{ep:03d}/w000/f{frame:06d}/o{slot:03d}"
        for ep, count in [(3, 2), (10, 1)]
        for frame in range(4)
        for slot in range(count)
    ]
    pq.write_table(
        pa.table({"row_id": ids}), kit / "data/track_3_sample_submission.parquet"
    )
    return c


def test_complete_roster_wires_actual_stage_outputs_without_execution(
    settings, tmp_path
):
    out = tmp_path / "plans"
    report = create(settings, out)
    assert [r["episode"] for r in report["episodes"]] == [3, 10]
    assert report["selected_mesh_slots"] == 3
    assert report["runtime_preflights_still_required"] is True
    assert report["docker_invoked"] is report["policy_training_executed"] is False
    root = Path(settings["work_root"]) / "episodes/e003"
    hands, reconstruction, policy = [
        json.loads((out / "e003" / (s + ".json")).read_text())
        for s in ("hands", "reconstruction", "policy")
    ]
    assert reconstruction["observations"] == str(root / "hands/observations")
    assert reconstruction["temporal_fits"] == str(root / "hands/temporal-hands")
    assert reconstruction["cameras"] == str(root / "hands/perception")
    assert policy["bundle"] == str(root / "reconstruction/bundles/e003")
    assert policy["support"] == str(root / "reconstruction/support/e003")
    assert hands["out"] == str(root / "hands")
    assert policy["iterations"] == 100
    assert policy["require_reference_contact_feasibility"] is True
    assert not Path(settings["work_root"]).exists()
    for row in report["episodes"]:
        for command in row["commands"]:
            assert (
                hashlib.sha256(Path(command["argv"][-1]).read_bytes()).hexdigest()
                == command["config_sha256"]
            )
    with pytest.raises(ValueError, match="fresh"):
        create(settings, out)


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "mutable-image",
        "missing-video",
        "unlisted-runner",
        "changed-runner",
        "short-commit",
        "labels",
        "existing-output",
        "input-overlap",
        "missing-hand-lock",
    ],
)
def test_invalid_prerequisite_fails_before_creating_plan(settings, tmp_path, change):
    c = dict(settings)
    if change == "unknown":
        c["token"] = "not-a-configuration-field"
    elif change == "mutable-image":
        c["training_image"] = "image:latest"
    elif change == "missing-video":
        next(Path(c["source_root"]).rglob("*000010.mp4")).unlink()
    elif change in {"unlisted-runner", "short-commit"}:
        path = Path(c["policy_context"]) / "context-manifest.json"
        manifest = json.loads(path.read_text())
        if change == "unlisted-runner":
            manifest["files"] = {}
        else:
            manifest["source_commit"] = "abc"
        path.write_text(json.dumps(manifest))
    elif change == "changed-runner":
        (
            Path(c["policy_context"]) / "code/scripts/track3/run_scored_policy.py"
        ).write_text("changed")
    elif change == "labels":
        put(Path(c["source_root"]) / "evaluation/data/labels.parquet", "forbidden")
    elif change == "existing-output":
        (Path(c["work_root"]) / "episodes/e010/policy").mkdir(parents=True)
    elif change == "input-overlap":
        c["work_root"] = c["source_root"]
    elif change == "missing-hand-lock":
        path = Path(c["robot_asset_lock"])
        manifest = json.loads(path.read_text())
        del manifest["files_sha256"]["urdfs/g1/dex3_right.urdf"]
        path.write_text(json.dumps(manifest))
    out = tmp_path / "plans"
    with pytest.raises(ValueError):
        create(c, out)
    assert not out.exists()


@pytest.mark.parametrize("episodes", [[], [3, 3], [2], [True]])
def test_only_distinct_official_episode_ids(settings, tmp_path, episodes):
    with pytest.raises(ValueError, match="distinct episodes"):
        create(settings, tmp_path / "plans", episodes)


def test_subset_is_explicit_and_keeps_full_roster_count(settings, tmp_path):
    result = create(settings, tmp_path / "plans", [10])
    assert result["required_episodes"] == 2
    assert result["selected_episodes"] == result["selected_mesh_slots"] == 1


def test_old_context_needs_explicit_diagnostic_baseline(settings, tmp_path):
    path = Path(settings["policy_context"]) / "context-manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["files"]["code/scripts/track3/reference_contact_gate.py"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="reference contact gate"):
        create(settings, tmp_path / "quality")
    result = create(dict(settings, diagnostic_baseline=True), tmp_path / "diagnostic")
    assert result["native_reference_contact_gate_required"] is False
    with pytest.raises(ValueError, match="explicit JSON boolean"):
        create(dict(settings, diagnostic_baseline="false"), tmp_path / "bad")
