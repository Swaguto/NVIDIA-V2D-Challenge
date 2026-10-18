"""Prepare uniform hands/reconstruction/policy configs for the official roster.

This validates static prerequisites and writes plans only. Runtime preflights,
GPU supervision, result backup and provider shutdown remain separate.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from audit_scored_coverage import official_frames
from run_scored_hands import validate as validate_hands
from run_scored_policy import TRAINING, verify_files
from run_scored_reconstruction import PERCEPTION, SAM3D, image_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_runner(context, manifest, stage):
    relative = f"code/scripts/track3/run_scored_{stage}.py"
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("source_commit", "")):
        raise ValueError("Stage context needs its full committed source SHA")
    runner = context / relative
    if not runner.is_file() or manifest["files"].get(relative) != sha(runner):
        raise ValueError("Stage runner is missing from its verified code closure")
    return runner


def create(settings, out, episodes=None):
    required = {
        "source_root",
        "data_subdir",
        "models",
        "perception_context",
        "policy_context",
        "robot_assets",
        "robot_asset_lock",
        "kit",
        "work_root",
    }
    allowed = required | {
        "perception_image",
        "sam3d_image",
        "training_image",
        "docker_host",
        "diagnostic_baseline",
    }
    if not required <= settings.keys() or settings.keys() - allowed:
        raise ValueError(
            "Use declared settings only; credentials are not configuration fields"
        )
    diagnostic = settings.get("diagnostic_baseline", False)
    if type(diagnostic) is not bool:
        raise ValueError("Diagnostic baseline must be an explicit JSON boolean")
    paths = {
        key: Path(settings[key]).expanduser().resolve()
        for key in required - {"data_subdir"}
    }
    out = out.resolve()
    readonly = [
        paths[k]
        for k in (
            "source_root",
            "models",
            "perception_context",
            "policy_context",
            "robot_assets",
            "kit",
        )
    ]
    if out.exists() or any(
        out.is_relative_to(p)
        or p.is_relative_to(out)
        or paths["work_root"].is_relative_to(p)
        or p.is_relative_to(paths["work_root"])
        for p in readonly
    ):
        raise ValueError(
            "Configs and work outputs must be fresh and separate from read-only input roots"
        )
    for key, path in paths.items():
        if key == "work_root":
            continue
        if not path.exists():
            raise ValueError("Missing static prerequisite: " + key)
    kit = paths["kit"]
    roster = kit / "data/track_3_evaluation_objects.json"
    names = json.loads(roster.read_text())["episodes"]
    frames = official_frames(kit, names)
    selected = sorted(frames) if episodes is None else list(episodes)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(type(ep) is not int or ep not in frames for ep in selected)
    ):
        raise ValueError("Select distinct episodes from the complete official roster")
    images = dict(
        perception=settings.get("perception_image", PERCEPTION),
        sam3d=settings.get("sam3d_image", SAM3D),
        training=settings.get("training_image", TRAINING),
    )
    for image in images.values():
        image_digest(image)
    policy_context = paths["policy_context"]
    policy_source = json.loads((policy_context / "context-manifest.json").read_text())
    if policy_source.get("schema") != "v2d.scored-policy-context.v1" or any(
        policy_source.get(k) is not False
        for k in (
            "private_inputs_included",
            "robot_assets_included",
            "model_weights_included",
            "credentials_included",
        )
    ):
        raise ValueError("Need a verified code-only policy context")
    verify_files(policy_context, policy_source["files"])
    if (
        not diagnostic
        and "code/scripts/track3/reference_contact_gate.py"
        not in policy_source["files"]
    ):
        raise ValueError("Fresh quality runs need a committed reference contact gate")
    perception_source = json.loads(
        (paths["perception_context"] / "context-manifest.json").read_text()
    )
    runners = {
        "hands": require_runner(
            paths["perception_context"], perception_source, "hands"
        ),
        "reconstruction": require_runner(
            paths["perception_context"], perception_source, "reconstruction"
        ),
        "policy": require_runner(policy_context, policy_source, "policy"),
    }
    asset_lock = json.loads(paths["robot_asset_lock"].read_text())
    if asset_lock.get("schema") != "v2d.dex3.robot-assets.v1" or not all(
        f"urdfs/g1/dex3_{side}.urdf" in asset_lock.get("files_sha256", {})
        for side in ("left", "right")
    ):
        raise ValueError("Need the locked actual robot assets")
    verify_files(paths["robot_assets"], asset_lock["files_sha256"])
    rows = []
    for ep in selected:
        label = f"e{ep:03d}"
        root = paths["work_root"] / "episodes" / label
        hands, reconstruction, policy = (
            root / name for name in ("hands", "reconstruction", "policy")
        )
        if any(p.exists() for p in (hands, reconstruction, policy)):
            raise ValueError("Planned episode outputs already exist: " + label)
        if any(
            out == p or out.is_relative_to(p) or p.is_relative_to(out)
            for p in (hands, reconstruction, policy)
        ):
            raise ValueError("Configuration folder overlaps a future episode output")
        for camera in "abc":
            video = (
                paths["source_root"]
                / settings["data_subdir"]
                / f"videos/chunk-000/observation.images.ego_cam_{camera}/episode_{ep:06d}.mp4"
            )
            if not video.is_file():
                raise ValueError("Required source video missing: " + str(video))
        hand_config = dict(
            episode=ep,
            split="evaluation",
            source_root=str(paths["source_root"]),
            data_subdir=settings["data_subdir"],
            models=str(paths["models"]),
            overlay=str(paths["perception_context"]),
            roster=str(roster),
            kit=str(kit),
            out=str(hands),
            max_seconds=1800,
            perception_image=images["perception"],
        )
        if settings.get("docker_host"):
            hand_config["docker_host"] = settings["docker_host"]
        # This reads only video/metadata layout, code closures and private model
        # directory presence. It neither loads weights nor executes inference.
        validate_hands(hand_config)
        recon_config = dict(
            episode=ep,
            source_root=str(paths["source_root"]),
            data_subdir=settings["data_subdir"],
            models=str(paths["models"]),
            observations=str(hands / "observations"),
            temporal_fits=str(hands / "temporal-hands"),
            cameras=str(hands / "perception"),
            overlay=str(paths["perception_context"]),
            roster=str(roster),
            out=str(reconstruction),
            max_seconds=2400,
            foreground_depth_only=False,
            foreground_translation_fallback=False,
            pose_hypothesis_selection=True,
            perception_image=images["perception"],
            sam3d_image=images["sam3d"],
        )
        if settings.get("docker_host"):
            recon_config["docker_host"] = settings["docker_host"]
        policy_config = dict(
            episode=ep,
            bundle=str(reconstruction / "bundles" / label),
            support=str(reconstruction / "support" / label),
            context=str(policy_context),
            robot_assets=str(paths["robot_assets"]),
            robot_asset_lock=str(paths["robot_asset_lock"]),
            kit=str(kit),
            out=str(policy),
            iterations=100,
            max_seconds=1200,
            anchor_policy="visual",
            require_reference_contact_feasibility=not diagnostic,
            training_image=images["training"],
        )
        configs = dict(
            hands=hand_config, reconstruction=recon_config, policy=policy_config
        )
        commands = []
        for stage, runner in runners.items():
            config_path = out / label / (stage + ".json")
            commands.append(
                dict(
                    stage=stage,
                    argv=[sys.executable, str(runner), "--config", str(config_path)],
                )
            )
        rows.append(
            dict(
                episode=ep,
                frames=frames[ep],
                objects=names[str(ep)],
                configs=configs,
                commands=commands,
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    for row in rows:
        for stage, config in row.pop("configs").items():
            path = out / f"e{row['episode']:03d}" / (stage + ".json")
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(config, indent=2) + "\n")
        for command in row["commands"]:
            command["config_sha256"] = sha(Path(command["argv"][-1]))
    report = dict(
        schema="v2d.scored-roster-stage-plan.v1",
        episodes=rows,
        selected_episodes=len(rows),
        required_episodes=len(frames),
        selected_mesh_slots=sum(len(r["objects"]) for r in rows),
        images=images,
        perception_source_commit=perception_source["source_commit"],
        policy_source_commit=policy_source["source_commit"],
        official_roster_sha256=sha(roster),
        robot_asset_lock_sha256=sha(paths["robot_asset_lock"]),
        static_prerequisites_verified=True,
        runtime_preflights_still_required=True,
        private_weights_loaded=False,
        docker_invoked=False,
        cloud_instance_created=False,
        policy_training_executed=False,
        diagnostic_baseline_requested=diagnostic,
        native_reference_contact_gate_required=not diagnostic,
        physical_task_accepted=False,
        exact_reconstruction_or_retraining_reproduction_claimed=False,
    )
    (out / "roster-plan.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, nargs="+")
    a = p.parse_args()
    result = create(json.loads(a.settings.read_text()), a.out, a.episodes)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))
