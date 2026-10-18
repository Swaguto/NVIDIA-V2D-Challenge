"""Run one scored reconstruction with pinned images and private mounted assets.

Host Python needs only the standard library and Docker. A surrounding compute
lease must enforce VM shutdown. This command does not start or provision a VM.
No provided object CAD or scored pose labels are mounted into the inference data.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

PERCEPTION = "ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:6e2039d0a54e14a188a1798efe57c3465d002f063c782c827b791e4418d2dce6"
SAM3D = "ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:dc9d0aedd2c6d50f53ce2d0677eca6f96e9631806d2bc12a7da4396629ce5357"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_digest(value):
    if not re.fullmatch(r"[a-z0-9./_-]+@sha256:[0-9a-f]{64}", value):
        raise ValueError("Runtime image must have an immutable SHA256 digest")
    return value


def validate_video_mount(root, data):
    """Check the entire common mount, including files outside its video view."""
    root = root.resolve()
    if any((data / name).exists() for name in ("mesh", "urdf", "data")):
        raise ValueError(
            "Inference data view must expose only videos and calibration metadata"
        )
    for path in root.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise ValueError("Source links escape the common video mount")
        if (path.is_dir() and path.name in {"mesh", "urdf", "data"}) or (
            path.is_file() and path.suffix not in {".mp4", ".json", ".txt"}
        ):
            raise ValueError(
                "Common inference mount must contain only videos and calibration metadata"
            )


def relocate_camera_paths(source, camera, out, label):
    """Copy the manifest with new paths to byte-identical calibrated cameras.

    Mount this file over the original manifest's existing container pathname.
    Its relative image/depth paths then retain their original parent directory.
    """
    original = source.read_bytes()
    manifest = json.loads(original)
    cam = manifest["color_camera"]["id"]
    if not re.fullmatch(r"[a-zA-Z0-9_]+", cam):
        raise ValueError("Unsafe camera ID")
    files = {}
    for frame in manifest["frames"]:
        filename = f"{frame['id']:06d}.json"
        old = (source.parent / frame["camera_pose_path"]).resolve()
        measured = camera / cam / filename
        if sha(old) != sha(measured):
            raise ValueError("Camera relocation would change numerical source data")
        record = json.loads(measured.read_text())
        if (
            record["episode"] != manifest["episode"]
            or record["frame_index"] != frame["id"]
            or record["gt_use"] != "independent"
            or record["camera_frame"] != cam
        ):
            raise ValueError("Camera relocation changed the frame/episode binding")
        files[filename] = sha(measured)
        frame["camera_pose_path"] = f"/stage/source/{label}/camera/{cam}/{filename}"
    proof = {
        "source_manifest_sha256": hashlib.sha256(original).hexdigest(),
        "camera_files_sha256": files,
        "camera_values_changed": False,
        "image_depth_hand_inputs_changed": False,
    }
    manifest["input_contract"]["camera_path_relocation"] = proof
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(proof, relocated_manifest_sha256=sha(out))


def validate(config):
    for key in (
        "foreground_depth_only",
        "foreground_translation_fallback",
        "pose_hypothesis_selection",
        "resume_from_bind",
    ):
        if type(config.get(key, False)) is not bool:
            raise ValueError("Inference switches require explicit JSON booleans")
    if config.get("pose_hypothesis_selection", False) and (
        config.get("foreground_depth_only", False)
        or config.get("foreground_translation_fallback", False)
    ):
        raise ValueError(
            "Whole-track selection uses its three separate fixed hypotheses"
        )
    if type(config["max_seconds"]) is not int or not 0 < config["max_seconds"] <= 7200:
        raise ValueError(
            "Episode execution needs a positive deadline no longer than two hours"
        )
    for name in ("perception_image", "sam3d_image"):
        image_digest(config[name])
    relative = Path(config["data_subdir"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Data subdirectory must stay within the common source mount")
    for name in (
        "source_root",
        "models",
        "observations",
        "temporal_fits",
        "cameras",
        "overlay",
    ):
        if not Path(config[name]).is_dir():
            raise ValueError("Missing private/source directory: " + name)
    data = Path(config["source_root"]) / relative
    if (
        not (data / "videos").is_dir()
        or not (data / "meta/camera_calibration.json").is_file()
    ):
        raise ValueError("Video/calibration view is incomplete")
    if any(
        not (data / name)
        .resolve()
        .is_relative_to(Path(config["source_root"]).resolve())
        for name in ("videos", "meta/camera_calibration.json")
    ):
        raise ValueError("Video/calibration links escape the common source mount")
    # The inference view must not expose reference meshes or pose tables.
    validate_video_mount(Path(config["source_root"]), data)
    roster = json.loads(Path(config["roster"]).read_text())["episodes"]
    names = roster[str(config["episode"])]
    if (
        not names
        or len(set(names)) != len(names)
        or not all(n.isidentifier() for n in names)
    ):
        raise ValueError("Official object slots must be distinct safe names")
    overlay = Path(config["overlay"])
    manifest = json.loads((overlay / "context-manifest.json").read_text())
    if manifest.get("schema") != "v2d.scored-perception-context.v1" or any(
        manifest.get(name) is not False
        for name in (
            "private_footage_included",
            "model_weights_included",
            "credentials_included",
        )
    ):
        raise ValueError("Expected a committed code-only perception overlay")
    for name, expected in manifest["files"].items():
        path = (overlay / name).resolve()
        if not path.is_relative_to(overlay.resolve()) or sha(path) != expected:
            raise ValueError("Committed runtime overlay differs")
    return names, manifest["source_commit"]


def run(config):
    names, commit = validate(config)
    out = Path(config["out"]).resolve()
    resume = config.get("resume_from_bind", False)
    previous = None
    reused = set()
    if resume:
        previous = json.loads((out / "receipt.json").read_text())
        expected_stages = (
            ["masks"]
            + [
                obj + "-" + stage
                for obj in names
                for stage in ("selection", "sam3d", "review")
            ]
            + ["view"]
        )
        expected_stages += (
            (
                ["poses-control", "poses-foreground-depth", "poses-foreground-motion"]
                + [obj + "-pose-selection" for obj in names]
            )
            if config.get("pose_hypothesis_selection", False)
            else ["poses"]
        )
        expected_stages += ["world"]
        stages = previous["stages"]
        if (
            [s["name"] for s in stages] != expected_stages + ["bind"]
            or any(s.get("returncode") != 0 for s in stages[:-1])
            or stages[-1].get("returncode") == 0
            or not previous.get("error")
        ):
            raise ValueError(
                "Bind resume requires exactly the completed upstream stages and failed bind"
            )
        if (
            previous["episode"] != config["episode"]
            or previous["object_slot_names"] != names
            or previous["images"]
            != {
                "perception": config["perception_image"],
                "sam3d": config["sam3d_image"],
            }
        ):
            raise ValueError("Resume changed the episode, meshes or immutable runtimes")
        for key in (
            "foreground_depth_only",
            "foreground_translation_fallback",
            "pose_hypothesis_selection",
        ):
            if bool(previous.get(key, False)) != bool(config.get(key, False)):
                raise ValueError("Resume cannot relabel upstream tracking policy")
        if (out / "receipt-before-bind-repair.json").exists():
            raise ValueError(
                "This bind repair already ran; inspect the first remaining failure"
            )
        reused = set(expected_stages)
    else:
        out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir(exist_ok=resume)
    if resume:
        if sha(Path(config["roster"])) != sha(out / "inputs/roster.json"):
            raise ValueError("Resume changed the original stored roster")
    else:
        shutil.copyfile(config["roster"], out / "inputs/roster.json")
    wrapper = out / "inputs/run_native.py"
    wrapper.write_text(
        "import runpy,sys\nfrom pathlib import Path\n"
        "script=sys.argv.pop(1);sys.path.insert(0,str(Path(script).parent));runpy.run_path(script,run_name='__main__')\n"
    )
    ep = config["episode"]
    label = f"e{ep:03d}"
    base = Path(config["models"])
    data = "/source/" + config["data_subdir"]
    image = config["perception_image"]
    daemon = ["docker"] + (
        ["--host", config["docker_host"]] if config.get("docker_host") else []
    )
    deadline = time.monotonic() + config["max_seconds"]
    relocated = None
    receipt_path = out / ("resume-bind-preflight.json" if resume else "receipt.json")
    receipt = {
        "schema": "v2d.scored-single-episode-reconstruction.v1",
        "episode": ep,
        "object_slot_names": names,
        "source_commit": commit,
        "images": {"perception": image, "sam3d": config["sam3d_image"]},
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "source_pose_labels_read": False,
        "provided_object_meshes_read": False,
        "physical_task_accepted": False,
        "stages": [],
        "foreground_depth_only": config.get("foreground_depth_only", False),
        "foreground_translation_fallback": config.get(
            "foreground_translation_fallback", False
        ),
        "pose_hypothesis_selection": config.get("pose_hypothesis_selection", False),
    }

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")

    def container(script, args, name, *, native=False, sam=False, gpu=False, limit=600):
        if name in reused:
            return
        container_name = f"v2d-scored-{label}-{name}"
        command = daemon + [
            "run",
            "--rm",
            "--name",
            container_name,
            "--network",
            "none",
            "--shm-size",
            "8g",
        ]
        if gpu:
            command += ["--gpus", "all"]
        mounts = [
            (Path(config["overlay"]) / "code", "/workspace/pr32", True),
            (Path(config["source_root"]), "/source", True),
            (Path(config["observations"]), "/workspace/issue6-full-inputs", True),
            (Path(config["temporal_fits"]), "/workspace/temporal-hands", True),
            (base, "/models", not sam),
            (out, "/stage", False),
            (base / "hand-runtime-v1/hamer", "/workspace/hamer-weights", True),
            (base / "hand-runtime-v1/hamer/_DATA", "/workspace/Dyn-HaMR/_DATA", True),
            (
                base / "hand-runtime-v1/hamer/_DATA/data/mano",
                "/workspace/mano/models",
                True,
            ),
            (base / "hand-runtime-v1/public", "/workspace/pr32-models", True),
        ]
        for source, destination, readonly in mounts:
            command += [
                "-v",
                str(source.resolve()) + ":" + destination + (":ro" if readonly else ""),
            ]
        if relocated is not None:
            command += [
                "-v",
                str(relocated)
                + f":/workspace/issue6-full-inputs/{label}/manifest.json:ro",
            ]
        environment = {
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility,graphics",
            "PYOPENGL_PLATFORM": "egl",
            "HF_HOME": "/models/sam3d/hf_home" if sam else "/models/huggingface",
            "TORCH_HOME": "/models/sam3d/torch_home",
            "HF_HUB_OFFLINE": "1",
            "LIDRA_SKIP_INIT": "1",
            "V2D_DINO_REVISION": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
            "V2D_SAM2_REVISION": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
            "V2D_CALIB": data + "/meta/camera_calibration.json",
            "FOUNDATIONPOSE_WEIGHTS_DIR": "/models/foundationpose",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
        }
        for key, value in environment.items():
            command += ["-e", key + "=" + value]
        if native or sam:
            interpreter = (
                "/workspace/sam3d-runtime/bin/python"
                if sam
                else "/workspace/hand-runtime/bin/python"
            )
            command += ["--entrypoint", interpreter]
        command += [config["sam3d_image"] if sam else image]
        if native:
            command += ["/stage/inputs/run_native.py"]
        command += [script, *map(str, args)]
        remaining = int(deadline - time.monotonic() - 10)
        if remaining <= 0:
            raise TimeoutError("Episode reconstruction deadline")
        started = time.monotonic()
        try:
            with (out / (name + ".log")).open("wb") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=min(limit, remaining),
                    check=False,
                )
        except subprocess.TimeoutExpired:
            subprocess.run(
                daemon + ["rm", "-f", container_name],
                capture_output=True,
                timeout=30,
                check=False,
            )
            receipt["stages"].append({"name": name, "timed_out": True})
            save()
            raise
        stage = {
            "name": name,
            "returncode": result.returncode,
            "wall_seconds": time.monotonic() - started,
        }
        receipt["stages"].append(stage)
        save()
        print(json.dumps(stage), flush=True)
        result.check_returncode()

    try:
        camera = Path(config["cameras"]) / label / "camera"
        fit = Path(config["temporal_fits"]) / label
        for path in [
            camera / "report.json",
            fit / "fit_receipt.json",
            fit / "observation_gate.json",
        ]:
            if not path.is_file():
                raise ValueError(
                    "Required calibrated fit/observation proof missing: " + str(path)
                )
        receipt["sources_sha256"] = {
            "roster": sha(Path(config["roster"])),
            "hand_observations": sha(
                Path(config["observations"]) / label / "manifest.json"
            ),
            "temporal_fit": sha(fit / "fit_receipt.json"),
            "observation_gate": sha(fit / "observation_gate.json"),
            "camera_report": sha(camera / "report.json"),
            "model_download_receipt": sha(base / "sam3d/download-receipt.json"),
        }
        if previous is not None:
            if receipt["sources_sha256"] != previous["sources_sha256"]:
                raise ValueError(
                    "Resume changed source observations, fit, camera or model receipt"
                )
            if (out / "bound" / label / "manifest.json").exists() or (
                out / "native" / label
            ).exists():
                raise ValueError(
                    "Bind repair must not overwrite partial downstream outputs"
                )
            shutil.copyfile(
                out / "receipt.json", out / "receipt-before-bind-repair.json"
            )
            shutil.copyfile(out / "bind.log", out / "bind-before-repair.log")
            receipt_path = out / "receipt.json"
            receipt["resume_history"] = {
                "previous_receipt_sha256": sha(out / "receipt-before-bind-repair.json"),
                "previous_source_commit": previous["source_commit"],
                "previous_config_sha256": previous["config_sha256"],
                "failed_bind_log_sha256": sha(out / "bind-before-repair.log"),
                "upstream_inference_repeated": False,
            }
            receipt["stages"] = [
                dict(s, reused=True, source_commit=previous["source_commit"])
                for s in previous["stages"][:-1]
            ]
        save()
        if previous is None:
            shutil.copytree(camera, out / "source" / label / "camera")
        else:
            for p in camera.rglob("*"):
                if p.is_file() and sha(p) != sha(
                    out / "source" / label / "camera" / p.relative_to(camera)
                ):
                    raise ValueError("Reused camera bytes differ")
        relocated = out / "inputs/relocated-observations.json"
        receipt["observation_camera_path_relocation"] = relocate_camera_paths(
            Path(config["observations"]) / label / "manifest.json",
            camera,
            relocated,
            label,
        )
        save()
        batch = "/workspace/pr32/scripts/track3/batch_public.py"
        common = [
            "--roster",
            "/stage/inputs/roster.json",
            "--episodes",
            ep,
            "--root",
            "/stage/source",
        ]
        container(
            batch, ["--phase", "masks", "--data", data, *common], "masks", gpu=True
        )
        for obj in names:
            selection = f"/stage/selections/{obj}.json"
            container(
                "/workspace/pr32/scripts/track3/select_reconstruction_frame.py",
                [
                    "--data",
                    data,
                    "--masks",
                    f"/stage/source/{label}/masks",
                    "--hand-masks",
                    f"/workspace/temporal-hands/{label}/visual_masks",
                    "--episode",
                    ep,
                    "--object",
                    obj,
                    "--out",
                    selection,
                ],
                obj + "-selection",
                limit=180,
            )
            chosen = json.loads((out / "selections" / (obj + ".json")).read_text())[
                "selected"
            ]["frame"]
            destination = f"/stage/geometry/{label}/{obj}"
            container(
                "/workspace/pr32/scripts/track3/sam3d_candidate.py",
                [
                    "--data",
                    data,
                    "--masks",
                    f"/stage/source/{label}/masks",
                    "--episode",
                    ep,
                    "--object",
                    obj,
                    "--frame",
                    chosen,
                    "--weights",
                    "/models/sam3d",
                    "--out",
                    destination,
                ],
                obj + "-sam3d",
                sam=True,
                gpu=True,
            )
            container(
                "/workspace/pr32/scripts/track3/render_sam3d_candidate.py",
                ["--candidate", destination, "--out", destination + "/review"],
                obj + "-review",
                sam=True,
                gpu=True,
                limit=120,
            )
        view = f"/stage/views/{label}"
        container(
            "/workspace/pr32/scripts/track3/reconstructed_view.py",
            [
                "--data",
                data,
                "--geometry",
                f"/stage/geometry/{label}",
                "--roster",
                "/stage/inputs/roster.json",
                "--episode",
                ep,
                "--out",
                view,
            ],
            "view",
            limit=60,
        )
        options = (
            ["--foreground-depth-only"]
            if config.get("foreground_depth_only", False)
            else []
        )
        if config.get("foreground_translation_fallback", False):
            options += ["--foreground-translation-fallback"]
        if config.get("pose_hypothesis_selection", False):
            hypotheses = {}
            for arm, flags in [
                ("control", []),
                ("foreground-depth", ["--foreground-depth-only"]),
                ("foreground-motion", ["--foreground-translation-prior"]),
            ]:
                host_root = out / "hypotheses" / arm / "source"
                if previous is None:
                    shutil.copytree(
                        out / "source" / label / "masks", host_root / label / "masks"
                    )
                root = f"/stage/hypotheses/{arm}/source"
                container(
                    batch,
                    [
                        "--phase",
                        "poses",
                        "--data",
                        view,
                        "--fp-python",
                        "/opt/conda/bin/python",
                        "--roster",
                        "/stage/inputs/roster.json",
                        "--episodes",
                        ep,
                        "--root",
                        root,
                        *flags,
                    ],
                    "poses-" + arm,
                    gpu=True,
                )
                hypotheses[arm] = root + f"/{label}/poses"
            (out / "inputs/pose-hypotheses.json").write_text(
                json.dumps(hypotheses, indent=2) + "\n"
            )
            for obj in names:
                container(
                    "/workspace/pr32/scripts/track3/select_pose_hypothesis.py",
                    [
                        "--episode",
                        ep,
                        "--object",
                        obj,
                        "--data",
                        view,
                        "--observations",
                        f"/workspace/issue6-full-inputs/{label}/manifest.json",
                        "--masks",
                        f"/stage/source/{label}/masks",
                        "--candidates",
                        "/stage/inputs/pose-hypotheses.json",
                        "--out",
                        f"/stage/source/{label}/poses",
                    ],
                    obj + "-pose-selection",
                    limit=300,
                )
        else:
            container(
                batch,
                [
                    "--phase",
                    "poses",
                    "--data",
                    view,
                    "--fp-python",
                    "/opt/conda/bin/python",
                    *common,
                    *options,
                ],
                "poses",
                gpu=True,
            )
        container(
            batch, ["--phase", "world", "--data", view, *common], "world", limit=120
        )
        manifest = f"/stage/bound/{label}/manifest.json"
        container(
            "/workspace/pr32/scripts/track3/bind_scored_objects.py",
            [
                "--manifest",
                f"/workspace/issue6-full-inputs/{label}/manifest.json",
                "--data",
                view,
                "--results",
                "/stage/source",
                "--out",
                manifest,
            ],
            "bind",
            limit=60,
        )
        container(
            "/workspace/pr32/reconstruction/validation/issue6/convert_and_validate.py",
            [
                "--episode",
                ep,
                "--manifest",
                manifest,
                "--temporal-root",
                f"/workspace/temporal-hands/{label}",
                "--conversion-root",
                f"/stage/conversions/{label}",
                "--temporal-manifest",
                f"/stage/manifests/{label}.json",
                "--output",
                f"/stage/native/{label}",
                "--mano-root",
                "/workspace/mano",
            ],
            "native",
            native=True,
            gpu=True,
            limit=900,
        )
        container(
            "/workspace/pr32/reconstruction/validation/issue6/repair_hand_continuity.py",
            [
                "--episode",
                ep,
                "--input-root",
                "/stage",
                "--modules",
                "/workspace/pr32/reconstruction/modules",
                "--output",
                f"/stage/continuity/{label}",
                "--original-run",
                f"/stage/native/{label}",
                "--temporal-root",
                f"/workspace/temporal-hands/{label}",
                "--converted-records",
                f"/stage/conversions/{label}/converted",
                "--mano-assets-root",
                "/workspace/mano",
                "--image-fit",
                "--bidirectional-image-flow",
            ],
            "continuity",
            native=True,
            gpu=True,
        )
        checkpoint = json.loads((fit / "fit_receipt.json").read_text())[
            "final_checkpoint"
        ]
        container(
            "/workspace/pr32/scripts/dex3_review/export_bundle.py",
            [
                "--episode",
                ep,
                "--run",
                f"/stage/continuity/{label}",
                "--checkpoint",
                checkpoint,
                "--mano",
                "/workspace/mano",
                "--out",
                f"/stage/bundles/{label}",
                "--scored-reconstructed-view",
                view,
                "--object-world-root",
                f"/stage/source/{label}/inferred",
            ],
            "bundle",
            native=True,
            limit=180,
        )
        container(
            "/workspace/pr32/scripts/track3/infer_support_plane.py",
            [
                "--manifest",
                manifest,
                "--masks",
                f"/stage/source/{label}/masks",
                "--bundle-manifest",
                f"/stage/bundles/{label}/manifest.json",
                "--out",
                f"/stage/support/{label}",
            ],
            "support",
            limit=180,
        )
        receipt["reconstruction_exported"] = True
    except BaseException as error:
        receipt["error"] = repr(error)
        raise
    finally:
        save()
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    run(json.loads(a.config.read_text()))
