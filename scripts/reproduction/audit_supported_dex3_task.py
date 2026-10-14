"""Audit stored physical readback against frozen inputs; never approve a task or score a policy."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--states", type=Path, required=True)
    p.add_argument("--states-sha256", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--video-sha256", required=True)
    p.add_argument("--annotation-archive", type=Path, required=True)
    p.add_argument("--source-color-dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError("Use a fresh audit directory")
    if sha(a.states) != a.states_sha256 or sha(a.video) != a.video_sha256:
        raise ValueError("Physical evidence hash mismatch")
    if (
        sha(a.case / "file-manifest.json")
        != "0954068550b897b5cae5465e24e8dc7edeae025223e5e4786068bffb22628fb2"
    ):
        raise ValueError("Wrong frozen development case")
    manifest = json.loads((a.case / "file-manifest.json").read_text())
    for name in ("bundle/episode.npz", "proposal/robot-motion-proposal.npz"):
        if sha(a.case / name) != manifest["files_sha256"][name]:
            raise ValueError("Changed source or motion input")
    with np.load(a.case / "bundle/episode.npz", allow_pickle=False) as z:
        source = {key: z[key].copy() for key in z.files}
    with np.load(
        a.case / "proposal/robot-motion-proposal.npz", allow_pickle=False
    ) as z:
        motion = {key: z[key].copy() for key in z.files}
    if motion["robot_motion_target_validated"].any():
        raise ValueError("Frozen target approval flags changed")
    display = np.zeros_like(motion["source_hand_valid"], dtype=bool)
    for name, value in motion.items():
        if "display_only" in name and value.shape == display.shape:
            display |= value.astype(bool)
    state = json.loads(a.states.read_text())
    frames = state["rendered_source_frames"]
    if [f["source_frame"] for f in frames] != source["frame_indices"].tolist():
        raise ValueError("Incomplete source frame coverage")
    by_step = {r["step"]: {} for r in state["hands"]}
    for row in state["hands"]:
        si = ("left", "right").index(row["side"])
        index = int(
            np.searchsorted(
                source["timestamps_s"], row["source_time_s"] + 1e-10, side="right"
            )
            - 1
        )
        if (
            row["source_hand_valid"] != bool(source["hand_valid"][index, si])
            or row["source_directly_observed"]
            != bool(source["hand_directly_observed"][index, si])
            or row["display_only"] != bool(display[index, si])
        ):
            raise ValueError("Observation or display provenance mismatch")
        by_step[row["step"]][row["side"]] = row
    step_at_frame = [
        min(
            by_step,
            key=lambda step: abs(
                by_step[step]["left"]["source_time_s"] - f["source_time_s"]
            ),
        )
        for f in frames
    ]
    object_summaries, hand_summaries = [], []
    for oi in (0, 1):
        rows = [r for r in state["objects"] if r["object"] == oi]
        for row in rows:
            index = int(
                np.searchsorted(
                    source["timestamps_s"], row["source_time_s"] + 1e-10, side="right"
                )
                - 1
            )
            if row["source_object_observed"] != bool(
                motion["source_object_observed"][index, oi]
            ):
                raise ValueError("Object observation mask changed")
        errors = [r["position_error_m"] for r in rows if r["source_object_observed"]]
        object_summaries.append(
            {
                "object": oi,
                "observed_position_error_median_p90_max_m": np.quantile(
                    errors, [0.5, 0.9, 1]
                ).tolist(),
                "final_position_error_m": rows[-1]["position_error_m"],
            }
        )
    for side in ("left", "right"):
        rows = [r for r in state["hands"] if r["side"] == side]
        errors = [
            np.linalg.norm(
                np.array(r["actual_root_pose_stage_xyzw"][:3])
                - r["target_root_pose_stage_xyzw"][:3]
            )
            for r in rows
        ]
        angles = [
            (
                Rotation.from_quat(r["actual_root_pose_stage_xyzw"][3:])
                * Rotation.from_quat(r["target_root_pose_stage_xyzw"][3:]).inv()
            ).magnitude()
            for r in rows
        ]
        hand_summaries.append(
            {
                "side": side,
                "wrist_target_error_median_p90_max_m": np.quantile(
                    errors, [0.5, 0.9, 1]
                ).tolist(),
                "wrist_rotation_error_median_p90_max_deg": np.rad2deg(
                    np.quantile(angles, [0.5, 0.9, 1])
                ).tolist(),
                "maximum_reported_solver_penetration_m": max(
                    r["solver_contacts"]["maximum_solver_penetration_m"] for r in rows
                ),
            }
        )
    with tarfile.open(a.annotation_archive) as t:
        annotation_bytes = t.extractfile(
            "evidence/e002-source-grasp-annotation.json"
        ).read()
    if (
        hashlib.sha256(annotation_bytes).hexdigest()
        != "f4de747c96da804158b063b86d5683dd28c618ae5c24558cb4f8038399d4b8bf"
    ):
        raise ValueError("Independent annotation changed")
    # Source labels are read only after measurements and provenance checks.
    annotation = json.loads(annotation_bytes)
    contact_rows = []
    for interval in annotation["visible_grasp_intervals"]:
        side = interval["hand"]
        oi = ("left", "right").index(side)
        core = list(
            range(
                interval["definite_core_start_frame"],
                interval["definite_core_end_frame"] + 1,
            )
        )
        present = [
            frame
            for frame in core
            if by_step[step_at_frame[frame]][side]["object_contact_force_magnitudes_N"][
                oi
            ]
            > 0
        ]
        contact_rows.append(
            {
                "interval": interval["interval_id"],
                "source_definite_core_frames": len(core),
                "physical_contact_frames": present,
                "contact_is_not_force_closure_or_a_grasp": True,
            }
        )
    a.out.mkdir(parents=True)
    report = {
        "schema": "v2d.supported-task-readback-audit.v1",
        "states_sha256": sha(a.states),
        "video_sha256": sha(a.video),
        "source_frames": len(frames),
        "object_records": len(state["objects"]),
        "hand_records": len(state["hands"]),
        "control_records": len(state["controls"]),
        "all_recorded_source_observation_and_display_flags_match": True,
        "all_686_input_target_approvals_false": True,
        "objects": object_summaries,
        "hands": hand_summaries,
        "source_grasp_comparison": contact_rows,
        "scope": "Actual physics readback against inferred frozen targets and frame-indexed independent source labels; not independent human-finger ground truth, complete penetration certification or official metric parity",
        "physical_task_accepted": False,
        "official_score_claimed": False,
    }
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if a.source_color_dir:
        import cv2
        import imageio_ffmpeg
        from PIL import Image, ImageDraw

        lock = json.loads(
            (a.case / "bundle/source/provenance/inputs-lock.json").read_text()
        )
        colors = {
            Path(r["path"]).name: r["sha256"]
            for r in lock["source_files"]
            if "/color/" in r["path"]
        }
        w, h = map(int, source["image_size"])
        video = cv2.VideoCapture(str(a.video))
        writer = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{2 * w}x{h + 36}",
                "-r",
                "20",
                "-i",
                "pipe:0",
                "-c:v",
                "libx264",
                "-crf",
                "22",
                "-preset",
                "fast",
                "-pix_fmt",
                "yuv420p",
                str(a.out / "source-vs-dynamic-task-failure.mp4"),
            ],
            stdin=subprocess.PIPE,
        )
        panels = []
        try:
            for index, f in enumerate(frames):
                name = f"{f['source_frame']:06d}.png"
                original = a.source_color_dir / name
                if sha(original) != colors[name]:
                    raise ValueError("Changed source image: " + name)
                ok, rgb = video.read()
                if not ok or rgb.shape[:2] != (h, w):
                    raise ValueError("Missing or incorrectly sized dynamic video frame")
                page = Image.new("RGB", (2 * w, h + 36), (20, 20, 20))
                page.paste(Image.open(original).convert("RGB"), (0, 36))
                page.paste(
                    Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)), (w, 36)
                )
                draw = ImageDraw.Draw(page)
                draw.text(
                    (8, 8),
                    f"Source recording | frame {f['source_frame']}",
                    fill="white",
                )
                draw.text(
                    (w + 8, 8),
                    "Actual dynamic Isaac replay | TASK FAILED | no policy score",
                    fill="white",
                )
                writer.stdin.write(np.asarray(page).tobytes())
                if index % 40 == 0 or index == len(frames) - 1:
                    panels.append(page.resize((1024, 335)))
        finally:
            video.release()
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("Source comparison encoding failed")
        sheet = Image.new("RGB", (1024, 335 * len(panels)))
        for i, panel in enumerate(panels):
            sheet.paste(panel, (0, i * 335))
        sheet.save(a.out / "source-comparison-samples.jpg", quality=90)
        report["source_color_files_verified"] = len(frames)
        report["comparison_video_sha256"] = sha(
            a.out / "source-vs-dynamic-task-failure.mp4"
        )
        (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
