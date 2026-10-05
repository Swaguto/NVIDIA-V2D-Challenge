# Track 3 RunPod validation handoff

This handoff is for finishing independent validation of merged PR #31, with the
white-pot episode first. Work on `feat/track3-runpod-validation` in the isolated
worktree `/private/tmp/track3_runpod_validation`. The original checkout at
`/Users/rohithiruku/v2d_repo` has unrelated uncommitted changes. Keep PR #33
as a draft until runtime results and artifacts are reviewed; do not merge it.

## State at handoff

- PR #31 is merged into `main` at `3d26acb`.
- Draft PR #33 contains a custom Dockerfile, image build workflow, SSH startup
  script, and `docker/track3-runpod/validate_e000.sh`. Branch head before this
  handoff was `83bfd5b52f0377e7003cf78e7a76a5ad74c7c30e`.
- GitHub Actions run `37364642833`, attempt 2, was building and pushing the
  image at handoff. Check its conclusion before using the image. The workflow
  publishes `ghcr.io/swaguto/track3-runpod:<image-source-sha>` and
  `ttl.sh/v2d-track3-<image-source-sha>:24h`. The TTL tag expires after 24h.
- OpenCode 1.18.1 already has a connected RunPod MCP server (`opencode mcp
  list`). Codex has a separate local RunPod MCP configuration. Never echo or
  commit the API key. The OpenCode connection can call RunPod tools directly.
- One older RunPod A40 pod, `kfnab1ciqtarg2`, is stopped. Leave it untouched.
  No new validation pod had been created, so no new GPU billing had started.
- RunPod reported A40 Secure Cloud at $0.49/hr at handoff, plus storage. Check
  live price and capacity before creating a pod. A40 availability was low.
- The public Hugging Face dataset paths for episode 0, the two white-pot GLBs,
  stereo videos, and calibration were verified by the dataset API.
- PR #31's committed `docs/track3_public_scores.csv` claims 20 public episodes,
  mean inferred ADD-AUC 0.7304 and e000 inferred ADD-AUC 0.735925. These are
  reference claims to reproduce, not new validation results.

## Finish image build

From the isolated worktree, inspect the [workflow run](https://github.com/Swaguto/NVIDIA-V2D-Challenge/actions/runs/37364642833)
or run `gh run view 37364642833 --json status,conclusion,jobs`. If it fails,
read `gh run view 37364642833 --log-failed`, fix the Dockerfile, commit, push,
and wait for the replacement workflow. Earlier build failures due to uppercase
GHCR tag, wrong PyTorch3D tag, missing pybind11 CMake location, and missing
`h5py` have already been fixed. The image source SHA is the workflow run's
`head_sha`, even if this handoff doc is committed later.

Use the public TTL tag if GHCR is private to RunPod. Verify the chosen image
can be pulled before creating a billed pod when possible.

## Create and access a validation pod

Ask OpenCode to call the connected RunPod MCP `get-gpu-type` tool for current
A40 Secure price/capacity, then `create-pod` with a body equivalent to:

```json
{
  "name": "track3-pr31-validation",
  "image": "ttl.sh/v2d-track3-<image-source-sha>:24h",
  "gpu": {"id": "NVIDIA A40", "count": 1, "minCudaVersion": "12.4"},
  "cloud": "SECURE",
  "disk": 150,
  "mounts": {"persistent": {"size": 100, "path": "/workspace"}},
  "ports": ["22/tcp"],
  "startSsh": true,
  "startJupyter": false
}
```

Use a separate pod, not the older stopped pod. The image's `start.sh` accepts
RunPod's `PUBLIC_KEY` and starts `sshd`; the account already has a registered
SSH key. Poll `get-pod` until running and use its returned direct SSH endpoint,
or the proxy endpoint if direct SSH is unavailable. The local private key is
`~/.ssh/id_ed25519`. Check `nvidia-smi` and `python -c 'import torch;
print(torch.cuda.is_available())'` after login. If the pod cannot pull the
image or SSH fails, inspect RunPod pod logs before changing anything.

The persistent volume keeps downloaded weights and outputs across pod restarts.
It costs money until removed; download all final artifacts before terminating
the validation pod and volume.

## White-pot pilot

On the pod, run:

```bash
git clone --branch feat/track3-runpod-validation \
  https://github.com/Swaguto/NVIDIA-V2D-Challenge.git \
  /workspace/NVIDIA-V2D-Challenge
bash /workspace/NVIDIA-V2D-Challenge/docker/track3-runpod/validate_e000.sh \
  2>&1 | tee /workspace/track3-e000.log
```

Use `tmux` or `nohup` so an SSH disconnect does not kill the run. The script
downloads public e000 stereo video, parquet metadata, white-pot CAD, and
FoundationPose weights. It runs stages in order: cameras, masks, poses,
scores, overlays; then renders the textured CAD surface overlay. It asserts
each stage's `batch_<phase>.json` manifest is `done` or `cached`.

Expected output root: `/workspace/track3-validation`. Inspect:

- `e000/scores_observed.json`, `e000/scores_inferred.json`: actual metrics;
  compare with the committed CSV, allowing modest environment variation.
- `e000/overlay.mp4`: tracked CAD outline on rectified video.
- `e000/cad_mesh_overlay_color.mp4`: filled, textured CAD mesh overlay, with
  stereo depth occlusion. This is the priority visual requested by the user.
- `e000/masks/report.json`, `e000/poses/e000/*_fp_report.json`, and `e000/logs`:
  diagnose coverage and failures.

Do not trust video file size alone: download the CAD overlay and inspect sample
frames at beginning, middle, and end for pot/lid alignment and occlusion.
Check that the pot model tracks the pot rather than jumping to the lid. The
previous run's pot observed coverage was 251/373, lid 373/373, and inferred
coverage was full. AUC 0.735925 is a diagnostic reconstruction score after
one constant episode alignment, not robot policy success.

If a stage fails, read its `e000/logs/<stage>.log`. The stage runner is
resumable, but a partially written output directory with no report may need
to be moved aside before retry because stages require fresh output dirs.
Keep failed directories for diagnosis instead of deleting them blindly.

## Validate all 20 episodes

After e000 works, use `hf download nvidia/video_to_data_challenge --repo-type
dataset` to fetch all `track_3/public/meta/*`, `data/chunk-000/*.parquet`,
`mesh/**/*.glb`, and left/right video MP4s under
`videos/chunk-000/observation.images.ego_cam_b/` and `ego_cam_c/`. Confirm the
download includes all 20 parquet episodes before running. Expand includes as
needed for the `hf` CLI's glob behavior. Reuse the e000 weights and volume.

With the same environment variables as `validate_e000.sh`, run each phase for
all episodes using `python scripts/track3/batch_public.py --phase <phase>
--data /workspace/data/track_3/public --root /workspace/track3-validation`.
Use phase order `cameras`, `masks`, `poses`, `scores`, `overlays`. Check every
`batch_<phase>.json`: all 20 IDs should be `done` or `cached`, with no failed
values. Run `render_cad_overlay.py` once per episode to produce
`eNNN/cad_mesh_overlay_color.mp4`; inspect each video, especially weak e012.
Then run `report_public.py --data /workspace/data/track_3/public --root
/workspace/track3-validation` and open `index.html` locally after downloading
the report and MP4s.

Compare fresh observed/inferred scores, coverage, episode roster, camera
errors, and CAD overlays against `docs/track3_public_scores.csv`. Explain any
differences; do not claim independent validation of unrun episodes. The
pipeline's camera/object predictions consume public video, calibration, object
names, and public CAD. The final scorer alone reads public ground-truth poses.
Check this separation if altering pipeline code.

## Code map and finish

- `scripts/track3/batch_public.py`: resumable phase runner and manifests.
- `stereo_odometry.py`: stereo camera motion.
- `temporal_masks.py`: Grounding DINO seed boxes, SAM2 mask propagation, and
  pot/lid identity-switch suppression.
- `foundationpose_track.py`: fits each public GLB against rectified RGB and
  stereo depth using FoundationPose, marking observed/missing frames.
- `scripts/eval/convert_camera_poses.py`: independent camera-to-world poses.
- `infer_occlusion.py`: fills missing object poses with marked inference.
- `scripts/eval/score_track3_official.py`: observed and inferred diagnostics.
- `render_overlay.py` and `render_cad_overlay.py`: CAD outlines and filled CAD
  surfaces on video; `report_public.py` builds the HTML gallery.
- `docker/track3-runpod/`: image, SSH entrypoint, e000 pilot script.

Download scores, logs, report HTML, and representative MP4s from the pod;
publish or attach them to the draft PR in a way reviewers can access. Update
PR #33 with image tag, exact commands, scores, frame inspection findings,
deviations from the committed CSV, and any code fixes. Keep it as a draft
until review. Stop or terminate the new pod after artifacts are secured to
end GPU billing, and account for persistent volume charges.
