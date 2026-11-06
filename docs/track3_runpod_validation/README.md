# Track 3 RunPod Validation (PR31 reproduction)

Date: 2026-10-05

## Image build
GitHub Actions run https://github.com/Swaguto/NVIDIA-V2D-Challenge/actions/runs/37364642833
completed `success` at head `83bfd5b52f0377e7003cf78e7a76a5ad74c7c30e`.
Published images:
- `ghcr.io/swaguto/track3-runpod:83bfd5b52f0377e7003cf78e7a76a5ad74c7c30e`
- `ttl.sh/v2d-track3-83bfd5b52f0377e7003cf78e7a76a5ad74c7c30e:24h` (manifest verified pullable before provisioning).

The GHCR image is the durable reference. Its manifest was fetched without
registry credentials on 2026-10-06. The TTL image expires after 24 hours.
The image workflow builds on changes to its inputs on `main` or the validation
branch, and also supports manual dispatch.

## Pod
- New pod `track3-pr31-validation` (`j6r4kfld26dimz`), NVIDIA A40, Secure Cloud EU-SE-1, 100 GB persistent volume.
- Existing stopped pod `kfnab1ciqtarg2` was left untouched.
- SSH key `~/.ssh/id_ed25519`, direct endpoint `root@194.68.245.54 -p 22020`.

## e000 pilot (`docker/track3-runpod/validate_e000.sh`)
- The script now accepts either the flat FoundationPose weights layout or the
  `no_diffusion/` subdirectory produced by the upstream Google Drive folder.
  Set `TRACK3_REF` to select a repository ref when cloning; it defaults to
  `main` for runs after merge.
- cameras/masks/poses/scores/overlays manifests all `done`.
- `scores_observed.json`: ADD-AUC 0.72947, RPE 0.869 cm, MPPE 3.019 cm.
- `scores_inferred.json`: ADD-AUC 0.73685, RPE 0.835 cm, MPPE 2.999 cm. PR #31 recorded ADD-AUC 0.735925; our inferred ADD-AUC is within +0.0009 (0.13%), attributed to dataset/pose noise and minor environment differences.
- CAD overlay inspection: at frames 60/186/360 the shaded CAD region overlaps the white_pot and white_pot_lid masks by ~95%; the lid tint is consistently inside the lid mask. No frame showed the pot mesh jumping onto the lid.

## Full 20-episode validation
Equivalent commands used (on pod, with the environment from `validate_e000.sh`):
```
for phase in cameras masks poses scores overlays; do
  python scripts/track3/batch_public.py --phase "$phase" \
    --data /workspace/data/track_3/public --root /workspace/track3-validation
done
# Then render each episode's CAD video with render_cad_overlay.py and build
# index.html with report_public.py.
```
All five `batch_<phase>.json` manifests: 20/20 done or cached, no failures.
All 20 `eNNN/cad_mesh_overlay_color.mp4` produced.

Scores and coverage matched PR #31 closely. Mean inferred ADD-AUC 0.7285 (PR #31: 0.7304). Per-episode inferred ADD-AUC deviations were within ±0.01 for 18/20 episodes.

Notable differences from `docs/track3_public_scores.csv`:
- **e012**: inferred RPE 23.6 cm vs 3.3 cm (ADD-AUC similar 0.303 vs 0.295). PR #31's low e012 RPE was from a manual `--reinit-every 30` rerun on the dev instance, not carried into the PR #31 default batch; the RunPod run used the unmodified default batch, so the pose drift reappears. Diagnostic only.
- **e039**: inferred ADD-AUC 0.517 vs 0.561.
- Everything else within ~0.01 AUC, RPE mostly within ±0.1 cm.

## Artifacts
- `index.html` and per-episode videos/logs were downloaded from the pod.
- e000 pilot artifacts: `e000/cad_mesh_overlay_color.mp4` (10.8 MB), `overlay.mp4` (11.0 MB), `scores_*.json`, `track3-e000.log`, `e000_new_*.png` sample frames.
- The [validation release](https://github.com/Swaguto/NVIDIA-V2D-Challenge/releases/tag/track3-runpod-validation-2026-10-06)
  exposes selected e000/e012 sample videos for review. Its `index.html` is a
  record of the full local gallery; the other episode videos are not attached
  to that release, so links to them in the HTML will not resolve there.
