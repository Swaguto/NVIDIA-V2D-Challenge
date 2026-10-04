# Track 3 — public run results (20/20 episodes)

Run date: 2026-10-04. Draft PR #31 (`feat/track3-public-cad-mesh`).
Machine-readable per-episode numbers: [`track3_public_scores.csv`](track3_public_scores.csv).

All 20 public episodes completed with poses, `overlay.mp4`, a shaded
`cad_mesh_overlay_color.mp4`, and official scores. Every prediction set is
GT-independent; public ground truth is read only by the final scorer.

## Headline

| metric (inferred / gap-filled set) | value |
| --- | --- |
| mean ADD-AUC | **0.7304** |
| median ADD-AUC | 0.7308 |
| mean RPE | 0.99 cm |
| mean MPPE | 3.00 cm |
| episodes with coverage 1.0 | 20 / 20 |
| episodes with CAD overlay | 20 / 20 |

The gap-filled (`inferred`) set is the submission basis: it has coverage 1.0 on
every episode, whereas the raw `observed` set has gaps on 4 episodes (below).

## Per-episode results

Sorted by ADD-AUC. `cov` is observed-pose coverage; the inferred set is 1.0 everywhere.

| episode | objects | cov | ADD-AUC | RPE (cm) | MPPE (cm) |
| --- | --- | ---: | ---: | ---: | ---: |
| e026 | plastic_dish_rack, white_pot | 1.000 | 0.941 | 1.60 | 1.21 |
| e013 | blue_cup, plastic_dish_rack | 1.000 | 0.926 | 1.33 | 1.37 |
| e027 | plastic_dish_rack, white_pot | 1.000 | 0.886 | 1.28 | 1.78 |
| e002 | white_pot, white_pot_lid | 0.915 | 0.858 | 0.82 | 2.02 |
| e022 | blue_cup, water_pitcher | 1.000 | 0.858 | 2.31 | 2.11 |
| e040 | mini_dust_pan | 1.000 | 0.855 | 0.00* | 2.13 |
| e043 | white_pot | 1.000 | 0.794 | 0.00* | 2.58 |
| e023 | wooden_piece_1, wooden_piece_2 | 1.000 | 0.787 | 0.56 | 2.61 |
| e018 | mini_dust_pan, mini_sweeper | 0.996 | 0.778 | 0.73 | 2.81 |
| e000 | white_pot, white_pot_lid | 0.836 | 0.736 | 0.83 | 3.01 |
| e031 | mini_sweeper, plastic_dish_rack | 1.000 | 0.726 | 1.35 | 2.79 |
| e045 | white_pot | 1.000 | 0.723 | 0.00* | 3.04 |
| e021 | blue_cup, water_pitcher | 1.000 | 0.714 | 2.39 | 3.14 |
| e044 | white_pot | 1.000 | 0.704 | 0.00* | 3.20 |
| e011 | beige_cup, blue_cup | 1.000 | 0.690 | 2.36 | 3.37 |
| e041 | white_pot | 1.000 | 0.677 | 0.00* | 3.52 |
| e039 | mini_dust_pan | 1.000 | 0.561 | 0.00* | 4.41 |
| e042 | white_pot | 1.000 | 0.558 | 0.00* | 4.40 |
| e024 | wooden_piece_1, wooden_piece_2 | 1.000 | 0.542 | 0.82 | 4.64 |
| e012 | blue_cup, plastic_dish_rack | 0.997 | 0.295 | 3.32 | 5.77 |

`*` RPE is **0.00 by definition** for single-object episodes — see caveats.

## What was fixed during the run

Two episodes suffered severe FoundationPose drift. Both were diagnosed as
*gradual* pose drift rather than bad masks, and both were fixed with periodic
re-registration (`foundationpose_track.py --reinit-every N`). The existing
jump-rejection gate (15 cm / 40°) never fired because the drift was slow
(e012's largest single step was 5.65 cm).

| episode / object | change | ADD-AUC | RPE (cm) | MPPE (cm) |
| --- | --- | --- | --- | --- |
| e012 `blue_cup` | `--reinit-every 30` | 0.175 → **0.295** | 25.68 → **3.41** | 10.62 → **5.79** |
| e018 `mini_sweeper` | `--reinit-every 20` | 0.395 → **0.778** | 23.43 → **0.74** | 13.39 → **2.81** |

Also:

- **e031 `mini_sweeper` masks.** The automatic seed locked onto the left image
  border (mask centroid drifted `x≈98→61`, area collapsing to ~5.6k px). Replaced
  with the corrected `masks_sweeper` set (centroid `x≈744→854`); provenance
  recorded in `e031/mask_corrections.json`. Installed before any pose run.
- **e044 `white_pot` coverage.** 72 frames were lost to CUDA OOM caused by
  concurrent GPU jobs during the batch run. Re-running the episode alone
  recovered all of them: observed coverage `0.711 → 1.000`, poses `177 → 249`.

### How the drift was diagnosed (no GT)

For a rigid object, mask area `a` and pose depth `z` must satisfy `a·z² ≈ const`.
Deviation from that constant is a GT-free signal that pose and mask disagree:

- e012 `blue_cup`: `a·z²` jumped `0.494 → 1.980` between frames 80→100 while
  mask area stayed flat (~16–20k px) — depth was ~38 cm too large.
- e018 `mini_sweeper`: `a·z²` p95 deviation `77%`, mask area collapsing
  `12055 → 2061` px over frames 46–61 (genuine occlusion, not propagation
  failure — seed stride 20, 1000 and the shipped masks are byte-identical there).

Re-registration intervals were chosen on this consistency signal, not on public
score, then confirmed by scoring.

## Caveats and known gaps

- **Observed-pose coverage gaps** (all resolved to 1.0 by occlusion inference):
  - `e000 white_pot` 251/373 and `e002 white_pot` 285/343. The mask generator
    deliberately **suppressed** these frames — `suppressed_identity_switch_frames`
    records detected identity switches (e.g. e000 frames 0–39 and 291–372). It
    would rather emit no mask than track the wrong object.
  - `e012` 180/181 and `e018` 350/353 — 1 and 3 frames lost to jump rejection.
- **RPE is uninformative for single-object episodes.** `relative_position_error_cm`
  returns `0.0` when `bodies < 2`, so e039–e045 (single `mini_dust_pan` or
  `white_pot`) report RPE 0.00 regardless of quality. Use ADD-AUC/MPPE there.
- **ADD-AUC saturates at large error.** Rotating a submission to identity
  changed RPE from 12.7 cm to 17.0 cm while ADD-AUC stayed exactly `0.1946`.
  Do not read ADD-AUC differences at ~15 cm error as signal.
- **Worst remaining episode is e012** (AUC 0.295). `blue_cup` depth still walks
  39.8 → 53 cm against a near-static camera, so residual drift remains after
  re-registration.
- Public scores are not a leaderboard position; no hidden-set behaviour is implied.

## Reproduction

```bash
# full run (skips episodes already complete)
scripts/track3/orchestrate_public.py --root <out> --skip <episodes>

# per-object pose with periodic re-registration
fp_venv/bin/python scripts/track3/foundationpose_track.py \
  --episode 12 --object blue_cup --data <public> \
  --masks <out>/e012/masks --out <dest> --reinit-every 30

# shaded CAD overlay
v2d_venv/bin/python scripts/track3/render_cad_overlay.py \
  --episode 12 --data <public> --predictions <out>/e012/world \
  --out <out>/e012/cad_mesh_overlay_color.mp4 --natural-material
```

## Artifacts

Per episode under the run root: `poses/`, `world/`, `inferred/`,
`scores_observed.json`, `scores_inferred.json`, `overlay.mp4`,
`cad_mesh_overlay_color.mp4`, `logs/`.

Rotations are serialized `[w, x, y, z]`, matching the official scorer. Pose
projection is `X = scale · (R · raw_mesh_vertex) + translation`, with no mesh
recentering. `gt_use` is `none`/`independent` throughout the perception path.
