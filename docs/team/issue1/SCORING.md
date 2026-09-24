# How we get scored (Track 3) — one page

## What the judges measure

Track 3 (Egocentric Video → Policy) is scored on **how precisely the robot moves each
object like the human did**. The reference is the motion-captured object 6-DoF pose
trajectory (`observation.objects` in the dataset). Your pipeline must produce, from
egocentric video, a robot policy whose **simulated object pose trajectory** matches that
reference. Submission is via the organizers' `eval_e2e.py`; we reproduce its four metrics
locally.

Everything reduces to: **compare an achieved object trajectory against the reference
trajectory, per object, per frame.**

## The four numbers (higher = better for the first three, lower = better for MPPE)

| Metric | Origin | Question it answers | Threshold / mechanics |
|---|---|---|---|
| **AUC** (a.k.a. ADD-AUC) | DexMachina | "Do the object's 3D surface points land where they should?" | For each frame, sample 500 points on the object's mesh, move them by the achieved vs reference transform, average the distance (ADD, metres). Accuracy = fraction of (frame, world) samples below each threshold in **1…9 cm**. **AUC = area under that accuracy-vs-threshold curve** (thresholds rescaled to [0,1], so AUC ∈ [0,1]). |
| **SP-SR** | SPIDER | "Was the whole episode close enough, on average?" | Per episode: frame-mean position error ≤ **10 cm** AND frame-mean geodesic orientation error ≤ **0.5 rad (~28.6°)**. Success fraction of episodes/worlds. |
| **MP-SR** | ManipTrans | "Was *every* object tracked tightly?" | Per object: position error < **3 cm** AND orientation error < **30°**. Every object must clear it (averaged across its pose parts). Success fraction of episodes/worlds. |
| **MPPE** | CHORD | "How far off is the object, continuously?" | Six keypoints on the object at **±5 cm** along X/Y/Z; per frame take the worst body, average across frames → cm. |

Two notes that keep your expectations honest:

- **AUC is the discriminator.** CHORD on clean mocap refs ≈ 0.918 ADD-AUC; CHORD end-to-end
  from mono ego video ≈ 0.647. Our plan is to beat that with **stereo + CAD-template tracking**.
- **SP-SR is generous** (10 cm / 28.6°): it saturates quickly. MP-SR and MPPE are the strict
  ones you should watch during training. See `docs/team/plan.md` §metrics for targets
  (AUC ≥ 0.90, SP-SR ≥ 0.90, MP-SR ≥ 0.87, MPPE < 6 cm on held-out public episodes).

## Our local proxy leaderboard

- **Proxy test set = 5 held-out public episodes** (`scripts/eval/reference_loader.py`):
  episodes **0** (Pot_with_lid), **11** (Cup_Stack), **23** (Planter_Stand), **31**
  (Brush_Dishrack), **39** (Dustpan_Solo) — one per distinct task. These GT trajectories are
  **never used for tuning/calibration**. The real (video-only) eval set under
  `data/hf/track_3/evaluation/` has no GT, so don't aim to "learn" it; the proxy is our
  honest stand-in.
- **Score any candidate in one command** (runtime ~0.4 s for the whole proxy set):
  ```bash
  python scripts/eval/score_candidate.py \
      --candidate data/outputs/<run>/episode_{:06d}.parquet \
      --episodes 0,11,23,31,39 --episode-pattern --json <run>.json
  ```
  Candidates are either a reconstruction parquet (same `observation.objects` schema) or a
  recorded rollout `.npy` (`[T, W, B, 7]` poses, xyzw, `--mode rollout`). Use
  `--align-rigid` when the candidate is in a different world frame than GT (Umeyama
  alignment before scoring; default **off** — match the GT frame in your pipeline).

## What has to be true for a confident score

1. **Frames align**: candidate and reference must be re-indexed to the same
   `frame_index` (global, not per-episode indices) — the data's `frame_index` is already
   global; match by it.
2. **Quaternion convention**: parquet stores **w-first** (`[x,y,z,qw,qx,qy,qz]`); the CHORD
   metrics use **xyzw**. `reference_loader.py` handles the swap; your candidate loader must
   too (`scripts/eval/score_candidate.py::load_candidate_parquet`).
3. **Units**: metres and radians everywhere.
4. **Symmetric handling**: ADD over mesh surface points is already symmetric-friendly
   (surface sampling, not vertex index matches).
5. **Poses in the object's scan frame** (the mesh puts them where the real object was):
   don't transform the reference — transform your *candidate* into that frame.

## Grounding & infrastructure (depends)

- GPU box + Docker + Isaac Lab + CHORD smoke test **running** (VNC stack live on
  `rising-gold-junglefowl`: `localhost:5900`).
- Baseline reproduction of the organizers' `flash_chord.evaluation.metrics`
  (`robotic_grounding/flash_chord/.../evaluation/metrics.py` upstream) — this file is our
  exact reference; `scripts/eval/chord_metrics.py` is that implementation vendored in.
- First end-to-end run: ego video → reconstruction → retarget each proxy episode → record
  rollout → score. If the oracle (GT-as-candidate) scores 1.000/0.000 and a noisy
  candidate scores monotonically worse, the harness is trustworthy.

## Where the code lives

- `scripts/eval/chord_metrics.py` — vendored CHORD metrics (ADD-AUC, SP-SR, MP-SR, MPPE).
- `scripts/eval/glb_mesh.py` — reads `public/mesh/*.glb`, area-weighted surface sampling.
- `scripts/eval/reference_loader.py` — GT loader (xyzw conversion) + **PROXY_EPISODES** split.
- `scripts/eval/score_candidate.py` — CLI scorer (parquet reconstruction or `.npy` rollout).
- Run `python scripts/eval/score_candidate.py -h` for all flags.