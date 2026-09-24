# V2D Challenge — Track 3 (Egocentric Video → Policy): Winning Architecture

*Prepared: Sep 24, 2026 · Leaderboard freeze: **Nov 4, 2026, 5:00 p.m. ET** (~41 days from today)*

**Track:** 3 — Egocentric Video → Policy · **Embodiment:** Unitree G1 + Dex3 (fixed, Isaac Lab) · **Submission:** `eval_e2e.py` artifacts (reconstruction + recorded policy evaluations) · **Scoring:** CHORD metrics (AUC, SP-SR, MP-SR, MPPE)

---

## 1. The assignment, decoded

**What Track 3 actually is.** Raw egocentric video from a **head-worn OAK rig** → a **trained robot policy** for the fixed embodiment (**Unitree G1 + Dex3 three-fingered hands** in Isaac Lab 2.3.1), scored on **CHORD's object-tracking metrics**: AUC (DexMachina ADD-AUC), SP-SR, MP-SR, MPPE. Method is unrestricted — reconstruct-then-retarget *or* implicit VLA/WAM. Teams submit **both a reconstruction artifact and recorded policy evaluations** via `eval_e2e.py`.

**Public split (the single most important fact in the challenge):**
- 20 episodes, 5,246 frames, 10 bimanual kitchen-style tasks: `Pot_with_lid`, `Cup_Stack`, `Cup_In_Dish_Rack`, `Dust_Brush`, `Water_Can_Blue_Cup`, `Planter_Stand`, `Pan_In_Dishrack`, `Brush_Dishrack`, `Dustpan_Solo`, `Pan_Solo`.
- Three **frame-exact, calibration-provided** ego streams: `ego_cam_a` (2028×1520 color) + **`ego_cam_b` / `ego_cam_c` = a 75 mm-baseline STEREO pair**. No exo cameras distributed.
- **Textured object meshes and URDFs** for all 10 objects; object identity = mesh folder name; per-episode object membership is fixed and declared in parquet schema metadata.
- **MoCap ground-truth object poses** in parquet (`[x, y, z, qw, qx, qy, qz]`, `world_T_object`, `visible` flag). **No camera-to-mocap-world extrinsic is provided** — "no camera-to-mocap-world extrinsic, so meshes cannot be projected into the images without solving that first."

**Data conventions (from `track_3/README.md`):**
- `observation.objects` — one row per video frame; `objects[i]` stays the same object across all frames; membership is constant within an episode.
- Pose is translation in meters, quaternion w-first, in the scanned mesh's frame.
- x/y centered per episode; z not centered (~0.61 m resting table height).
- `visible == false` ⇒ pose is all zeros — **filter on `visible` before using a pose**.

**Goals & constraints.**

| | |
|---|---|
| **Win condition** | #1 on Track 3 leaderboard: maximize object-tracking accuracy of a G1+Dex3 policy **and** deliver a valid 4D-reconstruction artifact |
| **Hard constraints** | Fixed embodiment (G1+Dex3, Isaac Lab 2.3.1 + RSL-RL PPO); eval via official script; ≤5 submissions/week (unlimited last 3 days); ~6 weeks |
| **Real constraints** | 20 demo episodes for all training + validation; hidden-GT test set; head-mounted ego = heavy hand self-occlusion + egomotion |
| **Exploits (not violations)** | Public GT is provided to everyone; using it to calibrate the fixed rig and validate is legitimate. Stereo + CAD meshes are provided modalities — use them |
| **Non-goals** | Real-world deployment, sim-to-real, teleop. Evaluation is in simulation |

---

## 2. Benchmark decision: CHORD over DexMachina — decisively

Use **CHORD** (`nvidia-isaac/video_to_data`, arXiv 2607.00033) as the scaffold.

1. **It is the challenge's own metric system and baseline.** Track 2/3 are scored with CHORD's metrics (AUC, SP-SR, MP-SR, MPPE), and CHORD's robotic-grounding code ships in the starter toolkit. The job is effectively "beat the organizers' method applied to ego video."
2. **It covers the whole Track-3 surface**, which DexMachina does not:
   - **Whole-body on G1+Dex3 from hand-only references** (CHORD Sec. 4.5, Appendix E) — exactly our setting.
   - **Video-reconstruction validation from a single ego/mono RGB clip** (Appendix C) and **quantified robustness to reconstructed-demo noise** (Appendix A: wrist up to 60 mm, orientation 30°, contacts 30 mm / 30°).
   - 4,739 tasks across ARCTIC, TACO, **HOT3D (egocentric)**, OakInk2, DexYCB, GRAB, H2O; real-world open/closed-loop deployment.
3. **Numbers say CHORD is the leader:**

| Method | AUC ↑ | SP-SR ↑ | MP-SR ↑ | JPE (°) ↓ | RPE (cm) ↓ | MPPE (cm) ↓ |
|---|---|---|---|---|---|---|
| **CHORD** | **0.918 ± 0.018** | **0.926 ± 0.022** | **0.894 ± 0.032** | **4.34 ± 0.25** | **5.14 ± 0.54** | **5.11 ± 0.39** |
| SPIDER | 0.804 ± 0.044 | 0.911 ± 0.039 | 0.789 ± 0.047 | 41.35 ± 2.44 | 6.99 ± 1.56 | 14.28 ± 2.79 |
| DexMachina | 0.737 ± 0.053 | 0.690 ± 0.068 | 0.569 ± 0.097 | 13.10 ± 2.31 | 9.92 ± 1.64 | 15.88 ± 0.70 |
| Human2Sim2Robot | 0.730 ± 0.056 | 0.686 ± 0.068 | 0.569 ± 0.085 | 11.80 ± 1.50 | 9.78 ± 1.03 | 16.59 ± 0.51 |
| ManipTrans | 0.506 ± 0.006 | 0.423 ± 0.011 | 0.323 ± 0.003 | 30.07 ± 0.61 | 9.91 ± 0.42 | 21.24 ± 6.19 |

   CHORD holds ~0.85–0.98 AUC out to 40–48 s horizons where baselines collapse.

**DexMachina's role:** one good idea we inherit — the **virtual object controller (VOC)** curriculum. CHORD already absorbed it. Treat DexMachina as a component, not a benchmark, for this track.

---

## 3. The gap to beat (SOTA, honest inventory)

| Method | Setting | Weakness for Track 3 |
|---|---|---|
| **CHORD** (organizers) | RL + contact-wrench guidance, VOC | Ego-mono E2E demo only hits **0.647 AUC** because monocular reconstruction is the bottleneck |
| DexMachina / ManipTrans / SPIDER / H2S2R | RL with contact-position guidance | Location-matching ≠ motion-effect (CHORD's thesis); all lose to CHORD on every metric |
| EgoAERO | Asset-free ego RGB-D → two-stage residual RL | Needs depth; weaker retargeting; no whole-body G1+Dex3 |
| EgoEngine | Ego RGB → robot demos (zero-shot VLA) | Does not optimize the object-tracking-in-sim metric |
| EgoVLA / VITRA / ACE-Ego-0 / Wh0 | VLA pretraining on ego video | Need robot post-training, ignore contacts, and VLA output is not natively scored well by ADD-AUC on object keypoints |
| EgoScaler | Ego → 6-DoF object trajectories | No hand/contact/wrench; rotation-from-SVD is weak |

**Leaderboard arithmetic:** the metric is object-tracking accuracy of the *policy*, which is *literally* what CHORD's `r_task` (SE(3) pose-tracking exponential kernel) + contact-wrench rewards optimize. So the highest-metric-density path is **RL-first, VLA-second**, with two dominant levers:

1. **Reconstruction quality of the reference** (object trajectory + contacts) — CHORD's own mono pipeline proved this is where 0.647 → 0.9+ comes from.
2. **Policy tracking precision** on that reference — CHORD at 0.918 on clean mocap refs.

**One guiding conviction: perception is the score bottleneck, not the policy.** CHORD on clean mocap refs ≈ 0.918; CHORD end-to-end on mono ego video ≈ 0.647. The entire ~0.27-AUC delta is reconstruction. The design attacks that delta directly.

---

## 4. Winning architecture

```
egocentric video (a + stereo b/c)
        │
        ▼  STAGE A — Perception & 4D HOI reconstruction
A1 shadow-free rig solve ............ camera trajectory + metric scale
                                        (NOT MoGe mono — real stereo)
A2 object .......................... grounding + SAM2 mask + CAD-template
                                        6-DoF tracking (FoundationPose)
A3 hand ............................ ViPE/HaMeR + Dyn-HaMR MANO,
                                        depth-anchored by stereo (not a
                                        viewing-ray shift on mono depth)
A4 contact ......................... penetration-free proximity → contacts,
                                        friction cones, object-frame wrenches
A5 joint refinement ................ Gaussian-splat scene + SLAM-based optimizer
                                        with stereo depth loss
A6 validation ...................... public GT object poses (regression-free)
        │
        ▼  STAGE B — Retargeting, whole-body
B1 MANO → Dex3 differential IK (QP)  wrist pose + fingertip residuals, joint limits
B2 G1 whole-body inpainting ........ hand-only refs → full body (Zhu et al. module,
                                        seeded with AMASS priors)
B3 quality gate ..................... capsule penetration < 2 cm; concavity-aware
        │
        ▼  STAGE C — Policy learning (G1+Dex3, Isaac Lab, PPO)
C1 CHORD reward stack .............. r_task (+r_rel for multi-object tasks)
                                        + r_imit + r_cws (+ r_fc fallback)
C2 curriculum ...................... VOC w/ decaying strength + mid-trajectory
                                        resets + stabilization window + task-relevant
                                        wrench perturbations
C3 noise-aware contact ............. stereo-confident contacts; per-task switch
                                        CWS ↔ force-closure; confidence weighting
C4 (optional) two-stage residual ... EgoAERO-style hand-tracking → residual, for
                                        the noisiest clips
C5 (optional, Industry award) ...... VITRA/pi0 post-trained on reconstructed
                                        pseudo-actions, reliability-weighted
        │
        ▼  STAGE D — Metric-optimized eval harness
D1 reproduce eval_e2e.py ........... decode AUC/SP-SR/MP-SR/MPPE definitions
D2 local leaderboard ............... hold out 4–5 public episodes as proxy test
D3 submission ..................... reconstruction artifact + recorded rollouts
```

### Stage A — Perception (the 0.647 → 0.9+ lever)

- **A1 Metric scale & camera trajectory.** CHORD anchors on monocular **MoGe** depth; we have a real **75 mm stereo pair** that is frame-exact with the color stream. Run **stereo SLAM / Stereo-DROID** over `ego_cam_b/c` with `ego_cam_a` extrinsics attached. This removes the two biggest Appendix-C errors: object metric scale and hand depth anchoring. Expected gain: object pose error to **<1 cm / <3°** during visibility, vs several cm for mono depth anchoring.
- **A2 Object.** Identity is known (mesh folder name = object; per-episode membership fixed). Use **Grounding DINO** (prompted with object name) → SAM2 masks → **FoundationPose with the CAD mesh** as template → per-frame 6-DoF. This is Track-1-grade tracking and eliminates SAM3D object-mesh reconstruction error *when assets are provided*. **Keep an asset-free fallback** (EgoAERO-style mask → Gaussian/mesh) in case the held-out set hides meshes/URDFs.
- **A3 Hand.** ViPE/HaMeR + **Dyn-HaMR** (dynamic-camera MANO), then **anchor translation along the viewing ray against stereo disparity** instead of MoGe. In ego video the visible hand is the wearer's — no identity ambiguity.
- **A4 Contact & wrenches.** Contact when fingertip-to-object distance < ~2 cm (CHORD's HOT3D rule); bridge ≤15-frame breaks; majority-vote per object. Build the **wrench matrix** (Coulomb friction cone polyhedralized into *d* edge forces) from **object-frame contact positions/normals**. **Critical detail:** contact position/normal noise (the 30 mm / 30° rows of CHORD's robustness ablation) is the fragile point — this is where stereo accuracy pays off and where A5's joint refinement matters.
- **A5 Joint refinement.** Fit a **Gaussian-splat scene** anchored to hand + object meshes + background, camera initialized from SLAM, optimizing photometric + stereo-depth + segmentation + temporal-smoothness (CHORD's final stage) — with the depth term fed by **real stereo** rather than mono estimates.
- **A6 Validation.** Use public GT object poses as a regression oracle: per-episode tracking error, contact precision, and — the single highest-value step — **solve the fixed camera-to-world transform once on the train split and confirm it transfers**, since the rig is static hardware and the pose convention (`world_T_object`, mesh frame) is documented.

### Stage B — Retargeting + whole body

- **B1** CHORD's **differential-IK (QP)** MANO → Dex3, tracking wrist pos + orientation + fingertip residuals within joint limits (~200 iterations/frame). Tune per-morphology alignment.
- **B2 Whole-body.** We only get hand refs (head-mounted ego, no body). Use their **whole-body inpainting module** (end-effector → full-body, trained on mocap priors) to produce the G1 trajectory — exactly what CHORD did for G1+Dex3 at **90.77% whole-body success**.
- **B3 Quality gate.** Reject/soften clips with >2 cm hand-object or hand-hand capsule penetration; concavity-aware (skip objects with convex-hull-to-mesh volume ratio > 3, e.g. dishracks).

### Stage C — Policy learning

The scoring surface. Build CHORD faithfully first (floor ≈ its clean-bench 0.918 / whatever it posts E2E), then push:

- **C1 Rewards**: `r = r_task + r_rel (multi-object) + r_imit + r_cws` (+ penalties for missed/unintended contacts). Keep the SE(3) pose kernel and the **relative-pose reward** for multi-object tasks (`Cup_In_Dish_Rack`, `Pan_In_Dishrack`, `Water_Can_Blue_Cup` need it).
- **C2 Curriculum**: VOC with decaying strength; reset to arbitrary mid-trajectory states with a brief full-VOC stabilization window; **wrench-sampled task-relevant perturbations** for robustness.
- **C3 Noise-aware contact**: confidence-weighted CWS support function; per-task peel-back to the **force-closure objective** (`r_fc`) where contact precision is low — CHORD's documented remedy for noisy video refs. Wherever stereo + CAD yields clean contacts, keep full CWS (correlates r ≈ 0.80 with success).
- **C4 Residual stage** for the hairiest clips (EgoAERO's two-stage: track-hand policy, then object/contact-conditioned residual) — insurance against contact jitter.
- **C5 VLA track (optional — Industry award + belt-and-suspenders)**: post-train **VITRA/pi0** on reconstructed pseudo-action trajectories with reliability-aware weighting (ACE-Ego-0's idea), co-trained with a sliver of simulated robot rollouts. Do **not** bet the leaderboard on it; RL is better aligned to ADD-AUC.

### Stage D — Evaluation-doctoral rigor (where most teams lose points)

The biggest structural risk is mis-scoring. First week: **reproduce `eval_e2e.py` exactly**, decode the precise ADD-AUC / SP-SR / MP-SR / MPPE implementations (threshold sets, keypoint sets, symmetric-ADD, downsampling, whether eval uses privileged state), and mirror Track 2's tier-1 reference-trajectory expectations. Build a **local proxy leaderboard** by holding out 4–5 public episodes (never tune on the GT of held-out folds). Submit early and often (5/week) to learn the true scorer's quirks.

---

## 5. Exactly where we beat CHORD / SOTA

| Limitation in CHORD / SOTA | Our fix (all within scope) |
|---|---|
| Mono-MoGe metric anchoring → **0.647 E2E AUC** | Real 75 mm **stereo** SLAM + depth; CAD-template object tracking |
| Unknown-object mesh reconstruction error | Provided **mesh + URDF** → FoundationPose template tracking; asset-free path retained as fallback |
| Raw CWS reward fragile to noisy video contacts | Stereo-confident contacts + confidence weighting + documented **force-closure fallback** per clip |
| No camera-to-world alignment provided | One-time rig calibration from public GT, validated to transfer (hardware is fixed) |
| Whole-body was second-class (inpainting) | Dedicated AMASS-seeded inpainting; G1+Dex3 retarget gate |
| No robustness to the 60 mm-wrist / 30 mm-contact noise regime | Matched-noise augmentation during RL (CHORD's own level-4/5 rows) + seed ensembling |
| VLA approaches don't optimize object-tracking metrics | RL-first (metric-aligned), VLA held as residual/auxiliary |

**Quantitative target:** object trajectory **<1 cm / <3°**; E2E policy **AUC ≥ 0.90, SP-SR ≥ 0.90, MP-SR ≥ 0.87, MPPE < 6 cm** on held-out public episodes with GT as reference. That matches/exceeds CHORD's clean-bench strength (0.918 AUC) while being *fully end-to-end*.

---

## 6. Execution timeline to Nov 4 freeze (~6 weeks)

- **W0 (days 1–3):** env bring-up (V2D Docker images, Isaac Lab, G1+Dex3); download data; **stand up `eval_e2e.py` and decode metrics**; calibrate camera→world on public split.
- **W1:** reproduce CHORD baseline E2E on 2–3 public tasks; establish numbers. Start mono reconstruction port (Appendix C replica) as control.
- **W2:** **stereo upgrade** (SLAM + depth anchor), CAD-template object tracking; validate against public GT per episode; freeze per-task map of reconstruction quality and contact precision.
- **W3:** retargeting (MANO → Dex3), whole-body inpainting, quality gates; train CHORD RL; measure AUC/SP-SR/MP-SR/MPPE on held-out folds. First submission (baseline) mid-week.
- **W4–W5:** the scoring push: confidence-weighted CWS, per-clip CWS ↔ `r_fc` selection, VOC/reset/stabilization curriculum, residual-stage insurance, seed ensembling, noise augmentation; iterate reconstruction → policy (both directions). 2–3 submissions/week.
- **W5.5:** final 3 days — unlimited submissions; pick best-validated artifacts; **don't game the public split** (post-challenge large-scale evaluation is the tiebreaker/narrative); assemble technical report + code release (required for all awards, incl. Industry Innovation Award).

---

## 7. Risks, limitations — and what's actually fixable in 6 weeks

**Fixable in-scope (primary bets):**
- Mono-depth bottleneck → **stereo**.
- Unknown-object mesh error → **CAD templates**.
- Contact-noise fragility → **confidence weighting + force-closure fallback**.
- Missing camera→world extrinsic → **GT calibration**.
- Missing full body → **whole-body inpainting**.
- URDF / sim asset quality → **early curation**.
- Metric-definition ambiguity → **eval reproduction in week 1**.

**Real-but-managed:**
- Head-mounted ego occlusion / motion blur at 20 fps → multi-view fusion of 3 cams + joint refinement stage.
- Overfitting to 20 episodes → held-out folds, perturbation, seeds, matched-noise augmentation.
- Held-out test set may drop meshes/URDFs → keep the asset-free path warm.
- Small team = compute ceiling → scale the RL pool only on the highest-value tasks; cache everything.

**Explicitly NOT fixable in-scope:**
- Body pose fundamentally absent from head-mounted ego (mitigated only via inpainting priors).
- Physical sim-to-real (not required — evaluation is in simulation).
- Hand pose under total, sustained occlusion (mitigated by contact/relative-object inference, never eliminated).

---

## 8. Resource ask

- **GPU node for RL:** ≥4× (ideally 8×) L40S/A100-class, Isaac Lab + RSL-RL.
- **Team:** 2–3 engineers — one RL/retargeting, one reconstruction/perception, one eval/infra. The perception half buys the big AUC; the RL half compounds it.
- **Priority ranking under crunch:**
  1. Eval harness + stereo reconstruction + CAD tracking + CHORD RL floor
  2. Contact robustness + curriculum
  3. Whole-body polish
  4. VLA / residual insurance
  5. Industry-Award narratives

---

## 9. Summary

This is a winnable track because the organizers' own method concedes ~0.27 AUC to monocular reconstruction, and the dataset gifts us exactly the three tools that close that hole: **a real stereo pair, CAD/URDF assets, and private GT for rig calibration**. Beating 0.647 (CHORD's mono E2E) is near-certain; beating the clean-reference 0.918 end-to-end is the stretch goal that wins #1.

---

### Key references

- Challenge: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- Starter toolkit: https://github.com/nvidia-isaac/video_to_data/tree/release/0.2.0/
- Dataset (Track 3): https://huggingface.co/datasets/nvidia/video_to_data_challenge/tree/main/track_3
- CHORD paper: arXiv 2607.00033 · project: https://nvidia-isaac.github.io/video_to_data/chord/
- DexMachina: arXiv 2505.24853 · project: https://project-dexmachina.github.io/
- Egocentric SOTA: EgoAERO (arXiv 2606.08057), EgoEngine (arXiv 2606.12604), EgoVLA (arXiv 2507.12440), Wh0 (arXiv 2606.22136), EgoZero (arXiv 2505.20290), ACE-Ego-0 (arXiv 2606.17200)