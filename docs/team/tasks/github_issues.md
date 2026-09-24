# GitHub Issue Cards — paste-ready

*How to use: create these from each card below. Suggested board columns: **Backlog → Ready → In Progress → Review → Done**. Suggested labels: `infra`, `vision`, `robot`, `eval`, `integration`, `blocker`, `stretch`. Suggested milestones: `W1`, `W2`, `W3`, `W4-5`, `Freeze`. Assign by role number (Team Brief §4).*

---

## Sprint 1 (W1–W2): make everything talk to each other

### Issue #1 — Set up GPU box, Docker, Isaac Lab + CHORD smoke test
- **Label:** `infra` · **Assignee (Role 2):** ______
- **Why:** Everyone's code runs on this machine. Until it's up, nobody works.
- **What to do:**
  1. Provision the cloud GPU (A100/H100/L40S recommended; **not** Blackwell sm_120; driver ≥ 580).
  2. Install Docker + NVIDIA Container Toolkit.
  3. Clone `video_to_data` (release/0.2.0) and build the reconstruction + robotic-grounding containers.
  4. Run the CHORD smoke test (one task, few steps) to prove the stack works.
  5. Sync the dataset to a shared folder; write a `README` with login/login instructions.
- **Done when:** A teammate can log in, run a container, and see the test task train a few steps.
- **Resources:**
  - Starter toolkit: https://github.com/nvidia-isaac/video_to_data/tree/release/0.2.0 (README has install + docker commands)
  - Robotics grounding setup: `video_to_data/robotic_grounding/docs/SETUP.md`
  - Isaac Lab: https://github.com/isaac-sim/IsaacLab · RSL-RL: https://github.com/leggedrobotics/rsl_rl
  - Dataset (local copy already downloaded): `~/track3_data/hf/track_3` (see Issue #1)
- **Needs from:** Captain → which cloud provider/instance.

### Issue #2 — Decode the official scoring (`eval_e2e.py`) + build a local leaderboard
- **Label:** `eval` · **Assignee (Role 3):** ______
- **Why:** We must know exactly how we'll be scored, and measure our own progress fast (AUC / SP-SR / MP-SR / MPPE).
- **What to do:**
  1. Read the challenge submission section + starter toolkit for `eval_e2e.py`.
  2. Write a one-page plain-English "how scoring works" doc.
  3. Implement the four metrics (AUC=ADD-AUC, SP-SR, MP-SR, MPPE) against object-pose references.
  4. Hold out 4–5 public episodes as a private proxy test set; build a script that scores any candidate reconstruction + recorded rollout against them.
- **Done when:** Scorekeeper can score any output in <1 min and the team understands the metrics.
- **Resources:**
  - Challenge page (evaluation): https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
  - CHORD paper metric table: https://arxiv.org/abs/2607.00033 (Sec 4.1) · HTML: https://arxiv.org/html/2607.00033v2
  - DexMachina AUC metric: https://github.com/MandiZhao/dexmachina
  - Local test-set episodes (no GT, these are the real test videos): `~/track3_data/hf/track_3/evaluation/videos`
  - Fork `video_to_data`; check `robotic_grounding` for eval logic.
- **Depends on:** #1 (running env).

### Issue #3 — Camera↔world calibration from public GT (the highest-value step)
- **Label:** `vision` · **Assignee (Role 4):** ______
- **Why:** Data docs say there's no camera-to-mocap-world transform. We compute it once from the public episodes (which have GT), then trust it for the test episodes, because the rig is fixed hardware.
- **What to do:**
  1. Inspect `camera_calibration.json` (intrinsics + cam-to-cam extrinsics; confirm the 75 mm stereo baseline between b/c).
  2. Using public GT object poses, solve the world→camera transform (rigid alignment over frames where objects are visible).
  3. Verify it transfers across episodes; report residual error.
- **Done when:** We can project the GT object mesh into the images correctly; error written in the report.
- **Resources:**
  - `track_3/README.md` (pose conventions, "solving that first" note): local at `~/track3_data/hf/track_3/README.md` or https://huggingface.co/datasets/nvidia/video_to_data_challenge/resolve/main/track_3/README.md
  - Calibration: `~/track3_data/hf/track_3/public/meta/camera_calibration.json`
  - GT parquets: `~/track3_data/hf/track_3/public/data`
- **Needs from:** Role 5 (which objects are visible per frame).

### Issue #4 — Stereo depth + camera trajectory (metric scale) [A1]
- **Label:** `vision` · **Assignee (Role 4):** ______
- **Why:** CHORD guesses distance from one eye (monocular MoGe) — that alone causes the ~0.27 AUC loss. Our stereo pair measures distance for real.
- **What to do:**
  1. Run stereo SLAM (e.g. DROID-SLAM stereo mode) on `ego_cam_b` + `ego_cam_c` to get true metric depth + camera path.
  2. Attach `ego_cam_a` (color) via its extrinsics so color and depth are aligned.
  3. Produce, per episode: per-frame depth, camera poses, scale check.
- **Done when:** One episode shows depth + camera path, with cm-level scale verified against GT table height (~0.61 m) and object sizes.
- **Resources:**
  - DROID-SLAM (stereo supported): https://github.com/princeton-vl/DROID-SLAM
  - MoGe (mono, for comparison): https://github.com/microsoft/MoGe
  - OAK camera docs: https://docs.luxonis.com/
  - CHORD Appendix C (what errors we're fixing): https://arxiv.org/html/2607.00033v2
  - Stereo pair = `ego_cam_b` + `ego_cam_c` (75 mm baseline); color = `ego_cam_a`.
- **Depends on:** #3 (calibration) for alignment.

### Issue #5 — Object tracking from CAD templates [A2]
- **Label:** `vision` · **Assignee (Role 5):** ______
- **Why:** We know the exact 3D model of every object. Template-based tracking beats "reconstruct the object from scratch" and removes that error source.
- **What to do:**
  1. Prompt Grounding DINO with the object name (mesh folder name) → SAM2 masks on a few keyframes.
  2. Run FoundationPose with the CAD mesh as template → per-frame 6-DoF pose.
  3. Validate against the public GT parquets; produce a per-episode error report (position cm + rotation deg).
  4. Keep an asset-free fallback note for test episodes in case meshes are hidden.
- **Done when:** Error report on all 20 public episodes; median error < a few cm / few degrees while visible.
- **Resources:**
  - FoundationPose: https://github.com/NVlabs/FoundationPose
  - Grounding DINO: https://github.com/IDEA-Research/GroundingDINO
  - SAM 2: https://github.com/facebookresearch/sam2
  - Object meshes: `~/track3_data/hf/track_3/public/mesh/<object_name>` · URDFs: `public/urdf` · GT: `public/data`
  - Object-name prompt = mesh folder name (e.g. `white_pot`).
- **Depends on:** #4 (depth/camera path); #3 (calibration).

### Issue #6 — Egocentric hand pose + touches (contact events) [A3/A4-part]
- **Label:** `vision` · **Assignee (Role 6):** ______
- **Why:** The robot learns from how the human's hands move and touch the object. This is the hardest, most occluded job.
- **What to do:**
  1. Run hand tracking (ViPE/HaMeR per-frame → Dyn-HaMR for temporal consistency) to get MANO hand + fingers per frame.
  2. Anchor hand depth using stereo disparity (not mono guessing) along the viewing ray.
  3. Detect touch/contact: fingertip-to-object distance < ~2 cm from the object poses (Role 5 output).
- **Done when:** Hand-overlay + contact annotations on ≥3 episodes, visually sane and time-smooth.
- **Resources:**
  - HaMeR: https://github.com/geopavlakos/hamer · MANO model: https://mano.is.tue.mpg.de/
  - ViPE: https://github.com/neeKo96/ViPE · Dyn-HaMR project: https://reddy-lab.github.io/dynhamr-project/
  - CHORD contact rule (HOT3D, 2 cm / 15-frame bridging): Appendix D of the CHORD paper
- **Depends on:** #4, #5 (depth + object poses to compute 3D fingertip–object distances).

### Issue #7 — Joint 3D refinement + quality report [A5]
- **Label:** `vision` · **Assignee (Role 7):** ______
- **Why:** All per-frame outputs are jittery/noisy; we merge them into one smooth, consistent 3D movie that the robot team can trust.
- **What to do:**
  1. Fit a Gaussian-splat scene to each episode, anchored to hand + object meshes + background, camera from #4's SLAM.
  2. Optimize photometric + depth (stereo) + segmentation + temporal-smoothness losses to jointly clean object/hand/camera.
  3. Output per-episode quality report (jitter, drift, contact precision).
- **Done when:** One episode fully refined, viewable in a 3D viewer; jitter/drift numbers written down.
- **Resources:**
  - Gaussian splatting: https://github.com/graphdeco-inria/gaussian-splatting
  - SAM3D (object mesh ref): https://github.com/RussRobin/SAM3D
  - CHORD Appendix C refexact pipeline: https://arxiv.org/html/2607.00033v2
- **Depends on:** #4, #5, #6 (inputs to merge).

### Issue #8 — Retarget human hand → Dex3 robot hand + quality gate [B1/B3]
- **Label:** `robot` · **Assignee (Role 8):** ______
- **Why:** The robot's hand is different from a human's; we must mathematically "rewrite" the human hand motion onto the 3-finger Dex3 hand, and reject impossible moves.
- **What to do:**
  1. Run CHORD's differential-IK (QP) MANO→Dex3 (wrist pose + fingertip residuals, joint limits).
  2. Run the quality gate: reject clips with >2 cm capsule penetration (hand-object / hand-hand); skip very concave objects.
  3. Visualize the retargeted clip in the simulator.
- **Done when:** Sim video of the robot hand mimicking the human on 1–2 tasks, no obvious interpenetration >2 cm.
- **Resources:**
  - CHORD retargeting details: Appendix D (IK QP optimizer, penetration checks) — https://arxiv.org/html/2607.00033v2
  - Code: `video_to_data/robotic_grounding` (retargeting utilities + README)
  - DexMachina retarget configs (reference): https://github.com/MandiZhao/dexmachina
- **Depends on:** #6/#7 (hand + refined object motion).

### Issue #9 — Whole-body inpainting for G1 [B2]
- **Label:** `robot` · **Assignee (Role 8):** ______
- **Why:** We only have hand motion from the video; the G1 humanoid is a full body. We fill in the rest of the body so the robot's torso/arms follow naturally.
- **What to do:**
  1. Use CHORD's whole-body inpainting module (end-effector → full body, mocap priors).
  2. Produce full G1+Dex3 trajectories from each hand-only reference.
  3. Sanity-check: robot can stand and reach the way the human did.
- **Done when:** Full-body G1+Dex3 trajectory plays cleanly in Isaac Lab for ≥2 tasks.
- **Resources:**
  - CHORD Appendix E (whole-body inpainting): https://arxiv.org/html/2607.00033v2
  - `video_to_data/robotic_grounding` (whole-body module ships here)
- **Depends on:** #8.

### Issue #10 — CHORD RL baseline on the first task [C1/C2]
- **Label:** `robot` · **Assignee (Role 9):** ______
- **Why:** This is the scoring surface: teach the robot, via reinforcement learning, to move the object exactly like the human.
- **What to do:**
  1. Import the reference bundle (object trajectory, hand reference, contacts) for one task into Isaac Lab.
  2. Implement the reward stack: `r_task` (object pose tracking) + `r_rel` (multi-object, if needed) + `r_imit` (imitation) + `r_cws` (contact wrench).
  3. Add the curriculum: Virtual Object Controller with decaying strength + mid-trajectory resets + stabilization window.
  4. Train with PPO (RSL-RL); report AUC/SP-SR/MP-SR/MPPE to the Scorekeeper.
- **Done when:** First trained policy + first accuracy numbers on the Scorekeeper's board.
- **Resources:**
  - CHORD paper method + Appendices (A/B rewards, obs/action): https://arxiv.org/abs/2607.00033
  - Isaac Lab: https://github.com/isaac-sim/IsaacLab · RSL-RL: https://github.com/leggedrobotics/rsl_rl
  - Training entry point: `python scripts/rsl_rl/train.py --task Sharpa-V2D-v0` inside `video_to_data/robotic_grounding`
- **Depends on:** #7, #8, #9 (reference bundle + retargeted/whole-body motion).

### Issue #11 — First end-to-end run + baseline leaderboard submission
- **Label:** `integration` · **Captain + Roles 3/7/9:** ______
- **Why:** Prove the whole pipeline works on one task and get on the board (max 5/week — start early).
- **What to do:**
  1. Chain: video → reference bundle → retarget → policy → recorded rollouts.
  2. Produce the submission artifact with the official script.
  3. Captain submits; Scorekeeper logs the returned score.
- **Done when:** A real score on the public leaderboard with the score written in our tracking sheet.
- **Resources:**
  - Challenge registration (needed for submissions): https://docs.google.com/forms/d/e/1FAIpQLSdZJYNsEPPGDeIRH2yb_Dui-lWcIxWRF2CON7UOIijzCw8zyA/viewform
  - Challenge page + submission rules: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- **Depends on:** #1–#10.

---

## Sprint 2 (W3–W5): make it great

### Issue #12 — Training-ready reference bundles for all episodes
- **Label:** `vision`+`robot` · **Assignee (Roles 4–7, owner Role 7):** ______
- **Why:** The interface between the Vision and Robot squads. One standardized file per episode (object pose, hand, contacts, wrenches, camera/world frame) that Robot squad can load without errors.
- **What to do:** Define the format once (with Role 3), then generate bundles for all public + evaluation episodes.
- **Done when:** Robot squad can train from any bundle without asking Vision questions.
- **Depends on:** #7.

### Issue #13 — Noise-aware contacts + robustness [C3]
- **Label:** `robot` · **Assignee (Role 9):** ______
- **Why:** Video contacts are noisy; a thin reward that trusts them too much trains badly. We add confidence weighting + a force-closure fallback where contact data is bad.
- **What to do:** Confidence-weighted contact-wrench reward; per-task switch CWS↔`r_fc`; matched-noise data augmentation during training; 3 seeds.
- **Done when:** Robustness ablations match CHORD's table and hold held-out AUC ≥ goal.
- **Resources:** CHORD Sec 3.2 + Appendix A (noise table: 30 mm / 30°): https://arxiv.org/html/2607.00033v2
- **Depends on:** #10, #12.

### Issue #14 — (Stretch) Residual stage + VLA experiments [C4/C5]
- **Label:** `stretch` · **Assignee (Roles 8/9):** ______
- **Why:** Insurance + the "Industry Innovation Award" angle; not the main bet.
- **What to do:** EgoAERO two-stage residual for the hairiest clips; optionally post-train VITRA/pi0 on reconstructed pseudo-actions with reliability weighting.
- **Resources:** EgoAERO (arXiv 2606.08057) · ACE-Ego-0 (arXiv 2606.17200) · VITRA.
- **Depends on:** #10, #13.

### Issue #15 — Scale-up: run the whole pipeline on all public + evaluation episodes
- **Label:** `infra`+`vision` · **Assignee (Roles 2/4/5/7):** ______
- **Why:** The final score is over the test episodes; we must process all of them reliably, not just the easy ones.
- **What to do:** Batch-render every episode through A1→A5; monitor failures; cache outputs.
- **Done when:** All public + evaluation episodes have reference bundles with quality reports.
- **Depends on:** #12.

### Issue #16 — Technical report + code release + freeze checklist
- **Label:** `integration` · **Captain (Role 1):** ______
- **Why:** Required for all awards (winner/runner-up + Industry Award). Code release + report submitted with the final artifacts.
- **What to do:** Draft report as we go; freeze checklist (submission limits, final 3 days unlimited, author list); ship code + docs.
- **Resources:** Challenge awards/timeline: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/ (Nov 4 5pm ET freeze).
- **Depends on:** all.

---

*Tip: turn on issue "dependencies" links and assign roles in the board. Keep descriptions updated as the sprint evolves — these cards are living docs.*