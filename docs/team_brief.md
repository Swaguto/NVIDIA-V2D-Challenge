# Team Brief — V2D Track 3 (simple version)

*For the whole team. One page of "what are we doing," then who does what. This is the plain-language companion to `plan.md` (the technical one).*

---

## 1. What are we doing? (elevator pitch)

A human puts on a **head-mounted camera rig** (like GoPro glasses) and does everyday kitchen tasks in front of a table: picking up a pot and its lid, stacking cups, putting a pan in a dish rack, filling a watering can, etc.

**We have to teach a humanoid robot to repeat those same tasks** — using **only the camera video**. No sensors on the human, no special suits. Just video.

The robot (called **G1 + Dex3**) is a humanoid with two three-fingered dexterous hands, and it must be trained in a physics simulator (Isaac Lab).

**Winning = the robot imitates the demonstrated motion of the objects most accurately.** The judges score us on "how exactly does the robot move the mug/pan/pot compared to the human in the video," plus the quality of our 3D reconstruction.

---

## 2. Why is this hard? Why can we win?

Think of it in two halves:

1. **Watch half (Vision):** turn the video into accurate 3D data — where's the object, where are the hands, when do they touch, what is the object's shape.
2. **Teach half (Robot):** turn that 3D data into robot training so the robot can physically reproduce the motion.

The people who made this challenge (NVIDIA, "CHORD" = their robot-teaching engine) are excellent at half 2, but their **half 1 is weak**: they used only *one* eye of the camera and *guessed* distances. Result: on real video their end-to-end score is only **≈ 65%**, while on perfect lab data they hit ≈ 92%.

**Our edge:** we get a real **stereo camera** (two eyes side-by-side, 75 mm apart) → true distances. And NVIDIA gives us the **exact 3D models of every object** (mug, pot, pan) → we can track them precisely instead of guessing.

> **In one line: the official engine (CHORD) is stuck at ~65% on video because of bad depth. Our team fixes that and unlocks ~90%.**

---

## 3. Our pipeline in plain words

```
 VIDEO IN
   (3 cameras on the head)
       │
 [1] SEE  → working out camera motion + true distances   (Vision team)
 [2] SPOT → finding & tracking the object every frame    (Vision team)
 [3] HANDS→ finding the hands & when they touch the object (Vision team)
 [4] CLEAN→ merging everything into one smooth 3D movie   (Vision team)
       │
 [5] REWRITE → translating human hand motion to robot hands (Robot team)
 [6] TRAIN → teaching the robot to repeat the task via CHORD (Robot team)
       │
 [7] SCORE → checking our robot against the official scoring (Scorekeeper)
       │
 SUBMIT to the leaderboard (Captain) → WIN
```

---

## 4. Team roster (9 roles)

Fill in names. Two squads + support.

### VISION SQUAD — "turns video into data" (people 4–7)
### ROBOT SQUAD — "turns data into robot skill" (people 8–9)
### SUPPORT — Captain, Cloud, Scorekeeper (people 1–3)

| # | Role | Name | One-line job | Main deliverable by end of Week 2 |
|---|------|------|--------------|-----------------------------------|
| 1 | **Captain / Submissions** | ______ | Keeps us on schedule, runs the official leaderboard submissions (max 5/week), assembles the final technical report | Submission plan + team schedule; first baseline submission in Week 3 |
| 2 | **Cloud & Build (Infra)** | ______ | Sets up GPU servers, Docker, the shared dataset, and Isaac Lab; every teammate's code runs on this machine | GPU box is up; Isaac Lab + CHORD runs a test task; everyone can log in |
| 3 | **Scorekeeper (Eval & Metrics)** | ______ | Decodes the official scoring script and builds our own local leaderboard so every change shows a number fast | Local leaderboard on 4 held-out episodes + a one-page "how scoring works" |
| 4 | **Camera & Scale (Stereo)** | ______ | Turns the side-by-side cameras into true distances + camera path; solves the one-time camera-to-world calibration | A depth + motion demo on one episode; calibration numbers (tables at cm-level accuracy) |
| 5 | **Object Tracker** | ______ | Finds and tracks the mug/pot/pan every frame using its given 3D model; checks against the ground-truth data we were given | Tracking error report for all 20 practice episodes (vs ground truth) |
| 6 | **Hand Tracker** | ______ | Finds the hands + fingers every frame, and detects touches ("hand meets object"); the hardest, most occluded job | Hand overlay + touch ("contact") annotations on 3 episodes |
| 7 | **Refiner & Validator** | ______ | Merges everyone's per-frame numbers into one smooth 3D movie; removes jitter and drift; confirms quality | One episode fully refined in a 3D viewer + a jitter fix report |
| 8 | **Body Translator (Retargeting + Whole body)** | ______ | "Rewrites" human hand/finger motion into robot hand motion (G1+Dex3) via inverse kinematics; fills in the rest of the robot body; rejects impossible moves | A video of the robot mimicking a human on 1–2 tasks in the simulator |
| 9 | **Robot Teacher (RL Training)** | ______ | Runs the CHORD reinforcement-learning engine; tunes rewards so the robot actually completes the task; makes the final recorded rollouts | First trained policy on 1 task + first accuracy score for the Scorekeeper |

---

## 5. What each role feeds the next (the "handoffs")

```
Object Tracker (5) ──► Refiner (7) ──► Body Translator (8) ──► Robot Teacher (9) ──► Scorekeeper (3) ──► Captain submits (1)
Hand Tracker (6) ────┘                    (whole body) ────────────────┘
Camera & Scale (4) ─► serves 5, 6, 7 (true distances and camera path for everyone)
Cloud & Build (2) ───► runs everything; Infra for all
```

**Rule of thumb for everyone:** your teammate's output must be *usable, not perfect*. Ship working data early and often (even slightly rough) so the next person isn't blocked. The pipeline can only be as good as its weakest link — and we fix the weak link by talking, not by perfecting alone.

---

## 6. Simple timeline (to the Nov 4 freeze)

| Week | Focus | Milestone |
|------|-------|-----------|
| W1 (now–Sep 28) | Stand-up | Everyone has logins + data + a tiny first deliverable; names in the roster |
| W2 (Sep 28–Oct 5) | First data flows | Vision squad produces first tracked references; Robot squad gets CHORD training on 1 task; Scorekeeper's leaderboard online |
| W3 (Oct 5–12) | End-to-end v1 | Full pipeline runs on one task; first baseline submission to the leaderboard |
| W4–5 (Oct 12–26) | Make it great | Iterate: better vision → better contacts → better training; 2–3 submissions/week; everyone checks their change on Scorekeeper's board |
| W5.5 (Oct 26–Nov 4) | Lock it in | Unlimited submissions in the last 3 days; pick best artifacts; write the technical report; **freeze Nov 4, 5 p.m. ET** |

---

## 7. Mini glossary (plain definitions)

- **AUC / SP-SR / MP-SR / MPPE** — the four scoring numbers the judges use. Don't memorize them; just know the Scorekeeper (3) tracks them for every change. Roughly: they measure "how precisely does the robot move the object like the human did."
- **Stereo pair** — two cameras side by side; your brain-style depth vision. This is our unfair advantage.
- **CAD / URDF / 3D model** — the exact digital twin of each object (mug, pot, pan). NVIDIA gave it to us, so we don't have to guess what the object looks like.
- **MANO** — the standard computer model of a human hand (bones, joints, fingertips). Used to describe hands in 3D.
- **IK (inverse kinematics)** — math that figures out which robot joints to bend to place the hand where we want.
- **Isaac Lab** — the physics simulator where the robot trains and where the final evaluation happens.
- **CHORD** — NVIDIA's robot-teaching method (reinforcement-learning + contact guidance). It is our engine, and we make it better by feeding it much better data.
- **MoCap (motion capture)** — the ground-truth measurements of the objects that NVIDIA recorded with special markers. We are given MoCap truth for 20 *practice* episodes to validate our work; the test episodes keep it hidden.

---

*Companion docs: `plan.md` (full technical plan) · `~/v2d_track3/` (repo + data) · `eval` details from Scorekeeper.*