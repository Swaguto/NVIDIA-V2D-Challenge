# Dex3 controller and physics validation

## Scope and status

**Latest update:** the grip-force and payload-moment correction now passes two
e002 lid acquisition and floating-wrist lift diagnostics. See
[GRIP_CONTROLLER_FIX.md](GRIP_CONTROLLER_FIX.md). There are 52 passing CPU
controls. The complete episode, pot grasp and learned policy remain unaccepted.
The original source-target failures below are preserved as historical evidence.

The October 15 implementation adds explicit controller, contact and reference
validation stages. All 37 local controls pass. GPU comparisons completed on a
stoppable Brev L40S using the published runtime and hash-verified source overlays.
The candidates pass the short free-space test; all three complete e002 task runs
fail the unchanged wrist gate and object retention. The four decomposed-geometry
static grasp starts also fail unsupported retention. No controller candidate has
been promoted into the training task. No physical task or learned policy is accepted.

The subsequent development training and acquisition work is recorded in
[TRAINING_VALIDATION.md](TRAINING_VALIDATION.md). The source and runtime now have
44 passing CPU controls across training, grasp geometry and reproduction. A
1,000-iteration unassisted training pilot runs, but its controlled window success
remains zero. The new acquisition candidates also fail retention.

The saved failed e002 run loses tracking before object contact: the left wrist
first exceeds 10 cm error at source time 0.075 s, versus its first sampled object
contact at 7.09 s. The right exceeds 10 cm at 0.015 s and has no sampled positive
object contact. Both wrists' local X torques change sign ten times in the first
0.05 source seconds and grow toward the original 60 Nm component cap. This is
evidence for controller isolation, not proof of a unique cause.

## Changes under test

- Preserve the existing `project_pd` controller for comparison. Its gains and
  force/torque caps are unchanged.
- Add optional `implicit_pd`: a backward-Euler PD prediction using the actual
  world-frame root 6×6 mass block, root COM velocity and gravity compensation.
  Position error is also measured at the root COM, with explicit xyzw rotations.
  The mass block approximates locked fingers; this is not full inverse dynamics.
- Add separately selected `implicit_pd_ff`: use the next reference pose to
  predict the endpoint and damp velocity error against the reference twist.
  This addresses the moving-target lag measured in the first full-task run.
  Gains, force/torque caps and acceptance limits are unchanged. It supplies no
  external object forces and does not alter the reference trajectory.
- Add separately selected `implicit_pid_ff` for the remaining contact-load error:
  linear integral gain 50 N/(m·s), with each integral component capped at 10 N.
  Updates that push farther into force saturation are frozen, and the state is
  reset when the observation-derived collision mask changes. PD gains and the
  total 60 N / 60 Nm component limits remain unchanged. This remains a controller
  experiment; it cannot establish that the target grasp is physically feasible.
- Initialize floating wrists once after Kit startup with zero velocity. Both
  comparison controllers use this initialization. No wrist pose is teleported
  during execution, and objects receive no motion playback or external forces.
- Free-space tests disable hand collisions and apply a declared 1 cm / 5° initial
  perturbation to test recovery. Full-task tests retain the original display-only
  collision masks and inferred support, without that diagnostic perturbation.
- Record the candidate/executor hashes, root mass eigenvalues, velocities, wrench
  frame, saturation and source provenance. Failures remain visible in new outputs.
  New runs also check and record every actual USD hand collider flag at each mask
  transition; a requested mask without matching readback aborts execution.

Controller-only limits, fixed before the GPU comparison, are p95 position error
≤2 cm, maximum ≤10 cm, p95 orientation error ≤10° and maximum ≤45°, for **each**
hand. No startup interval is excluded. Gravity/contact controls and complete
requested physics-step coverage must also pass. These are diagnostic limits,
not official score thresholds or a physical grasp certificate.

## Measured GPU results

The completed comparison uses 0.01 s physics steps, the original PD gains/caps,
and the original inferred e002 reference. Each full run covers 343 source frames
and 3,420 physics steps, with actual hand/object readback and decoded video checks.
The separate bounded-integral candidate adds the declared integral gain above.

| Full-task controller | Left position p95 | Right position p95 | First pot grasp: sampled positive contacts | Wrist gate |
| --- | ---: | ---: | ---: | --- |
| `implicit_pd` | 7.65 cm | 7.64 cm | 184/185 | Fail |
| `implicit_pd_ff` | 6.38 cm | 1.28 cm | 184/185 | Fail |
| `implicit_pid_ff` | 4.16 cm | 1.30 cm | 98/185 | Fail |

The original project controller fails the perturbed free-space test with left /
right position p95 errors of 1.414 / 1.058 m. All three candidates pass that same
41-frame prefix. The prefix mainly tests recovery from a fixed reference; it
does not establish tracking across all moving portions of the episode.

Feedforward removes much of the right wrist's motion lag. In that full run the
left wrist's position p95 is 1.83 cm without measured object contact and 6.81 cm
during contact (contact split threshold 0.001 N). This motivates separating
contact-load error from free motion. Integral compensation improves the left
position statistic but reduces sampled pot-contact coverage, so the numerical
improvement does not justify promoting it.

Every full run records zero sampled positive contacts in both right-lid grasp
cores (0/62 and 0/72), and zero in the final left-pot core (0/62). The new USD
readback confirms actual hand collider enablement in these eligible intervals.
Contact alone is not force closure. Both objects eventually leave the finite
inferred support in these runs; none reproduces the complete source task.

### Static geometry isolation

The new fixed-wrist repeat captures **32 cooked convexes per object** and verifies
all 404 grasp records, 202 control records and 21 decoded video frames. Both
gravity/contact controls pass. All four automatically selected inferred grasp
starts fail the unchanged unsupported one-second / 10 mm retention test:

| Sample | Final displacement | Maximum net normal contact force |
| --- | ---: | ---: |
| First left pot start, source frame 47 | 4.067 m | 110.75 N |
| Second left pot start, source frame 273 | 4.769 m | 19.40 N |
| First right lid start, source frame 47 | 4.936 m | 0 N |
| Second right lid start, source frame 236 | 4.934 m | 0 N |

These four fixtures omit the table, other object, other hand and squeeze preload.
The starts are inferred first-touch poses. Their failure isolates an unresolved
grasp/load problem; it does not prove that a supported source interaction is
impossible. The older convex-hull measurements remain separate historical evidence.

### Evidence and next gate

The [receipt directory](receipts/physics/brev-20261015/) preserves every comparison
gate, report, source-deployment manifest, decoded-frame receipt and full-task
contact audit, including failures. The first GPU attempt's empty-observation
summary error is preserved separately; its partial output was not accepted.
Raw states/videos/SDK logs are privately backed up with per-file SHA256 inventories.

Next work is contact-aware retargeting and a closed-loop manipulation controller:
check body/mesh frames and initial settling, find feasible Dex3 pad contacts,
establish supported grasp acquisition and object retention, then repeat a complete
episode with the same observation masks. Source grasp annotations remain validation
inputs. A fixed wrist or extra wrist force cannot substitute for this gate.
Task registration, confidence-aware training consumption and evaluator checkpoint
compatibility remain required before a learned-policy pilot.

## Run with the published runtime

Follow the [container setup](../../../docker/dex3-review/README.md) first. Use
the digest-pinned registry environment and a fresh `V2D_OUTPUT_DIR`. The overlay
mounts the checked-out candidate code read-only; it does not rebuild or relabel
the published image. The producing source checkout and image digest must both
be recorded for the experiment.

The host must supply an NVIDIA driver compatible with CUDA 13 (580 or newer).
Some Brev VM templates arrive with older drivers even when the GPU is an L40S.
Each physics stage checks the actual host driver and executes a CUDA tensor
calculation in the simulator environment before launching Isaac. An RTX render
is still required after that compute preflight.

Obtain the separate, verified physics inputs:

```bash
gh release download dex3-support-replay-20261013 \
  --repo Swaguto/NVIDIA-V2D-Challenge \
  --pattern CAD-support-dex3-simulator-inputs-v2.tar.gz --dir downloads
printf '\nV2D_CODE_DIR=%s\nV2D_PHYSICS_ARCHIVE=%s\n' \
  "$PWD" "$PWD/downloads/CAD-support-dex3-simulator-inputs-v2.tar.gz" \
  >> docker/dex3-review/.env
```

The input archive SHA256 is
`759b6e9fa642b47ec24306969688cc1ce15ead4cb0b80e4f8d086b727b93c680`.
Every one of the 1,663 case files is verified before execution. The source motion
evaluator and independent source grasp labels are restored from their committed,
hash-bound evidence archive.

```bash
# Separate invocations preserve each result, including an expected baseline failure.
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml -f docker/dex3-review/physics.compose.yaml \
  run --rm physics --stage baseline-free
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml -f docker/dex3-review/physics.compose.yaml \
  run --rm physics --stage candidate-free
docker compose --env-file docker/dex3-review/.env \
  -f docker/dex3-review/compose.yaml -f docker/dex3-review/physics.compose.yaml \
  run --rm physics --stage candidate-full
```

The two free-space tests cover the first two source seconds at the original
0.01 s physics timestep and 0.5 motion speed. Full-task execution is refused
unless the saved free-space gate still passes with the same candidate/executor
bytes and selected controller mode. To compare the velocity-feedforward candidate,
add `--candidate-controller implicit_pd_ff` to **both** candidate invocations in a
fresh output root. Use `implicit_pid_ff` for the bounded integral candidate.
Each stage verifies actual video decoding and frame count. The full task
also runs the independent source observation/contact audit before its wrist gate.
Source annotations are evaluation inputs only; they do not fit the controller.

Add `--prepare-only` for input checks in a separate fresh output root. Existing
stage directories are never overwritten. A gate failure returns nonzero and
preserves its video, states and report. Passing wrist tracking does not approve
object retention, inferred targets, collision geometry or policy performance.

## Team reproduction and reference handoff

### Separate unsupported grasp-load check

In a fresh output root, run the same Compose command with `--stage static-load`.
This selects four existing inferred grasp starts automatically, fixes the wrists
and runs each object unsupported for one second under gravity. The test captures
the actual cooked object convexes, verifies all 404 grasp / 202 control records
and 21 video frames, and records the original 10 mm retention criterion.
It supplies no table, other object, squeeze preload or learned residual. A
first-touch failure does not prove that the source manipulation is impossible.
This repeat addresses the earlier static test's documented convex-hull geometry
qualification. It does not replace the full dynamic task or approve its targets.
The wrapper preserves the original MPEG-4 physics video and creates a separate
H.264 browser derivative, checking all 21 frame timestamps and binding both
hashes in `browser-preview.json`. Historical static outputs can use
`package_static_preview.py --run /absolute/path/to/static-load/run/simulator`.
The initial static video image can retain the importer rest display; numerical
initial-pose readback establishes the starting state. Later images follow physics.

### Replay verification and diagnostic references

After the frozen-target container pilots, a teammate can check their deliveries:

```bash
python3 scripts/dex3_review/verify_reproduction.py \
  --inputs /absolute/path/to/dex3-review \
  --outputs /absolute/path/to/review-001 --episodes 11,40,41 \
  --out /absolute/path/to/team-reproduction-check.json
```

This checks reconstruction/IK hashes, exact renderer source, original numerical
tolerances, observation provenance, all four videos and their frame/PTS receipts.
It does not require pixel identity across GPUs. Full manual playback remains a
separate review.

`scripts/dex3_physics/export_reference.py` converts the preserved target contract
to **diagnostic** `motion_v1` dual-hand Parquet and a hash-bound uncertainty sidecar.
It preserves frame indices, gaps, observations and unknown contacts/wrenches;
xyzw target poses are explicitly converted to the schema's wxyz convention.
It does not invent MANO contact geometry or mark a reference training-ready.

The current dual-hand trainer still reads the legacy `ManoSharpaData` contract;
its robot registry includes Sharpa and whole-body G1, not floating Dex3.
Registering the task, integrating confidence masks/contact geometry and proving
checkpoint compatibility are required after the physical controller gate.
Do not run the existing Sharpa command against a file labelled Dex3.

The offline `submission_readiness.py` command reads the supplied official kit's
roster and inventories per-episode checkpoints, object meshes and rollout files.
It distinguishes development previews from the scored roster. Presence alone
does not establish checkpoint compatibility or replace the official packer.
