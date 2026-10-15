# Dex3 development training and grasp acquisition

## Status

The training/evaluation plumbing is implemented and GPU tested. **Physical grasp
retention remains unsolved.** No candidate or learned checkpoint is promoted as
a working task solution. These runs cover e002 development data and do not
establish hidden-episode performance or organizer acceptance of the custom schema.

## Implementation

- Dual floating Dex3 hands with seven finger joints and three digits each.
- Pinned official FlashCHORD runtime, Newton/MuJoCo-Warp physics and upstream PPO.
- Source confidence masks preserved; the pilot selects consecutive observed rows.
- Gravity, hand/object and object/object collisions, and finite inferred support.
- Zero virtual object assistance and zero assisted settling steps.
- Unknown human contact/wrench labels do not become measured Dex3 labels.
  Objectives requiring those labels are rejected. The exploration experiment's
  force regularizer uses actual simulated contacts.
- Source-hashed runtime extensions, safetensors metadata, initial checkpoint,
  every-iteration local metrics and actual scene physics readback.
- Private, digest-pinned Docker image and local frozen inputs.

The extension is required to read this dual-hand schema. The unmodified official
`scripts/evaluate_policy.py` entry point loads the checkpoints **inside the
extension image**. An unmodified stock image cannot load this custom schema.
This does not establish compatibility with every embodiment in the scored roster.

## Controlled training result

The longest eligible interval is source frames 47–121: 75 frames at 20 Hz out of
343. The original missing/confidence rows remain in a sidecar. Mass/inertia are
nominal and gravity orientation comes from inferred support, not measurements.

| Policy | Environment steps | Window success | Lid position error, mean | Mean ADD |
| --- | ---: | ---: | ---: | ---: |
| Initial | 0 | 0% | 18.51 cm | 9.94 cm |
| Learned, iteration 1000 | 1,536,000 | 0% | 18.19 cm | 10.18 cm |

Each policy was evaluated with four worlds over all 75 frames. Identical saved
experiment configuration, frozen collision files and image context were verified.
The training reward changed while task success did not improve; reward alone is
not an acceptance criterion. The lid still slips, and wrist failure also remains.
Full metrics and SHA-256 receipts are in `training-policy-comparison-v1.json`.

An earlier 100-iteration comparison used independently generated CoACD caches.
Their pot hulls differed despite identical cache keys. Those preliminary numbers
are not a controlled learning comparison. The loader now refuses unfrozen caches;
the 1,000-iteration run was trained from scratch against the frozen pack.
Frozen geometry permits a controlled comparison but does not promise bit-exact
stochastic training on other hardware or across all GPU scheduling conditions.

## Physical grasp tests

The geometric search checks exposed opposing surfaces, URDF FK against actual
SDK link poses, bounded wrist/finger changes, joint margins and cooked convex
penetration. These are candidate checks, not force-closure certificates.

The paired acquisition stage compares two source grasps with two proposals. It
closes fingers for 0.25 seconds on a temporary tray, settles to 0.5 seconds, then
disables and hides every tray and tests one unsupported second. Independent
free-fall, permanent-support and released-support controls pass. No object pose
playback or external object wrench is used after initialization.

All three acquisition attempts fail the unchanged 10 mm retention limit. The
older joint-limit candidate briefly retained one lid with 45 mm displacement
and a very large contact force; it is rejected. Candidates with interior joint
margins avoid that artifact but drop the lid. Aligning the tray to actual cooked
geometry also fails. See `grasp-acquisition-summary-v1.json` for all outcomes.

The tray, fixed wrist and nominal object mass are diagnostic fixtures. They cannot
be substituted for the evaluation scene. Unsupported starts, acquisition and
complete motion require separate validation.

The new interior-joint proposals also fail when initialized directly in their
closed pose: the two lids move 4.84 m during the unsupported second. Acquisition
timing alone therefore does not explain their failure to support the nominal load.

## Reproduction and next gate

Follow `docker/dex3-training/README.md` on Linux amd64 with an NVIDIA GPU. The
tested image and hashes are in `training-container-verification-v1.json`.
The episode pack and checkpoints remain local; manually transfer them through
an approved private channel. The repository/image alone does not supply inputs.
The local bundle receipt and exact saved-policy replay commands are in
[LOCAL_TRAINING_REPRODUCTION.md](LOCAL_TRAINING_REPRODUCTION.md).

Before extending training to more episodes, establish a supported close followed
by unsupported retention with bounded forces, then validate lift and release in
the full scene. The existing source-oriented geometric seeds have failed that
gate. A successful physical acquisition/contact strategy is still required;
additional PPO iterations alone have not repaired it.
