# Full-roster Track 3 baseline — 18 October 2026

The **complete local baseline passed the untouched official packer**. Its
physical quality is not accepted: all 20 selected short policies have local
MP-SR zero against inferred references. No hidden leaderboard score is known,
and nothing has been submitted.

## Delivered coverage

| Item | Verified result |
| --- | --- |
| Scored video inputs | 20 episodes, 60 videos, 5,181 source frames |
| Own reconstructed mesh slots | 36 / 36 with provenance and file hashes |
| Complete short policies | 20 / 20, 100 PPO iterations each |
| Selected policy training | 153,600 environment steps per episode, seed 42 |
| Official combined artifact | 9,120 rows, 3,727,302 bytes, 36 mesh payloads |
| Private replay packs | 20 / 20, 383,501,755 bytes total; kept private |
| Physically accepted tasks | 0 / 20 |

The packer's local receipt was verified at 14:26:04 UTC. The artifact SHA256 is
`d9c0b20d03ddc7f37c0ae0959ebcad48f1f0e1dc1ad3e4a72d092a09f4ca21de`;
its recorded code commit is
`a4510fb4e1736008e9090a8535a4b365b2c0e6fb`.
Input poses or scores were not repaired to satisfy packing. This is a valid
baseline artifact for captain review, not an approved competitive release.

See the [slot inventory](scored-coverage-progress-20261018.json) and
[per-episode metrics and audit hashes](full-roster-baseline-result-20261018.json).
The local review gallery includes source videos and recorded unassisted
Newton/MuJoCo physics for all 20 cases. It is an offline collision-surface replay,
not an Isaac Sim render; eight worlds were evaluated and world zero is shown.

## What is failing

A native CPU audit reset both robot hands and every object to their exact
reference poses at **every source knot**. It independently checked robot FK and
the frozen file closures. Every episode has hand/object or hand/support
penetration above 1 mm at some knots: **5,761 of 10,362 hand/frame slots** are
affected. Maximum per-episode penetration ranges from approximately 10.5 to
45.4 mm. These measurements do not assess contact forces or force closure.

This establishes reference inconsistency, not a proof that every policy failure
has one cause. Missing metric hand depth, robot/human morphology differences,
shape/pose errors and retargeting all remain relevant. e020's object-relative
gap proposal reduces a large hand/object separation but increases intersecting
frames. Its local contact-projection attempts also leave unresolved contacts
and introduce velocity violations. Those proposals were not promoted or trained.

A subsequent source-constrained articulated e037 candidate clears **27 → 0**
hand/frame penetration failures across all 151 knots. Independent source
projection, full native FK and actual URDF finger position/velocity checks pass.
The full task remains blocked by **78 object/support penetration knots**; no new
policy or score improvement is claimed. The original selected baseline is
preserved. [Native CPU repair and limitations](NATIVE_CPU_CONTACT_REPAIR_20261018.md).

The independently frozen reserved baseline was run on e014/e032/e036 using the
previously recorded method and source/image pins. All three MP-SR results are
zero; some objects are lost badly in the unassisted rollout. Those results
reject a claim that this baseline already generalizes successfully. They were
not used to retune the reserved test and then claim held-out improvement.

The e002 checkpoint-2000 control remains retained. The two-stream controlled
reset/reward investigations did not establish a repeatable improvement beyond
evaluation variability. Higher training reward alone did not qualify a change.

## Reproduction status

- [Frozen policy replay](SCORED_POLICY_REPLAY.md) verifies the input/checkpoint
  closure and re-runs full unassisted evaluation without training. A three-case
  fresh-container rehearsal on e003/e015/e038 completed and all contracts passed.
  It uses relocated
  private packs on the existing host; a fresh container is not an independent
  teammate or fresh-host reproduction.
  [Replay comparisons](scored-replay-rehearsal-20261018.json) show different
  numerical scores: e003 AUC 0.1478 → 0.1818, e015 0.54452 → 0.54417,
  e038 0.21890 → 0.22119. MP-SR remains zero in each case. Identical scores are
  not established; the cause of the replay variability is not isolated.
  An optional [passive startup trace](REPLAY_STARTUP_TRACE_20261018.md) now
  captures the previously missing reset/capture/first-interval buffers in the
  separate diagnostic tool. Two CPU tests pass; GPU integration remains
  unverified and no completed replay was rerun.
- [Fresh roster plans](SCORED_ROSTER_PIPELINE.md) generate ordered hand,
  reconstruction and policy configurations with pinned code runners, images and
  private paths. The uniform baseline does not silently substitute the selected
  e003 geometry or e015 seed experiment.
- Existing replay packs include the selected frozen inputs and learner
  checkpoints, but exclude footage, MANO/inference weights and credentials.
  They remain local/private at the owner's request.
- An external teammate's clean setup, private package/input access, result
  comparison and shutdown receipt remain to be verified. No teammate was
  messaged and no private input pack was published.

## Next quality gate

Repair object initialization and contact-aware retargeting on a non-reserved
single-object task. Require source-frame fit, native full-reference contacts,
joint limits, velocity limits and full unassisted rollouts before extending PPO.
New roster plans enable a 1 mm native reference-penetration gate before PPO.
It was checked against 14 actual paired full-audit archives; all 14 correctly
block training. An explicit diagnostic-baseline option preserves intentional
baseline experiments. [Spoon source-fit checks](SPOON_SOURCE_FIT_20261018.md)
improve sampled geometry fit on three cases but do not yet repair a complete
contact-consistent trajectory.
Seed-only geometry fit improvements are candidates until they survive other
source frames and physics. Keep the packed baseline and every failed control
unchanged during this investigation.

Each GPU session has a deadline, private hash-verified backups and confirmed
provider shutdown. The latest replay's 190 retained files were verified and
Brev was confirmed STOPPED at 14:32:02 UTC. Brev must stop when idle.
Later guarded depth-probe launch attempts never reached RUNNING; the stop
preflight still returns an invalid-security-token error after a fresh CLI login. Twelve
subsequent checks confirmed STOPPED; no GPU depth inference or training occurred.
[Native contact repair now runs on the Mac CPU](NATIVE_CPU_CONTACT_REPAIR_20261018.md).
Its static control reproduces all 151 archived e037 knots. The articulated
hand-only stage passes its source/contact/motion checks, while the full task
still fails object/support contact requirements.

The separate [CPU stereo experiment](FAST_STEREO_CPU_20261018.md) has now
completed and independently verified all **151 e037 depth frames** with the
source clock and frozen inputs intact. It does not establish ground-truth depth
accuracy. The complete object fit is rejected for source/contact/motion failures.
A separate [source-derived camera/reference repair](E037_FULL_REFERENCE_REPAIR_20261018.md)
now passes all 151 object source fits, unchanged original motion caps, fresh
full native object/hand contacts, source hand projections and URDF motion
limits. A further between-frame clipping failure was repaired: the latest
independent discrete sweep clears all 3,805 samples. This is an inferred
reference candidate for a physics trial, not a grasp or score improvement.
The new geometry has not been used for PPO. Normal Brev login/listing now
work, but provider stop commands still fail; the VM was independently confirmed
STOPPED at 20:39:46 UTC. No new paid work was started.
The earlier development record is
preserved in [the history](SCORED_BASELINE_HISTORY_20261018.md).
