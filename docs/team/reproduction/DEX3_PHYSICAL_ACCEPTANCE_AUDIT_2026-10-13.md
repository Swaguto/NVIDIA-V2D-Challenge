# Dex3 development task: completion and physical-accuracy audit

**Later scope correction:** the hand clearance certificates below cover the
URDF collision meshes, not every rendered visual triangle. The unchanged
replay has visual-surface intersections missed by those collision meshes.
Further cosmetic geometry work is deferred at the user's direction because
Track 3 has no separate hand penetration or visibility score. See the
[current scoring scope and certificate qualification](SCORING_SCOPE_2026-10-13.md).
The earlier frozen evidence is retained; physical acceptance remains false.

The frozen e002 reconstruction and actual Isaac replay are reproducible, but
the requested physically credible development task is **not yet established**.
This audit completes the source comparison inspection: all 343 chronological
source/observation/actual-Isaac rows on 58 sheets were inspected. It adds no
fitted pose, source observation, timing label, geometry change or simulator run.

The experiment reference remains `dex3-support-replay-20261013`, commit
`85cc0d01c8e93b20c620c354251e29290f43efd4`. Its published method, trajectory,
proofs, videos and nine artifact hashes are unchanged. The new
[complete source-review receipt](dex3-motion-replay/support-full-source-comparison-review-v2.json)
records this later inspection separately from the original frozen review.

The later [public object accuracy check](PUBLIC_OBJECT_ACCURACY_2026-10-13.md)
supersedes the earlier claim that an independent object reference was unavailable.
Organizer public MoCap allows a scan-frame relative-pose comparison: observed
median translation error is 6.774 mm and orientation error 3.582 degrees.
The largest 62.936 mm discrepancy overlaps exact repeated reference poses and
needs source/timing diagnosis. All reference rows remain included. Hand/support
measurements and physical acceptance are unchanged.

The [current final-fit contact audit](FUNCTIONAL_DEX3_CONTACT_AUDIT_2026-10-13.md)
now measures the actual CAD-selected finger faces. All 377 closed states have
two opposing pad witnesses within 2 mm; the same six of 381 source-labeled
grasp states are missed. Those six are false negatives, not unsupported source
labels. Four have no recovered hand pose; left 272 and right 309 have observed
poses but lack the inferred robot contact. The table below's older wording
"unsupported" concerns the robot proxy, not uncertainty in those positive
source grasp labels. No robot motion or acceptance flag changes.

## Requirement-by-requirement result

| Goal requirement | Evidence | Disposition |
| --- | --- | --- |
| Retarget PR #35 after its parent merge | GitHub PR #35 base is `main`, head `ae8fa33f8f48a005cb22726ee96686fa30dda20b` | Complete |
| Review the newer reproduction branch separately | [Baseline review](DEX3_BASELINE_REVIEW_2026-10-13.md) names PR #37 head `bc1a60132f30f7307ade59623d8e035ef0e20e4f`, base `recent-reproducibility-code`, and its independent recovery/export boundary | Technical self-review complete; no human approval or merge claimed |
| Preserve observed versus display-held provenance | Exact source valid masks remain 272 left / 284 right. Prefix hold masks name frames 0–45 for each hand; inference kind is 27 there. All 686 motion-target flags are false | Complete for provenance; display continuity adds no accuracy |
| Reproduce the development reconstruction | [15-stage receipt](dex3-motion-replay/support-reconstruction-receipt-v3.json), locked kit/input hashes, actual fresh Linux execution and negative controls | Complete at the documented fitted-input boundary; raw inference/searches are not repeated |
| Show actual Dex3 in Isaac without artificial hand on/off flicker | Published 343-frame Isaac video, readback, independent A40/Windows audits and all-frame visual inspection | Complete for this sequence; cropping, occlusion and posture changes are separate |
| Check and fix hand placement against source | All 343 comparison rows inspected, prior full-resolution contact/free-motion diagnostics, bounded CAD fits and free-path repairs | Checks and model repairs complete; source hand/finger accuracy remains unproven and mixed |
| Check object tracking against source | Calibration/image guards retain original tracks. Later organizer MoCap relative check: observed median 6.774 mm / 3.582 degrees; all 343 records repeated on Linux and Windows. Largest discrepancy overlaps repeated reference values | Independently checked relative scan poses; absolute placement and support height remain unverified |
| Check contact timing against source | Read-only source annotation supports 375/381 labeled core states; left 46,272,332,333 and right 46,309 remain unsupported | Checked; six states remain unaccepted, labels never drive fitting |
| Check and fix penetration | All 3,955 segments pass original CAD/estimated-sheet and declared material/self/cross checks; 377 closed anchors retained. An 8 mm hypothetical sheet shift causes crossings at six of seven automatically selected witnesses | Complete for the original models; robustness to recorded support-height uncertainty fails at selected witnesses |
| Provide reviewable methods and documentation | [Step-by-step guide](SUPPORT_AWARE_DEX3_REPLAY_2026-10-13.md), locked release inputs/methods/evidence, versioned receipts and PR #38 | Complete for fitted reconstruction/replay; no all-episode or private-evaluation guarantee |
| Reuse GitHub code without parallel active implementations | Existing CAD reproducer extended, shared extraction/hash/runtime functions and existing Isaac wrapper reused | Complete; versioned experiment snapshots are evidence, not a second live entry point |
| Turn off unused compute; defer Issue #9 | Owned A40 `dgbgc1vu123kll` stopped at 23:42:21 UTC October 13 and independently rechecked `EXITED` at 00:07:38 UTC October 14 | Complete; Issue #9 and training have not started |

PR #35 CPU/reproduction/docs/pre-commit checks pass; its GPU Build & Test was
**cancelled**, not passed. PR #37 controls pass. PR #38 had no automatic checks
at the frozen experiment head. All three PRs remain drafts with zero human
reviews at the recorded publication audit. Human review/merge is outstanding;
this agent's technical review does not substitute for a teammate's approval.

## What the complete source inspection shows

The unobserved opening does not reproduce the visible human approach. Both
robot hands are held at a later inferred pair for frames 0–45. That is explicitly
display-only, even though the hands appear near the objects. Frame 46 is also
unobserved and inferred. A viewer must not treat that opening as contact evidence.

After observations begin, the broad pot-handle grasp, lid lift, lateral placement,
pot movement, returns and releases follow the visible source sequence. Dex3 and
human morphology differ, and the free right robot hand is more open than the
human hand through parts of the first withdrawal and final release. The left
free-hand posture and rapid orientation change near frame 267 also remain
unverified. Camera-edge clipping affects right-hand interpretation.

Invalid observations have no recovered-hand overlay. Missing interior left
observations are 126,128,140,141,143,144,166,214,225,226,237,265,266; the missing
interior right observation is 122. Both source masks are invalid at 0–46 and
332–342 although parts of the real hands are visible. The inferred tail motion
does not become a recovered observation. These findings support retaining all
target-approval flags as false; they do not supply measured corrections.

This was qualitative inspection of the sheets. It did not independently measure
every source pixel, finger identity, occluded fingertip or 3D pose. Selected
full-resolution diagnostics supplement the sheets; nearest distance to the
estimated MANO mesh is not independent hand ground truth.

## Why physical acceptance remains open

The combined CAD/estimated-sheet minimum separation bound is **0.000188 mm**.
Filtering the stored proof to the sheet alone gives a **0.010711 mm** minimum
lower bound; the recorded support-height discrepancy is **8–10 mm**. The sheet is
finite, inferred and has no measured thickness. A positive model bound cannot
establish physical clearance against that uncertainty. The unsigned CAD checks
and declared 2 mm material-depth limit also have narrower scope than true solid
penetration. The source-depth residuals do not identify a unique corrective
object or support transform; previously tried calibration alternatives failed
held-out image guards.

The later [support-height sensitivity diagnostic](dex3-motion-replay/support-height-uncertainty-probe-v1.json)
reuses the frozen motion and existing CAD motion/query/control modules. It first
verifies every case input, the original proof and sheet hashes. It selects the
closest actual sampled sheet distance in each left/right, closed/free and
observation-bracket class, yielding seven witnesses. It queries the unchanged
original mesh with sheet translations of 0,−10,−8,+8,+10 mm along the plane
normal. Every zero-shift witness agrees with the original proof to 1e-9 m and
has no surface intersection.

Six witnesses intersect under an 8 mm shift. Observed closed left bracket81–82
intersects the palm at+8 mm; observed closed right bracket106–107 intersects
thumb links at+8 mm. Observed left free bracket231–232 also intersects at+8 mm.
The original continuous proof has3,372 of4,698 intervals lacking the sufficient
10 mm translation-ball clearance bound. That bound refusal alone is not a
collision; the explicit shifted-witness queries are the concrete model
counterexamples. This is finite sensitivity evidence, not a new continuous
proof, a measured support correction or proof that the real table intersects.

To repeat this additional diagnostic, use the frozen fitted reconstruction and
environment from the replay guide, plus the
[recorded diagnostic source](dex3-motion-replay/support-height-uncertainty-probe-v1.py)
at this documentation revision:

```sh
/data/v2d/geometry-runtime/bin/python docs/team/reproduction/dex3-motion-replay/support-height-uncertainty-probe-v1.py \
  --case /data/v2d/support-reconstruction/15-case \
  --methods /data/v2d/support-method-kit/methods \
  --sheet /data/v2d/support-method-kit/sheet \
  --proof /data/v2d/support-inputs/case/validation/original-CAD-continuous/report.json \
  --out /data/v2d/new-support-height-probe
```

The diagnostic source hash is
`a00d02f638d293249b97a3b1f3915a8cd4306232694b632d315f25848d94ec4c`.
Compare input/module hashes, seven witness selections, six shifted crossing
witnesses,3,372 refused interval bounds, links and numerical distances. The
elapsed-time field varies. This validation snapshot neither generates motion
nor adds a second active reproduction pipeline.
The documented command was also run against the clean 15-stage reconstruction
and extracted simulator proof. Its selections, crossings and numerical outputs
match the original diagnostic exactly; only source formatting and elapsed-time
metadata differ from the first local probe.

The user confirmed no additional measurements are available. Hidden/camera-edge
finger placement and exact support height therefore cannot be promoted to
verified values from this recording. The existing renderer disables gravity
and collision resolution: readback proves execution of the intended motion,
not force-supported manipulation. Better visibility or another identical replay
would not settle these issues.

The unresolved acceptance condition is evidence for correct hand/object contact
and physical clearance in one complete development task. The public object
reference narrows the diagnosis; next work must improve source-supported hand
placement/timing and clearance under support uncertainty. Preserve the refusals
and uncertainty; do not relabel inferred poses as validated targets, alter frozen
hashes, relax thresholds or call the full goal complete. Issue #9 remains deferred.
Untouched public/private episode validation and scoring are not claimed.
