# Track 3 scoring scope and deferred hand geometry work

The user directed us to leave work that is not evaluated by the challenge.
The current official submission kit was downloaded from the
[NVIDIA challenge site](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/#submission)
on October 14, 2026 UTC. Its SHA256 is
`8d2bce64ecab6da4eee938fe2859bd60e767535af30068b386e2dc20a72cf43e`.
The kit's `tracks/track_3/README.md` and `metric_code/README.md` specify six
metrics: AUC, SP-SR, MP-SR, RPE, MPPE and CD-O.

Five metrics evaluate object trajectories from recorded policy evaluations;
CD-O evaluates reconstructed object surfaces. The submission includes policy
checkpoints, evaluator provenance, meshes in the same body frames as rollout
poses and a producing code commit. Object symmetry and mesh registration are
part of the current pose scoring rules. Object trajectory scale remains scored.
The kit does not list a separate hand visibility, hand mesh penetration,
hand self-intersection or table-clearance metric for Track 3. Track 1's PEN
metric must not be treated as a Track 3 metric.

Further cosmetic Dex3 mesh and assembly-overlap work is deferred. The original
robot models, frozen replay, observation masks and display-only pose flags are
unchanged. Experimental model replacements were rejected and reverted; no new
production motion or Isaac run resulted. Contact errors remain a possible
downstream policy-training risk because they can prevent accurate physical
object motion. That is an indirect consequence, not an additional scored metric
or evidence of a score improvement from these experiments.

## Qualification of the earlier geometric certificates

The frozen geometric certificates query the hand's URDF collision meshes.
They do not establish clearance for every rendered visual triangle. A later
read-only comparison of the unchanged 343-frame replay found visual-mesh/object
intersections in 320 of 686 hand states and visual-mesh/estimated-sheet
intersections in 25 states. The corresponding collision-mesh sample queries
reported zero. Six thumb visual meshes extend beyond their collision mesh
convex hulls, by up to approximately 26.5 mm. These are triangle intersection
counts on source-frame samples, not material penetration depths, continuous
visual clearance, measured real-world errors or a policy score.

Consequently, the earlier claim of nonintersection remains restricted to the
declared collision geometry and estimated support model. Actual Isaac readback
still verifies execution of the frozen kinematic motion and display continuity;
it does not verify contact-force manipulation. Source hand accuracy, physical
acceptance and every motion-target approval remain unaccepted.

The diagnostics and rejected experiments are retained in the local checkpoint
`runs/dex3-baseline-20261011/scoring-scope-checkpoint-v23.json`. They are not a
new reconstruction pipeline or a replacement release. No further geometry
experiment is running. The GPU was not restarted.

## Work that remains useful for evaluation

Preserve the reproducible reconstruction/replay baseline and explicit inference
provenance. Use development object references for accuracy diagnostics without
copying reference poses into predictions. Check the existing evaluation code
against this current kit before claiming official metric parity: older local
five-metric diagnostics do not establish parity with its mesh registration and
symmetry handling. Untouched public episodes and private evaluation remain
unvalidated by the current e002 replay. A kinematic replay is not a scored policy
rollout; policy training remains a separate later project stage.

The original goal remains incomplete. Deferring unscored polishing does not
certify hand accuracy, physical task success, a private score or human PR review.
