# Source evidence at the remaining Dex3 contact boundaries

The frozen e002 motion replay is complete and reproducible. Its physical target
gate remains unaccepted. An additional automatic check processes all 343 frames,
both hands and both objects using the retained recovery geometry reviewed in
PR #37. It supplies evidence for assessing the six contact misses, rather than
extending grasps to match their validation labels.

The method uses current-image detector-bounded masks, independently verified
native geometry, actual calibrated stereo pixels, the original camera and
object poses. Original human observation flags remain unchanged. Geometry
consistency passing is a distinct condition from accuracy or contact acceptance.
Object poses must have original observed support before their pixels contribute
to this diagnostic. Backfilled and inferred pot poses are excluded.

The unchanged thresholds require all nine pixels in a 3 by 3 neighborhood to
have valid depth and own-hand mask support, depth spread at most 10 mm, a hand
front depth error at most 10 mm, and CAD surface proximity at most 5 mm.
Competing objects require more than 2 mm difference in their proximity.
Potential intervals require at least eight points in each of three contiguous
frames. Adjacent pixels are not independent depth measurements. The CAD area
sampling is deterministic: one million points per object, seed 4814 plus the
object index. The method reads no grasp annotations or ground truth.

The geometric calculation precedes comparison with the existing validation
misses. The complete run finds 14,991 left/pot candidate pixels over 222 frames
and 1,615 right/lid candidate pixels over 116 frames. The other two hand/object
pairings have zero candidates. These counts measure qualifying proximity;
they do not establish accurate recovered fingers or tactile contact.

| Boundary | Original hand observation | Verified current-image proposal | Qualifying points on intended object |
|---|---|---|---:|
| Left 46 | Missing | Missing | 0 |
| Left 272 | Observed | Available | 4 |
| Left 332 | Missing | Available | 0 |
| Left 333 | Missing | Available | 0 |
| Right 46 | Missing | Available | 0 |
| Right 309 | Observed | Available | 1 |

The pot is backfilled at frame 46 and inferred at 332–333, so it cannot provide
original observed object support at those boundaries. Even the two boundaries
with observed hands fall below the existing eight-point minimum. Therefore
this diagnostic does not justify a contact timing extension. Replay coverage
remains 375/381; source flags and the frozen trajectory remain unchanged.

The proximal-thumb caveat concerns **human source anatomy**. The independent
robot contact audit uses actual original Dex3 distal thumb (`thumb_2`) and index
(`index_1`) triangle faces. Correct robot contact geometry does not establish
that a hidden human fingertip was observed or accurately recovered.

The first experimental protocol snapshot was written before the current-image
input requirements were added. Its terminal report recorded the effective
requirements correctly; the snapshot alone must not be used as the effective
protocol. The corrected method writes the protocol after binding these inputs.
The earlier run and this metadata problem remain retained as experimental
history. The fresh corrected run completed: all 14 arrays and the compressed
payload bytes match the preceding run exactly. Every saved protocol field now
agrees with the terminal report's effective requirements. See the
[verification receipt](dex3-motion-replay/contact-evidence-verification-v1.json).

## Visual ownership check and conservative mask agreement

The automatically selected maximum-support source views were inspected: left
178 and right 278. Several left candidates appear on the visible white pot
handle. The current-image hand mask includes that area. This is a semantic
ownership concern even when the depth and CAD proximity checks pass: an object
pixel is not an observed finger contact. The original left mask was inspected
separately and excludes part of this handle area.

An automatic additional filter requires all nine pixels also to belong to the
original own-hand mask and not the other original hand mask. It processes every
retained candidate, uses locked frame/hand mask identities, and can only remove
parent candidates. It removes 1,451 candidates: the left count becomes 14,491
and the right count 664. Source observation flags, geometry and the replay are
unchanged. This agreement of two estimated masks is a conservative filter,
not semantic ground truth or proof that every remaining pixel is a hand.
It must not be treated as a fix for the six grasp boundaries or used to promote
the recovered geometry to accurate contact targets.

## Reproduce this evidence calculation

The [method archive](dex3-motion-replay/contact-evidence-methods-v1.tar.gz)
contains the two configurable entry points. Verify its contents against the
[byte manifest](dex3-motion-replay/contact-evidence-methods-manifest-v1.json).
The [summary](dex3-motion-replay/contact-evidence-summary-v1.json) includes all
automatic intervals and the separate validation-boundary comparison.

Use the existing validated episode bundle, its exact derived voxel material,
the verified retained hand-quality run from PR #37, and authorized source depth,
validity and original hand masks. The source root is the repository's
`robotic_grounding` directory, including the recorded source-review renderer.
The recorded CPU runtime uses Python 3.11.17, NumPy 2.2.6, SciPy 1.15.3,
Numba 0.61.2 and llvmlite 0.44.0. A completely fresh installation of this
additional diagnostic environment has not been independently validated.

Extract the method archive into `/data/review/methods`. Relocate locked source
files with the same explicit path-map format as the
[source accuracy diagnostic](../issue8/SOURCE_ACCURACY.md); hashes and identities
must match. Use new output directories:

```sh
python /data/review/methods/quality_stereo_contact_regions_v2.py \
  --source /data/repository/robotic_grounding \
  --bundle /data/episode-bundle --material /data/derived-material \
  --quality /data/retained-quality-run --path-map /data/paths.json \
  --out /data/new-current-image-evidence

python /data/review/methods/filter_contact_mask_agreement_v1.py \
  --source /data/repository/robotic_grounding \
  --bundle /data/episode-bundle --parent /data/new-current-image-evidence \
  --path-map /data/paths.json --out /data/new-mask-agreement-evidence
```

The first output payload SHA256 is
`bb60315578053f5c95df41a61f506f661549e29c8a3e43d386f60ebcf5d8700b`;
the filtered payload is
`17621c8f340c1e7d6ad1d20a0c0b4d96a49b587bfb1c6c768670c8cb1b04e98a`.
Exact array and file reproduction was measured locally in this recorded runtime.
Neither cross-machine execution of this new diagnostic nor a complete upstream
inference/fitting rerun is claimed. Code access alone does not supply licensed
raw source inputs or imply accepted contact targets.

No simulator rerun is implied by this check: it changes evidence calculations,
not the robot motion. The frozen release tag and archives remain unchanged.
The transition check is a separate explicitly declared interpolation candidate;
partial certified intervals do not establish clearance for the full replay.
Support accuracy, hidden-finger accuracy, full upstream reconstruction and
untouched-episode generalization remain unaccepted. Issue #9 remains deferred.
