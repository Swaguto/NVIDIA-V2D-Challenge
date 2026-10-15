# Container verification status

## Completed

- Branch `codex/continuity-dex3-brev` preserves the hand continuity and Dex3 histories,
  executed preview sources, runtime pins and twelve-episode review receipts.
- Native hand-continuity and IK Linux amd64 images are built. Their checked package
  closures contain 35 and 72 distributions respectively, with no mismatches.
- Authorized private MANO support-source imports passed without loading model weights.
- e011 media packaging ran in the native Docker image: all four videos decode to
  137 frames with strictly increasing presentation timestamps. The first test fixture
  omitted its numerical receipt; that failed output was preserved and the complete
  fixture passed in a new directory.
- The private input pack verified all 11,749 files across twelve episodes before
  extraction. A separate hash-checked pack restores 40 runtime support source files.
- All 14 migration controls pass locally and in GitHub CI. Thirteen existing
  continuity/image controls also passed on the Mac (with existing NumPy runtime warnings).
- The source link audit found 3,336 symlinks and no external/broken targets in its
  declared scan roots. This does not establish completeness; e039 is still unavailable.

## Strict fresh-IK diagnostic

The emulated Linux CPU reruns did not match frozen targets at the declared 1e-6
absolute qpos tolerance. The maximum finger-angle differences were
1.833927903316379e-5 rad for e011 and 0.019262072979340406 rad for e021.
Source axes, unknown masks and joint limits passed. The strict comparisons remain
**failed**; no acceptance threshold was changed. Default replay uses the exact
frozen targets. New IK remains a diagnostic until checked on the native target host.

## In progress / remaining

The full Isaac image downloaded its official 5.9 GB extension-cache wheel and is
now downloading the remaining SDK/runtime dependencies.
Its final build, package check and complete three-image export are pending. Current
image IDs and the producing commit are recorded in `receipts/container/`.

The original native twelve-episode replay and 3,096-frame simulator checks predate
containerization. They do not prove a GPU Docker/Brev render. This Mac has no NVIDIA
GPU; the first complete render on a compatible Brev host remains a separate gate.
Physics-enabled manipulation and learned-policy evaluation are unvalidated.
