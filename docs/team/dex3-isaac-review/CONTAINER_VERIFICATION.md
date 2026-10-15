# Container verification status

## Completed

- Branch `codex/continuity-dex3-brev` preserves the hand continuity and Dex3 histories,
  executed preview sources, runtime pins and twelve-episode review receipts.
- All three Linux amd64 images built on Brev from code commit
  `a91fa0e36968c25eaca44349caf628450d8b4c71`. Their checked package closures contain
  35 native-continuity, 72 IK and 178 simulator distributions, with no mismatches.
  The pinned Isaac Lab source check also passed.
- The Brev host is an L40S / driver 580.126.09 / Intel Xeon Platinum 8462Y+.
  The native continuity image passed a real CUDA 12.4 tensor calculation on that GPU.
- Authorized private MANO support-source imports passed without loading model weights.
- Full GPU Docker replays completed on Brev for e011/e040/e041: 137/260/299
  frames (696 total), with four decoded/verified videos per episode. All 248
  output/log files were copied back and SHA256-verified. Eight e011 sample rows
  were visually inspected; this is not a full manual review of all twelve videos.
- Earlier e011 media packaging ran in the native Docker image: all four videos decode to
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
**failed**; no acceptance threshold was changed.

Native x86 Brev reruns then **passed** the same comparison: e011's maximum qpos
difference was 4.218847493575595e-14 and e021's was 1.7680301667155618e-12.
Frame axes, unknown masks and joint limits passed for both. This establishes
reproduction for these two checks on this host; it does not certify all episodes
or arbitrary hardware. Default preview replay still uses the exact frozen targets.

## In progress / remaining

The original three-image archive is 16,440,235,501 bytes, SHA256
`439404151798b173195da9f1fc962028eef73e160208d6a733f9dcf596f4557f`.
Its slow Mac transfer was interrupted; that local partial copy is not an accepted
backup. The full archive remains on the Brev VM. Private GHCR publication is
approved and in progress. Registry layers are bounded to 8 GiB payloads and the
repacked simulator prefix is checked file-by-file against the installed builder.

The original native twelve-episode replay and 3,096-frame simulator checks predate
containerization. The separate Brev container pilot covers three episodes, not
all twelve. This Mac has no NVIDIA GPU.
Physics-enabled manipulation and learned-policy evaluation are unvalidated.
