# Container verification status

## Completed

- Branch `codex/continuity-dex3-brev` preserves the hand continuity and Dex3 histories,
  executed preview sources, runtime pins and twelve-episode review receipts.
- The first three Linux amd64 pilot images built on Brev from code commit
  `a91fa0e36968c25eaca44349caf628450d8b4c71`. Their checked package closures contain
  35 native-continuity, 72 IK and 178 simulator distributions, with no mismatches.
  The pinned Isaac Lab source check also passed.
- The Brev host is an L40S / driver 580.126.09 / Intel Xeon Platinum 8462Y+.
  The native continuity image passed a real CUDA 12.4 tensor calculation on that GPU.
- Authorized private MANO support-source imports passed without loading model weights.
- Full GPU Docker replays completed on Brev for e011/e040/e041: 137/260/299
  frames (696 total), with four decoded/verified videos per episode. All 248
  output/log files were copied back and SHA256-verified. Eight sample rows per episode
  were visually inspected; this is not a full manual review of all twelve videos.
- Earlier e011 media packaging ran in the native Docker image: all four videos decode to
  137 frames with strictly increasing presentation timestamps. The first test fixture
  omitted its numerical receipt; that failed output was preserved and the complete
  fixture passed in a new directory.
- The private input pack verified all 11,749 files across twelve episodes before
  extraction. A separate hash-checked pack restores 40 runtime support source files.
- All 16 migration controls pass locally and in GitHub CI. Thirteen existing
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

## Published images and final-layout check

All three current images are published **privately** on GHCR from
`5128f63e250ca5626e7b093ccb6d26cdfad304b6`, linked to this repository. Each registry
manifest's config digest equals the built image ID; each anonymous pull is denied.
Exact pull digests and compressed layer bounds are in
[`brev-5128f63/ghcr-images.json`](receipts/container/brev-5128f63/ghcr-images.json).
Use `docker/dex3-review/registry.env.example` to pull these images by digest.

Registry layers are bounded to 8 GiB payloads. All 146,692 simulator files/symlinks
matched the builder's bytes, file modes and link targets. The final registry-layout
image then completed e011 on the GPU: all four packaged videos match the first
Brev e011 run byte for byte. Its 72 IK / 178 Sim checks and the current native
image's 35-package check passed. All 330 pilot/registry output and log files
(138,056,449 bytes) were copied to the Mac and SHA256-verified.

The original `a91fa0e` intermediate archive was 16,440,235,501 bytes, SHA256
`439404151798b173195da9f1fc962028eef73e160208d6a733f9dcf596f4557f`.
Its Mac transfer was interrupted; local partials are not an accepted archive.
The current private GHCR images supersede this intermediate distribution.
Historical source, image IDs, pilot results and archive receipt are preserved.

The original native twelve-episode replay and 3,096-frame simulator checks predate
containerization. The separate Brev container pilot covers three episodes, not
all twelve. This Mac has no NVIDIA GPU.
Physics-enabled manipulation and learned-policy evaluation are unvalidated.

## Compute closure

All cloud builds, rendering and registry publication completed. Brev rejected the
stop request because `v2d-dex3-container-check` (`da0f8igen`) does not support stop.
All current artifacts are independently preserved. After explicit owner approval,
`brev delete` completed successfully. At 2026-10-15 15:09:37 UTC, `brev ls --json`
returned `{"workspaces": null}`: the temporary VM is deleted and no Brev instances
remain listed in this organization. See the
[compute closure receipt](receipts/container/brev-5128f63/compute-closure.json).
Original RunPod resources were not changed.
