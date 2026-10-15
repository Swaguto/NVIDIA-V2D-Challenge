# Brev physics validation receipts — October 15, 2026

Read the [protocol and measured results](../../../PHYSICS_VALIDATION.md).
The complete physical task remains failed. These receipts preserve successful
controls and failed manipulation outcomes without changing acceptance limits.

## Source and runtime

| Experiment | Source checkout | Result |
| --- | --- | --- |
| `baseline-free`, `implicit-free`, `implicit-full` | `59cc82e0345134933d1e8528bbf6b00653074696` | Baseline prefix fails; candidate prefix passes; complete task fails |
| `feedforward-free`, `feedforward-full` | `761890f0c648573ad636bba3e974f31cc8a7dac5` | Prefix passes; complete task fails |
| `integral-free`, `integral-full` | `07a6b1f5b5d599eea5df63b6960791302feb1d81` | Prefix passes; complete task fails |
| `static-load` | `b40213a25fff764311835c143a9de4043708be59` | Physics controls pass; all four unsupported grasp starts fail |

The immutable runtime is
`ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:8585588357d898929bf6f80f0011441cbcab07a3d95e2f5f4d0e24bc5c177190`,
built from `5128f63e250ca5626e7b093ccb6d26cdfad304b6`.
Each physics checkout was mounted read-only. Deployment manifests bind every
selected source file and source archive. Per-stage GPU preflights establish
L40S / driver 580.178.04 / pinned CUDA 13 tensor execution. Simulator reports,
actual rendered videos and decoded-frame receipts establish subsequent execution.

The original failed attempt at `0336fdb` hit an empty-observation summary error.
Its failure sidecar is retained separately; it has no accepted tracking gate.
The later executor saves partial states before summaries and handles that prefix.

## Integrity and interpretation

Each stage preserves its invocation, simulator report, video check and tracking
or load gate. Full stages also preserve the independent source-contact audit.
`integral-full/collision-readback.json` records the actual USD enablement of all
eight colliders per hand at every mask transition. A nonzero final command exit
for these completed failures is expected; successful video generation alone is
not acceptance. Contact coverage alone is not a successful grasp.

`final-backup-verification.json` binds 16,154 unique declared files and
4,573,326,213 bytes. The matching compressed inventory contains every SHA256;
raw states, original videos and SDK logs remain in the private Mac backup.
Per-attempt inventories deliberately overlap logs; the final inventory deduplicates
only identical hashes. The VM is **STOPPED**; see `compute-closure.json`.

The browser-compatible static derivative was packaged locally from the unchanged
Brev MPEG-4 original. `static-load/browser-preview.json` binds both hashes and
all 21 frame timestamps. This packaging check used Mac FFmpeg 8.1.2; it is separate
from the GPU source checkout above. The new wrapper automates the same packaging
for future runs. The original physics video/report hashes remain unchanged.

`fresh-host-team-check.json` and `fresh-host-e011/` establish the complete 137-frame,
four-video frozen-target replay on this new registry host. They do not establish
successful physics or independent teammate review. The twelve diagnostic
reference exports preserve uncertainty and are explicitly not training-ready.

Local controls: **37 passed**. Physics/reference GitHub CI passed at `59cc82e`
([run 38766587597](https://github.com/Swaguto/NVIDIA-V2D-Challenge/actions/runs/38766587597)),
`761890f` ([38775670969](https://github.com/Swaguto/NVIDIA-V2D-Challenge/actions/runs/38775670969)),
`07a6b1f` ([38777179859](https://github.com/Swaguto/NVIDIA-V2D-Challenge/actions/runs/38777179859))
and `b40213a` ([38778585890](https://github.com/Swaguto/NVIDIA-V2D-Challenge/actions/runs/38778585890)).
