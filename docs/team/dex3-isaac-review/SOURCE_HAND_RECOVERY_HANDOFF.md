# Fresh hand recovery → reconstruction handoff

Status: the first real eight-image SAM2/HaMeR probe completed on Watcloud. The
handoff and refinement contracts have 28 passing CPU tests. Source geometry and
physical grasp acceptance remain unverified; see the measured checkpoint below.

## What is wired now

1. Bind the recovery request to the original episode payload **and its canonical
   manifest**. Every original hand JSON is checked against that manifest. The
   request must preserve the selected hand's identity, direct-observation flag,
   subject shape and scale. The private e037 revision-2 request verifies all
   **302 original hand records** and reuses the eight existing source crops.
2. Run the separate source-image recovery adapter in the pinned perception
   runtime. It records hashes for the pose, source-point JSON and mask together.
   Model setup errors retain an incomplete receipt. A process completing with
   rejected masks is not a successful hand recovery.
3. Back up and hash-verify the private output, stop the GPU, then run the CPU
   merger into a fresh directory. Rejected, incomplete, foreign or altered
   output is refused. Supported canonical observations cannot be overwritten.
4. The merger copies every unselected original prior byte for byte, replaces
   only the selected missing priors, and retains original confidence arrays.
   Both `--prior-records` and `--image-observations` are supplied to the later
   continuity fit. A new mask alone would still leave the old pose prior.

```sh
python scripts/track3/merge_recovered_source_hand.py \
  --request /private/path/request.json --validate-sources-only

python scripts/track3/merge_recovered_source_hand.py \
  --request /private/path/request.json \
  --recovery /private/path/fresh-source-results \
  --out /private/path/new-source-handoff
```

Use `new-source-handoff/priors` as canonical source-index `--prior-records` and
`new-source-handoff/image-observations` as `--image-observations` in a **new**
`repair_hand_continuity.py` run. Preserve its original-run, temporal-fit, calibrated
camera, licensed MANO and source-clock inputs. The selected source frame numbers
must not be remapped as temporal-fit indices.

The handoff is intentionally **unqualified**: no source fit, robot retargeting,
contact check or training occurs in the merger. Its receipt requires source-fit
nonregression, honest missing flags, retargeting under the original morphology
bounds and actual URDF limits, the unchanged **1 mm native reference-contact
gate**, swept checks, grasp proximity and motion checks before PPO. No source
confidence or contact truth is inferred from a fresh model prediction.

## CPU verification

Twenty focused tests pass. The protocol integration tests use explicitly
synthetic model callbacks and the **real** `load_native_priors` and
`load_image_estimate` consumers. They check original-byte preservation, source
indices, missing-point confidence, mask rejection, output hashes, authority
flags, supported-frame protection and incomplete model-setup receipts. They
are tests of the handoff, not hand reconstruction accuracy.

## Compute and privacy

The original Brev plan remains unqueued during its outage. Watcloud is now the
selected provider, with bounded Slurm allocations and verified result backups.
Use the existing pinned perception image and a committed code overlay;
no image rebuild or full source-video inference is required for this eight-frame
probe. Models and inputs are separate read-only private mounts; inference output
stays private. Do not publish footage, model weights or this private input pack.

A local-only minimal probe payload now contains 315 hash-verified files: eight
existing source images, 302 canonical hand records and the required request,
confidence/geometry payload and provenance metadata. Its input archive is
12,191,559 bytes; it contains no model weights or full source videos. Validation
against copied, relocated inputs succeeds without reading the original paths.
This is a partial source-recovery payload, not a complete episode replay pack.

The recovery has a streaming SHA256 implementation compatible with the stated
Python 3.10+ host contract. The committed code context is versioned separately
from the unchanged private input archive. The copied-input CPU rehearsal did
not execute Docker or GPU inference; the separate Watcloud run is recorded below.

## Watcloud refinement after the first real inference

The first eight-image inference now ran successfully on Watcloud. Its private
outputs were hash-verified locally and the Slurm allocation completed. See
[the measured result](WATCLOUD_MIGRATION_20261018.md); physical acceptance remains
pending. The copied-input rehearsal above and the real inference have separate
receipts and source identities.

An optional per-frame `inferred_exclusion` binds a source object-mask path and
SHA256, the source image hash, kind `inferred_source_object_segmentation`, and
`ambiguous_regions_are_unknown: true`. Object/hand overlap becomes unknown for
the hand. Positive source points inside it lose support; SAM2 receives negative
prompts in the excluded region, and candidate masks retain no excluded pixels.
Fewer than four remaining source points causes explicit abstention. These
inferred masks never supply metric or contact labels.

For a segmentation-only experiment, use `--reuse-hand-recovery` together with
`--reuse-request`. The complete parent recovery and file hashes are checked.
Images, crops, evidence, calibration, identity, shape and all other request
fields must match; only the exclusion controls may differ. Pose/camera values
are reused unchanged, with the parent record and request identities retained.
The HaMeR model and its weights are not loaded in this mode. This avoids repeating
unchanged pose inference or transferring its checkpoint for a mask refinement.

The 28 focused CPU tests include this refinement, negative-prompt handling,
abstention, parent-pose preservation and tampered/cross-source rejection. Model
accuracy is not established by the file protocol. The separate Watcloud
SAM2-only probe now exercised negative prompts and unchanged-pose reuse for all
eight frames, with its allocation released after verified backup. See the
measured migration report; its handoff still has no physical qualification.
