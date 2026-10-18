# Fresh hand recovery → reconstruction handoff

Status: CPU file protocol verified; real SAM2/HaMeR inference and physical results
remain unverified. This does not repair or qualify a grasp by itself.

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

Nineteen focused tests pass. The protocol integration tests use explicitly
synthetic model callbacks and the **real** `load_native_priors` and
`load_image_estimate` consumers. They check original-byte preservation, source
indices, missing-point confidence, mask rejection, output hashes, authority
flags, supported-frame protection and incomplete model-setup receipts. They
are tests of the handoff, not hand reconstruction accuracy.

## Compute and privacy

Brev remains unqueued during the owner-confirmed outage. Before any new paid
session, verify reliable stop control, arm the independent guard and use a short
lease. Use the existing pinned perception image and a committed code overlay;
no image rebuild or full source-video inference is required for this eight-frame
probe. Models and inputs are separate read-only private mounts; inference output
stays private. Do not publish footage, model weights or this private input pack.
