# Original SAM seed preservation experiment

e015's source frame zero exposes a pose-refinement tradeoff:

| Rack estimate | Source silhouette IoU | Median measured depth distance | Minimum height above inferred table plane |
| --- | ---: | ---: | ---: |
| Original SAM seed | 0.8636 | 8.59 mm | +1.38 mm |
| FoundationPose output | 0.8678 | 7.53 mm | −15.27 mm |

Refinement changed translation by 21.57 mm and rotation by 3.11°. The spoon's
original seed is also above the plane; its initial refinement worsens both
silhouette and measured-depth fit. These are source/reference diagnostics, not
hidden ground-truth accuracy measurements. Native collision checks separately
confirm initial intersections in the current baseline. Training later aborts on
non-finite tracking metrics; this does not prove the sole cause of instability.

The optional `preserve_model_seed` reconstruction setting, or
`batch_public.py --preserve-model-seed`, now proposes retaining the original SAM
pose at its seed frame. It verifies the original mask/image/pointmap/raw mesh/
transform closure, baked mesh hash, body pose and at least 200 measured pixels.
The usual measured-depth and temporal pose-jump gates still run. Old caches with
a different initialization policy cannot silently enter this arm.

This option **does not certify improved source fit or physical feasibility**.
It remains off by default. Refined seeds continue to require their existing
calibrated-source contract; this option cannot relabel them as raw seeds.

CPU contracts and both actual e015 source snapshots pass. A paired GPU trial
must retain identical source inputs and mesh bytes, then compare complete tracks,
native contacts, training stability and unassisted evaluations before promotion.

The separate measured-surface rack candidate was rejected because it worsened
source fit. Its improved spoon candidate alone does not fix the rack reference.
