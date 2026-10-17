# Estimated contact guidance: development experiment

**CPU mechanics pass; GPU validation and learned-policy results are pending.** This adds the official pinned FlashCHORD contact-wrench-space support term using explicitly estimated human surface contacts. It does not establish a complete CHORD reproduction, successful grasp, or hidden-episode performance.

## Why this change

Previous grip learning used measured robot load/support rewards with unknown human contact labels disabled. A saved-rollout audit found that the failed first-carry window received roughly55%of its positive mean reward from object tracking,25%from pad support and1.4%from carried-load matching. Those averages include all active hands; they are a diagnostic, not proof that a particular weight caused the failure.

The preserved source bundle contains observed MANO meshes. At the raw estimated contact points, a convex-halfspace audit found that58.84%of pot points and51.89%of lid points lie more than2mm inside at least one native collision piece. This is a mismatch between reconstructed surface estimates and the current32-piece colliders, not a measurement of robot penetration. Raw estimated guidance would contain targets that cannot be touched at their original locations.

Keeping only hints that project at most2mm onto an exposed collider removed every opposing normal pair in the first pot-carry window. A separately bounded reference-estimation correction of at most12mm recovered opposing directions in all172observed frames of that182-frame window. The largest actual projection was9.59mm for the pot and11.85mm for the lid. Every stored point was rechecked against the complete convex union after float32 storage; the largest inward roundoff was below3nanometres.

The robot's existing2mm sampled-penetration gate is unchanged. The12mm bound applies only to constructing approximate reference hints from uncertain reconstructed geometry. It changes neither source observations nor the frozen robot/object trajectories, masses, gravity or motor limits. Collider fidelity remains a separate question.

## Preparation contract

1. Use only directly observed, valid hand meshes. Invalid frames stay unknown.
2. Find exact nearby object triangles using bounded spatial queries. Project hand surface points within5mm onto locally consistent object surfaces; exclude boundary faces, local intersections deeper than2mm, and ambiguous object assignments.
3. Reproject those estimates onto exposed native convex surfaces, with a maximum12mm correction and30°normal change for this development experiment. Reject faces buried inside another collision part. Use inward native face normals as estimated force directions.
4. Preserve original measured-contact flags as unknown. Bind candidate data, source bundle, object order, source clock, native collision contract and preparation receipts by SHA256.
5. Copy the frozen input directory and add a separate estimated-contact file. Verify that every existing asset and target remains byte-identical.

This preparation is deterministic under the tested CPU library versions, but bit-exact behavior across platforms is not promised. Inputs and derived contact arrays stay private/local for transfer to authorized compute; they are excluded from code images and repository commits.

## CPU preparation

Install the pinned optional dependencies in [requirements.contact-prep.txt](../../../docker/dex3-training/requirements.contact-prep.txt). Set the paths below to the privately transferred original bundle and corrected-clock inputs. Use fresh output directories.

```bash
python scripts/dex3_training/prepare_estimated_contacts.py \
  --bundle "$V2D_SOURCE_BUNDLE" --out "$V2D_RAW_ESTIMATES"
python scripts/dex3_training/filter_contacts_for_collision.py \
  --native "$V2D_BASE_INPUTS" --estimate "$V2D_RAW_ESTIMATES" \
  --out "$V2D_NATIVE_ESTIMATES" --maximum-reference-projection-m 0.012
python scripts/dex3_training/attach_estimated_contacts.py \
  --base "$V2D_BASE_INPUTS" --estimate "$V2D_NATIVE_ESTIMATES" \
  --out "$V2D_CWS_INPUTS"
```

The source bundle is the existing private `bundle/episode.npz` with its adjacent manifest. For this e002 experiment its SHA256 is `4de1b9507e42ed66e4341b93cc38563885fc2eda43db72935a546967671b6277`; the unchanged base task manifest is `c4d95304f28b1c484059a10c2f5bd19412f5f468b003769f9ef1f1882bb82da4`. The prepared, collision-compatible input manifest is `b0a072632f93f273da1cb85deb3c3ed83545d996b6e1ec8594dbe9ec64e4b17f`.

## Runtime and validation

`dex3_grip_estimated_cws` inherits the wider-action assisted experiment and adds a CWS reward weight of10. The term compares wrench-support functions using the pinned implementation's512basis directions and8-ray friction cones. Missed-contact, unintended-contact and force-closure terms remain disabled. Unobserved frames have no estimated support target and receive no label-based contact penalty.

Enabling CWS requires both an explicit configuration flag and a verified estimated-surface reference. Default experiments retain the unknown-contact guard. A previous image digest does not contain this new feature; use its separately validated image once published.

The GPU gate must compare reference supports against an independent CPU calculation, verify zero reward for absent robot contact and one for identical active support, reject unbound references, and exercise the captured unassisted evaluator. Then train a fresh controlled experiment and audit the complete frame-zero replay in all worlds. No full-task or generalization claim follows from prepared contact hints.
