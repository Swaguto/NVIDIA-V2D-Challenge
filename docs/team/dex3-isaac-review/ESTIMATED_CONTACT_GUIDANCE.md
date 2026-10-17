# Estimated contact guidance: development experiment

**CPU mechanics, GPU contract checks and actual PPO smoke pass. Full policy quality is pending.** This adds the official pinned FlashCHORD contact-wrench-space support term using explicitly estimated human surface contacts. It does not establish a complete CHORD reproduction, successful grasp, or hidden-episode performance.

## Why this change

Previous grip learning used measured robot load/support rewards with unknown human contact labels disabled. A saved-rollout audit found that the failed first-carry window received roughly 55% of its positive mean reward from object tracking, 25% from pad support and 1.4% from carried-load matching. Those averages include all active hands; they are a diagnostic, not proof that a particular weight caused the failure.

The preserved source bundle contains observed MANO meshes. At the raw estimated contact points, a convex-halfspace audit found that 58.84% of pot points and 51.89% of lid points lie more than 2 mm inside at least one native collision piece. This is a mismatch between reconstructed surface estimates and the current 32-piece colliders, not a measurement of robot penetration. Raw estimated guidance would contain targets that cannot be touched at their original locations.

Keeping only hints that project at most 2 mm onto an exposed collider removed every opposing normal pair in the first pot-carry window. A separately bounded reference-estimation correction of at most 12 mm recovered opposing directions in all 172 observed frames of that 182-frame window. The largest actual projection was 9.59 mm for the pot and 11.85 mm for the lid. Every stored point and a 50 µm outward probe were rechecked against the complete convex union after float32 storage; the largest inward roundoff was below 3 nanometres.

The robot's existing 2 mm sampled-penetration gate is unchanged. The 12 mm bound applies only to constructing approximate reference hints from uncertain reconstructed geometry. It changes neither source observations nor the frozen robot/object trajectories, masses, gravity or motor limits. Collider fidelity remains a separate question.

## Preparation contract

1. Use only directly observed, valid hand meshes. Invalid frames stay unknown.
2. Find exact nearby object triangles using bounded spatial queries. Project hand surface points within 5 mm onto locally consistent object surfaces; exclude boundary faces, local intersections deeper than 2 mm, and ambiguous object assignments.
3. Reproject those estimates onto exposed native convex surfaces, with a maximum 12 mm correction and 30° normal change for this development experiment. Reject faces buried inside another collision part. Use inward native face normals as estimated force directions.
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

The source bundle is the existing private `bundle/episode.npz` with its adjacent manifest. For this e002 experiment its SHA256 is `4de1b9507e42ed66e4341b93cc38563885fc2eda43db72935a546967671b6277`; the unchanged base task manifest is `c4d95304f28b1c484059a10c2f5bd19412f5f468b003769f9ef1f1882bb82da4`. The prepared, collision-compatible input manifest is `7cfb37e61e070dac916d78b869ea644e577c786167391c0a8267325763961b89`.

## Runtime and validation

`dex3_grip_estimated_cws` inherits the wider-action assisted experiment and adds a CWS reward weight of 10. The term compares wrench-support functions using the pinned implementation's 512 basis directions and 8-ray friction cones. Missed-contact, unintended-contact and force-closure terms remain disabled. Unobserved frames have no estimated support target and receive no label-based contact penalty.

Enabling CWS requires both an explicit configuration flag and a verified estimated-surface reference. Default experiments retain the unknown-contact guard. A previous image digest does not contain this new feature; use its separately validated image once published.

The GPU gate must compare reference supports against an independent CPU calculation, verify zero reward for absent robot contact and one for identical active support, reject unbound references, and exercise the captured unassisted evaluator. Then train a fresh controlled experiment and audit the complete frame-zero replay in all worlds. No full-task or generalization claim follows from prepared contact hints.

Session 88 passed that gate with a maximum CPU/GPU support discrepancy of
`2.3839840079808994e-7` and unchanged physical-model SHA256
`92d12e7f10a0b4a27b623e64e0b5c251c4f49190ff4b5c590dd99f0981747bba`.
Its 8-iteration PPO smoke recorded nonzero CWS reward. The private code image was
pushed and pulled by immutable digest; it contains neither inputs nor weights.

Copy [registry.estimated-cws.env.example](../../../docker/dex3-training/registry.estimated-cws.env.example)
to a local environment file and set the private input and writable output paths.
Use a fresh output directory for each invocation.

```bash
docker compose --env-file /path/to/local-cws.env \
  -f docker/dex3-training/compose.estimated-cws.yaml run --rm estimated-cws
docker compose --env-file /path/to/local-cws.env \
  -f docker/dex3-training/compose.estimated-cws.yaml run --rm estimated-cws \
  /opt/dex3-extension/train.py --config-dir /opt/dex3-extension/configs \
  experiment=dex3_grip_estimated_cws output_dir=/outputs/ppo
```

### Removing reset assistance

`dex3_grip_cws_unassisted` is a separate, explicit actor-initialized experiment
prepared after the matched comparison. It removes the 20-step reset assistance
and uses only frame-zero starts with zero VOC throughout training. It copies
actor mean parameters and observation normalization while starting a fresh
critic, optimizer, exploration noise and training counters. The input, physical
model, action space and observation schema must match the parent. The validator
permits only this declared removal of reset settling; other reset changes still
fail. This continuation is not present in the source-31c0d89 image above and
requires its own GPU validation and image before reproduction.

## Scene compatibility

Estimated contacts retain the `dex3_dual_hand_v1` scene schema so the runtime
resolves the unchanged support table. Their separate
`local_development_estimated_surface` provenance tag and hash-validated contact
receipt distinguish them from references with unknown contacts. Session 87
stopped during scene setup before any training because the initial integration
changed the scene schema. Its logs were backed up and the VM was stopped.

