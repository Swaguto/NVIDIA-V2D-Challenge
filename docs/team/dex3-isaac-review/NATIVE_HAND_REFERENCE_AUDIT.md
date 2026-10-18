# Native hand reference audit

`scripts/dex3_training/audit_hand_reference_contacts.py` checks every reference
knot on CPU before new short policy runs. It verifies the frozen input closure,
applies exact object poses, resets wrist slide/quaternion and explicitly ordered
finger joints, and checks all native hand-body poses against independent URDF
forward kinematics. Native joint prefixes are matched unambiguously.

The report records actual native contact samples and summarizes hand/object and
hand/support penetration by source frame. Its 1 mm threshold is a reporting
tolerance, not an official evaluator rule or physical-task acceptance criterion.
Positive/zero distance alone does not establish a grip. This does not measure
contact forces, step physics or check hand self-collision.

Run in the pinned training runtime with the source directory containing
`check_native_scene.py` available:

```sh
python scripts/dex3_training/audit_hand_reference_contacts.py \
  --inputs /private/frozen-pack --out /private/fresh-hand-contact-report.json
```

Fresh code-only policy contexts include this diagnostic; a new runner rejects
an older context that omits it. Archived runs retain their original producers
and can be audited separately without rewriting their receipts.

## Integration check

Both the original e020 and its object-relative command candidate were audited
over all 202 source knots under the exact pinned native runtime. All native
hand-body FK checks and frozen file closures passed. The candidate retained
the control's object/robot/support/collision assets. Its new right-hand
trajectory follows the can more closely but increases colliding frames, so it
is not promoted. No GPU training was used for this check.
