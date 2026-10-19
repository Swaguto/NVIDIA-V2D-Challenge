# Native hand self-contact audit

The hand/object/support audit does not assess collisions between links of the
same hand. Before physical validation, assess both scopes on the changed
reference. Keep the original 1 mm gate and native collision filters.

```bash
python scripts/dex3_training/audit_hand_self_contacts.py \
  --inputs /private/frozen-reference \
  --native-model /private/native-model.mjb \
  --native-model-sha256 <verified-model-sha256> \
  --out /private/hand-self-contacts.json
```

Use a model from the same pinned native runtime and geometry. The audit loads
the cached model without recompiling. It verifies frozen input hashes, explicit
joint/body bindings and independent URDF FK at every reference knot. It reports
both hands, including empty contact knots. Unknown or malformed body bindings,
gaps and clocks fail rather than silently removing failures.

`hand_self_contact_gate_passed` concerns native active collision geometry at
the reference knots. It does not certify measured forces, source accuracy,
continuous swept motion, motor feasibility, a grasp or task success. Assess
those separately. Never reinterpret an old hand/object certificate as covering
self-contact or a newly changed command sequence.

## e037 retained results

The original cached reference has self-penetration over 1 mm in 151 left and
46 right knots. A new derived left-thumb correction passes an independent
151-knot self/source/motion check: maximum left self-penetration is 0.50 mm,
all 604 original image-anchor errors are nonregressing, and actual joint limits
and original motion bounds pass. The human source data, authority flags, wrist
roots, right hand and non-thumb fingers are unchanged. Robot thumb angles are
derived retargets, not measured human joint angles.

There are also 450 sampled intermediate left-self states, maximum 0.501 mm;
these samples are not a continuous or full-task swept certificate. Right-hand
self-collisions and five hand/object penetration failures remain. No complete
physical task, policy or score improvement is qualified, and no PPO was run.
