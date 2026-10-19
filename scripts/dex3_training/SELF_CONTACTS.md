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

A subsequent right late-contact-mode component preserves that left fix and
reduces right failing knots from 46 to 13, with every right knot nonregressing.
Fresh verification checks all 1,072 available left/right image anchors and all
original motion bounds; 136 missing right anchors remain explicit and unchanged.
The first full right paths and bridge failures are retained. Five object
penetration failures persist, and 450 new complete-pose interpolation samples
also retain failures. This combined component is not a whole-task, continuous
swept, motor, grasp or policy certificate. No GPU/PPO ran.

Analytic source-ray branches and a finite temporal path subsequently clear the
first five right-hand failures. Fresh full-clock checks preserve both earlier
components, all 1,072 available image anchors and original motion limits. Right
self failures are now eight (110–117); all five object failures remain. Early
reference penetration reaches 0.9934 mm and sampled intermediate penetration
0.9910 mm, so nominal margins are narrow. No continuous swept, physical or
policy success is established. Fixed-root compatibility diagnostics remain
qualified by missing per-leaf evidence; audit source/root authority before a new
wrist hypothesis. Human observations, missing flags and held-out failures remain.

The subsequent authority audit confirms that wrist orientation is model-derived
and softly fitted during retargeting. Two new coupled wrist/finger poses clear
the remaining hard knots in isolation while preserving image/model-orientation
nonregression and original component bounds. Independent native/URDF checks
agree, but their sparse insertion violates actual finger speed and original
temporal bounds; two dense paths and a conditional construction also fail their
acceptance checks. No new reference replaces the three-component freeze. The
eight right-self and five object failures remain; no GPU/PPO or task success
is qualified. See the retained source-wrist authority archive and wrist report.
