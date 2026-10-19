# Wrist retargeting and contact checkpoint

These are CPU experiments on non-reserved e037. No policy was trained, no
completed baseline or replay was rerun, and no complete physical task passed.
Source footage, model weights and result arrays remain private.

## Motion-constrained wrist reconstruction

The previous object-relative proposal changed only frames 36–69. That window
cannot satisfy the original acceleration cap: the repaired control already
exceeds it at acceleration row 70, whose three position knots are all fixed
under that restriction. This is a feasibility issue at the interval boundary.

The new proposal permits changes to right wrist commands at frames 25–99,
where source hand observations are neither direct nor metric. Direct and metric
commands, all source validity flags, the left hand, finger angles, object poses,
meshes and URDFs remain unchanged. A valid but inferred source hand flag is not
promoted to a direct observation.

The first SLSQP translation fit reaches its iteration limit and fails motion
checks. A convex OSQP projection then passes exact motion checks, but reports
inaccurate convergence; that control remains retained. A final SLSQP refinement
from the convex initializer converges in eight iterations. The completed
angular fit is reused, rather than rerun. Its analytical constraint Jacobian
agrees with central differences within 6.25e-10 in the sampled coordinates.

| Full reference motion summary | Original cap | Final proposal |
| --- | ---: | ---: |
| Maximum linear speed, m/s | 0.6223014934 | 0.5737553493 |
| Maximum angular speed, rad/s | 8.6099547292 | 8.5320580159 |
| Maximum linear acceleration, m/s² | 13.5397961244 | 13.5397947704 |
| Maximum angular acceleration, rad/s² | 111.3218987085 | 111.3218875764 |
| p95 linear speed, m/s | 0.5272371386 | 0.5272370859 |
| p95 angular speed, rad/s | 5.0782837183 | 5.0170609885 |

All six original caps pass. The five retained inferred-image checks also
improve over the previous repaired command: median errors at frames
40/45/50/55/67 change from 147.19/260.07/290.65/266.41/67.34 pixels to
46.77/58.73/79.01/79.81/43.68 pixels. Absolute errors remain substantial.
These are inferred source landmarks, not direct measurements or an independent
episode generalization test. No source image is used in this trajectory solve.

## Fresh native result: rejected

Right command frames are regenerated from the actual URDF before separately
freezing and compiling the new reference. The unchanged full 151-knot native
gate finds **five hand slots over 1 mm**, with worst penetration **6.302 mm**.
Failures occur at frames 35, 36, 37, 47 and 55, on the right index links.
Object/support penetration remains zero. The motion pass therefore does not
qualify this reference for training.

At frame 50, whole-digit separation lower bounds improve substantially, but
the thumb still sits 27.46 mm from the spoon. The index can overlap it. A zero
separation lower bound is not evidence of a stable grip.

## Finger reachability and the missing pinch

One-frame follow-ups keep actual URDF joint limits and use geometry-only
objectives. They never fit to held-out source image/depth values. Subsequent
image comparisons retain their inferred and nonmetric status.

1. A fixed-wrist search reduces the thumb distance upper bound to about
   2.6 mm, while the middle finger's bound remains over 4 mm. It is not
   accepted. The first global
   search exhausts its iteration budget; that failure is retained even though
   its subsequent Powell stage returns successfully.
2. A coupled proposal allows at most 8 mm per root translation component and
   0.08 rad per rotation component. Its selected translation changes by
   approximately 4.95 mm. All three distal surface-distance upper bounds fall
   below 0.006 mm at frame 50, with no object penetration. This narrow geometric
   improvement does **not** establish contact or force closure.
3. A separate witness audit verifies external object surfaces, then finds
   thumb/index outward-normal dot product **+0.8202**: they approach nearly
   the same side. There are **zero native hand/object contact pairs** at this
   pose. A palm/thumb self-contact of about 0.250 mm is retained in the report.
4. An explicit opposed-site control selects a thumb site with normal dot
   product **−0.7581** relative to the index. Under the same wrist bounds, the
   two local searches leave the thumb **21.36 mm** from that facing site. The
   proposal is rejected. This limited search is not proof of global
   unreachability.

The reachability producers save the lowest finite objective seen during their
searches, which can be an intermediate point rather than the optimizer's
returned stationary point. Optimizer return success is reported separately
and does not qualify the saved candidate. No one-knot candidate has a full
trajectory, velocity, swept-contact or force certificate.

## Next work

Compare the retained single-frame RGB hand's inferred palm orientation with
the current object-relative wrist and the actual Dex3 palm/hinge convention.
Audit chirality, coordinate transforms and contact-site association before a
new source-supported orientation proposal. Reuse the saved geometry; do not
rerun completed inference or expand the RGB probe.

Then solve wrist orientation and object-local opposing contact sites together
over time, with actual URDF joint/velocity limits. Every changed reference
needs fresh source and motion checks, the **unchanged full 1 mm native gate**,
swept checks and a grasp assessment before PPO. Establish a reference that
forms an opposing grasp before investing in additional PPO.

All earlier source-fit failures, including 31/32 held-out skin-ray coverage,
remain unchanged. No whole source hand or complete physical task is qualified.
All selected baseline policies still have MP-SR zero. Independent teammate
reproduction and a hidden leaderboard score remain unverified.

Exact experimental code and restoration instructions are archived under
`experiments/scored_baseline_20261019/wrist_retarget/`. It is not enabled as a
default preprocessing policy. Brev stayed stopped; these studies used no GPU
job or PPO step.
