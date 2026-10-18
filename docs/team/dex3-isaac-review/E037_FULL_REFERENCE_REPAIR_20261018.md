# e037 complete-reference repair investigation

This is a private-input CPU development experiment. No new policy, accepted
physical task or hidden leaderboard improvement is established. Original
baseline artifacts, reserved episodes and every rejected control are retained.

## Completed controls

| Check | Result |
| --- | --- |
| Original full object-pose fit with learned depth | 140/151 qualify; 11 rejected |
| Fresh full native audit of that fit | 6 object/support knots and 20 hand/frame slots exceed 1 mm |
| Worst native penetration | Object 13.177 mm; hand 8.963 mm |
| Original → fitted peak angular speed | 5.044 → 7.126 rad/s |
| Original → fitted peak linear acceleration | 15.476 → 23.125 m/s² |
| Finite source-neighbourhood repair | Recovers 9 rejected pools; frames 126/128 remain empty |

The initial complete fit is rejected for source, contact and motion reasons.
Completing a process does not qualify its output. The unchanged motion selector
correctly refuses an empty source-qualified pool, without inserting a pose.

## Background-plane evidence

Source images 124–130 show the spoon on the tabletop with the hands clear. No
gross mask failure is apparent in this window. The estimated background surface
has a 7–12 mm median discrepancy from the existing support. Independent spatial
holdout residuals of local plane fits are roughly 0.7–1.3 mm. These measurements
come from the new estimated depth; depth bias and camera drift are not isolated.

A two-knot normal/height alignment makes the previously rejected object fits at
126/128 qualify and clears their native object/support contacts. It changes the
inferred camera centre by 7.9/6.5 mm and preserves source hand projections within
2.6e-13 pixels. The full sample gate still rejects left-hand/support penetration
of **1.059/3.034 mm**. No threshold was loosened.

## Full-clock alignment hypothesis

- Background-plane observations qualify at **137/151 frames**. Frames 45–58
  remain explicitly unsupported or inferred. Original source confidence and
  contact/wrench flags are not upgraded.
- Directly observed hand projections are excluded from background fitting.
  These exclusions are derived regions, not measured hand masks. Background
  membership remains uncertain.
- The first smooth alignment fails held-out source checks at 67/113/115.
  Its output remains rejected.
- The second version adds training-inlier centroid constraints at **all 137
  supported frames**. The held-out and correction-motion gates stay unchanged.
  All 137 spatial holdouts then pass. Held-out points never enter optimization.
- Maximum proposed camera correction is **12.798 mm** normal displacement and
  **0.028964 rad** rotation. Adjacent normal corrections stay within 1.5 mm.
  No tangential camera translation or yaw measurement is inferred from a plane.
  Source hand reprojection changes by at most **5.1e-13 pixels** when all world
  references transform consistently.
- Most camera motion statistics improve; peak angular speed changes slightly
  from 0.206845 to 0.208049 rad/s. This is not a full object/robot motion
  certificate. The object-path motion caps remain those of the untouched
  original trajectory.

The original support, source files and scaled object geometry are unchanged.
The alignment is a separate inferred-world hypothesis, not ground-truth camera
recovery. The earlier conflicting RGB matching costs and frame-125 SGBM fit
regression remain relevant; see [the depth investigation](FAST_STEREO_CPU_20261018.md).

## Source-qualified candidates and temporal selection

Rechecking the previous pools under the new alignment reuses 130 frames and
refits 21. Frame 93 initially remains empty. A bounded finer search, including
view-ray proposals that preserve the projected object centre, produces 754
qualifying alternatives at 93. All 151 source-frame pools are then nonempty.

The new portable `temporal_pose_path.py` uses second-order dynamic programming
with linear/angular speed and acceleration limits at the exact timestamps. Its
five focused CPU tests pass. It does not itself certify source fit or contact.

The first aligned temporal selection fails at frame 62. Source-validated local
pose additions resolve that bottleneck; the next selection fails at 79. A
bounded adaptive expansion resolves 79, then the next selection fails at 95.
A final bounded expansion at 94–96 then supplies a complete path. All **151
chosen poses independently pass the original source-fit checks**, and the
complete trajectory passes the unchanged original motion maxima and p95 speed
caps. These are reference checks, not successful physical rollout segments.

The local expansion first exposed invalid-matmul warnings on this Mac. That
worker was stopped and its partial output retained. The replacement uses
explicit finite-checked contractions. Completed additions at 60–63 were
independently revalidated before use; the intentionally stopped larger search
is not labelled complete. No connection to the separate GPU replay variability
has been established.

## Complete object source and motion result

| Object motion statistic | Original | New inferred path |
| --- | ---: | ---: |
| Maximum linear speed, m/s | 0.802981 | 0.679760 |
| Maximum angular speed, rad/s | 5.043756 | 4.597923 |
| Maximum linear acceleration, m/s² | 15.476045 | 13.111465 |
| Maximum angular acceleration, rad/s² | 152.514879 | 135.818117 |
| p95 linear speed, m/s | 0.492505 | 0.445686 |
| p95 angular speed, rad/s | 3.569995 | 3.055094 |

Every selected pose is rechecked against its own source frame. Source flags,
full source clock and original correction bounds remain intact. Geometry is the
separately recorded 1.3-scale reconstruction hypothesis with its own nine-hull
cache; the support has not changed. This is not measured shape or depth truth.

## Fresh hand retarget and native audit

Applying the same inferred world alignment to both hands and their source
comparisons clears all object/support reference intersections, but leaves
**51 hand/frame failures**, up to 11.430 mm, with the previous repaired hands.
The aligned original unrepaired robot reference has 75 failures. Earlier hand
certificates therefore do not transfer.

A new articulated temporal solver retains the original global correction,
per-point source projection, joint position and actual URDF velocity bounds.
It compares with the original robot reference transformed by the same world
alignment, avoiding accumulated morph allowances. The first cold start is
infeasible. A warm start then clears all 151 static knots and independently
passes all 904 directly observed source-point comparisons. Their absolute
errors remain about 15 px median; non-regression does not establish accurate
metric hand depth. Fourteen background planes remain explicitly inferred.

## Between-knot failure and repair

The first complete static pass still clips the right middle finger through the
spoon **between frames 67 and 68**. Independent linear position/finger and
shortest-arc quaternion interpolation at no more than 2 ms spacing finds
**11 failed samples, maximum 3.961 mm**, among 3,805 samples. This failed control
is retained. Source-knot contact checks alone were insufficient.

Two initial sweep-constrained SQP proposals are also rejected. Their first
steps introduce new knot failures and later convex subproblems are infeasible.
The accepted reference candidate adds a local proximal objective and bounded
SQP steps around the already source-qualified hand path. Source, native-contact,
absolute morph, temporal and URDF acceptance gates remain unchanged. Existing
conservative knot constraints are retained while interior constraints act on
the source endpoints. No intermediate source observations are invented.

A **new independent freeze and fresh native compiler** verify the final candidate:

| Check | Result |
| --- | --- |
| Object source fits and original motion caps | 151 / 151 pass |
| Source-knot object/support penetration >1 mm | 0 / 151 |
| Source-knot hand/frame penetration >1 mm | 0 / 302 |
| Native reported maximum source-knot penetration | 0 m |
| Hand source projections | All 904 points non-regressing; medians/p95 pass |
| Finger URDF position and velocity limits | All 14 joints pass |
| Original global and temporal morph bounds | Pass |
| Native hand FK and geometry buffers | Pass; all compared geometry arrays exact |
| Independent discrete interpolation samples | 3,805 / 3,805 clear; maximum spacing 2 ms |
| Source observations/confidence/contact labels changed | No |
| Physics steps, new training steps | 0 |
| Stable grasp, force closure, policy or leaderboard improvement | Unverified |

The final frozen manifest SHA256 is
`7e12fcd85eab8b54e1d694a95be7037c624a71bb827139e6a18f4ca4420bf650`.
The command candidate SHA256 is
`be8fdb232ab3cba7418ff127683efee625929446461b66783c97079cbb53c81f`.
Private arrays remain outside Git. This discrete geometric sweep does not prove
continuous collision avoidance or the motion of an actual controlled hand.
No force-bearing grasp is established by making reference meshes nonpenetrating.

## Later carry audit: grasp reference remains rejected

The static/discrete gate pass is preserved. A subsequent source-hand
audit **rejects this candidate as a carry/grasp reference**; do not start PPO
on it. See [the carry diagnosis](E037_CARRY_SOURCE_GAP_20261018.md).

After that upstream failure is repaired, a new frozen candidate must qualify
all original gates before a bounded GPU trial: fresh native/source
preflight, full unassisted zero-residual controller rollout, measured motor and
contact readbacks, then a short learning experiment only if the reference and
controller survive. Evaluate independent repeats against fixed inputs before
promoting any policy. New locally inferred references and geometry can change
local AUC itself, so comparing numbers across different references cannot prove
leaderboard improvement.

Normal Brev login and listing were restored with the owner's email. Provider
listing confirms the existing VM **STOPPED at 20:39:46 UTC**. STOP commands still
return a provider invalid-security-token error; the Brev console also displays
a service-disruption notice. No START or paid job was attempted after that
failure. Keep the VM stopped until shutdown control works. The cloud outage
cause is not proven from the banner alone.

The 1 mm gate, unassisted dynamics and independent evaluation repeats still
stand. All 20 selected policies retain MP-SR zero. No completed cohort, replay,
finalizer or renderer was rerun; no new policy or submission was promoted.
Private footage, arrays and weights remain outside Git. The source archive
contains 137 exact scripts: 135 executed and two prepared but unexecuted old
unaligned validators. All failed controls are preserved.
