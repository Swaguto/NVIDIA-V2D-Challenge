# e002 pot grasp correction

This is the historical separate-effort checkpoint. Its simulator outcomes are
preserved, but its combined motor budget was not certified: later probes found
that separate feedforward bypassed the native drive cap. See the
[current measured repair](motor-contact-repair-results-v1.json) for the shared
drive command and fresh lift validation.

Both left/pot source phases pass isolated supported acquisition and a floating
wrist lift. This is a development diagnostic; complete episode and learned policy
acceptance remain false. The earlier 0% PPO result is unchanged.

| Source frame | Unsupported acquisition max drift | Actual lift | Max relative slip | Max wrist position error | Max wrist rotation error |
| --- | ---: | ---: | ---: | ---: | ---: |
| 47 | 3.44 mm | 51.02 mm | 1.67 mm | 8.10 mm | 1.11° |
| 273 | 2.05 mm | 51.43 mm | 1.55 mm | 8.76 mm | 1.18° |

## Mechanism and limits

Two opposed pads alone could not resist the pot's gravity moment about its
handle. Dense exposed-surface samples and whole-hand collision checks produce a
lower-lever grasp. A third, upper handle contact from the middle finger supplies
the counter moment. A bounded point-contact linear program balances nominal
0.3 kg gravity and moment with conservative friction coefficient 1, contact normal
forces between 2 and 10 N, and motor feedforward at most 0.55 Nm. The command is
mapped through each contact Jacobian transpose.

The tested finger stiffness is 8 Nm/rad; original damping 0.2 and declared drive
cap 0.76 Nm remain unchanged. The maximum clipped motor estimate is 0.625 Nm in
acquisition and 0.579 Nm in lift; actual combined torque was not measured.
Floating wrist payload compensation requires
measured contact on thumb, index and middle, allowing a 30 ms dropout hold.
The wrist caps remain 60 N / 60 Nm. Objects receive no external wrench, weld,
attachment or pose writes after initialization.

Paired position-only controls use the same geometry, target poses and stiffness;
both drop approximately 4.93 m. Independently captured cooked convex pieces match
between each paired hand/object to less than 1 micrometre. Gravity, permanent
support and support removal controls pass. Each run preserves all 604 fixture
states, 353 control states and 31 video frames.

This grasp requires substantial robot retargeting: about 15.2 cm vertical wrist
correction and about 140 degrees of rotation for the first phase. Human
observations and masks are unchanged. A finite 11-configuration open-to-closed
check allows up to 1.5 mm piece overlap; one initial non-pad link has only 46
micrometres of axis separation. Full approach clearance, full task dynamics,
continuous collision freedom and hidden episode performance are not established.

## Preparation and exact evidence

- `antipodal_seed.py`: dense exposed samples, load-aware candidate selection,
  bounded root/finger fit and independent cooked-piece collision checks.
- `support_finger.py`: searches and refines a third contact and checks nominal
  force/moment feasibility under motor caps.
- `object_grasp.py`: rigid object-relative reuse across source phases, bounded
  small seating correction, independent destination collision checks.
- `grip_force.py`: load-balanced effort planning, whole-pad pregrasp separation,
  declared stiffness and required contact digits.

The exact executed source overlay archive is SHA-256
`637a3a6bb2d7344d7cad4770602966586f931eb197befa97a5f9553e77a7fedc`.
It is an uncommitted candidate based on `6736901`; the receipt records this
explicitly. Subsequent source-phase completion changes affect preparation only.
The unchanged Isaac base image is pinned by digest in `GRIP_CONTROLLER_FIX.md`.

`pot-grip-result-v1.json` contains measured summaries, file hashes, paired cooked
geometry checks, gain readbacks and lifecycle receipt. Raw seeds, source archives,
geometry and videos remain local at the owner's `dex3_pot_grasp_20261015` folder.
They have not been published to GitHub/GHCR. Use the existing private input pack
and `grasp-acquisition` / `grip-lift` container stages with the exact tested
`pot-three-pad-effort-v3.json` seed; no new image is required.

Failed attempts are retained locally: two-pad normal-force increases to 8/10 N
and stiffness 8 alone still left 15–21 mm drift. These were not promoted. The
third-contact acquisition (`pot-acquisition-v7`) and lift (`pot-lift-v3`) are the
accepted isolated fixtures. Brev is confirmed stopped after all 16,381 output
files (4,743,522,394 bytes) were backed up with matching SHA-256 hashes.
Conservative total compute bound is $8.8173 of the approved $10; this is not an
invoice.
