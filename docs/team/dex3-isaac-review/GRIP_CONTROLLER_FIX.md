# Dex3 grip force and wrist payload correction

## Result

Two e002 right-hand lid grasps now pass supported closure, removal of support,
and a separate 5 cm lift using a dynamically controlled floating wrist.
The same geometry and finger targets with zero squeeze effort drop the lid.
The acceptance limits were fixed before the lift run and remain unchanged.

| Lid grasp | Actual lift | Maximum object drift relative to wrist | Maximum wrist position error | Maximum wrist rotation error | Lift gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Source frame 47 | 49.06 mm | 3.77 mm | 8.36 mm | 1.10° | Pass |
| Source frame 236 | 48.48 mm | 6.02 mm | 8.11 mm | 1.08° | Pass |

The preceding fixed-wrist acquisition comparison also passes both grasps:
3.95 mm / 5.34 mm maximum displacement over one unsupported second, versus
approximately 4.94 m drops for both position-only controls. The four panels
alternate position-only and squeeze control for the two grasps.

This is a verified correction for these two development fixtures. The complete
episode, left pot grasp and learned policy are **not accepted**. The earlier
PPO checkpoint has not been retrained with this controller, and its recorded
0% window success remains unchanged. Hidden-episode generalization is unmeasured.

## Why the previous attempts failed

1. Opposing contact geometry supplied too little squeeze. With the existing
   2 Nm/rad joint stiffness and 0.75 mm geometric preload, a local Jacobian
   calculation predicts about 0.28–0.30 N at the index pad. Merely reaching that
   position is insufficient to generate the friction needed to carry the
   nominal 0.3 kg lid.
2. An outward fingertip point target could leave another part of the thumb mesh
   touching the handle. Opening now constrains the complete cooked pad geometry
   and independently checks solid separation. The four initial distal-pad gaps
   are 2.00–3.28 mm. Joint limits and bounds are enforced.
3. The floating wrist needs payload compensation. The first lift held both lids
   but one wrist missed the 5° rotation gate at 5.71°. Adding the payload's lever
   arm moment reduced the maximum error to approximately 1.1°, with unchanged
   gains, force/torque caps and acceptance limits.

## Implementation and evidence

- `scripts/dex3_physics/grip_force.py` maps a 6 N opposing pad-force command to
  real finger efforts using the Jacobian transpose. Feedforward is capped at
  0.55 Nm. The original combined simulator motor limit remains 0.76 Nm.
- `--stage grasp-acquisition` compares identical open/closed poses and wrist
  geometry when a pregrasp is supplied. Only squeeze effort differs between
  its paired cases. Fingers close for 0.25 s; effort ramps during the next
  0.25 s; all temporary trays are disabled and hidden at 0.5 s.
- `--stage grip-lift` uses floating wrists with the existing implicit controller,
  lifts 5 cm from 0.5 to 1.0 s, then holds until 1.5 s. Wrist force and torque
  limits remain 60 N / 60 Nm. Compensation includes the nominal payload weight
  and its moment about the actual wrist COM. It requires measured thumb and
  index contacts of at least 0.5 N each, with a 30 ms hold after contact loss.
- Objects receive no external wrench, attachment or pose writes after
  initialization. Wrist and finger actuation transmit forces through contact.
- Independent free-fall, permanent support and support-removal controls pass.
  All 604 fixture states, 353 control states and 31 video frames are retained
  in each acquisition/lift run. H.264 derivatives preserve every timestamp.
- The final lift reaches approximately 0.42 Nm maximum estimated finger effort,
  10.7 N maximum commanded wrist force component and 0.32 Nm wrist torque
  component. The source motor and wrist limits remain unchanged.

The solver's contact buffer at time zero can retain contacts from SDK startup,
before the explicit initial-state writes. That raw snapshot is preserved and
marked invalid for the current pose in new records. Numerical pose checks and
independent cooked-mesh separation establish the initial geometry. Executed-step
peak solver penetration in the final lift is approximately 1.1 mm / 0.9 mm;
this is not a zero-penetration or continuous-clearance certificate.

`grip-controller-fix-v1.json` records every attempt, including the failed first
lift, source revisions, video/state hashes, raw startup readings and measured
outcomes. The controller source for the final repeat is commit
`f67b2db`; it runs on the previously published private Isaac image by binding
the current checkout read-only. The training image is a separate runtime.

The left pot reference is still unresolved: its first source pad directions
fail the opposed-contact check, and the bounded geometric search found no valid
candidate for either selected pot grasp. Applying the lid's torque pattern to
that reference is not justified.

## Team reproduction

Obtain `dex3-grip-fix-inputs-local-v1.tar.gz` from the owner through an approved
private channel. It remains local and has not been published to GitHub or GHCR.
Verify its size and SHA-256 against `local-grip-input-bundle-receipt-v1.json`,
then extract into a fresh directory. Its manifest binds the frozen source archive
and two private grasp proposals. No MANO weights or credentials are included.

Use Linux amd64, NVIDIA Container Toolkit and the GPU/driver requirements in the
existing container documentation. From a checkout containing the controller fix:

```bash
export V2D_IMAGE=ghcr.io/swaguto/nvidia-v2d-challenge-dex3-review@sha256:8585588357d898929bf6f80f0011441cbcab07a3d95e2f5f4d0e24bc5c177190
export V2D_CODE_DIR=/absolute/path/to/NVIDIA-V2D-Challenge
export V2D_DATA_DIR=/private/extracted-grip-inputs
export V2D_PHYSICS_ARCHIVE="$V2D_DATA_DIR/input-pack.tar.gz"
export V2D_GRIP_SEED="$V2D_DATA_DIR/grip-effort-6N-open4mm.json"
export V2D_OUTPUT_DIR=/private/fresh-grip-results
mkdir -p "$V2D_OUTPUT_DIR"
# Set ACCEPT_EULA=Y and OMNI_KIT_ACCEPT_EULA=YES after accepting the NVIDIA license.
docker compose -f docker/dex3-review/compose.yaml \
  -f docker/dex3-review/physics.compose.yaml \
  -f docker/dex3-review/grip.compose.yaml run --rm physics \
  --stage grasp-acquisition --grasp-seed /physics-inputs/grasp-seeds.json
docker compose -f docker/dex3-review/compose.yaml \
  -f docker/dex3-review/physics.compose.yaml \
  -f docker/dex3-review/grip.compose.yaml run --rm physics \
  --stage grip-lift --grasp-seed /physics-inputs/grasp-seeds.json
```

Both stages use separate child directories and refuse to overwrite earlier runs.
All inputs and source code are mounted read-only. Results include the invocation,
physics states, gate, SDK logs, original video and browser-preview receipt.
The Compose configuration has been checked against the published digest and
contains no image build. Back up results and stop the GPU VM when it is idle.

To regenerate the private force proposal from the saved cooked closed-grasp run:

```bash
python scripts/dex3_physics/grip_force.py \
  --run /private/interior-static/grasp-seed/run/simulator \
  --urdfs /private/robot/urdfs/g1 \
  --seed /private/interior-grasp-seeds.json \
  --pad-force-N 6 --pregrasp-open-mm 4 --out /private/new-grip-proposal.json
```

This generator preserves observations and source object poses. Its output is an
experimental robot command requiring a new physics check for each new case.
