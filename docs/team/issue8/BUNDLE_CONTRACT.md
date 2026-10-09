# Track 3 episode bundle pilot contract

Schema `v2d.track3.episode.v1` connects the accepted Issue 7 scene exporter to the
Issue 8 Dex3 consumer. It is the initial producer/consumer proposal for Issue 12;
all-episode generation and a training-loader integration remain future work.

An episode is a directory containing `manifest.json`, `episode.npz`, the frozen
`source/` records and `validation.json`. The manifest declares units, frames,
FPS, hand and joint ordering, object provenance, and SHA256 values. The consumer
verifies every declared source and payload before solving IK. Portable artifacts
live outside the tracked source checkout because they contain private geometry.

## Coordinates and time

Positions use metres and column-vector transforms. `camera_c2w` maps the rectified
OpenCV camera (X right, Y down, Z forward) to the Issue 7 reference-camera gauge.
Object and hand outputs use that same world. Object uniform scale is separate
from the proper rigid transform. The bundle retains the camera intrinsics and
image size. An Isaac scene must explicitly declare any additional world-to-stage
transform; the world should not be silently interpreted as gravity-aligned.

Source frame indices are complete and consecutive. Timestamps are source frame
index divided by the declared FPS. Episode 2 contains frames 0–342 at 20 Hz.

## Hands and unknowns

The hand axis is `[left, right]`. The 21 joint names are wrist, then thumb,
index, middle, ring and pinky, with four joints per finger. The geometry arrays
retain world joints and vertices, wrist transforms, faces, validity and direct
observation masks. Invalid geometry is NaN, not zero or an interpolated pose.
The consumer resets a hand's warm start across each missing interval.

Wrist orientation comes from the frozen MANO global orientation, with the same
camera-X mirror conjugation used by refinement for the left canonical hand,
followed by camera-to-world rotation. Fingertip orientations are unknown and
are not optimized. The retarget adapter accounts for the difference between this
mirrored-left convention and CHORD's default left-MANO palm convention; other
dataset retargeters remain unchanged.

All fingertip contact probabilities and world wrenches are explicitly unknown in
this pilot. Coarse source-reviewed hand–object labels cannot supply per-fingertip
contact probabilities or measured forces. Wrench order is force XYZ in newtons,
then torque XYZ in newton-metres, with its own validity mask.

## Objects and provenance

The bundle contains canonical triangle geometry, per-frame object-to-world poses
and positive uniform scales. It preserves the upstream observed/inferred pose
labels. Episode 2 uses provided pot/lid CAD for development; the manifest marks
these meshes ineligible as participants' reconstructed submission meshes.

The source directory freezes the scene, completed run receipt, input manifest,
all camera/object/hand exports, original input locks, and the independent lock
verification receipt. The exporter refuses to replace an existing output.

## Robot consumer

`scripts/retarget/track3_to_dex3.py` uses the existing CHORD Pink/DAQP differential
IK. The output contract `v2d.track3.dex3.v1` declares each side's joint order,
position and velocity limits, FK frame names, input hashes and runtime versions.
Its floating-base quaternion is **xyzw**; FK frame poses are **wxyz**. Conversion
to simulator root state must respect this difference.

The final finger configuration between adjacent supported source frames is
bounded by the previous configuration ± the URDF velocity limit times source
dt. Internal IK iteration limits alone do not prove a physically feasible
source-frame velocity. The output reports tracking errors and source-frame
velocities, with missing robot configurations remaining NaN.
