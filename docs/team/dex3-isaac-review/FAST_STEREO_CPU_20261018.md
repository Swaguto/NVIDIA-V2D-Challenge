# CPU stereo and spoon geometry investigation — 18 October 2026

A smaller pretrained model lets the source-depth investigation proceed on the
Mac while Brev's start/stop API is unavailable. This is an **e037 development
experiment**, not a trained policy, accepted full trajectory or leaderboard
result. The selected baseline and reserved episodes are preserved.

## Reproducible inference contract

- Model: [NVIDIA C-Fast-FoundationStereo](https://huggingface.co/nvidia/c-fast-foundationstereo),
  revision `53a84e28a368f37e699776250215fcb450d6a109`.
- File: `onnx/c_fast_foundationstereo_bp2_dynbatch_dynhw.onnx`, 103,228,473 bytes.
  SHA256: `2c79bbb274dc0aef687a8732b987077e9e3aa26ff9dd653141cda48c57dad5d3`.
- The [published ONNX contract](https://huggingface.co/nvidia/c-fast-foundationstereo/discussions/2)
  specifies RGB float32 tensors, ImageNet normalization outside the graph,
  dimensions divisible by 32, and disparity in input-image pixels.
- Actual run: ONNX Runtime 1.31.0 CPU provider, NumPy 2.2.6, two CPU threads,
  Python 3.12.13 on Mac ARM. The private dependency folder adds FlatBuffers
  25.12.19 and Protobuf 7.36.2 without replacing the existing CPU environment.
- Source images are the same calibrated rectified pairs used by the retained
  SGBM control. Inference uses 320×512; output disparity is divided by the actual
  horizontal resize factor 0.4 before metric backprojection. Reverse mirrored
  inference supplies a fixed one-source-pixel cycle check. Failed checks remain
  missing estimates.
- A synthetic 16-pixel shift produced median disparity 16.0117 pixels and
  median absolute error 0.0117 pixels. This checks the implementation's units
  and input convention; it does not measure real-episode depth accuracy.

The initial frame-125 run used about 1.05 GB peak resident memory and roughly
nine seconds for its two inference passes. The other six fixed sample pairs
completed under an independent CPU guard in 93.3 seconds, reusing the verified
frame-125 output. No images were uploaded and no GPU session was started.
Weights, source images and depth arrays remain private and outside Git.

## Seven-frame evidence and its limitations

The fixed sample axis is 0, 25, 50, 75, 100, 125, 150. Nearby table measurements
were sufficient at six frames; frame 50 has no qualifying nearby table fit.

- On those six frames, the fraction of eroded foreground depth estimates below
  the locally fitted table dropped to zero. At frame 125 it was previously
  68.03%. Held-out nearby-background median plane residuals decreased from
  2.66–3.22 mm to 0.476–0.969 mm. Masks, cameras and original support were fixed.
- The independent RGB comparison is mixed: on identical textured object pixels,
  the new model's median census matching cost is worse at six frames and tied
  at one. It is **not** justified to infer ground-truth depth accuracy from the
  smoother plane alone. The full photometric comparison is retained.
- The same retained six-DoF fitter was tested on the same three diagnostic
  frames and two geometry arms. With the learned depth, five of six arms
  qualified; the original-size frame-125 arm still failed. The 1.3 scale arm
  then qualified on all seven samples. No acceptance threshold was relaxed.
- The unchanged depth-support/jump rejection logic accepts these seven scaled
  poses against both depth estimates. The fuller SGBM source-fit score improves
  at six frames but regresses at frame 125, from 0.13166 to 0.10436. The learned
  source-fit score improves there from 0.24650 to 0.32937. These are conflicting
  estimated-depth hypotheses, not hidden pose labels or proven accuracy.

## Fresh native geometry checks

The first scaled import correctly failed because the Mac lacked CoACD. The
upstream importer incorporates scale into its cache key, so the old cache could
not be reused as proposed. That failed producer and output are preserved.

The repair installs the pinned CoACD 1.0.11 CPU package in a separate folder and
uses the **unchanged upstream collision-cache producer** to create a new cache.
The new nine convex parts match the actual Newton body-local hulls after fresh
MuJoCo import within **1.84e-9 m**. The effective evaluation OBJ uses the same
1.3 scale as the native URDF. Body origin and support stay fixed. Nominal mass
is unchanged; inertia is scaled by 1.3². These remain nominal assumptions.

| Native sample check | Result |
| --- | --- |
| Object/support penetration at seven fitted knots | **0 m at all seven** |
| Complete hand/object/support contact sample check | 6 / 7 pass the existing 1 mm threshold |
| Remaining sample failure | Frame 100, right hand/spoon: **4.649 mm** |
| Hand FK and frozen input closures | Pass |
| Full 151-knot gate, forces, swept contacts, dynamic rollout | Not completed for this geometry |

The previous hand-only repair must be checked again after changing geometry and
object targets. It is not a contact certificate for this new candidate.

## Next work

Full-clock CPU depth inference has started in a separate private job, with a
45-minute parent cap, 40-minute worker cap, 3 GiB RSS cap and 1.5 GiB disk
reserve. It reuses the seven outputs and verifies their source pixels against
the sequential video decode. Its completion must be established from the live
guard and depth report; this document does not claim it has finished.

After that: fit and check the complete object trajectory, repair the hand
commands against the new geometry, and run the unchanged full 1 mm native gate,
source-fit and motion checks. No PPO should start until those checks qualify.
Brev remains stopped; its shutdown-control failure still blocks paid GPU work.

See the [measurement summary](e037-fast-stereo-cpu-summary-20261018.json).
