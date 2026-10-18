# Numbered object mask ambiguity

The e025 source review found distinct board masks through most of the sequence,
followed by a fused mask at release. In frames **342–355 of 356**, more than 70%
of the smaller mask is shared with the other instance. The final reviewed frame
has 99% overlap relative to the smaller mask. A source-frame contact sheet was
visually inspected; one mask expands to include both assembled boards.

The original mask outputs and the first physics baseline remain preserved.
Neither automatic seeding nor a count of 356 masks certifies identity throughout
the video.

## Repair

For numbered same-category instances, the fresh mask pipeline now detects this
shared-foreground conflict and excludes **both ambiguous observations** from
pose fitting. It does not guess which object is in front. Raw ambiguous masks
are moved into `ambiguous-instance-masks/` inside the new output directory, with
their bytes retained. The report lists affected frames and effective mask
counts. Old numbered-mask caches without this policy marker are rejected by
the scored runner.

Source videos, calibrated depth, camera data, hand observations and the complete
source-frame clock remain unchanged. Later inferred object targets remain
explicitly inferred; an absent observation is not a new measured pose.

The 70% conflict rule is restricted to numbered instances. This is a conservative
quality rule, not proof of canonical identity or hidden-episode generalization.
Fresh e025 tracking/policy validation is still required. Unit tests and a CPU
probe of the actual masks verify conflict handling and raw-mask preservation;
they do not establish a successful physical grasp.
