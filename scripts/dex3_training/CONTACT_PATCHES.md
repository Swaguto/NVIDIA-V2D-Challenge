# Object-local contact hypotheses

`v2d_dex3_flashchord.contact_patches` separates a surface region on an object
from a hypothesized robot contact mode. A mode can require two or three Dex3
digits. Missing human landmarks do not imply a mandatory third robot contact.

`ObjectContactPatch` binds finite object-local triangles and explicit outward
normals to an object ID and geometry SHA-256. The producer must establish that
these triangles are external surfaces of the intended collision union. Queries
return the closest point on the provided triangles, with its facet normal;
this is a geometric witness, not an observed or active native contact.

`RobotContactMode.validate_patch_bindings` rejects missing, wrong-object and
stale-geometry patches. Its result explicitly does not certify physical success.
Pose queries require a proper rigid transform; reflections and scale fail.

## Use compiled collision geometry

For the pinned MuJoCo 3.8.1 runtime, `native_mesh.compiled_mesh_vertices` reads
the compiled convex graph's vertex-global-ID table. With no graph, it returns
all compiled mesh vertices, matching the native exhaustive support search.
Unknown versions and malformed storage fail closed. The returned vertices are
in compiled geom-local coordinates; use `data.geom_xpos/xmat` to place them.

**Do not rebuild collision hulls from every `mesh_vert` entry.** A limited native
convex graph can exclude raw mesh protrusions. Those protrusions can produce
false near-contact witnesses even when native geometry remains separated. This
reader changes the diagnostic representation, not the native model or assets.
The layout is bound to the pinned [native support implementation](https://github.com/google-deepmind/mujoco/blob/3.8.1/src/engine/engine_collision_convex.c)
and [graph producer](https://github.com/google-deepmind/mujoco/blob/3.8.1/src/user/user_mesh.cc).

## Required checks after proposing a mode

- Preserve source uncertainty, visibility and held-out failures.
- Realize the mode with the actual URDF and original root/finger motion limits.
- Check actual native contacts, including unmatched contacts; proximity alone
  does not establish contact or an opposing grasp.
- Evaluate required object wrenches and pose-bound actuator efforts with
  `contact_load_feasibility`. Solved forces are not measured forces, and outer
  friction feasibility is not a dynamic or force-closure certificate.
- Any changed reference needs fresh source, full 1 mm native, swept, motor and
  grasp checks before GPU validation or learning.

The e037 investigations remain rejected controls. The compiled-vertex correction
closes seven saved native support gaps numerically, but the corrected two-digit
trial still has no thumb contact and fails both required load checks.

## Common-root facet seeds

`contact_translation.common_translation_contact_seed` solves a geometric LP for
two or three fixed hand/object facet pairs. It chooses barycentric points on
both facets and one shared bounded root translation. Finger configurations and
root orientation remain fixed; separate roots per digit are not allowed.
An independent residual check verifies the returned point closures and bounds.

Optional translation halfspaces can carry conservative whole-hull separation
constraints. This prevents a chosen facet from meeting while another part of
the same hull penetrates. The producer must derive and bind these planes to the
actual compiled geometry and concrete pose. Fixed-plane infeasibility does not
prove global workspace infeasibility. Neither facet closure nor these planes
certify active contacts, forces, source accuracy or grasp success.

The finite e037 atlas and all its candidates are retained failures. The earlier
native hand/object/support audit explicitly reports that hand self-contacts were
not assessed. A new self-contact audit is required before treating a reference
as ready for physical validation; a passing hand/object gate alone is insufficient.
