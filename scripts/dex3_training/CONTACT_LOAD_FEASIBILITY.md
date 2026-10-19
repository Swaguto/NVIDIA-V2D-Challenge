# Contact-load feasibility

Use `v2d_dex3_flashchord.contact_load_feasibility` to reject a frozen set of
assumed active contacts that cannot balance a declared object wrench, even
under optimistic friction and unlimited normal force.

```python
from v2d_dex3_flashchord.contact_load_feasibility import (
    ContactGeometry, outer_contact_load_feasibility,
)

# Contact basis ROWS are normal into the object, tangent 1, tangent 2.
# The caller must establish which native geom is the object.
contact = ContactGeometry.from_mujoco_geometry(
    point_world_m=[0, 0, 0],
    frame_geom1_to_geom2_world=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    dimension=3,
    friction=[0.5, 0.5, 0, 0, 0],
    object_is_geom2=True,
)
result = outer_contact_load_feasibility(
    [contact],
    com_world_m=[0, 0, 0],
    required_wrench_world=[0, 0, 2.943, 0, 0, 0],
)
# This one-sided contact cannot balance a pure vertical wrench.
assert result["feasible_in_outer_model"] is False
```

## Contracts and interpretation

- Use one world frame, metres, newtons and newton-metres. Wrenches are ordered
  force then moment, about the supplied COM. For gravity support, request minus
  the gravity force. The caller is responsible for mass, inertia and motion
  provenance; the API does not infer them from source images.
- `ContactGeometry` requires a finite point, an orthonormal 3×3 basis, integer
  dimension 1/3/4/6 and five nonnegative friction coefficients. Either basis
  handedness is permitted after reversing the native object-side force axes.
  Friction order is sliding 1/2, spin, rolling 1/2. Sliding coefficients are
  dimensionless; spin/rolling coefficients have units of metres.
- The contact geometry is assumed active. Geometry alone does not establish
  contact activation or measured forces. Do not pass support contacts as hand
  contacts or inferred source contacts as measured truth.
- Independent component bounds form an **outer** friction box. Feasibility can
  include solutions outside the exact native cone. Total solved normal force
  is minimized without force budgets or motor caps. The result does not certify
  actual force production, force closure, grasp stability or dynamic success.
- Infeasibility concerns only the supplied frozen geometry, friction and load.
  It does not prove task impossibility or rule out different contacts.
- Unresolved solver results and failed independent residual checks raise
  `RuntimeError`. Preserve them as numerical failures. Invalid inputs raise
  `ValueError`. Do not reinterpret either as physical infeasibility.
- `moment_length_m` only conditions the moment equations. It changes no contact
  geometry or gate. Equilibrium and friction residuals are independently checked
  in SI units with absolute numerical tolerance 1e-7 (equilibrium also allows
  1e-9 times each required wrench component).

Use the separate `wrench_capacity.capacity` API when measured normal-force
budgets are available. Do not substitute solved forces for measured budgets.

Before learning from any proposed reference, require source fit, actual URDF
and motion limits, the full 1 mm native reference-contact gate, swept checks,
actual motor feasibility and grasp assessment. This diagnostic alone satisfies
none of those qualification steps.
