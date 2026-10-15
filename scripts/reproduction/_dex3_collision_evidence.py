"""Read cooked convexes and solver contact separations without changing the scene."""

from pathlib import Path


def capture_cooked_colliders(stage, assets, write, destination: Path):
    import numpy as np
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from pxr import PhysicsSchemaTools, Usd, UsdGeom, UsdPhysics, UsdUtils

    cooking = get_physx_cooking_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    cache = UsdGeom.XformCache()
    records = []
    for label, asset in assets:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(asset.cfg.prim_path)):
            if prim.GetTypeName() != "Mesh" or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            body = prim
            while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
                body = body.GetParent()
            if not body:
                raise ValueError("Collider has no rigid-body ancestor")
            result_data = []

            def callback(result, convexes, result_data=result_data, prim=prim):
                if result != PhysxCollisionRepresentationResult.RESULT_VALID:
                    raise ValueError(
                        "Cooked collision representation unavailable: "
                        + str(prim.GetPath())
                    )
                for convex in convexes:
                    result_data.append(
                        {
                            "vertices_mesh_local_m": [
                                [v.x, v.y, v.z] for v in convex.vertices
                            ],
                            "indices": list(convex.indices),
                            "polygons": [
                                {
                                    "index_base": polygon.index_base,
                                    "num_vertices": polygon.num_vertices,
                                    "plane": [
                                        polygon.plane.x,
                                        polygon.plane.y,
                                        polygon.plane.z,
                                        polygon.plane.w,
                                    ],
                                }
                                for polygon in convex.polygons
                            ],
                        }
                    )

            cooking.request_convex_collision_representation(
                stage_id=stage_id,
                collision_prim_id=PhysicsSchemaTools.sdfPathToInt(str(prim.GetPath())),
                run_asynchronously=False,
                on_result=callback,
            )
            if not result_data:
                raise ValueError(
                    "No cooked convex shapes returned: " + str(prim.GetPath())
                )
            # The SDK does not promise a stable convex ordering.
            result_data.sort(
                key=lambda item: (
                    len(item["indices"]),
                    len(item["vertices_mesh_local_m"]),
                    item["vertices_mesh_local_m"][0],
                )
            )
            transform = (
                cache.GetLocalToWorldTransform(prim)
                * cache.GetLocalToWorldTransform(body).GetInverse()
            )
            records.append(
                {
                    "asset": label,
                    "prim": str(prim.GetPath()),
                    "body_prim": str(body.GetPath()),
                    "body_name": body.GetName(),
                    "mesh_to_body_matrix_row_convention": np.asarray(
                        transform
                    ).tolist(),
                    "approximation": UsdPhysics.MeshCollisionAPI(prim)
                    .GetApproximationAttr()
                    .Get(),
                    "convex_count": len(result_data),
                    "convexes": result_data,
                }
            )
    write(
        destination,
        {"schema": "v2d.cooked-collision-evidence.v1", "colliders": records},
    )
    return [
        {
            "asset": row["asset"],
            "prim": row["prim"],
            "convex_count": row["convex_count"],
        }
        for row in records
    ]


def contact_separation_summary(sensor, dt):
    import numpy as np

    _, _, _, separation, counts, starts = sensor.get_contact_data(dt=dt)
    separation = separation.numpy().reshape(-1)
    counts, starts = counts.numpy().reshape(-1), starts.numpy().reshape(-1)
    values = [
        separation[int(start) : int(start + count)]
        for count, start in zip(counts, starts, strict=True)
        if count
    ]
    values = np.concatenate(values) if values else np.empty(0)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite solver contact separation")
    return {
        "contact_points": len(values),
        "maximum_solver_penetration_m": float(max(0.0, -values.min()))
        if len(values)
        else 0.0,
    }
