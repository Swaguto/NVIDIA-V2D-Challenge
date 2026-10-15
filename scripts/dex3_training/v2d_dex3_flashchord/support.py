"""Preserve a finite triangle support footprint as thin static convex prisms."""

import numpy as np
import newton
import warp as wp
from pxr import Usd


def has_mesh_support(path):
    stage = Usd.Stage.Open(str(path))
    return stage is not None and any(
        p.GetTypeName() == "Mesh" for p in stage.Traverse()
    )


def add_mesh_support(builder, path, thickness_m=0.01):
    from flash_chord.scene.support import SupportBinding
    from flash_chord.scene.collision import ShapeSpan

    stage = Usd.Stage.Open(str(path))
    start_q, start_dof, start_shape = (
        builder.joint_coord_count,
        builder.joint_dof_count,
        builder.shape_count,
    )
    body = builder.add_body(
        xform=wp.transform_identity(), is_kinematic=True, label="support"
    )
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        points = np.asarray(prim.GetAttribute("points").Get(), dtype=np.float64)
        counts = np.asarray(prim.GetAttribute("faceVertexCounts").Get())
        indices = np.asarray(prim.GetAttribute("faceVertexIndices").Get())
        if (
            not len(points)
            or not len(counts)
            or not (counts == 3).all()
            or not np.isfinite(points).all()
        ):
            raise ValueError("Support must contain finite source triangles")
        for index, triangle in enumerate(indices.reshape(-1, 3)):
            top = points[triangle]
            if np.linalg.norm(np.cross(top[1] - top[0], top[2] - top[0])) < 1e-12:
                raise ValueError("Degenerate support triangle")
            vertices = np.concatenate((top, top - [0, 0, thickness_m])).astype(
                np.float32
            )
            faces = np.array(
                [
                    0,
                    1,
                    2,
                    3,
                    5,
                    4,
                    0,
                    3,
                    4,
                    0,
                    4,
                    1,
                    1,
                    4,
                    5,
                    1,
                    5,
                    2,
                    2,
                    5,
                    3,
                    2,
                    3,
                    0,
                ],
                dtype=np.int32,
            )
            builder.add_shape_mesh(
                body=body,
                mesh=newton.Mesh(vertices, faces),
                label=f"support_{prim.GetName()}_{index}",
            )
    if builder.shape_count == start_shape:
        raise ValueError("No finite support prisms")
    return SupportBinding(
        body_id=body,
        free_q_ids=tuple(range(start_q, start_q + 7)),
        free_dof_ids=tuple(range(start_dof, start_dof + 6)),
        shapes=ShapeSpan(start_shape, builder.shape_count),
    )
