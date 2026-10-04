# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parse a glTF binary (.glb) mesh with only stdlib+numpy and sample surface points.

The V2D Track 3 meshes are plain GLB files (no external buffers/draco), so a tiny
reader is enough.  ADD needs a per-object point cloud in the object's own frame:
we draw uniformly over triangle area so the sampled points reflect real geometry
(handles e.g. the hollow white-pot lid correctly).
"""

from __future__ import annotations

import json
import struct
from typing import BinaryIO

import numpy as np

_GLB_HEADER = struct.Struct("<III")  # magic, version, length
_CHUNK_HEADER = struct.Struct("<II")  # length, type
_JSON_TYPE = 0x4E4F534A  # "JSON"
_BIN_TYPE = 0x004E4942  # "BIN\0"

_COMPONENT_BYTES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
_COMPONENT_DTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT3": 9, "MAT4": 16}

_POSITION_ATTRIBUTE = "POSITION"


def load_vertices(buffer: BinaryIO) -> tuple[np.ndarray, np.ndarray]:
    """Read every triangle primitive in the active scene, retaining node transforms."""
    buffer.seek(0)
    header = _GLB_HEADER.unpack(buffer.read(12))
    if header[0] != 0x46546C67:
        raise ValueError("not a glTF binary")
    json_data: bytes | None = None
    bin_data: bytes | None = None
    while True:
        chunk = buffer.read(8)
        if not chunk:
            break
        length, ctype = _CHUNK_HEADER.unpack(chunk)
        data = buffer.read(length)
        if ctype == _JSON_TYPE:
            json_data = data
        elif ctype == _BIN_TYPE:
            bin_data = data
    if json_data is None:
        raise ValueError("glb has no JSON chunk")
    if bin_data is None:
        raise ValueError("glb has no BIN chunk")
    gltf = json.loads(json_data.decode("utf-8"))

    def read_accessor(acc_index: int, expected_type: str) -> np.ndarray:
        accessor = gltf["accessors"][acc_index]
        if "sparse" in accessor or "bufferView" not in accessor:
            raise ValueError("sparse GLB accessors are unsupported")
        view = gltf["bufferViews"][accessor["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise ValueError("external GLB buffers are unsupported")
        count = accessor["count"]
        component_type = accessor["componentType"]
        if accessor.get("type") != expected_type or component_type not in _COMPONENT_DTYPE:
            raise ValueError(f"unsupported accessor for {expected_type}")
        if expected_type == "VEC3" and component_type != 5126:
            raise ValueError("POSITION must use float32")
        if expected_type == "SCALAR" and component_type not in (5121, 5123, 5125):
            raise ValueError("triangle indices must be unsigned integers")
        dtype = np.dtype(_COMPONENT_DTYPE[component_type]).newbyteorder("<")
        byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        size = _COMPONENT_BYTES[component_type] * _TYPE_COUNT[expected_type]
        stride = view.get("byteStride", size)
        end = byte_offset + (count - 1) * stride + size if count else byte_offset
        if stride < size or byte_offset < 0 or end > len(bin_data) or end > view.get("byteOffset", 0) + view["byteLength"]:
            raise ValueError("accessor bufferView out of range")
        return np.ndarray((count, _TYPE_COUNT[expected_type]), dtype=dtype,
                          buffer=bin_data, offset=byte_offset, strides=(stride, dtype.itemsize)).copy()

    def node_matrix(node):
        if "matrix" in node:
            matrix = np.asarray(node["matrix"], dtype=float).reshape(4, 4, order="F")
        else:
            q = np.asarray(node.get("rotation", [0, 0, 0, 1]), dtype=float)
            if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
                raise ValueError("invalid GLB node rotation")
            x, y, z, w = q / np.linalg.norm(q)
            rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
            matrix = np.eye(4)
            matrix[:3, :3] = rotation @ np.diag(node.get("scale", [1, 1, 1]))
            matrix[:3, 3] = node.get("translation", [0, 0, 0])
        if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("invalid GLB node transform")
        return matrix

    vertex_parts, face_parts, offset = [], [], 0
    def add_mesh(mesh_index, matrix):
        nonlocal offset
        for primitive in gltf["meshes"][mesh_index].get("primitives", []):
            if _POSITION_ATTRIBUTE not in primitive.get("attributes", {}):
                continue
            if primitive.get("mode", 4) != 4:
                raise ValueError("ADD surface sampling requires triangle primitives")
            vertices = read_accessor(primitive["attributes"][_POSITION_ATTRIBUTE], "VEC3").astype(float)
            faces = (read_accessor(primitive["indices"], "SCALAR").ravel().astype(np.int64)
                     if "indices" in primitive else np.arange(len(vertices)))
            if len(faces) % 3 or (faces < 0).any() or (faces >= len(vertices)).any():
                raise ValueError("invalid GLB triangle indices")
            vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
            vertex_parts.append(vertices)
            face_parts.append(faces.reshape(-1, 3) + offset)
            offset += len(vertices)

    nodes = gltf.get("nodes", [])
    def visit(index, parent, ancestors):
        if index in ancestors:
            raise ValueError("GLB node hierarchy contains a cycle")
        node = nodes[index]
        matrix = parent @ node_matrix(node)
        if "mesh" in node:
            add_mesh(node["mesh"], matrix)
        for child in node.get("children", []):
            visit(child, matrix, ancestors | {index})
    if nodes:
        scenes = gltf.get("scenes", [])
        roots = (scenes[gltf.get("scene", 0)].get("nodes", []) if scenes else
                 sorted(set(range(len(nodes))) - {i for node in nodes for i in node.get("children", [])}))
        for index in roots:
            visit(index, np.eye(4), set())
    else:
        for index in range(len(gltf.get("meshes", []))):
            add_mesh(index, np.eye(4))
    if not vertex_parts:
        raise ValueError("no active POSITION triangle geometry in GLB")
    return np.concatenate(vertex_parts), np.concatenate(face_parts).ravel()


def sample_surface(
    vertices: np.ndarray,
    indices: np.ndarray | None = None,
    num_points: int = 4096,
    seed: int | None = 0,
) -> np.ndarray:
    """Sample ``num_points`` points uniformly over triangle area.

    ``vertices`` is ``(V, 3)``, ``indices`` is ``(T, 3)`` ints (or ``None`` for a
    triangle fan).  Returns ``(num_points, 3)`` in the same frame as ``vertices``.
    """
    rng = np.random.default_rng(seed)
    if indices is None:
        tri_count = len(vertices) // 3
        indices = np.arange(tri_count * 3, dtype=np.int64).reshape(tri_count, 3)
    else:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
        tri_count = len(indices)
    triangles = vertices[indices]  # (T, 3, 3)
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(v1 - v0, v2 - v0)
    # Twice the area; guard against degenerate triangles.
    areas = np.linalg.norm(cross, axis=-1) * 0.5
    total = areas.sum()
    if total <= 0 or not np.isfinite(total):
        raise ValueError("mesh has zero or invalid surface area")
    probs = areas / total
    picks = rng.choice(tri_count, size=num_points, p=probs)
    r1 = rng.random(num_points)
    r2 = rng.random(num_points)
    sqrt_r1 = np.sqrt(r1)
    u, v = 1.0 - sqrt_r1, sqrt_r1 * r2
    a, b = v0[picks], v1[picks]
    points = (
        a
        + u[:, None] * (b - a)
        + (sqrt_r1 * (1.0 - r2))[:, None] * (v2[picks] - a)
    )
    return points


def load_mesh_points(glb_path: str, num_points: int = 4096, seed: int = 0) -> np.ndarray:
    """Convenience: vertices, area-weighted surface points for one .glb file."""
    with open(glb_path, "rb") as handle:
        vertices, indices = load_vertices(handle)
    return sample_surface(vertices, indices, num_points=num_points, seed=seed)
