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


def load_vertices(buffer: BinaryIO) -> np.ndarray:
    """Return ``(V, 3)`` float64 mesh vertex positions from a .glb stream."""
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

    # Find the first primitive exposing POSITION.
    accessor_index: int | None = None
    indices_index: int | None = None
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if _POSITION_ATTRIBUTE in attributes:
                accessor_index = attributes[_POSITION_ATTRIBUTE]
                indices_index = primitive.get("indices")
                break
        if accessor_index is not None:
            break
    if accessor_index is None:
        raise ValueError("no POSITION attribute in glb")

    def read_accessor(acc_index: int, expected_type: str, expected_comp: int) -> np.ndarray:
        accessor = gltf["accessors"][acc_index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        count = accessor["count"]
        component_type = accessor["componentType"]
        if accessor.get("type") != expected_type or component_type != expected_comp:
            raise ValueError(f"accessor must be {expected_type} 0x{expected_comp:x}")
        dtype = _COMPONENT_DTYPE[component_type]
        byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
        byte_length = _COMPONENT_BYTES[component_type] * _TYPE_COUNT[expected_type] * count
        raw = bin_data[byte_offset : byte_offset + byte_length]
        if len(raw) != byte_length:
            raise ValueError("accessor bufferView out of range")
        return np.frombuffer(raw, dtype=dtype)

    vertices = read_accessor(accessor_index, "VEC3", 5126).astype(np.float64).reshape(-1, 3)
    indices: np.ndarray | None = None
    if indices_index is not None:
        indices = read_accessor(indices_index, "SCALAR", 5125).astype(np.int64)
    return vertices, indices


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