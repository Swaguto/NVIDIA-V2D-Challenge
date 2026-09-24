"""Load V2D Track 3 public reference episodes and mesh vertex clouds.

Poses in the parquet are stored w-first (``[x, y, z, qw, qx, qy, qz]``) metres/radians;
every metric helper here returns **xyzw** quaternions to match the CHORD metrics module.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq

_W2Xyzw = np.array([0, 1, 2, 4, 5, 6, 3], dtype=np.int64)

# Public episodes deliberately held out as the private proxy test set.
# Never tune/normalize/overfit against these ground-truth trajectories.
# One per distinct task (Pot_with_lid, Cup_Stack, Planter_Stand, Brush_Dishrack, Dustpan_Solo).
PROXY_EPISODES: tuple[int, ...] = (0, 11, 23, 31, 39)

TRAIN_EPISODES: tuple[int, ...] = tuple(
    sorted(set(range(46)) - set(PROXY_EPISODES))
)


@dataclass(frozen=True)
class EpisodeReference:
    """Ground-truth object trajectory for one public episode."""

    episode_index: int
    object_names: tuple[str, ...]
    pose_xyzw: np.ndarray  # (T, B, 7) metres/radians, xyzw
    visible: np.ndarray  # (T, B) bool
    frame_index: np.ndarray  # (T,)
    capture_time: np.ndarray  # (T,)
    fps: float

    @property
    def steps(self) -> int:
        return self.pose_xyzw.shape[0]

    @property
    def bodies(self) -> int:
        return self.pose_xyzw.shape[1]


def load_reference(episode_index: int, base_dir: str | os.PathLike) -> EpisodeReference:
    """Load one public episode's GT object trajectory into xyzw metric form."""
    base = Path(base_dir)
    parquet = base / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    pf = pq.ParquetFile(parquet)
    table = pf.read()
    meta = pf.schema_arrow.metadata
    object_names = tuple(json.loads(meta[b"objects"]))
    objects = table.column("observation.objects").to_pylist()

    steps = len(objects)
    bodies = len(object_names)
    pose = np.empty((steps, bodies, 7), dtype=np.float64)
    visible = np.empty((steps, bodies), dtype=bool)
    for t, frame_objects in enumerate(objects):
        by_name = {o["name"]: o for o in frame_objects}
        for b, name in enumerate(object_names):
            pose[t, b] = np.asarray(by_name[name]["pose"], dtype=np.float64)
            visible[t, b] = bool(by_name[name]["visible"])
    pose_xyzw = pose[..., _W2Xyzw]

    info = json.loads((base / "meta" / "info.json").read_text())
    return EpisodeReference(
        episode_index=episode_index,
        object_names=object_names,
        pose_xyzw=pose_xyzw,
        visible=visible,
        frame_index=np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64),
        capture_time=np.asarray(table.column("capture_time").to_pylist(), dtype=np.float64),
        fps=float(info["fps"]),
    )


def object_mesh_dir(object_names: Sequence[str], base_dir: str | os.PathLike) -> dict[str, Path]:
    """Map each object name to its GLB mesh path (name == mesh folder name)."""
    base = Path(base_dir)
    mesh_root = base / "mesh"
    result: dict[str, Path] = {}
    for name in object_names:
        candidates = (
            mesh_root / name / f"{name}.glb",
            mesh_root / name / f"{name}_visual.glb",
            mesh_root / name / f"{name}_collision.glb",
            mesh_root / f"{name}.glb",
            mesh_root / f"{name}_visual.glb",
        )
        matched = next((candidate for candidate in candidates if candidate.exists()), None)
        if matched is None:
            raise FileNotFoundError(f"mesh not found for {name!r}: tried {candidates[0]}")
        result[name] = matched
    return result


@dataclass(frozen=True)
class VertexClouds:
    """Per-object surface points in the object's scan frame, plus the source names."""

    object_names: tuple[str, ...]
    points: Mapping[str, np.ndarray]  # name -> (N, 3)
    counts: Mapping[str, int] = field(default_factory=dict)


def load_vertex_clouds(
    object_names: Sequence[str],
    base_dir: str | os.PathLike,
    num_points: int = 4096,
    seed: int = 0,
) -> VertexClouds:
    """Area-weighted surface samples for the ADD metric (one mesh per object)."""
    from glb_mesh import load_mesh_points

    meshes = object_mesh_dir(object_names, base_dir)
    points: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for name, path in meshes.items():
        pts = load_mesh_points(str(path), num_points=num_points, seed=seed)
        points[name] = pts
        counts[name] = int(len(pts))
    return VertexClouds(tuple(object_names), points, counts)


def batch_vertices_for(reference: EpisodeReference, clouds: VertexClouds) -> np.ndarray:
    """``(B, Vmax, 3)`` vertex arrays stacked in body order (padded to equal length)."""
    vertex_list = [clouds.points[name] for name in reference.object_names]
    vmax = max(int(p.shape[0]) for p in vertex_list)
    out = np.zeros((len(vertex_list), vmax, 3), dtype=np.float64)
    for b, points in enumerate(vertex_list):
        out[b, : points.shape[0]] = points
    return out