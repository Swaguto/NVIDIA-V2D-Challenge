"""Vertices used by the pinned MuJoCo convex collision support function.

Compiled ``mesh_vert`` also contains vertices excluded by ``maxhullvert``.
Rebuilding a hull from every vertex can therefore invent contact surfaces.
The layout below follows MuJoCo 3.8.1 engine_collision_convex.c and user_mesh.cc;
unknown versions fail closed. This reader does not modify the compiled model.
"""
import numpy as np


def _integers(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in 'iu' or array.ndim != 1:
        raise ValueError(f'{name} must be a one-dimensional integer array')
    return array


def compiled_mesh_vertices(model, mesh_id, *, mujoco_version):
    """Return a copy of the geom-local collision vertices for one mesh.

    With a graph, read its vertex-global-id table, exactly as native support
    does. With graph address -1, native support scans all mesh vertices.
    Rendering, inertia, exact contact generation and force feasibility are
    separate contracts. No poses or ``mesh_pos/quat`` are applied here: these
    compiled coordinates are transformed by ``data.geom_xpos/xmat``.
    """
    if mujoco_version != '3.8.1':
        raise ValueError('Convex graph layout is verified only for MuJoCo 3.8.1')
    if isinstance(mesh_id, (bool, np.bool_)) or not isinstance(mesh_id, (int, np.integer)):
        raise ValueError('mesh_id must be an integer')
    addresses = _integers(model.mesh_vertadr, 'mesh_vertadr')
    counts = _integers(model.mesh_vertnum, 'mesh_vertnum')
    graphs = _integers(model.mesh_graphadr, 'mesh_graphadr')
    if len(addresses) != len(counts) or len(graphs) != len(counts) or not 0 <= mesh_id < len(counts):
        raise ValueError('Invalid mesh id or inconsistent mesh tables')
    vertices = np.asarray(model.mesh_vert, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Need finite compiled mesh vertices[N,3]')
    start, count = int(addresses[mesh_id]), int(counts[mesh_id])
    if start < 0 or count < 4 or start + count > len(vertices):
        raise ValueError('Invalid compiled vertex span')
    vertices = vertices[start:start + count]
    address = int(graphs[mesh_id])
    if address == -1:
        return vertices.copy()
    graph = _integers(model.mesh_graph, 'mesh_graph')
    if address < 0 or address + 2 > len(graph):
        raise ValueError('Invalid convex graph address')
    following = graphs[graphs > address]
    end = int(following.min()) if len(following) else len(graph)
    graph = graph[address:end]
    vertex_count, face_count = int(graph[0]), int(graph[1])
    if (vertex_count < 4 or vertex_count > count or face_count < 4
            or len(graph) != 2 + 3*vertex_count + 6*face_count):
        raise ValueError('Invalid convex graph header or storage span')
    ids = graph[2 + vertex_count:2 + 2*vertex_count]
    if (np.any(ids < 0) or np.any(ids >= count)
            or len(np.unique(ids)) != vertex_count):
        raise ValueError('Invalid convex graph vertex-global-id table')
    return vertices[ids].copy()
