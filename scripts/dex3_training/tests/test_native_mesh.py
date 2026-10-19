"""A discarded protrusion must never become a native contact surface."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from v2d_dex3_flashchord.native_mesh import compiled_mesh_vertices


def model():
    # A tetrahedral native support graph excludes a fifth, protruding raw vertex.
    # Four vertices/faces require 38 graph ints. Only the global-id table is read.
    graph = np.zeros(38, dtype=int)
    graph[:2] = [4, 4]
    graph[6:10] = [0, 1, 2, 3]
    return SimpleNamespace(mesh_vertadr=np.array([0]), mesh_vertnum=np.array([5]),
        mesh_graphadr=np.array([0]), mesh_graph=graph,
        mesh_vert=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 0, 0]], float))


def read(m, mesh_id=0, version='3.8.1'):
    return compiled_mesh_vertices(m, mesh_id, mujoco_version=version)


def test_native_support_excludes_discarded_protrusion_and_preserves_order():
    m = model()
    m.mesh_graph[6:10] = [3, 1, 0, 2]
    result = read(m)
    np.testing.assert_array_equal(result, m.mesh_vert[[3, 1, 0, 2]])
    assert result[:, 0].max() == 1  # A raw-vertex hull would claim x=9.
    result[:] = 99
    assert m.mesh_vert[0, 0] == 0


def test_no_graph_uses_all_vertices_exactly_as_native_exhaustive_search():
    m = model()
    m.mesh_graphadr[0] = -1
    np.testing.assert_array_equal(read(m), m.mesh_vert)


def test_per_mesh_offsets_and_next_graph_boundary_are_respected():
    m = model()
    m.mesh_vert = np.concatenate([m.mesh_vert + 20, m.mesh_vert])
    m.mesh_vertadr = np.array([0, 5]); m.mesh_vertnum = np.array([5, 5])
    m.mesh_graphadr = np.array([0, 38]); m.mesh_graph = np.tile(m.mesh_graph, 2)
    np.testing.assert_array_equal(read(m, 1), m.mesh_vert[5:9])


@pytest.mark.parametrize('mesh_id', [-1, 1, True, 0.0])
def test_invalid_mesh_id_is_rejected(mesh_id):
    with pytest.raises(ValueError):
        read(model(), mesh_id)


@pytest.mark.parametrize('mutation', [
    lambda m: setattr(m, 'mesh_graph', m.mesh_graph[:-1]),
    lambda m: m.mesh_graph.__setitem__(0, 3),
    lambda m: m.mesh_graph.__setitem__(1, 0),
    lambda m: m.mesh_graph.__setitem__(6, -1),
    lambda m: m.mesh_graph.__setitem__(6, 5),
    lambda m: m.mesh_graph.__setitem__(6, 1),
    lambda m: m.mesh_graphadr.__setitem__(0, -2),
    lambda m: m.mesh_vertadr.__setitem__(0, 100),
    lambda m: m.mesh_vert.__setitem__((0, 0), np.nan),
    lambda m: setattr(m, 'mesh_graph', m.mesh_graph.astype(float)),
])
def test_invalid_compiled_storage_fails_closed(mutation):
    m = model(); mutation(m)
    with pytest.raises(ValueError):
        read(m)


def test_unknown_native_layout_is_not_silently_accepted():
    with pytest.raises(ValueError, match='3.8.1'):
        read(model(), version='future')
