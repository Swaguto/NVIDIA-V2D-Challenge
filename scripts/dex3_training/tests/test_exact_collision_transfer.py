"""Geometry transfer must preserve vertices and mechanical XML."""

import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from freeze_isaac_collision_geometry import write_hulls


def test_collision_hulls_keep_body_coordinates_and_inertia(tmp_path):
    points = np.array([[0, 0, 0], [0.02, 0, 0], [0, 0.03, 0], [0, 0, 0.01]])
    collider = {
        "mesh_to_body_matrix_row_convention": np.eye(4).tolist(),
        "convexes": [{"vertices_mesh_local_m": points.tolist()}],
    }
    link = ET.fromstring(
        '<link name="finger"><inertial><mass value="0.3"/></inertial><visual/><collision><geometry><box size="1 1 1"/></geometry></collision></link>'
    )
    before = ET.tostring(link.find("inertial"))
    records = write_hulls(tmp_path, link, [collider], "finger")
    text = (tmp_path / records[0]["file"]).read_text().splitlines()
    vertices = np.array(
        [[float(x) for x in line.split()[1:]] for line in text if line.startswith("v ")]
    )
    faces = np.array(
        [
            [int(x) - 1 for x in line.split()[1:]]
            for line in text
            if line.startswith("f ")
        ]
    )
    np.testing.assert_array_equal(vertices, points)
    center = vertices.mean(0)
    for tri in faces:
        a, b, c = vertices[tri]
        assert np.cross(b - a, c - a) @ (a - center) > 0
    assert ET.tostring(link.find("inertial")) == before
    assert link.find("visual") is not None
    assert len(link.findall("collision")) == 1
