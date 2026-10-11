import numpy as np
import trimesh

from issue6.geometry import surface_contacts


def test_shared_edge_normal_ignores_subpicometre_distance_roundoff(monkeypatch):
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 3, 1]],
        process=False,
    )
    original = trimesh.triangles.closest_point

    def perturbed(triangles, points):
        closest = original(triangles, points)
        # Both faces meet at the closest edge; one runtime rounds the first
        # triangle's point a fraction of a picometre away from the query.
        closest[0, 1] += 5e-13
        return closest

    monkeypatch.setattr(trimesh.triangles, "closest_point", perturbed)
    monkeypatch.setattr(trimesh.proximity, "_corresponding", perturbed)
    point = np.array([[0.3, -0.1, -0.1]])
    _, distance, normal, _ = surface_contacts(point, mesh, np.eye(4))
    np.testing.assert_array_equal(normal[0], mesh.face_normals[0])
    # Strict distance/contact classification still uses the actual minimum,
    # independently of the canonical normal face.
    np.testing.assert_allclose(distance[0], np.sqrt(0.02), rtol=0, atol=1e-15)


def test_distinct_closest_surface_is_not_replaced_by_lowest_face():
    mesh = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        faces=[[0, 1, 2], [0, 3, 1]],
        process=False,
    )
    _, distance, normal, _ = surface_contacts(np.array([[0.3, 0.001, 0.2]]), mesh, np.eye(4))
    np.testing.assert_array_equal(normal[0], mesh.face_normals[1])
    np.testing.assert_allclose(distance[0], 0.001, atol=1e-15)
