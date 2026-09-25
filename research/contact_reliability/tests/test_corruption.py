import numpy as np
import pytest
from v2d_reliability.corruption import perturb


def test_reproducible_nonmutating_corruption():
    original = np.ones((5, 2, 1, 3)); valid = np.ones((5, 2, 1), dtype=bool)
    args = (original, valid, "dev", ["dev"], "pose_jitter")
    a, _ = perturb(*args); b, _ = perturb(*args)
    assert np.array_equal(a, b) and not np.array_equal(a, original)
    assert (original == 1).all()
    _, missing = perturb(original, valid, "dev", ["dev"], "dropout", magnitude=1)
    assert not missing.any()
    _, shifted = perturb(original, valid, "dev", ["dev"], "timing")
    assert not shifted[0].any() and shifted[1:].all()
    with pytest.raises(ValueError):
        perturb(original, valid, "holdout", ["dev"], "pose_jitter")
def test_contacts_are_separate_reproducible_and_proxy_protected():
    import numpy as np
    import pytest
    from v2d_reliability.corruption import corrupt_contacts
    points = np.zeros((3, 2, 1, 4, 3)); normals = np.zeros_like(points); normals[..., 2] = 1
    active = np.ones(points.shape[:-1], dtype=bool)
    args = (points, normals, active, 2, [2], "contact_normal_rotation", .5)
    p, n, a = corrupt_contacts(*args)
    np.testing.assert_array_equal(p, points)
    np.testing.assert_allclose(np.linalg.norm(n, axis=-1), 1)
    np.testing.assert_array_equal(n, corrupt_contacts(*args)[1])
    assert not np.array_equal(n, normals)
    with pytest.raises(ValueError): corrupt_contacts(points, normals, active, 0, [0], "contact_dropout", .1)
