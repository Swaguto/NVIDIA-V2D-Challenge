"""Analytic, branch, singularity and input-contract checks for point IK."""
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest

from v2d_dex3_flashchord.planar_thumb_ik import (
    PlanarThumbChain, thumb_point, thumb_point_branches,
)

CHAIN = PlanarThumbChain((.0255, 0., 0.), (-.0025, .0193), .0458, .035)
WIDE = np.array([[-math.pi, math.pi]] * 3)


def test_known_straight_point():
    np.testing.assert_allclose(thumb_point(CHAIN, [0, 0, 0]), [.023, .1001, 0], atol=1e-15)


def test_known_quarter_turns():
    chain = PlanarThumbChain((0, 0, 0), (0, 0), 2., 1.)
    np.testing.assert_allclose(thumb_point(chain, [math.pi/2, -math.pi/2, 0]), [0, 0, -3], atol=1e-15)
    np.testing.assert_allclose(thumb_point(chain, [0, 0, math.pi/2]), [-1, 2, 0], atol=1e-15)


@pytest.mark.parametrize('angles', [(.2,-.8,-.9),(-.6,-.3,-1.2),(.5,.4,.8),(-.4,2.5,.2)])
def test_recovers_branch_in_both_radial_halfplanes(angles):
    point = thumb_point(CHAIN, angles)
    branches = thumb_point_branches(CHAIN, point, WIDE)
    assert not branches.axial_singular
    assert any(np.max(abs(np.array(q)-angles)) < 1e-10 for q in branches.joints_rad)
    for q in branches.joints_rad:
        np.testing.assert_allclose(thumb_point(CHAIN, q), point, atol=1e-12)


def test_two_elbow_solutions_are_distinct():
    point = thumb_point(CHAIN, [.3,-.7,-.8])
    bounds = [[0,1],[-math.pi,math.pi],[-math.pi,math.pi]]
    branches = thumb_point_branches(CHAIN, point, bounds)
    assert len(branches.joints_rad) == 2
    assert {math.copysign(1,q[2]) for q in branches.joints_rad} == {-1,1}


def test_position_limits_remove_other_branches():
    q = np.array([.3,-.7,-.8]);p = thumb_point(CHAIN,q)
    bounds=np.c_[q-.01,q+.01]
    result=thumb_point_branches(CHAIN,p,bounds)
    assert len(result.joints_rad)==1
    assert thumb_point_branches(CHAIN,p,[[.4,.5],[-.8,-.6],[-.9,-.7]]).joints_rad==()


def test_angle_branch_wrapping_respects_declared_interval():
    q=np.array([.2,3.5,-.6]);p=thumb_point(CHAIN,q)
    result=thumb_point_branches(CHAIN,p,[[0,.5],[3.4,3.6],[-.7,-.5]])
    assert len(result.joints_rad)==1
    np.testing.assert_allclose(result.joints_rad[0],q,atol=1e-12)


def test_tangent_elbow_is_deduplicated():
    p=thumb_point(CHAIN,[.2,-.5,0])
    result=thumb_point_branches(CHAIN,p,[[0,.5],[-1,0],[-1,1]])
    assert len(result.joints_rad)==1


def test_unreachable_and_axial_singular_are_distinguished():
    result=thumb_point_branches(CHAIN,[1,1,1],WIDE)
    assert result.joints_rad==() and not result.axial_singular
    singular=thumb_point_branches(CHAIN,[CHAIN.origin[0],.06,CHAIN.origin[2]],WIDE)
    assert singular.joints_rad==() and singular.axial_singular


@pytest.mark.parametrize('bounds', [[[0,1]],[[1,0]]*3,[[0,7]]*3,[[0,float('inf')]]*3])
def test_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):thumb_point_branches(CHAIN,[.05,.06,0],bounds)


@pytest.mark.parametrize('point', [[0,0],[0,float('nan'),0],[0,0,float('inf')]])
def test_rejects_bad_point(point):
    with pytest.raises(ValueError):thumb_point_branches(CHAIN,point,WIDE)


@pytest.mark.parametrize('length', [0,-.01,float('nan'),float('inf')])
def test_rejects_invalid_lengths(length):
    with pytest.raises(ValueError):PlanarThumbChain((0,0,0),(0,0),length,.03)
