"""Visible asymmetric features must distinguish otherwise ambiguous CAD poses."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/track3'))
from pose_support import projected_support


def test_visible_handle_distinguishes_rotation_of_same_bowl():
    y,x=np.mgrid[0:81,0:81]
    bowl=(x-40)**2+(y-40)**2<=12**2
    handle=(abs(x-40)<=3)&(y>=50)&(y<=72)
    mask=bowl|handle
    yy,xx=np.nonzero(mask)
    points=np.column_stack([(xx-40)/100,(yy-40)/100,np.ones(len(xx))])
    K=np.array([[100.,0,40],[0,100.,40],[0,0,1]])
    depth=np.full(mask.shape,2.);depth[mask]=1.
    valid=np.ones_like(mask)
    excluded=np.zeros_like(mask)
    good=projected_support(points,K,mask,depth,valid,excluded)
    wrong=points.copy();wrong[:,0],wrong[:,1]=-points[:,1],points[:,0]
    bad=projected_support(wrong,K,mask,depth,valid,excluded)
    assert good['target_pixels_supported_within_3px_fraction']==1.
    assert bad['projected_support_score']>good['projected_support_score']+.05
    assert bad['target_pixels_supported_within_3px_fraction']<.9


def test_other_object_behind_cad_cannot_hide_projected_false_positive():
    mask=np.zeros((40,40),bool);mask[10:15,10:15]=True
    points=np.array([[x,y,1.] for y in range(10,15) for x in range(10,15)]
                    +[[x,y,1.] for y in range(25,30) for x in range(25,30)])
    excluded=np.zeros_like(mask);excluded[25:30,25:30]=True
    valid=np.ones_like(mask);depth=np.full(mask.shape,2.)
    behind=projected_support(points,np.eye(3),mask,depth,valid,excluded)
    depth[excluded]=.5
    nearer=projected_support(points,np.eye(3),mask,depth,valid,excluded)
    assert behind['visible_projected_samples']==50
    assert behind['projected_outside_mask_cost']>.4
    assert nearer['visible_projected_samples']==25
    assert nearer['projected_outside_mask_cost']==0.
