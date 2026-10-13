"""Exact nearest boundary-square distance for the declared closed-cell union.

This audits the existing inferred voxel material, not physical object truth.
Every possibly closer face is retained by a triangle-inequality radius bound.
"""
import numpy as np
from scipy.spatial import cKDTree

class Boundary:
    def __init__(self,solid,origin,pitch):
        self.solid=np.asarray(solid,dtype=bool);self.origin=np.asarray(origin);self.pitch=float(pitch)
        assert self.pitch>0 and self.solid.ndim==3 and self.solid.any()
        centers=[];axes=[];signs=[]
        for axis in range(3):
            for sign in (-1,1):
                neighbor=np.roll(self.solid,-sign,axis=axis)
                edge=[slice(None)]*3;edge[axis]=-1 if sign==1 else 0
                neighbor[tuple(edge)]=False
                indices=np.argwhere(self.solid&~neighbor)
                p=self.origin+indices*self.pitch;p[:,axis]+=sign*self.pitch/2
                centers.append(p);axes.extend([axis]*len(p));signs.extend([sign]*len(p))
        self.centers=np.concatenate(centers);self.axes=np.asarray(axes);self.signs=np.asarray(signs)
        self.half=np.full_like(self.centers,self.pitch/2);self.half[np.arange(len(self.centers)),self.axes]=0
        self.tree=cKDTree(self.centers)

    def inside(self,points):
        index=np.floor((points-self.origin)/self.pitch+.5).astype(np.int64)
        grid=np.all((index>=0)&(index<np.array(self.solid.shape)),axis=1)
        inside=np.zeros(len(points),dtype=bool);inside[grid]=self.solid[tuple(index[grid].T)]
        return inside

    def unsigned(self,points):
        points=np.asarray(points,dtype=float)
        assert points.ndim==2 and points.shape[1]==3 and np.isfinite(points).all()
        _,nearest=self.tree.query(points,k=1)
        first=np.maximum(np.abs(points-self.centers[nearest])-self.half[nearest],0)
        upper=np.linalg.norm(first,axis=1)
        radii=upper+np.sqrt(2)*self.pitch/2+1e-12
        distance=np.empty(len(points));nearest_face=np.empty(len(points),dtype=np.int64)
        queries=0;largest=0
        for start in range(0,len(points),256):
            candidates=self.tree.query_ball_point(points[start:start+256],radii[start:start+256],eps=0)
            for local,ids in enumerate(candidates):
                ids=np.asarray(ids,dtype=np.int64);row=start+local
                assert len(ids)>0
                delta=np.maximum(np.abs(points[row]-self.centers[ids])-self.half[ids],0)
                squared=np.einsum('ij,ij->i',delta,delta)
                selected=int(np.argmin(squared));distance[row]=np.sqrt(squared[selected]);nearest_face[row]=ids[selected]
                queries+=len(ids);largest=max(largest,len(ids))
        return distance,nearest_face,{'boundary_square_queries':queries,'maximum_candidate_faces_per_point':largest}

def controls():
    cases=[np.ones((1,1,1),dtype=bool),np.ones((2,3,4),dtype=bool),np.pad(np.ones((1,1,1),dtype=bool),1)]
    cavity=np.ones((3,3,3),dtype=bool);cavity[1,1,1]=False;cases.append(cavity)
    disconnected=np.zeros((4,4,4),dtype=bool);disconnected[0,0,0]=True;disconnected[3,3,3]=True;cases.append(disconnected)
    generator=np.random.default_rng(1150);worst=0.;count=0
    for solid in cases:
        boundary=Boundary(solid,np.array([.13,-.21,.07]),.01)
        points=generator.uniform(-.01,np.array(solid.shape)*.01+.01,size=(19,3))+boundary.origin
        actual,_,_=boundary.unsigned(points)
        for point,value in zip(points,actual):
            delta=np.maximum(np.abs(point-boundary.centers)-boundary.half,0)
            expected=np.linalg.norm(delta,axis=1).min()
            worst=max(worst,abs(value-expected));count+=1
    cube=Boundary(np.ones((1,1,1),dtype=bool),np.zeros(3),.02)
    points=np.array([[0,0,0],[.005,0,0],[.02,0,0],[.02,.02,0],[.02,.02,.02]])
    actual,_,_=cube.unsigned(points)
    np.testing.assert_allclose(actual,[.01,.005,.01,np.sqrt(2)*.01,np.sqrt(3)*.01],atol=1e-12)
    np.testing.assert_array_equal(cube.inside(points),[True,True,False,False,False])
    assert worst<1e-12
    return {'single_cube_analytic_distance':True,'inside_outside_control':True,'exhaustive_faces_points':count,
            'exhaustive_faces_max_error_m':worst,'cavity_and_disconnected_faces_checked':True}
