"""Bounded 6-DoF source-fit probe with finite support; no production promotion."""
import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation
from measured_pose_refinement import support_constraints, finite_support_overlap
from select_pose_hypothesis import observed_fit


def refine(vertices, faces, pose, depth, valid, mask, K, plane, footprint):
    vertices, faces, pose, depth, valid, mask, K, plane, footprint = map(
        np.asarray, (vertices, faces, pose, depth, valid, mask, K, plane, footprint)
    )
    tree = cKDTree(vertices)
    before = observed_fit(vertices, faces, tree, pose, depth, valid, mask, K)
    measured = valid & mask & np.isfinite(depth) & (depth > .1) & (depth < 2.5)
    yy, xx = np.nonzero(measured)
    if len(xx) < 100:
        return pose.copy(), dict(accepted=False, reason='insufficient measured foreground', before=before)
    ids = np.linspace(0, len(xx)-1, min(2048,len(xx)),dtype=int)
    xx, yy = xx[ids], yy[ids]
    zz = depth[yy,xx]
    points = np.c_[(xx-K[0,2])*zz/K[0,0],(yy-K[1,2])*zz/K[1,1],zz]
    sampled = vertices[np.linspace(0,len(vertices)-1,min(2048,len(vertices)),dtype=int)]
    outside = cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
    source_camera = np.einsum('ni,ji->nj',vertices,pose[:3,:3],optimize=False)+pose[:3,3]
    heights = np.einsum('ni,i->n',source_camera,plane[:3],optimize=False)+plane[3]
    support = support_constraints(vertices,pose,plane,footprint)
    supported = support is not None and heights.max() >= 0
    normal = plane[:3]
    axis = np.eye(3)[np.argmin(abs(normal))]
    u = np.cross(normal,axis);u /= np.linalg.norm(u)
    basis = np.stack([u,np.cross(normal,u)],axis=1)
    hull = ConvexHull(np.einsum('ni,ij->nj',footprint,basis,optimize=False))
    def transform(x):
        out = pose.copy()
        out[:3,3] += .01*x[:3]
        out[:3,:3] = np.einsum('ij,jk->ik',Rotation.from_rotvec(.1*x[3:]).as_matrix(),pose[:3,:3],optimize=False)
        return out
    def objective(x):
        T = transform(x)
        local = np.einsum('ni,ij->nj',points-T[:3,3],T[:3,:3],optimize=False)
        camera = np.einsum('ni,ji->nj',sampled,T[:3,:3],optimize=False)+T[:3,3]
        if not np.isfinite(local).all() or (camera[:,2] <= .05).any():return 1000.
        distances = np.sort(tree.query(local,workers=1)[0])[:int(.8*len(points))]
        robust = np.where(distances<.01,.5*distances**2,.01*(distances-.005)).mean()
        uu=K[0,0]*camera[:,0]/camera[:,2]+K[0,2];vv=K[1,1]*camera[:,1]/camera[:,2]+K[1,2]
        ix=np.clip(np.rint(uu).astype(int),0,mask.shape[1]-1);iy=np.clip(np.rint(vv).astype(int),0,mask.shape[0]-1)
        over=np.maximum(0,outside[iy,ix]-5)*camera[:,2]/K[0,0]
        over[(uu<0)|(uu>=mask.shape[1])|(vv<0)|(vv>=mask.shape[0])]=.1
        return float(robust+.5*np.minimum(over,.1).dot(np.minimum(over,.1))/len(over)+(.002*np.linalg.norm(.01*x[:3])/.05)**2+(.002*np.linalg.norm(.1*x[3:])/np.deg2rad(20))**2)
    def finite_constraints(x):
        T=transform(x);camera=np.einsum('ni,ji->nj',vertices,T[:3,:3],optimize=False)+T[:3,3]
        side=np.einsum('ni,ki->nk',np.einsum('ni,ij->nj',camera,basis,optimize=False),hull.equations[:,:2],optimize=False).max(0)+hull.equations[:,2]
        return np.r_[np.min(np.einsum('ni,i->n',camera,normal,optimize=False)+plane[3])-.002,-.001-side]
    constraints=[dict(type='ineq',fun=lambda x:.06**2-np.dot(.01*x[:3],.01*x[:3])),dict(type='ineq',fun=lambda x:np.deg2rad(20)**2-np.dot(.1*x[3:],.1*x[3:]))]
    if supported:constraints.append(dict(type='ineq',fun=finite_constraints))
    center=vertices-vertices.mean(0);eigen=np.linalg.eigh(np.einsum('ni,nj->ij',center,center,optimize=False))[1]
    axes=np.einsum('ij,jk->ik',pose[:3,:3],eigen,optimize=False)
    starts=[np.zeros(6)]
    for col in [-1,-2]:
        for sign in [-1,1]:starts.append(np.r_[np.zeros(3),sign*axes[:,col]*np.deg2rad(10)/.1])
    records=[];accepted=[]
    for start in starts:
        if supported:
            h=finite_constraints(start)[0]
            start[:3]+=normal*max(0,-h)/.01
        solution=minimize(objective,start,method='SLSQP',bounds=[(-6,6)]*3+[(-np.deg2rad(20)/.1,np.deg2rad(20)/.1)]*3,constraints=constraints,options=dict(maxiter=100,ftol=1e-11))
        candidate=transform(solution.x);after=observed_fit(vertices,faces,tree,candidate,depth,valid,mask,K)
        bounded=np.linalg.norm(.01*solution.x[:3])<=.06+1e-7 and np.linalg.norm(.1*solution.x[3:])<=np.deg2rad(20)+1e-7
        clear=not supported or np.min(finite_constraints(solution.x))>=-1e-7
        improved=after['quality']>before['quality'] and after.get('silhouette_iou',-1)>=before.get('silhouette_iou',0) and after.get('median_measured_vertex_distance_m',np.inf)<before.get('median_measured_vertex_distance_m',0)
        passes=bool(solution.success and bounded and clear and improved and np.isfinite(solution.x).all())
        record=dict(optimizer_success=bool(solution.success),message=str(solution.message),accepted=passes,before=before,after=after,translation_delta_m=(.01*solution.x[:3]).tolist(),rotation_delta_rad=(.1*solution.x[3:]).tolist(),candidate_camera_from_object=candidate.tolist(),finite_support_applied=supported,full_support_constraints_pass=bool(clear),bounded=bool(bounded),function_evaluations=int(solution.nfev))
        records.append(record)
        if passes:accepted.append((after['quality'],candidate,record))
    best=max(accepted,key=lambda x:x[0],default=None)
    return (pose.copy() if best is None else best[1]),dict(accepted=best is not None,before=before,attempts=records,selected_attempt=None if best is None else records.index(best[2]),body_origin_or_scale_changed=False,source_inputs_modified=False,observations_filled=False,physical_task_accepted=False)
