"""Independent ray/triangle skin-depth audit; do not compare occluded bones to skin.

Reuse the rejected rigid fit unchanged. Does not override its outcome. Visible
estimated depth measures the front surface, not a labelled anatomical joint.
"""
from pathlib import Path
import hashlib,json
import numpy as np
S=Path(__file__).resolve().parents[1];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ray_depth(vertices,faces,pixels,K):
 tri=vertices[faces];a=tri[:,0];edge1=tri[:,1]-a;edge2=tri[:,2]-a;rays=np.c_[(pixels[:,0]-K[0,2])/K[0,0],(pixels[:,1]-K[1,2])/K[1,1],np.ones(len(pixels))]
 h=np.cross(rays[:,None,:],edge2[None]);det=np.einsum('fi,rfi->rf',edge1,h);nonzero=abs(det)>1e-12;inv=np.divide(1,det,out=np.zeros_like(det),where=nonzero);ss=-a;u=np.einsum('fi,rfi->rf',ss,h)*inv;q=np.cross(ss,edge1);v=np.einsum('ri,fi->rf',rays,q)*inv;t=np.einsum('fi,fi->f',edge2,q)[None]*inv;valid=nonzero&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>0);return np.min(np.where(valid,t,np.inf),axis=1)
if __name__=='__main__':
 O=S/'e037-source-hand-surface-depth-audit-v1';O.mkdir();P=S/'e037-source-hand-motion-fit-v1';r=json.load(open(P/'report.json'));B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
 # Unit triangle at known z; one hit and one miss.
 control=ray_depth(np.array([[-1,-1,2],[1,-1,2],[0,1,2.]]),np.array([[0,1,2]]),np.array([[0,0],[2,2.]]),np.eye(3));assert abs(control[0]-2)<1e-12 and np.isinf(control[1])
 rows=[]
 with np.load(B) as b,np.load(P/'source-hand-candidate.npz') as z:
  for old in r['rows']:
   f=old['frame']
   if not np.isfinite(z['hand_vertices_world'][f]).all():continue
   cam=b['camera_c2w'][f];verts=np.einsum('ni,ij->nj',z['hand_vertices_world'][f]-cam[:3,3],cam[:3,:3]);uv=z['landmarks_px'][f];valid=np.isfinite(uv).all(-1)&np.isfinite(z['estimated_depth_m'][f]);ids=np.flatnonzero(valid);d=ray_depth(verts,b['hand_right_faces'],uv[ids],b['intrinsics']);hold=np.isin(ids,r['heldout_joint_ids']);measured=z['estimated_depth_m'][f,ids]-.005;error=abs(d-measured);good=np.isfinite(d)&hold
   rows.append(dict(frame=f,heldout_depth_rays=int(hold.sum()),heldout_mesh_hits=int(good.sum()),heldout_front_surface_median_error_m=float(np.median(error[good])) if good.any() else None,heldout_front_surface_p95_error_m=float(np.quantile(error[good],.95)) if good.any() else None,previous_joint_centre_depth_error_m=old['heldout_estimated_depth_median_error_m'],rigid_fit_outcome_unchanged=old['status']))
 report=dict(schema='v2d.source-hand-surface-depth-audit.v1',producer_sha256=sha(Path(__file__)),candidate_sha256=sha(P/'source-hand-candidate.npz'),ray_triangle_control_passed=True,original_failed_joint_gate_unchanged=True,training_approved=False,rows=rows)
 (O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([x for x in rows if x['frame'] in [25,35,45,50,67,75,99]],indent=2))
