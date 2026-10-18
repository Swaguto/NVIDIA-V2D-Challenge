"""Private measured-surface experiment; no simulation, pose labels or CAD.

Fit one uniform shape scale and anchor translation from the existing masked
stereo pointmap. Keep the SAM body origin and rotation. This is a candidate,
never a replacement for the frozen baseline or a physical certificate.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import cv2,numpy as np,trimesh
from scipy.optimize import minimize
from scipy.spatial import cKDTree

REPO=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(REPO/'scripts/track3'))
from select_pose_hypothesis import observed_fit

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(source,out,support_report=None,bundle=None,initial_candidate=None):
 out.mkdir(parents=True,exist_ok=False)
 m=json.loads((source/'manifest.json').read_text());mesh=trimesh.load(source/'mesh.glb',force='mesh',process=False)
 vertices=np.asarray(mesh.vertices);pose=np.asarray(m['seed_camera_from_object']);R=pose[:3,:3];initial=pose[:3,3]
 points=np.load(source/'pointmap.npy',allow_pickle=False);valid=np.isfinite(points).all(-1)&(points[...,2]>.1)&(points[...,2]<2.5)
 mask=cv2.imread(str(source/'mask.png'),0)>127;valid&=mask
 z=points[...,2];camera=json.loads((source/'input-intrinsics.json').read_text());K=np.array([[camera['fx'],0,camera['cx']],[0,camera['fy'],camera['cy']],[0,0,1.]])
 y,x=np.nonzero(valid)
 if len(y)<200:raise ValueError('Too little measured foreground')
 sample=np.linspace(0,len(x)-1,min(len(x),4096),dtype=int);p=points[y[sample],x[sample]]
 tree=cKDTree(vertices);outside=cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,3)
 vi=np.linspace(0,len(vertices)-1,min(len(vertices),4096),dtype=int);v=vertices[vi]
 plane=None;plane_sources={}
 if support_report is not None:
  bm=json.loads((bundle/'manifest.json').read_text())
  if bm['episode']!=m['episode'] or bm.get('provided_meshes_read') is not False or bm.get('reference_pose_labels_read') is not False:raise ValueError('Need matching label-free camera bundle')
  world=np.asarray(json.loads(support_report.read_text())['plane_world'])
  if not np.isclose(np.linalg.norm(world[:3]),1):raise ValueError('Support plane normal invalid')
  with np.load(bundle/'episode.npz',allow_pickle=False) as f:
   index=list(f['frame_indices']).index(m['seed_frame']);c2w=f['camera_c2w'][index]
  plane=np.r_[c2w[:3,:3].T@world[:3],world[3]+world[:3]@c2w[:3,3]]
  plane_sources=dict(support_report_sha256=sha(support_report),camera_bundle_manifest_sha256=sha(bundle/'manifest.json'),source_table_plane_changed=False,constraint_scope='Seed body minimum against independently inferred support plane; this pilot is known to be over the table, not a general off-table classifier')
 def height(a):
  if plane is None:return None
  return float(np.exp(a[3])*np.min((vertices@R.T)@plane[:3])+(initial+a[:3])@plane[:3]+plane[3])
 def objective(a):
  delta=a[:3];scale=np.exp(a[3]);t=initial+delta
  with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
   local=(p-t)@R/scale;cam=(v*scale)@R.T+t
  if not np.isfinite(local).all() or not np.isfinite(cam).all() or (cam[:,2]<.05).any():return 1e3
  d=tree.query(local,workers=1)[0]*scale;trim=np.sort(d)[:int(.8*len(d))]
  # Huber limits corrupt mask/depth leverage without filling unknown pixels.
  robust=np.where(trim<.01,.5*trim**2,.01*(trim-.005)).mean()
  u=K[0,0]*cam[:,0]/cam[:,2]+K[0,2];w=K[1,1]*cam[:,1]/cam[:,2]+K[1,2]
  ix=np.clip(np.rint(u).astype(int),0,mask.shape[1]-1);iy=np.clip(np.rint(w).astype(int),0,mask.shape[0]-1)
  metric_outside=np.maximum(0,outside[iy,ix]-5)*cam[:,2]/K[0,0]
  invalid=(u<0)|(u>=mask.shape[1])|(w<0)|(w>=mask.shape[0]);metric_outside[invalid]=.1
  penalty=20*max(0,.002-height(a))**2 if plane is not None else 0
  return float(robust+.5*np.minimum(metric_outside,.1).dot(np.minimum(metric_outside,.1))/len(v)+(0.003*a[3]/.2)**2+(0.002*np.linalg.norm(delta)/.05)**2+penalty)
 start=np.zeros(4)
 if initial_candidate is not None:
  prior=json.loads(initial_candidate.read_text())
  if prior['source_manifest_sha256']!=sha(source/'manifest.json'):raise ValueError('Initial candidate source differs')
  start=np.r_[prior['anchor_translation_delta_m'],np.log(prior['uniform_scale'])]
 if plane is not None:
  start[:3]+=plane[:3]*max(0,.002-height(start))
 fit=minimize(objective,start,method='SLSQP',bounds=[(-.03,.03),(-.03,.03),(-.12,.12),(np.log(.7),np.log(1.3))],constraints=[{'type':'ineq','fun':lambda a:height(a)-.002}] if plane is not None else [],options={'maxiter':200,'ftol':1e-11})
 scale=float(np.exp(fit.x[3]));newpose=pose.copy();newpose[:3,3]+=fit.x[:3];newmesh=mesh.copy();newmesh.vertices=vertices*scale
 before=observed_fit(vertices,np.asarray(mesh.faces),tree,pose,z,valid,mask,K)
 after=observed_fit(np.asarray(newmesh.vertices),np.asarray(newmesh.faces),cKDTree(newmesh.vertices),newpose,z,valid,mask,K)
 newmesh.export(out/'mesh.glb')
 result=dict(schema='v2d.measured-surface-metric-candidate.v1',episode=m['episode'],object=m['object'],source_manifest_sha256=sha(source/'manifest.json'),source_mesh_sha256=sha(source/'mesh.glb'),source_pointmap_sha256=sha(source/'pointmap.npy'),body_origin_changed=False,body_rotation_changed=False,uniform_scale=scale,anchor_translation_delta_m=fit.x[:3].tolist(),before=before,after=after,objective_before=objective(np.zeros(4)),objective_after=float(fit.fun),optimizer_success=bool(fit.success),function_evaluations=int(fit.nfev),candidate_mesh_sha256=sha(out/'mesh.glb'),seed_camera_from_object=newpose.tolist(),method_frozen=False,physical_task_accepted=False,reference_pose_labels_read=False,provided_meshes_read=False,source_files_modified=False,producer_sha256=sha(Path(__file__)))
 result.update(support_plane_constraint=plane_sources,seed_minimum_support_height_before_m=height(np.zeros(4)),seed_minimum_support_height_after_m=height(fit.x))
 result.update(initial_candidate_sha256=sha(initial_candidate) if initial_candidate else None,initial_parameters=start.tolist(),optimizer_message=str(fit.message))
 (out/'candidate-report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--support-report',type=Path);ap.add_argument('--bundle',type=Path);ap.add_argument('--initial-candidate',type=Path);a=ap.parse_args();run(a.source,a.out,a.support_report,a.bundle,a.initial_candidate)
