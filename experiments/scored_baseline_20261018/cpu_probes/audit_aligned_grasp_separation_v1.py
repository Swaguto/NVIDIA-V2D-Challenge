"""Bound reconstructed spoon/finger separation without claiming grasp forces.

For convex native pieces, any unit-axis separation is a distance lower bound;
any vertex pair is a distance upper bound. Minima across all object hulls bound
the distance to their union. Positive native distance queries are not used.
Source flags are reported separately and never promoted by these bounds.
"""
import datetime,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from audit_hand_reference_contacts import canonical_id
P=S/'plane-aligned-articulated-native-validation-v2';OUT=S/'plane-aligned-grasp-separation-v1';OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'schema':'v2d.native-convex-grasp-separation-bounds.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'producer_sha256':sha(Path(__file__)),'input_manifest_sha256':sha(P/'frozen/manifest.json'),'native_model_sha256':sha(P/'native-model.mjb'),'physics_stepped':False,'new_training_steps':0,'physical_task_accepted':False,'force_closure_certified':False,'reference_modified':False,'source_confidence_modified':False}
save=lambda:(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');started=time.monotonic();save()
def mm(a,b):
 assert np.isfinite(a).all() and np.isfinite(b).all()
 return np.einsum('ij,jk->ik',a,b,optimize=False)
def bounds(a,an,b,bn):
 axes=np.r_[an,bn];axes=axes/np.linalg.norm(axes,axis=1)[:,None]
 ap=mm(a,axes.T);bp=mm(b,axes.T)
 lower=float(np.maximum(np.maximum(ap.min(axis=0)-bp.max(axis=0),bp.min(axis=0)-ap.max(axis=0)),0).max())
 upper=float(np.sqrt(np.einsum('ijk,ijk->ij',a[:,None]-b[None],a[:,None]-b[None],optimize=False).min()))
 assert np.isfinite([lower,upper]).all() and 0<=lower<=upper+1e-7
 return lower,upper
# Analytic controls exercise both a gap and overlap without relying on native
# positive-distance behavior. The overlap lower bound is allowed to be zero.
box=np.array([[a,b,c] for a in [-.5,.5] for b in [-.5,.5] for c in [-.5,.5]])
axes=np.eye(3)
lo,hi=bounds(box,axes,box+[2.,0,0],axes);assert abs(lo-1)<1e-12 and abs(hi-1)<1e-12
lo,hi=bounds(box,axes,box+[.5,0,0],axes);assert lo==0 and hi>=0
report['analytic_separation_controls_passed']=True
try:
 proof=json.loads((P/'report.json').read_text());assert proof['complete_static_source_contact_motion_stage_passed']
 manifest=json.loads((P/'frozen/manifest.json').read_text())
 for name,h in manifest['files'].items():assert sha(P/'frozen'/name)==h
 with np.load(P/'frozen/targets.npz',allow_pickle=False) as z:q=z['robot_qpos'].copy();obj=z['object_poses'].copy();times=z['timestamps_s'].copy()
 with np.load(P/'frozen/aligned-source-reference.npz',allow_pickle=False) as z:flags={k:z[k].copy() for k in ['hand_valid','hand_directly_observed','hand_metric_valid']}
 model=mujoco.MjModel.from_binary_path(str(P/'native-model.mjb'));data=mujoco.MjData(model)
 bodies=[model.body(i).name for i in range(model.nbody)];joints=[model.joint(i).name for i in range(model.njnt)]
 object_id=canonical_id(bodies,'wooden_spoon_object');oa=int(model.jnt_qposadr[model.body_jntadr[object_id]])
 object_geoms=[i for i,b in enumerate(model.geom_bodyid) if b==object_id];assert len(object_geoms)==9
 binding={};groups={}
 for si,side in enumerate(['left','right']):
  binding[side]=([int(model.jnt_qposadr[canonical_id(joints,f'{side}_wrist_{a}')]) for a in ['x','y','z','rotvec']],[int(model.jnt_qposadr[canonical_id(joints,n)]) for n in manifest['joint_names'][side]])
  for digit,link in [('thumb',2),('index',1),('middle',1)]:
   groups[side+'/'+digit+'/distal']=[i for i,b in enumerate(model.geom_bodyid) if bodies[b].startswith('dex3_'+side+'_') and bodies[b].endswith(f'hand_{digit}_{link}_link')]
   groups[side+'/'+digit+'/whole_digit']=[i for i,b in enumerate(model.geom_bodyid) if bodies[b].startswith('dex3_'+side+'_') and ('hand_'+digit+'_') in bodies[b]]
 assert all(groups.values())
 local={}
 for g in set(object_geoms+[i for v in groups.values() for i in v]):
  mesh=int(model.geom_dataid[g]);assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
  a=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);v=np.asarray(model.mesh_vert[a:a+n],float);h=ConvexHull(v)
  normals=h.equations[:,:3];_,ids=np.unique(np.round(normals,10),axis=0,return_index=True)
  local[g]=(v[h.vertices],normals[ids])
 rows=[]
 for frame,t in enumerate(times):
  if time.monotonic()-started>240:raise TimeoutError('CPU separation audit wall cap')
  data.qpos[:]=model.qpos0;data.qpos[oa:oa+3]=obj[frame,0,:3];data.qpos[oa+3:oa+7]=obj[frame,0,[6,3,4,5]]
  for si,side in enumerate(['left','right']):
   wrists,fingers=binding[side]
   data.qpos[wrists[:3]]=q[frame,si,:3];a=wrists[3];data.qpos[a:a+4]=q[frame,si,[6,3,4,5]];data.qpos[fingers]=q[frame,si,7:]
  mujoco.mj_fwdPosition(model,data)
  world={g:(mm(v,data.geom_xmat[g].reshape(3,3).T)+data.geom_xpos[g],mm(n,data.geom_xmat[g].reshape(3,3).T)) for g,(v,n) in local.items()}
  pairs={}
  for g in set(i for v in groups.values() for i in v):
   a,an=world[g];results=[]
   for og in object_geoms:
    b,bn=world[og];results.append(bounds(a,an,b,bn))
   pairs[g]=(min(r[0] for r in results),min(r[1] for r in results))
  result={name:{'lower_m':min(pairs[g][0] for g in geoms),'upper_m':min(pairs[g][1] for g in geoms)} for name,geoms in groups.items()}
  rows.append({'frame':frame,'time_s':float(t),'source_flags':{s:{k:bool(v[frame,si]) for k,v in flags.items()} for si,s in enumerate(['left','right'])},'distance_bounds':result})
  if frame%25==0:print(json.dumps({'frame':frame,'right_distal_lower_mm':{d:result['right/'+d+'/distal']['lower_m']*1000 for d in ['thumb','index','middle']}}),flush=True)
 summaries=[]
 for group in groups:
  lower=np.array([r['distance_bounds'][group]['lower_m'] for r in rows]);upper=np.array([r['distance_bounds'][group]['upper_m'] for r in rows]);si=['left','right'].index(group.split('/')[0]);observed=flags['hand_valid'][:,si]&flags['hand_directly_observed'][:,si]
  summaries.append({'group':group,'minimum_lower_m':float(lower.min()),'median_lower_m':float(np.median(lower)),'p95_lower_m':float(np.quantile(lower,.95)),'minimum_upper_m':float(upper.min()),'median_upper_m':float(np.median(upper)),'frames_definitely_farther_than_1cm':np.flatnonzero(lower>.01).tolist(),'frames_with_distance_upper_bound_under_2mm':np.flatnonzero(upper<.002).tolist(),'directly_observed_frames':int(observed.sum()),'directly_observed_median_lower_m':float(np.median(lower[observed])) if observed.any() else None})
 (OUT/'frame-bounds.json').write_text(json.dumps({'frames':rows},indent=2)+'\n')
 report.update(frames=len(rows),native_object_hulls=9,summary=summaries,frame_bounds_sha256=sha(OUT/'frame-bounds.json'),distance_bound_scope='Independent convex face-axis lower and vertex-pair upper bounds. No signed-depth inference or contact label from a zero lower bound. Distances do not establish opposing contact normals or force closure.')
 for name,h in manifest['files'].items():assert sha(P/'frozen'/name)==h
 report['frozen_closure_verified_after']=True
except BaseException as error:report['error']=repr(error);raise
finally:
 report['wall_seconds']=time.monotonic()-started;save();print(json.dumps({k:v for k,v in report.items() if k!='summary'},indent=2),flush=True)
