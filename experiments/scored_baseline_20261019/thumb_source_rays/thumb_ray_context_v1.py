"""Read-only native/source context, starting from the independently verified combined left/right freeze."""
from pathlib import Path
import hashlib, itertools, json, sys, time
S=Path(__file__).resolve().parents[1]
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
import numpy as np, mujoco
from audit_hand_reference_contacts import canonical_id
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
F=S/'e037-right-late-independent-validation-v1/frozen'
MP=S/'e037-motion-constrained-wrist-native-v1/native-model.mjb'
assert sha(F/'manifest.json')=='505b62f8073337a36004be42ee17e4a9d19e7f0561f72c98f2928dbef1e12a2b'
assert sha(MP)=='755fc51729d4c4fc03d04108d39bec2b249074351863f096f9aa498c9ac28566'
manifest=json.loads((F/'manifest.json').read_text())
for n,h in manifest['files'].items():assert sha(F/n)==h
with np.load(F/'targets.npz') as z:targets={k:z[k].copy() for k in z.files}
q=targets['robot_qpos'];objects=targets['object_poses'];ts=targets['timestamps_s']
with np.load(F/'aligned-source-reference.npz') as z:aligned=z['robot_qpos'].copy();cameras=z['camera_c2w'].copy();human=z['hand_joints_world'].copy()
with np.load(F/'source-confidence.npz') as z:flags={k:z[k].copy() for k in z.files}
with np.load(S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz') as z:K=z['intrinsics'].copy()
stage=np.array(json.loads((F/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
model=mujoco.MjModel.from_binary_path(str(MP));data=mujoco.MjData(model)
bn=[model.body(i).name for i in range(model.nbody)];jn=[model.joint(i).name for i in range(model.njnt)]
wrists={s:[canonical_id(jn,f'{s}_wrist_{a}') for a in ['x','y','z','rotvec']] for s in ['left','right']}
fingers={s:np.array([canonical_id(jn,n) for n in manifest['joint_names'][s]]) for s in ['left','right']}
rj=fingers['right'];rdofs=model.jnt_dofadr[rj];rqa=model.jnt_qposadr[rj]
limits=model.jnt_range[rj].copy()
rgeoms=[g for g,b in enumerate(model.geom_bodyid) if bn[b].startswith('dex3_right_')]
pairs=list(itertools.combinations(rgeoms,2));pair_index={tuple(sorted(p)):i for i,p in enumerate(pairs)}
tipids=[canonical_id(bn,n) for n in ['right_hand_palm_link','right_thumb_tip','right_index_tip','right_middle_tip']]
def set_state(frame,angles=None):
    data.qpos[:]=model.qpos0;data.qvel[:]=0
    for si,s in enumerate(['left','right']):
        v=q[frame,si];w=wrists[s]
        data.qpos[model.jnt_qposadr[w[:3]]]=v[:3];a=int(model.jnt_qposadr[w[3]])
        data.qpos[a:a+4]=v[[6,3,4,5]];data.qpos[model.jnt_qposadr[fingers[s]]]=v[7:]
    if angles is not None:data.qpos[rqa]=angles
    for oi,obj in enumerate(manifest['objects']):
        bid=canonical_id(bn,obj['name']+'_object');a=int(model.jnt_qposadr[model.body_jntadr[bid]]);v=objects[frame,oi]
        data.qpos[a:a+3]=v[:3];data.qpos[a+3:a+7]=v[[6,3,4,5]]
    mujoco.mj_fwdPosition(model,data)
def uv(C):
    assert np.isfinite(C).all() and (C[:,2]>.01).all()
    return np.c_[K[0,0]*C[:,0]/C[:,2]+K[0,2],K[1,1]*C[:,1]/C[:,2]+K[1,2]]
def evaluate(frame,angles):
    set_state(frame,angles)
    gaps=np.zeros(len(pairs));gj=np.zeros((len(pairs),7));records=[]
    for c in data.contact[:data.ncon]:
        ga,gb=int(c.geom1),int(c.geom2)
        if ga not in rgeoms or gb not in rgeoms:continue
        records.append(dict(native_bodies=[bn[int(model.geom_bodyid[ga])],bn[int(model.geom_bodyid[gb])]],native_geom_ids=[ga,gb],signed_gap_m=float(c.dist)))
        k=pair_index[tuple(sorted([ga,gb]))]
        if c.dist>=gaps[k]:continue
        gaps[k]=c.dist;n=np.asarray(c.frame[:3]);pa=c.pos-.5*c.dist*n;pb=c.pos+.5*c.dist*n
        ja=np.zeros((3,model.nv));jb=ja.copy()
        mujoco.mj_jac(model,data,ja,None,pa,int(model.geom_bodyid[ga]));mujoco.mj_jac(model,data,jb,None,pb,int(model.geom_bodyid[gb]))
        gj[k]=np.einsum('i,ij->j',n,jb[:,rdofs]-ja[:,rdofs],optimize=False)
    camera=cameras[frame];A=np.einsum('ij,kj->ik',camera[:3,:3].T,stage,optimize=False)
    offset=camera[:3,:3].T@camera[:3,3]
    C=np.einsum('ij,nj->ni',A,data.xpos[tipids],optimize=False)-offset
    observed=np.einsum('ni,ij->nj',human[frame,1,[0,4,8,12]]-camera[:3,3],camera[:3,:3],optimize=False)
    pixels=uv(C);source_mask=np.isfinite(observed).all(axis=1)&(observed[:,2]>.01)&bool(flags['hand_valid'][frame,1]);obs=np.full((4,2),np.nan);obs[source_mask]=uv(observed[source_mask]);err=np.linalg.norm(pixels-obs,axis=1);ej=np.full((4,7),np.nan);pixel_jac=np.full((4,2,7),np.nan)
    for i,b in enumerate(tipids):
        if not source_mask[i]:continue
        J=np.zeros((3,model.nv));mujoco.mj_jac(model,data,J,None,data.xpos[b],b)
        cj=np.einsum('ij,jk->ik',A,J[:,rdofs],optimize=False);c=C[i]
        pj=np.array([[K[0,0]/c[2],0,-K[0,0]*c[0]/c[2]**2],[0,K[1,1]/c[2],-K[1,1]*c[1]/c[2]**2]])
        pixel_jac[i]=np.einsum('ij,jk->ik',pj,cj,optimize=False)
        ej[i]=np.einsum('i,ij,jk->k',(pixels[i]-obs[i])/max(err[i],1e-12),pj,cj,optimize=False)
    return dict(gaps=gaps,gap_jac=gj,errors=err,error_jac=ej,self_contacts=records,source_mask=source_mask,pixel_residual=pixels-obs,pixel_jac=pixel_jac)
def bounds(frame):
    origin=aligned[frame,1,7:]
    return np.maximum(limits[:,0],origin-.25),np.minimum(limits[:,1],origin+.25)
