"""Independent finite/einsum controls for camera conversion; reuse prepared pixels and inputs."""
from pathlib import Path
import hashlib,json,sys
S=Path(__file__).resolve().parents[2];sys.path.insert(0,str(S/'compute'))
import numpy as np
from scipy.spatial.transform import Rotation
from numpy_mano_source_geometry_v2 import NumpyMano
np.seterr(all='raise');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=S/'e037-native-RGB-hand-input-v1';inp=json.loads((P/'input.json').read_text());rp=S/'e037-watcloud-fresh-hand-source-audit-v1/unqualified-handoff/priors/right/000050.json';rec=json.loads(rp.read_text());assert sha(rp)==inp['fixed_source_record_sha256']
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');beta=rec['mano']['betas'];J0=m.shaped(beta)[1][0];base=np.r_[rec['mano']['global_orient'],rec['mano']['hand_pose']];rng=np.random.default_rng(42);T=np.asarray(inp['Crect_from_A']);rows=[]
for i in range(10):
 p=base+rng.normal(0,.1,48);original_t=np.asarray(rec['cam_t'])+rng.normal(0,.01,3);v,j,_=m.record(rec,pose=p,translation=original_t)
 for kind,Q,b in [('synthetic',Rotation.from_rotvec(rng.normal(0,.3,3)).as_matrix(),rng.normal(0,.05,3)),('factory',T[:3,:3],T[:3,3])]:
  new=p.copy();new[:3]=Rotation.from_matrix(np.einsum('ij,jk->ik',Q,Rotation.from_rotvec(p[:3]).as_matrix())).as_rotvec();new_t=np.einsum('ij,j->i',Q,original_t+J0)+b-J0
  vv,jj,_=m.record(rec,pose=new,translation=new_t);expect_v=np.einsum('ni,ji->nj',v,Q)+b;expect_j=np.einsum('ni,ji->nj',j,Q)+b
  assert all(np.isfinite(a).all() for a in [new,new_t,vv,jj,expect_v,expect_j]);error=max(float(abs(vv-expect_v).max()),float(abs(jj-expect_j).max()));assert error<(1e-12 if kind=='synthetic' else 1e-6)
  rows.append(dict(trial=i,kind=kind,all_vertices_and_joints_error_m=error))
out=dict(schema='v2d.native-RGB-camera-conversion-finite-control.v2',producer_sha256=sha(Path(__file__)),input_sha256=sha(P/'input.json'),input_exports_not_repeated=True,original_control_float_warning_retained=True,warning_source_unproven=True,independent_einsum_finite_controls_passed=True,floating_point_errors_raise=True,controls=rows,physical_task_accepted=False,native_contact_gate_m=.001)
(P/'finite-conversion-control-v2.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'controls':len(rows),'max_error_m':max(x['all_vertices_and_joints_error_m'] for x in rows)}))
