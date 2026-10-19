"""Check exact fingertip rays in a fitted estimated mesh; do not label the video."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
S=Path(__file__).resolve().parents[2];sys.path.insert(0,str(S/'compute'))
from numpy_mano_source_geometry_v2 import NumpyMano
from audit_source_hand_surface_depth_v1 import ray_depth
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
O=S/'e037-model-tip-visibility-audit-v1';O.mkdir();f=50
request_path=S/'e037-fresh-source-hand-request-v2/request.json';request=json.loads(request_path.read_text());row=next(x for x in request['frames'] if x['frame_idx']==f)
record_path=S/f'e037-watcloud-fresh-hand-source-audit-v1/unqualified-handoff/priors/right/{f:06d}.json';record=json.loads(record_path.read_text())
candidate=S/'e037-watcloud-fresh-hand-source-fit-v1/source-hand-candidate.npz'
with np.load(candidate) as z:i=int(np.flatnonzero(z['frame_indices']==f)[0]);pose=z['mano_pose'][i].copy();translation=z['camera_translation'][i].copy()
with np.load(request['canonical_bundle']) as z:K=z['intrinsics'].copy()
m=NumpyMano(S/'models/mano/models/MANO_RIGHT.pkl');v,j,_=m.record(record,pose=pose,translation=translation)
tip_ids=np.array([4,8,12,16,20]);vertex_ids=np.array([745,317,444,556,673]);closure=float(abs(j[tip_ids]-v[vertex_ids]).max());assert closure<1e-12
uv=np.c_[K[0,0]*j[tip_ids,0]/j[tip_ids,2]+K[0,2],K[1,1]*j[tip_ids,1]/j[tip_ids,2]+K[1,2]]
front=ray_depth(v,m.faces,uv,K);source=np.asarray(row['keypoints']);rows=[]
for k,joint in enumerate(tip_ids):
 gap=float(j[joint,2]-front[k]) if np.isfinite(front[k]) else None
 rows.append(dict(joint_id=int(joint),vertex_id=int(vertex_ids[k]),model_pixel=uv[k].tolist(),source_image_point_is_estimated=True,source_image_point_has_support=bool(source[joint,2]>0),source_point_error_px=float(np.linalg.norm(uv[k]-source[joint,:2])) if source[joint,2]>0 else None,front_surface_depth_m=float(front[k]) if np.isfinite(front[k]) else None,model_vertex_depth_m=float(j[joint,2]),gap_behind_front_surface_m=gap,occluded_in_this_estimated_mesh=bool(gap is not None and gap>1e-5)))
report=dict(schema='v2d.estimated-mesh-fingertip-visibility-audit.v1',producer_sha256=sha(Path(__file__)),source_candidate_sha256=sha(candidate),source_request_sha256=sha(request_path),source_record_sha256=sha(record_path),frame=f,tip_joint_to_surface_vertex_closure_m=closure,ray_uses_exact_model_projected_pixel=True,model_visibility_is_not_source_video_visibility=True,source_confidence_or_depth_weights_modified=False,original_failed_joint_screen_overridden=False,geometric_visibility_numeric_epsilon_m=1e-5,native_reference_contact_tolerance_m=.001,native_contact_gate_changed=False,training_approved=False,physical_task_accepted=False,rows=rows)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
