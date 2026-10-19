"""Independently compare actual HaMeR output geometry with its exported parameters on CPU."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];sys.path.insert(0,str(S/'compute'))
import numpy as np
from scipy.spatial.transform import Rotation
from numpy_mano_source_geometry_v2 import NumpyMano
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-native-HaMeR-geometry-closure-v1';O.mkdir(exist_ok=True)
prediction=S/'e037-watcloud-native-RGB-v1/outputs/native-RGB/prediction.npz';receipt_path=prediction.parent/'receipt.json';receipt=json.loads(receipt_path.read_text());assert sha(prediction)==receipt['prediction_sha256']
model_path=S/'models/mano/models/MANO_RIGHT.pkl';m=NumpyMano(model_path)
with np.load(prediction) as z:pred={k:z[k].copy() for k in z.files}
pose=np.r_[Rotation.from_matrix(pred['global_orient'].reshape(3,3)).as_rotvec(),Rotation.from_matrix(pred['hand_pose'].reshape(15,3,3)).as_rotvec().ravel()]
beta=pred['betas'].reshape(10);vertices,joints,_=m(pose,beta)
raw_v=pred['pred_vertices'].reshape(778,3);raw_j=pred['pred_keypoints_3d'].reshape(-1,3);assert raw_j.shape==(21,3)
assert all(np.isfinite(x).all() for x in [vertices,joints,raw_v,raw_j])
vd=vertices-raw_v;jd=joints-raw_j;offset=np.median(vd,axis=0)
report=dict(schema='v2d.actual-HaMeR-to-exported-parameter-geometry-closure.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),prediction_sha256=sha(prediction),GPU_receipt_sha256=sha(receipt_path),private_MANO_sha256=sha(model_path),geometry_helper_sha256=sha(S/'compute/numpy_mano_source_geometry_v2.py'),predicted_betas_used_without_subject_override=True,prediction_is_right_hand=True,all_vertices_max_abs_error_m=float(abs(vd).max()),all_joints_max_abs_error_m=float(abs(jd).max()),median_vertex_offset_m=offset.tolist(),vertices_max_error_after_constant_offset_m=float(abs(vd-offset).max()),vertex_centroid_error_m=(vertices.mean(0)-raw_v.mean(0)).tolist(),vertex_error_max_norm_m=float(np.linalg.norm(vd,axis=1).max()),joint_error_max_norm_m=float(np.linalg.norm(jd,axis=1).max()),raw_geometry_parameter_closure_passed=bool(max(abs(vd).max(),abs(jd).max())<=1e-6),numeric_closure_tolerance_m=1e-6,source_or_physics_gate_changed=False,original_records_modified=False,reference_accepted=False,physical_task_accepted=False,new_GPU_inference=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(O/'closure-differences.npz',vertex_delta_m=vd,joint_delta_m=jd);print(json.dumps(report,indent=2))
