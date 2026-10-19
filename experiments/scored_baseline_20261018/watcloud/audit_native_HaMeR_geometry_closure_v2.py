"""Verify the explicit fingertip adapter against saved actual model outputs; preserve v1 failure."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
import numpy as np
from source_hand_landmarks import TIP_JOINT_IDS,TIP_VERTEX_IDS,convert_mano_landmarks
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-native-HaMeR-geometry-closure-v2';O.mkdir(exist_ok=True)
parent=S/'e037-native-HaMeR-geometry-closure-v1';report=json.loads((parent/'report.json').read_text());prediction=S/'e037-watcloud-native-RGB-v1/outputs/native-RGB/prediction.npz';assert sha(prediction)==report['prediction_sha256']
with np.load(prediction) as z:raw_v=z['pred_vertices'].astype(float);raw_j=z['pred_keypoints_3d'].astype(float)
with np.load(parent/'closure-differences.npz') as z:v=raw_v+z['vertex_delta_m'];j=raw_j+z['joint_delta_m']
converted=convert_mano_landmarks(v,j,source_layout='manotorch21',target_layout='hamer_smplx21');roundtrip=convert_mano_landmarks(v,converted,source_layout='hamer_smplx21',target_layout='manotorch21');assert np.array_equal(roundtrip,j)
error=float(abs(converted-raw_j).max());assert error<=report['numeric_closure_tolerance_m']
non_tips=[i for i in range(21) if i not in TIP_JOINT_IDS];assert np.array_equal(converted[non_tips],j[non_tips])
tip_rows=[]
for joint in TIP_JOINT_IDS:
 distance=np.linalg.norm(raw_v-raw_j[joint],axis=1);vertex=int(distance.argmin());assert distance[vertex]==0
 tip_rows.append(dict(joint_id=joint,actual_GPU_output_vertex_id=vertex,actual_GPU_tip_vertex_error_m=float(distance[vertex]),old_native_to_GPU_tip_delta_m=float(np.linalg.norm(j[joint]-raw_j[joint]))))
result=dict(schema='v2d.actual-HaMeR-explicit-landmark-closure.v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),adapter_sha256=sha(R/'scripts/track3/source_hand_landmarks.py'),prediction_sha256=sha(prediction),failed_control_report_sha256=sha(parent/'report.json'),reused_saved_geometry_no_LBS_or_GPU_rerun=True,source_layout='manotorch21',target_layout='hamer_smplx21',layouts=TIP_VERTEX_IDS,joint_max_abs_error_m=error,unchanged_closure_tolerance_m=report['numeric_closure_tolerance_m'],all_21_actual_GPU_joints_close=True,non_tip_skeleton_exactly_preserved=True,canonical_roundtrip_exact=True,canonical_mesh_or_records_modified=False,heldout_samples_or_source_authority_changed=False,original_fit_failures_retained=True,tip_rows=tip_rows,native_reference_contact_gate_m=.001,new_training_steps=0,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
