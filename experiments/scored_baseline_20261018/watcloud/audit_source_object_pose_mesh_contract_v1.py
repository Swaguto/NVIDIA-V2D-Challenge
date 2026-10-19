"""Close the new native overlay to the exact source fitter's own mesh/pose; no fit rerun."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2]
import numpy as np,trimesh
from scipy.spatial.transform import Rotation
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-source-object-pose-mesh-contract-v1';O.mkdir()
P=S/'plane-aligned-native-reference-v1/frozen';trajectory=S/'plane-aligned-pose-temporal-path-v4/inferred-object-trajectory.npz';fit_report=S/'plane-aligned-pose-temporal-path-v4/report.json';fitter=json.loads(fit_report.read_text());assert sha(trajectory)==fitter['candidate_sha256']
source_mesh=S/'scored-cohort-14-results/reconstruction-e037/geometry/e037/wooden_spoon/mesh.glb';mesh=trimesh.load(source_mesh,force='mesh',process=False);vertices=np.asarray(mesh.vertices)*1.3
visual=P/'objects/wooden_spoon-eval-scaled.obj';evaluation=trimesh.load(visual,force='mesh',process=False);assert evaluation.vertices.shape==vertices.shape;mesh_delta=float(abs(evaluation.vertices-vertices).max());assert mesh_delta<1e-8
with np.load(trajectory) as z:source_pose=z['candidate_camera_from_object'][50].copy();source_world=z['candidate_world_from_object'][50].copy();source_camera=z['candidate_camera_c2w'][50].copy()
with np.load(P/'targets.npz') as z:target=z['object_poses'][50,0].copy()
with np.load(P/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
assert np.array_equal(camera,source_camera)
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal'])
native=np.eye(4);native[:3,:3]=Rotation.from_quat(target[3:]).as_matrix();native[:3,3]=target[:3];unrotated=np.eye(4);unrotated[:3,:3]=np.einsum('ji,jk->ik',stage,native[:3,:3]);unrotated[:3,3]=np.einsum('ji,j->i',stage,native[:3,3]);from_native=np.linalg.inv(camera)@unrotated;pose_delta=float(abs(from_native-source_pose).max());assert pose_delta<1e-10
report=dict(schema='v2d.source-fitter-to-native-object-mesh-pose-closure.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),source_hashes={str(p.relative_to(S)):sha(p) for p in [P/'manifest.json',trajectory,fit_report,source_mesh,visual]},frame=50,source_mesh_to_effective_eval_mesh_max_vertex_component_error_m=mesh_delta,source_fitter_to_native_recovered_camera_pose_max_component_error=pose_delta,source_cameras_byte_equal=True,original_source_fit_metrics_reused_not_rerun=fitter['rows'][50]['source_fit_after'],native_overlay_is_same_mesh_pose_used_by_source_fitter=True,source_shape_and_pose_error_not_causally_separated=True,earlier_relative_source_fit_qualification_is_not_absolute_geometry_accuracy=True,all_assets_and_targets_unchanged=True,hidden_or_provided_meshes_read=False,physics_stepped=False,new_GPU_work=False,new_PPO_steps=0,required_native_reference_contact_gate_m=.001,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
