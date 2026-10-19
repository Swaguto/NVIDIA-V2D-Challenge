"""Passive source skin/association audit of NEW selected potential sites.

Use saved meshes and licensed topology/skin labels only. No LBS/inference/source
fit, contact-load re-solve, native forward, command mutation or truth upgrade.
"""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path[:0]=[str(S/'compute'),str(S/'mediapipe-cpu-deps-v1'),str(R/'scripts/track3')]
import cv2,numpy as np
from trimesh.triangles import closest_point
from source_hand_depth import front_surface_depth
from numpy_mano_source_geometry_v2 import NumpyMano
O=S/'e037-wrench-site-source-skin-v1';start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();cv2.setNumThreads(1)
plan_path=S/'compute/wrench-site-source-skin-CPU-plan-v1.json';plan=json.loads(plan_path.read_text())
site_path=S/'e037-source-patch-wrench-sites-v1/report.json';sites=json.loads(site_path.read_text());assert sites['no_native_compile_or_forward_or_physics_steps']
for path,h in sites['inputs_sha256'].items():assert sha(Path(path))==h
model_path=S/'models/mano/models/MANO_RIGHT.pkl';mano=NumpyMano(model_path)
faces=np.asarray(mano.faces,int);weights=np.asarray(mano.weights);assert weights.shape==(778,16);dominant=weights.argmax(1)
# Explicit MANO skeletal segment ids from the same saved 16-joint tree/reorder.
segments={'thumb':[13,14,15],'index':[1,2,3],'middle':[4,5,6]}
assert np.array_equal(mano.reorder[:13],[0,13,14,15,16,1,2,3,17,4,5,6,18])
P=S/'e037-motion-constrained-wrist-native-v1/frozen'
with np.load(S/'e037-source-wrist-convention-audit-v1/saved-palm-orientation-diagnostics.npz') as z:native_from_C=z['native_from_C'].copy()
with np.load(P/'aligned-source-reference.npz') as z:camera=z['camera_c2w'][50].copy()
stage=np.asarray(json.loads((P/'command-proposal-report.json').read_text())['stage_rotation_from_inferred_support_normal']);origin=stage@camera[:3,3]
bundle=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(bundle) as z:old_camera=z['camera_c2w'][50].copy();K=z['intrinsics'].copy()
hand_mask_path=S/'e037-watcloud-mask-refinement-handoff-v1/image-observations/right/000050-mask.png';object_mask_path=S/'scored-cohort-14-results/reconstruction-e037/source/e037/masks/wooden_spoon/000050.png'
hand_mask=cv2.imread(str(hand_mask_path),0);object_mask=cv2.imread(str(object_mask_path),0);assert hand_mask is not None and hand_mask.shape==object_mask.shape
source_paths=[S/'e037-native-RGB-hand-source-audit-v1/unfitted-source-geometry.npz',S/'e037-native-RGB-multiview-source-fit-v1/source-hand-candidate.npz',S/'e037-native-RGB-hard-skin-coverage-fit-v2/source-hand-candidate.npz'];rows=[]
for i,(case,path) in enumerate(zip(sites['rows'],source_paths,strict=True)):
    seed=case['proposed_site_seed'];assert seed is not None
    with np.load(path) as z:vertices_C=z['vertices_camera_C'].copy() if i==0 else (z['vertices_world'][0]-old_camera[:3,3])@old_camera[:3,:3]
    verts=vertices_C@native_from_C.T+origin;triangles=verts[faces];site_rows=[]
    for digit in segments:
        site=np.asarray(seed['sites'][digit]['point']);object_normal=np.asarray(seed['sites'][digit]['normal'])
        nearest=closest_point(triangles,np.broadcast_to(site,(len(triangles),3)));distance=np.linalg.norm(nearest-site,axis=1);whole=int(distance.argmin())
        # All triangle vertices belong to the requested finger, with at least one
        # dominated by its distal segment. This is model association, not visibility.
        selected=np.flatnonzero(np.isin(dominant[faces],segments[digit]).all(1)&(dominant[faces]==segments[digit][-1]).any(1));assert len(selected)
        k=int(selected[np.argmin(distance[selected])]);triangle=triangles[k];normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
        C=(site-origin)@native_from_C;assert C[2]>0
        uv=np.array([K[0,0]*C[0]/C[2]+K[0,2],K[1,1]*C[1]/C[2]+K[1,2]])
        front=float(front_surface_depth(vertices_C,faces,uv[None],K)[0]);pixel=np.rint(uv).astype(int);inside=bool(0<=pixel[0]<hand_mask.shape[1] and 0<=pixel[1]<hand_mask.shape[0])
        site_rows.append(dict(digit=digit,nearest_inferred_whole_hand_skin_distance_m=float(distance[whole]),nearest_whole_hand_face_dominant_segment_ids=dominant[faces[whole]].tolist(),associated_inferred_distal_skin_distance_m=float(distance[k]),associated_skin_outward_dot_object_outward=float(np.dot(normal,object_normal)),associated_finger_triangle_count=len(selected),projected_source_C_pixel=uv.tolist(),site_source_C_depth_m=float(C[2]),inferred_hand_front_depth_m=front if np.isfinite(front) else None,site_minus_inferred_hand_front_depth_m=float(C[2]-front) if np.isfinite(front) else None,inside_source_image=inside,inferred_hand_mask_at_site=bool(hand_mask[pixel[1],pixel[0]]>0) if inside else None,inferred_object_mask_at_site=bool(object_mask[pixel[1],pixel[0]]>0) if inside else None,contact_or_visibility_directly_observed=False,source_metric_or_contact_truth=False))
    rows.append(dict(source_label=case['source_label'],source_flags=case['source_flags'],source_pose_and_heldout_failures_unchanged=True,sites=site_rows))
report=dict(schema='v2d.new-wrench-site-source-skin-association.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),plan_sha256=sha(plan_path),inputs_sha256={str(p):sha(p) for p in [site_path,model_path,bundle,hand_mask_path,object_mask_path]+source_paths},licensed_topology_and_segment_labels_used_privately_only=True,no_new_LBS_or_inference_or_source_fit=True,no_native_or_load_solver_rerun=True,actual_undistorted_C_intrinsics_used_with_C_points=True,RGB_pixels_not_substituted_under_stereo_intrinsics=True,all_mask_and_model_visibility_indicators_remain_inferred=True,source_contact_confidence_not_upgraded=True,full_reference_source_contact_motion_motor_or_grasp_not_qualified=True,required_fresh_full_native_reference_contact_gate_m=.001,rows=rows,new_GPU_work=False,new_PPO_steps=0,training_approved=False,physical_task_accepted=False,wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
