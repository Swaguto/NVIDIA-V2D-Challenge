"""NEW original source prerequisite for one saved root/F branch combination.

Pure actual URDF FK, no native position/physics/source fitting or alternate pose.
All unchanged source caps and missing flags remain authoritative.
"""
from pathlib import Path
import datetime, hashlib, json, os, sys, time
import xml.etree.ElementTree as ET
import numpy as np

S=Path('/Users/rohithiruku/v2d_scored_baseline_20261017');C=S/'compute'
R=Path('/Users/rohithiruku/v2d_hand_continuity_fix')
sys.path.insert(0,str(R/'scripts/dex3_training'))
from v2d_dex3_flashchord.kinematics import URDFKinematics
from thumb_conic_geometry_v1 import plain, journal
O=S/'e037-preserved-branch-source-v1';start=time.monotonic()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
F=S/'e037-thumb-ray-independent-validation-v1/frozen'
P=S/'e037-right-late-independent-validation-v1/frozen'
L=S/'e037-left-thumb-independent-validation-v1/frozen'
W=S/'e037-strict-source-wrist-mode-v1/UNQUALIFIED-strict-source-root-proposal.npz'
U=F/'robot/urdfs/g1/dex3_right.urdf';KFILE=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
inputs=[F/'manifest.json',F/'targets.npz',F/'aligned-source-reference.npz',F/'source-confidence.npz',F/'command-proposal-report.json',F.parent/'report.json',P/'targets.npz',L/'targets.npz',W,U,KFILE,R/'scripts/dex3_training/v2d_dex3_flashchord/kinematics.py']
with (O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 journal(log,dict(kind='BEFORE',scope='one saved root mode plus immutable F palm/thumb source prerequisite',inputs_sha256={str(p):sha(p)for p in inputs},source_image_tolerance_px=1e-6,original_depth_m=.01,no_old_source_job_replay=True,no_index_middle_or_root_proposal=True))
 manifest=read(F/'manifest.json');assert sha(F/'manifest.json')=='f438adf3da846ec1c76c31c7b6d6c12b46bea76004aa7745cfada504378da606'
 with np.load(F/'targets.npz')as z:q=z['robot_qpos'].copy()
 with np.load(P/'targets.npz')as z:prior=z['robot_qpos'].copy()
 with np.load(L/'targets.npz')as z:left=z['robot_qpos'].copy()
 with np.load(W)as z:w=z['robot_qpos'].copy()
 with np.load(F/'aligned-source-reference.npz')as z:camera=z['camera_c2w'].copy();human=z['hand_joints_world'].copy()
 with np.load(F/'source-confidence.npz')as z:valid=z['hand_valid'].copy()
 with np.load(KFILE)as z:K=z['intrinsics'].copy()
 stage=np.asarray(read(F/'command-proposal-report.json')['stage_rotation_from_inferred_support_normal'])
 caps={r['frame']:r for r in read(F.parent/'report.json')['source_rows']if r['hand']=='right'}
 frames=np.flatnonzero(np.any(w[:,1,:7]!=q[:,1,:7],axis=1)).tolist()
 early=np.flatnonzero(np.any(q[:,1,11:14]!=prior[:,1,11:14],axis=1)).tolist()
 late=np.flatnonzero(np.any(prior[:,1,11:14]!=left[:,1,11:14],axis=1)).tolist()
 assert frames==list(range(103,122))and np.array_equal(w[:,:,7:],q[:,:,7:])and np.array_equal(w[:,0],q[:,0])
 binding=dict(root_mode_changed_frames=frames,early_thumb_repair_frames=early,late_thumb_repair_frames=late,root_mode_overlap_with_late_repair_frames=sorted(set(frames)&set(late)),all_finger_commands_and_entire_left_component_exactly_F=True,original_right_root_or_thumb_component_certificate_NOT_inherited=True)
 save('exact-component-provenance.json',binding);journal(log,dict(kind='AFTER',operation='component_provenance',returned=binding))
 kin=URDFKinematics(U);bychild={j.find('child').attrib['link']:j for j in kin.joints}
 paths={}
 for link in ['right_hand_palm_link','right_thumb_tip']:
  chain=[];cur=link
  while cur!=kin.root_name:
   j=bychild[cur];chain.append(j.attrib['name']);cur=j.find('parent').attrib['link']
  paths[link]=list(reversed(chain))
 moving=set(manifest['joint_names']['right'][:4]);assert not any(moving.intersection(v)for v in paths.values())
 save('actual-URDF-branch-independence.json',dict(urdf_sha256=sha(U),root_link=kin.root_name,paths=paths,index_middle_joint_names=sorted(moving),no_index_middle_joint_is_ancestor=True,necessity='At fixed saved root and F thumb commands, index/middle cannot alter these palm/thumb camera points.'))
 rows=[];missing=[]
 for frame in frames:
  proposal=dict(frame=frame,root=w[frame,1,:7],finger_commands=q[frame,1,7:],tip_links=['right_hand_palm_link','right_thumb_tip'],human_landmarks=[0,4],original_cap_row=caps[frame])
  journal(log,dict(kind='BEFORE',operation='NEW_fixed_root_preserved_branches_FK_source',proposal=proposal))
  poses=kin.poses(w[frame,1,:7],dict(zip(manifest['joint_names']['right'],q[frame,1,7:],strict=True)))
  returned=[]
  for i,(link,hidx)in enumerate(zip(proposal['tip_links'],proposal['human_landmarks'],strict=True)):
   robot=poses[link][:3,3];rc=(robot@stage-camera[frame,:3,3])@camera[frame,:3,:3]
   hc=(human[frame,1,hidx]-camera[frame,:3,3])@camera[frame,:3,:3]
   available=bool(valid[frame,1]and np.isfinite(hc).all()and hc[2]>.01)
   assert available==caps[frame]['image_anchor_available'][i]
   base=dict(frame=frame,tip=link,landmark=hidx,actual_URDF_world_point=robot,actual_camera_point=rc,human_camera_point=hc,available=available)
   if not available:base.update(status='MISSING_preserved_no_gate_invented');missing.append(base)
   elif not(np.isfinite(rc).all()and rc[2]>.01):base.update(status='REJECTED_original_depth');rows.append(base)
   else:
    pixel=lambda c:np.array([K[0,0]*c[0]/c[2]+K[0,2],K[1,1]*c[1]/c[2]+K[1,2]])
    uv=pixel(rc);observed=pixel(hc);error=float(np.linalg.norm(uv-observed));cap=float(caps[frame]['source_before_after_px'][0][i])
    base.update(robot_pixel=uv,observed_pixel=observed,error_px=error,original_cap_px=cap,original_image_tolerance_px=1e-6,excess_px=error-cap,status='PASS_original_image'if error<=cap+1e-6 else'REJECTED_original_image')
    rows.append(base)
   returned.append(base)
  save(f'frame-{frame:03d}-actual-return.json',dict(proposal=proposal,returned=returned));journal(log,dict(kind='AFTER',operation='NEW_fixed_root_preserved_branches_FK_source',returned=returned))
 save('all-new-preserved-branch-source-returns.json',dict(rows=rows,missing=missing))
 rejects=[r for r in rows if r['status'].startswith('REJECTED')]
 summary=dict(schema='v2d.new-preserved-branch-source-prerequisite.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_pure_actual_URDF_states=len(frames),new_palm_thumb_source_checks=len(rows),missing_anchors_retained=len(missing),rejected_original_source_checks=len(rejects),rejected_frames=sorted({r['frame']for r in rejects}),maximum_source_excess_px=max([r.get('excess_px',0)for r in rows],default=0),all_finger_commands_exact_F=True,index_middle_cannot_repair_these_fixed_root_points=True,conditional_fixed_root_F_palm_thumb_family_rejected=bool(rejects),global_task_or_allowed_root_family_excluded=False,actual_new_candidate_returned=False,old_mode_orientation_root_motion_passes_NOT_relabelled_whole_source=True,F_and_all_old_results_UNCHANGED=True,source_authority_and_original_tolerances_UNCHANGED=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,wall_seconds=time.monotonic()-start)
 journal(log,dict(kind='AFTER',operation='scientific_scope',returned=summary))
 summary['all_outputs_sha256']={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log','report.json']}
 save('report.json',summary)
 print(json.dumps(summary))
