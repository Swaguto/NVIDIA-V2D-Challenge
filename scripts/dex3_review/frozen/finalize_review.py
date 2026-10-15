import datetime,hashlib,json,shutil,tarfile
from pathlib import Path
T=Path('/Users/rohithiruku/dex3_isaac_episode_review_20261014')
G=Path('/Users/rohithiruku/v2d_episode_previews/recovered_20261014/continuity_fix/dex3_isaac')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=json.load(open(G/'results.json'));proof=json.load(open(G/'playback-verification.json'));audit=json.load(open(T/'completed-review-audit.json'))
assert len(rows)==len(proof)==audit['complete_count']==12
assert all(v['ended'] and v['rate']==1 and v.get('error') is None for v in proof.values())
assert sum(x['frames'] for x in rows)==3096
uniform={x['renderer_sha256'] for x in audit['episodes']};assert len(uniform)==1
verified={}
for row in rows:
 ep=row['episode'];p=G/ep
 receipt=json.load(open(p/'isaac-receipt.json'));q=json.load(open(p/'retarget-quality.json'));m=json.load(open(p/'media-verification.json'))
 assert receipt['robot_asset_lock_verified'] and not receipt['physical_task_accepted']
 assert q['joint_position_limit_violations']==0
 assert all(v['finger_velocity_limit_violating_pairs']==0 for v in q['per_hand'].values())
 assert all(v['presentation_timestamps_strictly_increasing'] for v in m.values())
 for name,digest in json.load(open(p/'delivery-hashes.json')).items():
  assert sha(p/name)==digest
  verified[ep+'/'+name]=digest
 shutil.copy2(T/'completed-review-audit.json',G/'completed-review-audit.json')
report={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'episodes':[x['episode'] for x in rows],'source_frames':3096,'delivered_videos':48,'actual_isaac_camera_videos':24,'human_input_videos':12,'comparison_videos':12,'all_source_frames_and_presentation_timestamps_verified':True,'full_normal_rate_browser_playback_verified':12,'robot_asset_locks_verified':True,'joint_position_limit_violations':0,'finger_velocity_limit_violating_pairs':0,'source_axes_and_visual_metric_masks_preserved':all(x['source_axis_and_masks_match'] for x in audit['episodes']),'unknown_hand_assigned_color_pixels':max(max(x['unknown_hand_maximum_assigned_color_pixels']) for x in audit['episodes']),'maximum_root_state_error_m':max(x['root_error_m'] for x in audit['episodes']),'maximum_joint_state_error_rad':max(x['joint_error_rad'] for x in audit['episodes']),'uniform_renderer_sha256':list(uniform)[0],'gravity_enabled':False,'collision_resolution_enabled':False,'physical_grasp_success_tested':False,'user_acceptance':False,'scope':'Private visual review of inferred kinematic Dex3 replays; not physical task acceptance or hidden-evaluation readiness.','known_limitations':['Fingertip and object alignment remains approximate.','Unknown input intervals remain absent, including the long right-hand interval in e021.','The first five e002 frames have no native world camera pose.','e042 input finger articulation remains approximate.'],'next_phase':'Team reproduction and Brev/Docker packaging await user visual review.'}
(G/'batch-review.json').write_text(json.dumps(report,indent=2))
shutil.copy2(G/'batch-review.json',T/'batch-review.json')
# An exact gallery snapshot includes media and review evidence, private to the existing pod.
manifest={str(p.relative_to(G)):sha(p) for p in G.rglob('*') if p.is_file() and p.name not in ('backup-hashes.json','backup-verification.json')}
(G/'backup-hashes.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(T/'gallery-backup.tar','w') as tar:
 for p in G.rglob('*'):
  if p.is_file():tar.add(p,arcname=str(Path('dex3_isaac')/p.relative_to(G)),recursive=False)
print('FINALIZED',len(rows),'episodes',len(manifest),'files',sum(p.stat().st_size for p in G.rglob('*') if p.is_file()),'bytes')
