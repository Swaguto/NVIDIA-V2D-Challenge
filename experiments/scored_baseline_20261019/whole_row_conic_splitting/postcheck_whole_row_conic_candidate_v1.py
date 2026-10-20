"""Actual NEW raw terminal object pose gates; no old candidates/queries.

Called only after raw closed-model screening. Training uses actual changed
NN/ranks/rounded median/full triangles; original fullpixel/heldout stays POST.
No candidate can inherit robot/native/self/swept/motor/grasp certificates.
"""
from whole_row_conic_split_v1 import *
def postcheck(O,x,emit):
 from full_pose_inputs_v1 import load
 from full_pose_construction_v1 import actual_motion
 from object_full_pose_maps_v1 import compose
 from full_pose_training_oracle_v1 import FullPoseTrainingOracle
 d=load();m=json.loads((M/'NEW-canonical-metadata-RETURN.json').read_text());r=m['radius'];scale=m['scale'];local=x[:1057].reshape(151,7);t=local[:,3:6]/scale;D=1+np.einsum('ij,ij->i',t,t,optimize=False)
 delta=r*local[:,:3]/D[:,None];length=np.linalg.norm(t,axis=1);theta=np.zeros_like(t);nz=length!=0;theta[nz]=t[nz]*(2*np.arctan(length[nz])/length[nz])[:,None]
 poses=np.array([compose(d['world'][f],delta[f],theta[f],np.eye(3)) for f in range(151)])
 save(O/'ONE-NEW-sealed-raw-terminal-proposal-BEFORE.npz',x=x,delta=delta,theta=theta,world_from_object=poses)
 pid=sha(O/'ONE-NEW-sealed-raw-terminal-proposal-BEFORE.npz');emit(dict(kind='NEW_terminal_pose_sealed',proposal_id=pid))
 motion=actual_motion(poses,d['times'],d['motion_caps']);save(O/'ONE-NEW-actual-object-motion-RETURN.npz',**motion['arrays']);dump(O/'ONE-NEW-actual-object-motion-RETURN.json',{k:v for k,v in motion.items() if k!='arrays'})
 cells=[]
 for f in range(151):
  cam=compose(d['camera'][f],delta[f],theta[f],d['Rcw'][f]);V=d['V']@cam[:3,:3].T+cam[:3,3];finite=bool(np.isfinite(V).all());depth=bool(finite and np.all(V[:,2]>.01) and cam[2,3]>.05)
  with np.load(M/f'frame-{f:03d}-NEW-hard-cell-coefficients-RETURN.npz') as z:target=z['target_integer_uv']
  if depth:
   uv=np.column_stack([d['K'][0,0]*V[:,0]/V[:,2]+d['K'][0,2],d['K'][1,1]*V[:,1]/V[:,2]+d['K'][1,2]]);ints=np.clip(uv,-20000,20000).astype(np.int32);same=bool(np.array_equal(ints,target))
  else:uv=np.empty((0,2));ints=np.empty((0,2),np.int32);same=False
  save(O/f'f{f:03d}-NEW-actual-integer-depth-RETURN.npz',camera=cam,vertices=V,uv=uv,integer_uv=ints)
  cells.append(dict(frame=f,finite=finite,strict_depth=depth,actual_saved_integer_cells_equal=same))
 dump(O/'ONE-NEW-actual-integer-depth-decisions-RETURN.json',cells)
 result=dict(proposal_id=pid,actual_object_motion_passes=motion['passes'],actual_integer_cells=all(t['actual_saved_integer_cells_equal'] for t in cells),actual_strict_depth=all(t['strict_depth'] for t in cells),actual_training_queries=0,no_qualification=True)
 if not(all(motion['passes']) and result['actual_integer_cells'] and result['actual_strict_depth']):result['scope_status']='REJECTED_actual_motion_cells_or_depth';return result
 def loader(f):
  base=d['stats'][f]
  with np.load(base['partition']['path']) as z:mask=z['training_mask'];domain=z['domain']
  with np.load(base['point_data']['path']) as z:points=z['points_C_m']
  return dict(training_mask=mask,domain=domain,points_C_m=points)
 def saver(name,values):
  p=O/(name+'.npz');save(p,**values);return dict(path=str(p),sha256=sha(p))
 oracle=FullPoseTrainingOracle(d,loader,emit,saver);source=[oracle.query(f,delta[f],theta[f],pid) for f in range(151)];dump(O/'ONE-NEW-actual151-training-RETURN.json',source)
 result.update(actual_training_queries=151,training_pass=all(t['passes_TRAINING_construction'] for t in source),source_RETURN_ids_match=all(t['proposal_id']==pid for t in source))
 result['scope_status']='TRAINING_ONLY_pass_NEEDS_independent_original_POST_and_whole_reference_gates' if result['training_pass'] and result['source_RETURN_ids_match'] else 'REJECTED_actual_training_source'
 return result
