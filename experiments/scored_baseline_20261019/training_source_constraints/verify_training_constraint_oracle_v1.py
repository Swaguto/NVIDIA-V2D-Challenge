from pathlib import Path
import datetime,hashlib,json,os
import numpy as np,trimesh
from object_training_constraint_oracle_v1 import TrainingSourceOracle
from thumb_conic_geometry_v1 import plain,journal
S=Path(__file__).resolve().parents[1];O=S/'e037-training-constraint-oracle-contract-v1';T=S/'e037-training-source-constraints-v1';D=S/'e037-object-dynamic-runtime-v1';read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(n,v):
 p=O/n;p.write_text(json.dumps(plain(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return p
def arrays(name,values):
 p=O/(name+'.npz');np.savez_compressed(p,**values);p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):journal(log,v)
 emit(dict(kind='BEFORE_NEW_training_oracle_contracts',NEW16semantic_BVHcontrols_reused_BY_HASH=True,no_failed_object_point_or_fullpixel_or_heldout_gate_used=True))
 # Synthetic cached F must never evaluate image/point/scorer constructors.
 V=np.array([[-1.,-1.,2.],[1.,-1.,2.],[0.,1.,2.]]);faces=np.array([[0,1,2]]);K=np.array([[10.,0,16],[0,10.,16],[0,0,1.]]);base=dict(training_intersection=1,training_union=1,training_mesh_vertex_median_m=.5,training_quality=.5*np.exp(-.5/.02))
 def forbidden(_):raise AssertionError('zero reference cache must not load/evaluate training data')
 oracle=TrainingSourceOracle(V,faces,K,V[None],np.eye(4)[None],np.eye(3)[None],[base],forbidden,emit,arrays);zero=oracle.query(0,np.zeros(3),'SYNTHETIC-zero-cache');save('NEW-zero-cache-contract-RETURN.json',zero);assert zero['passes_TRAINING_construction']and zero['old_baseline_raster_point_or_scorer_NOT_replayed']
 # A genuinely different synthetic nonzero point crosses source near plane;
 # UNKNOWN is retained, not accepted via a proxy or a numerical tolerance.
 domain=(np.indices((32,32)).sum(0)%2==0);data=dict(domain=domain,training_mask=domain,points_C_m=np.tile([0.,0.,2.],(100,1)))
 unknown=TrainingSourceOracle(V,faces,K,V[None],np.eye(4)[None],np.eye(3)[None],[base],lambda _:data,emit,arrays).query(0,[0,0,-3.],'SYNTHETIC-source-unknown');save('NEW-nearplane-unknown-contract-RETURN.json',unknown);assert not unknown['passes_TRAINING_construction']and unknown['status'].startswith('UNRESOLVED')
 # Missing proposal identity is rejected before it can become an evaluation.
 rejected=False
 try:oracle.query(0,np.zeros(3),'')
 except ValueError as ex:rejected=True;save('NEW-proposal-provenance-contract-RETURN.json',dict(actual_error=str(ex),no_query_return_or_pose_created=True))
 assert rejected
 # Actual151F queries are cache-only identity checks, not baseline statistic
 # reruns or any NEW episode pose proposal.
 records=read(T/'all151-NEW-training-source-statistics.json');F=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(F/'manifest.json');mesh=trimesh.load(F/manifest['objects'][0]['mesh'],force='mesh',process=False)
 with np.load(D/'dynamic-source-data.npz')as z:C0=z['C0'];K0=z['K'];cam=z['original_camera_from_object'];RC=z['camera_from_world_linear']
 real=TrainingSourceOracle(np.asarray(mesh.vertices),np.asarray(mesh.faces),K0,C0,cam,RC,records,forbidden,emit,arrays);allzero=[]
 for f in range(151):
  ret=real.query(f,np.zeros(3),f'BASELINE-IDENTITY-CACHE-{f:03d}');assert ret['passes_TRAINING_construction']and ret['status']=='CACHED_F_training_values_BY_HASH';assert ret['training_median_m']==records[f]['training_mesh_vertex_median_m'];allzero.append(ret)
 save('all151-cache-only-F-source-predicate-identities.json',allzero)
 out=dict(schema='v2d.NEW-training-source-oracle-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),NEWsynthetic_contracts=3,actual_F_cache_only_identities=151,no_actual_nonzero_episode_point_proposed_or_evaluated=True,no_baseline_source_statistic_renderer_scorer_or_solver_replay=True,hard_training_predicate_NOT_original_gate_replacement=True,original_gates_and_source_authority_UNCHANGED=True,no_trajectory_qualified=True);save('report.json',out);emit(dict(kind='AFTER_NEW_training_constraint_oracle_contracts',actual_RETURN=out));print(json.dumps(out))
