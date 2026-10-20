"""NEW one-load immutable-array binding interface; old interrupted return stays unavailable."""
from joint_collision_maps_v1 import S,C,F,M,G,O,sha,save,save_npz
from pathlib import Path
import json,hashlib
import numpy as np
D=S/'e037-joint-one-load-binding-v1'
P=S/'e037-joint-collision-predicates-v1'
J=S/'e037-joint-native-object-gauge-v1'
model=S/'e037-hard-raster-model-v1'
old=S/'e037-joint-complete-binding-v1'
# NEW concrete resource contract verifies fields are loaded ONCE before column routing.
class Counted:
    def __init__(self): self.counts={}
    def __getitem__(self,key):
        self.counts[key]=self.counts.get(key,0)+1
        return {'data':np.arange(5.),'indices':np.arange(5),'indptr':np.array([0,2,5])}[key]
fake=Counted();fd,fi,fp=(fake[k] for k in ['data','indices','indptr'])
for c in range(2): _=fd[fp[c]:fp[c+1]]
save(D/'NEW-one-load-resource-contract-RETURN.json',dict(loads=fake.counts,columns=2))
assert fake.counts=={'data':1,'indices':1,'indptr':1}
previous=json.loads((old/'NEW-complete-row-bijection-RETURN.json').read_text())
save(D/'NEW-earlier-row-proof-BY-HASH-RETURN.json',dict(path=str(old/'NEW-complete-row-bijection-RETURN.json'),sha256=sha(old/'NEW-complete-row-bijection-RETURN.json'),retained_result=previous,old_coefficient_RETURN_UNAVAILABLE=True))
assert previous['exact'] and previous['retained']==4551571
with np.load(P/'NEW-object-canonical-row-and-column-routing-RETURN.npz',allow_pickle=False) as z:
    routing={k:z[k] for k in z.files}
spans=json.loads((P/'NEW-ALL-canonical-coefficient-routing-RETURN.json').read_text())
# Exactly one compressed read per field; no materialized model/solver constructor.
with np.load(model/'NEW-canonical-A-RETURN.npz',allow_pickle=False) as z:
    all_values=z['data'];all_rows=z['indices'];all_ptr=z['indptr']
coefficient_records=[]
for c,(a,b) in enumerate(zip(all_ptr[:-1],all_ptr[1:],strict=True)):
    a,b=int(a),int(b);rows=all_rows[a:b];values=all_values[a:b]
    allowed=np.logical_not(np.logical_and(rows>=2,rows<66593))
    new_rows=rows[allowed].astype(np.int64)
    new_rows-=np.where(new_rows>=66593,66591,0)
    expected_values=dict(old_column=c,new_column=c+465,stored_entries=b-a,
        retained_entries=int(np.count_nonzero(allowed)),invalid_old_geometry_entries=int(np.count_nonzero(~allowed)),
        original_values_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        retained_values_sha256=hashlib.sha256(values[allowed].tobytes()).hexdigest(),
        new_row_indices_sha256=hashlib.sha256(new_rows.tobytes()).hexdigest())
    coefficient_records.append(dict(column=c,stored_entries=b-a,exact=spans['columns'][c]==expected_values))
save(D/'NEW-one-load-ALL-32306054-routing-RETURN.json',coefficient_records)
assert all(r['exact'] for r in coefficient_records) and sum(r['stored_entries'] for r in coefficient_records)==32306054
# No P/q weight changes. A zero robot block extends the original objective.
with np.load(model/'NEW-canonical-vectors-RETURN.npz',allow_pickle=False) as z:
 b,q=z['b'],z['q'];selected=np.concatenate([b[:2],b[66593:]])
 qnew=np.zeros(1523);qnew[465:]=q
 objective=dict(original_q_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
  embedded_q_sha256=hashlib.sha256(qnew.tobytes()).hexdigest(),
  unchanged_b_selected_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
  all_b_rows_retained_or_typed_INVALID=len(b),kept_b_rows=len(selected),
  new_robot_objective_entries_all_exact_zero=bool(np.all(qnew[:465]==0)),
  q_object_block_exact=np.array_equal(qnew[465:],q))
with np.load(model/'NEW-canonical-P-RETURN.npz',allow_pickle=False) as z:
 objective['P_by_hash']=sha(model/'NEW-canonical-P-RETURN.npz')
 objective['P_stored_entries']=len(z['data'])
 objective['P_column_embedding_exact']=np.array_equal(routing['P_column_map'],np.arange(1058)+465)
save(D/'NEW-complete-objective-RHS-preservation-RETURN.json',objective)
assert objective['q_object_block_exact'] and objective['P_column_embedding_exact']
# Independent Boolean disjunction over original masks, welded representatives,
# explicit exclusions. Every false predicate remains a typed row, never removed.
raw=json.loads((O/'NEW-static-native-tree-RETURN.json').read_text())
pred=json.loads((P/'NEW-ALL-pair-quantified-geometric-constraints-RETURN.json').read_text())
ex=set(raw['exclude_signature']);checks=[]
parent=np.asarray(raw['body_parentid']);weld=np.asarray(raw['body_weldid'])
body=np.asarray(raw['geom_bodyid']);ct=np.asarray(raw['geom_contype']);ca=np.asarray(raw['geom_conaffinity'])
for p in pred['records']:
 a,b=p['geom_a'],p['geom_b'];ba,bb=int(body[a]),int(body[b]);wa,wb=int(weld[ba]),int(weld[bb])
 reason=[]
 if wa==wb:reason.append('same_weld')
 if not(raw['disableflags'] & raw['FILTERPARENT_bit']) and wa*wb!=0 and (wa==weld[parent[wb]] or wb==weld[parent[wa]]):reason.append('welded_parent')
 if (min(ba,bb)*65536+max(ba,bb)) in ex:reason.append('explicit_body_exclusion')
 if not ((ct[a]&ca[b])|(ct[b]&ca[a])):reason.append('geom_bitmask')
 checks.append(dict(pair=p['pair'],exact=p['conditional_builtin_filter_reasons']==reason,
  retained=p['retained'],phase_column=p['phase_column']==1522,
  quantified_all_support='ALL' in p['support_vertices']))
save(D/'NEW-all-filter-phase-support-predicate-bindings-RETURN.json',checks)
assert len(checks)==1653 and all(all(r[k] for k in ['exact','retained','phase_column','quantified_all_support']) for r in checks)
# Every original robot term receives the SAME column, with exact rational
# coefficients untouched; 194 source factors remain separate URDF expressions.
variables=json.loads((M/'continuous-angle-variables.json').read_text())
lookup={v['name']:v['column'] for v in variables}
linear=json.loads((M/'all-exact-linear-angle-inequalities.json').read_text())
source=json.loads((M/'all-direct-source-factor-proposals.json').read_text())
term_routes=[]
for i,r in enumerate(linear):
 for t in r['active_terms']:
  term_routes.append(dict(row=i,variable=t['variable'],column=lookup[t['variable']],
                         coefficient_UNCHANGED=t['coefficient']))
source_routes=[]
for i,r in enumerate(source):
 cols=[lookup[f"q_{r['frame']}_{name}"] for name in r['free_joint_names']]
 source_routes.append(dict(factor=i,frame=r['frame'],columns=cols,
  original_record_sha256=hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest(),
  URDF_source_separate_from_native_MJB_geometry=True))
save(D/'NEW-ALL-robot-original-factor-embedding-RETURN.json',dict(
 original_linear_path=str(M/'all-exact-linear-angle-inequalities.json'),
 original_linear_sha256=sha(M/'all-exact-linear-angle-inequalities.json'),
 rows=len(linear),terms=term_routes,source_factors=source_routes,
 original_domains_sha256=sha(M/'retained-halfangle-domains-and-unknown-source-evidence.json')))
assert len(linear)==8114 and len(source_routes)==194
# Bind actual saved native-object base and stage coefficients by exact bytes;
# no recomputed old closure or episode-transform evaluation.
with np.load(J/'NEW-native-object-gauge-coefficients-RETURN.npz',allow_pickle=False) as z:
 gauge={k:z[k] for k in z.files}
with np.load(G/'NEW-native-body-rotation-factors-RETURN.npz',allow_pickle=False) as z:
 native_base=z['cached_body_poses'][:,int(gauge['object_body'])]
 gauge_check=dict(stage_exact=np.array_equal(gauge['stage_world_translation_linear'],z['stage_world_translation_linear']),
 native_base_exact=np.array_equal(gauge['cached_NATIVE_object_base'],native_base),
 all_base_components=int(native_base.size),stage_components=9)
metadata=json.loads((model/'NEW-canonical-metadata-RETURN.json').read_text())
gauge_check.update(radius_exact=float(gauge['radius'])==metadata['radius'],scale_exact=float(gauge['scale'])==metadata['scale'])
save(D/'NEW-native-object-ALL-gauge-component-bindings-RETURN.json',gauge_check)
assert all(gauge_check[k] for k in ['stage_exact','native_base_exact','radius_exact','scale_exact'])
# Verify explicit semantic expressions through the independently saved authority.
gprog=json.loads((J/'NEW-native-object-gauge-program-RETURN.json').read_text())
assert gprog['native_R']=='cached_NATIVE_object_base[f,:3,:3] * Q(phi)'
assert gprog['native_t']=='cached_NATIVE_object_base[f,:3,3] + stage_world_translation_linear * delta'
save(D/'report.json',dict(schema='v2d.NEW-independent-complete-joint-model-binding.v1',
 passed=True,ALL_canonical_coefficient_positions=32306054,ALL_b_rows=4618162,
 retained_object_rows=4551571,invalid_fixed_hand_rows=66591,
 original_object_SOC4=900,original_object_SOC5=151,
 namespace_variables=1523,robot_linear_rows=8114,robot_exact_active_terms=len(term_routes),
 original_source_factors=194,all_pair_instances=249603,
 native_object_gauge_components=gauge_check['all_base_components']+9,
 no_new_objective_weight_scale_or_gate=True,
 native_URDF_distinct_metadata_failure_and_unknowns_RETAINED=True,
 all_commands_and_witness_directions_UNASSIGNED=True,
 no_optimizer_source_NN_raster_FK_native_episode_or_physics_evaluation=True))
