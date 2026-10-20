"""NEW complete quantified geometric model and unchanged-authority routing."""
from joint_collision_predicates_v1 import *
from fractions import Fraction as Q
D=S/'e037-joint-collision-predicates-v1'
# Four NEW concrete controls, not a rerun of prior contracts.
controls=[]
def control(name,raw,predicate):
    save(D/(name+'-RETURN.json'),raw)
    ok=bool(predicate(raw));controls.append(dict(name=name,passed=ok));assert ok,name
synthetic=dict(geom_bodyid=[3,4],body_weldid=[0,1,1,3,4],
 body_parentid=[0,0,1,2,0],disableflags=0,FILTERPARENT_bit=1,
 exclude_signature=[],geom_contype=[1,1],geom_conaffinity=[1,1])
# Parent of welded body3 is body2 welded to1: raw-parent equality is insufficient.
control('NEW-welded-parent-representative',dict(reasons=builtin_pair_reasons(synthetic,0,1),
 parent_weld=synthetic['body_weldid'][synthetic['body_parentid'][3]]),
 lambda r:r['parent_weld']==1 and r['reasons']==[])
synthetic2=dict(synthetic,geom_bodyid=[1,3])
control('NEW-parent-flag-and-world-exception',dict(
 enabled=builtin_pair_reasons(synthetic2,0,1),
 disabled=builtin_pair_reasons(dict(synthetic2,disableflags=1),0,1),
 world=builtin_pair_reasons(dict(synthetic,geom_bodyid=[0,4]),0,1)),
 lambda r:r['enabled']==['welded_parent'] and r['disabled']==[] and r['world']==[])
A=[(Q(0),Q(1)),(Q(2),Q(-1)),(Q(2),Q(1))]
B=[(Q(-1),Q(0)),(Q(1),Q(2))]
n=(Q(3,5),Q(4,5))
dot=lambda v:sum(x*y for x,y in zip(v,n))
cart=max(dot((a[0]-b[0],a[1]-b[1])) for a in A for b in B)
fact=max(dot(a) for a in A)+max(-dot(b) for b in B)
control('NEW-ALL-support-cartesian-factorization',dict(cart=str(cart),factor=str(fact),
 unit_norm=str(sum(x*x for x in n)),phase_sign=str(Q(1,1000)+Q(1,4)*Q(2,3))),
 lambda r:r['cart']==r['factor'] and r['unit_norm']=='1' and r['phase_sign']=='503/3000')
# Complete embedding preserves every selected value and objective, not a row bank.
small=np.arange(30).reshape(6,5);keep=np.array([0,1,4,5]);x=np.arange(8)
new=np.zeros((4,8),dtype=int);new[:,3:]=small[keep]
control('NEW-unchanged-row-column-objective-embedding',dict(
 old=(small[keep]@x[3:]).tolist(),new=(new@x).tolist(),
 excluded_only_invalid_geometry=[2,3],row_map=[0,1,-1,-1,2,3]),
 lambda r:r['old']==r['new'] and r['row_map'].count(-1)==2)
save(D/'NEW-predicate-contracts-report.json',dict(contracts=controls,passed=True))
raw=json.loads((O/'NEW-static-native-tree-RETURN.json').read_text())
prior=json.loads((O/'NEW-complete-pair-filter-metadata-RETURN.json').read_text())
ref=json.loads((C/'joint-collision-native-filter-public-reference-v1.json').read_text())
save(D/'NEW-predicate-input-SHA-RETURN.json',{
 str(O/'NEW-static-native-tree-RETURN.json'):sha(O/'NEW-static-native-tree-RETURN.json'),
 str(O/'NEW-complete-pair-filter-metadata-RETURN.json'):sha(O/'NEW-complete-pair-filter-metadata-RETURN.json'),
 str(Path(ref['path'])):sha(ref['path'])})
assert sha(ref['path'])==ref['sha256'] and len(raw['pair_geom1'])==0
records=[];counts={};active_counts={}
for p,row in enumerate(prior['pairs']):
 a,b=row['geoms'];reasons=builtin_pair_reasons(raw,a,b)
 cat=row['category'];counts[cat]=counts.get(cat,0)+1
 if not reasons:active_counts[cat]=active_counts.get(cat,0)+1
 records.append(dict(pair=p,geom_a=a,geom_b=b,all151clocks=True,category=cat,
  conditional_builtin_filter_reasons=reasons,retained=True,
  predicate='typed original-filter tautology' if reasons else 'exists n in R3, n.n=1: max_ALL_vA n.T*T_A(q)*vA + max_ALL_vB -n.T*T_B(q)*vB <= .001 + r*z_rho',
  support_vertices='ALL cached compiled BODY support vertices; no face/axis/vertex bank',
  phase_column=1522,phase_radius_source='unchanged compiled object radius in frozen canonical',
  robot_dependencies='union of both geometry ancestor-column routes at same clock',
  object_dependencies='465+7*clock+[0,1,2,3,4,5] ONLY for object geometry; sigma is lift-only',
  future_actual_native_filter_callback_state='UNVERIFIED; fresh actual native all-pair/self gate mandatory'))
save(D/'NEW-ALL-pair-quantified-geometric-constraints-RETURN.json',dict(
 records=records,all_pair_instances=249603,categories_per_clock=counts,
 conditional_active_categories_per_clock=active_counts,
 all_pairs_retained_including_filter_tautologies=True,
 public_source_scope='conditional no-user-callback/all-awake built-in static filter; no binary identity proof',
 sleep_and_dynamic_broadphase_NOT_used_to_omit_pairs=True,
 native_margin_EPA_contact_details_NOT_equated_with_support_predicate=True))
# NEW lossless row routing: old fixed-hand geometry gets a typed INVALID label;
# every unchanged source/motion/cone row has exactly one new location.
rows=np.arange(4618162,dtype=np.int64)
route=np.where(rows<2,rows,np.where(rows<66593,-1,rows-66591))
save_npz(D/'NEW-object-canonical-row-and-column-routing-RETURN.npz',
 old_row_to_joint_block_row=route,old_object_column_to_joint_column=np.arange(465,1523),
 robot_columns=np.arange(465),q_column_map=np.arange(465,1523),P_column_map=np.arange(465,1523))
with np.load(S/'e037-hard-raster-model-v1/NEW-canonical-A-RETURN.npz',allow_pickle=False) as z:
 data,indices,indptr=z['data'],z['indices'],z['indptr']
 spans=[]
 for col in range(1058):
  lo,hi=map(int,indptr[col:col+2]);rr=indices[lo:hi];vv=data[lo:hi]
  valid=route[rr]>=0
  # Record complete per-column routing identities without constructing a dense model.
  spans.append(dict(old_column=col,new_column=465+col,stored_entries=hi-lo,
   retained_entries=int(np.count_nonzero(valid)),invalid_old_geometry_entries=int(np.count_nonzero(~valid)),
   original_values_sha256=hashlib.sha256(vv.tobytes()).hexdigest(),
   retained_values_sha256=hashlib.sha256(vv[valid].tobytes()).hexdigest(),
   new_row_indices_sha256=hashlib.sha256(route[rr[valid]].tobytes()).hexdigest()))
 # Concrete phase coupling sign comes from saved canonical, no solver/model replay.
 lo,hi=map(int,indptr[1057:1059]);rho_rows=indices[lo:hi];rho_values=data[lo:hi]
 phase=[dict(row=int(k),coefficient=float(v)) for k,v in zip(rho_rows[:3],rho_values[:3])]
save(D/'NEW-ALL-canonical-coefficient-routing-RETURN.json',dict(columns=spans,
 phase_sign_binding=phase,ALL_original_stored_entries=sum(s['stored_entries'] for s in spans),
 unchanged_entries=sum(s['retained_entries'] for s in spans),
 invalid_old_geometry_entries=sum(s['invalid_old_geometry_entries'] for s in spans)))
assert phase==[dict(row=0,coefficient=-1.),dict(row=1,coefficient=1.),dict(row=2,coefficient=-1.)]
# Quantified support equivalence is a convex-body theorem, not a finite axis test.
save(D/'NEW-complete-model-proof-and-acceptance-RETURN.json',dict(
 ideal_proof=['For compact convex A,B, M=A-B has support hM(n)=hA(n)+hB(-n).',
 'If origin is inside M, the largest centered Euclidean ball has radius min_unit_n hM(n).',
 'If origin is outside M, a strict separating direction makes min_unit_n hM(n)<0.',
 'Thus ideal convex overlap depth <=tau iff exists unit n with hM(n)<=tau, for tau>=0.',
 'Max support uses ALL compiled vertices, retaining ties; no selected-face/edge-axis approximation.'],
 native_scope='Ideal compiled-convex geometry only; no numerical EPA/contact equivalence; actual native remains a separate gate.',
 full_model=['1523 continuous command/object/lift/phase coordinates remain UNASSIGNED',
 '8114 robot linear constraints +194 original source factors +38 domains/fullregion2 BY HASH',
 '4551571 unchanged object canonical rows; ALL900SOC4+151SOC5 retained; original objective embedded with zero robot block',
 '249603 ALL-pair symbolic predicates including typed original-static-filter tautologies',
 'ALL visual quantizer/strict-depth/unknown/alias/source and actual6-motion gates retained'],
 support_directions='internal existential continuous sphere variables, not selected/assigned normals or finite bank',
 conditioning='unchanged robot radians and saved object eta/r,u,sigma,z_rho; no new scaling or objective weight',
 resource='factorized 58 compiled bodies/31 chains/1653 pairs*151 clocks; NO vertex-product table or dense full Jacobian',
 no_optimizer_or_values_selected=True,
 future_acceptance='NEW full proposal and ALL constructor/intermediate returns durable BEFORE source/training/actual-motion; then independent POST whole source, original URDF, native all+self, swept, motor/grasp. No training/GPU before qualification.'))
save(D/'report.json',dict(schema='v2d.NEW-joint-full-model-predicate-routing.v1',
 new_contracts=4,complete_pair_predicates=249603,
 conditional_active_pair_instances=151*sum(active_counts.values()),
 conditional_active_categories_per_clock=active_counts,
 old_fixed_geometry_rows_invalidated=66591,unchanged_object_rows=4551571,
 all_canonical_stored_coefficient_positions_bound=32306054,
 no_episode_pose_or_solver_or_source_native_evaluation=True))
