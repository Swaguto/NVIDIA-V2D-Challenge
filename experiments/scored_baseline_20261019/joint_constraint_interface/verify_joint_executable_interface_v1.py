"""Independent NEW complete executable-matrix and operator bindings only."""
from joint_constraint_derivatives_v1 import *
from fractions import Fraction
D=S/'e037-joint-executable-binding-v1'
I=S/'e037-joint-executable-interface-v1'
model=S/'e037-hard-raster-model-v1'
# NEW full-namespace actual API contract, not the earlier four-coordinate sign toy.
toy=object.__new__(JointExecutableInterface)
toy.A=sparse.csc_matrix(([.3,-.4,.7],([0,1,2],[0,57,1057])),shape=(3,1058))
toy.robot=sparse.csc_matrix(([.2,-.5],([0,1],[17,464])),shape=(2,465))
toy.P=sparse.diags(np.linspace(.1,.4,1523),format='csc')
v=np.linspace(-.2,.3,1523);u=np.array([.4,-.3,.2]);w=np.array([.7,-.1])
f=toy.linear_actions(v);adj=toy.linear_adjoints(u,w)
api=dict(left=float(u@f['object_slack_direction']+w@f['robot_linear_slack_direction']),
 right=float(v@adj),shapes=[len(adj),len(f['object_slack_direction']),len(f['robot_linear_slack_direction'])],
 hessian_action_shape=len(f['objective_Hv']))
save(D/'NEW-full1523-actual-API-adjoint-contract-RETURN.json',api)
assert abs(api['left']-api['right'])<1e-15 and api['shapes']==[1523,3,2]
inputs=json.loads((I/'NEW-interface-input-bindings-RETURN.json').read_text())
sha_bind={p:sha(p)==expected for p,expected in inputs['input_sha256'].items()}
save(D/'NEW-executable-immutable-input-bindings-RETURN.json',sha_bind)
assert all(sha_bind.values())
# Both arrays fetched ONCE; independent per-column Boolean complement routing.
new=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz')
with np.load(model/'NEW-canonical-A-RETURN.npz',allow_pickle=False) as z:
 olddata=z['data'];oldrows=z['indices'];oldptr=z['indptr']
checks=[]
for c,(a,b) in enumerate(zip(oldptr[:-1],oldptr[1:],strict=True)):
 a,b=int(a),int(b);rr=oldrows[a:b];vv=olddata[a:b]
 valid=np.logical_not(np.logical_and(rr>=2,rr<66593))
 er=rr[valid].copy();er-=np.where(er>=66593,66591,0)
 lo,hi=map(int,new.indptr[c:c+2])
 checks.append(dict(column=c,original_entries=b-a,retained_entries=hi-lo,
   exact_values=np.array_equal(vv[valid],new.data[lo:hi]),exact_rows=np.array_equal(er,new.indices[lo:hi])))
save(D/'NEW-ALL-executable-A-component-bindings-RETURN.json',checks)
assert sum(r['original_entries'] for r in checks)==32306054
assert all(r['exact_values'] and r['exact_rows'] for r in checks)
assert new.shape==(4551571,1058) and new.nnz==31839917
# No old matrix-vector/model/optimizer query: exact coefficient copies only.
del olddata,oldrows,oldptr,new
records=json.loads((M/'all-exact-linear-angle-inequalities.json').read_text())
variables=json.loads((M/'continuous-angle-variables.json').read_text());lookup={v['name']:v['column'] for v in variables}
rat=lambda x:Fraction(int(x['numerator']),int(x['denominator']))
rob=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz').tocsr()
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:const={k:z[k] for k in z.files}
robot_checks=[]
for row,rec in enumerate(records):
 lo,hi=map(int,rob.indptr[row:row+2]);actual=dict(zip(rob.indices[lo:hi].tolist(),rob.data[lo:hi].tolist(),strict=True))
 expected={lookup[t['variable']]:float(rat(t['coefficient'])) for t in rec['active_terms']}
 robot_checks.append(dict(row=row,exact_coefficients=actual==expected,
 rhs_exact_saved_conversion=float(rat(rec['folded_upper']))==const['robot_b'][row],
 constant_row_retained=(len(actual)==0)==rec['constant_only']))
save(D/'NEW-ALL8114-robot-executable-bindings-RETURN.json',robot_checks)
assert all(r['exact_coefficients'] and r['rhs_exact_saved_conversion'] and r['constant_row_retained'] for r in robot_checks)
assert rob.nnz==8798 and rob.shape==(8114,465)
with np.load(model/'NEW-canonical-vectors-RETURN.npz',allow_pickle=False) as z:
 original_b=z['b'];original_q=z['q']
 numeric=dict(b_first2_exact=np.array_equal(const['object_b'][:2],original_b[:2]),
 b_rest_exact=np.array_equal(const['object_b'][2:],original_b[66593:]),
 q_object_exact=np.array_equal(const['q'][465:],original_q),robot_q_zero=bool(np.all(const['q'][:465]==0)))
oldP=sparse.load_npz(model/'NEW-canonical-P-RETURN.npz').tocoo();P=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz').tocoo()
old_entries=sorted(zip(oldP.row+465,oldP.col+465,oldP.data,strict=True))
new_entries=sorted(zip(P.row,P.col,P.data,strict=True))
numeric['P_exact_embedded_entries']=old_entries==new_entries
metadata=json.loads((model/'NEW-canonical-metadata-RETURN.json').read_text())
numeric.update(nonnegative_exact=int(const['nonnegative'])==metadata['nonnegative']-66591,
 radius_exact=float(const['radius'])==metadata['radius'],scale_exact=float(const['scale'])==metadata['scale'])
oldcaps=json.loads((C/'object-authority-checkpoint-v1.json').read_text())['original_object_motion_caps']
numeric['ALL6_actual_caps_exact']=np.array_equal(const['object_caps'],list(oldcaps.values()))
save(D/'NEW-ALL-RHS-objective-cone-conditioning-bindings-RETURN.json',numeric)
assert all(numeric.values())
# The entire source factor is bound, so root/camera/stage/K/disk/depth cannot be
# silently exchanged between original URDF and native numerical geometry maps.
source=json.loads((M/'all-direct-source-factor-proposals.json').read_text())
routes=json.loads((I/'NEW-ALL-source-executable-routing-RETURN.json').read_text())
source_checks=[]
for r,saved in zip(routes,source,strict=True):
 cols=[lookup[f"q_{saved['frame']}_{n}"] for n in saved['free_joint_names']]
 source_checks.append(dict(factor=saved['id'],all_original_values_exact=r['original_factor']==saved,
 columns_exact=r['columns']==cols,native_URDF_separate=r['native_URDF_metadata_identity_NOT_assumed']))
save(D/'NEW-ALL194-source-executable-binding-RETURN.json',source_checks)
assert len(source_checks)==194 and all(all(r[k] for k in ['all_original_values_exact','columns_exact','native_URDF_separate']) for r in source_checks)
protocol=json.loads((I/'NEW-region2-domain-return-protocol-RETURN.json').read_text())
save(D/'NEW-unknown-domain-gate-scope-RETURN.json',dict(protocol=protocol,
 exact_original_evidence_sha256=sha(M/'retained-halfangle-domains-and-unknown-source-evidence.json'),
 fixed_palm_immutable_thumb_middle_and_late_changed_root_FULL_source_gates_STILL_REQUIRED=True,
 no_194factor_or_closed_model_pass_is_whole_source_qualification=True))
assert protocol['original_domain_sha256']==sha(M/'retained-halfangle-domains-and-unknown-source-evidence.json')
save(D/'NEW-final-interface-declaration-RETURN.json',dict(
 primitive_interface_path=str(C/'joint_constraint_interface_v1.py'),primitive_sha256=sha(C/'joint_constraint_interface_v1.py'),
 complete_derivative_path=str(C/'joint_constraint_derivatives_v1.py'),complete_derivative_sha256=sha(C/'joint_constraint_derivatives_v1.py'),
 immutable_preflight_root=str(I),immutable_manifest_sha256=sha(I/'NEW-interface-input-bindings-RETURN.json'),
 contracts=['12 NEW primitive executable checks','3 NEW correlated tie/zero-cone/sign checks','1 NEW actual full1523 API adjoint check'],
 ALL_retained_object_rows=4551571,ALL_robot_linear=8114,ALL_source=194,ALL_domains=38,
 ALL_pair_scopes=249603,ALL_compiled_vertices_and_ties=True,
 continuous_direction_unconstrained_assignment_NOT_selected=True,
 directional_support='max over ALL tied CORRELATED joint+direction slopes, never mean-J directional claim',
 classical_cone_gradient_at_zero_tail_NOT_claimed=True,
 mean_tie_J_SEARCH_ONLY=True,
 evaluation_NOT_declared_or_selected=True,
 any_future_evaluation_needs_NEW_full_plan_and_all_input_hashes_sealed=True,
 all_original_training_objectmotion_robotmotion_URDF_source_fullpixelheldout_POST_native_self_swept_motor_grasp_gates_REQUIRED=True))
save(D/'report.json',dict(schema='v2d.NEW-independent-joint-executable-interface-binding.v1',passed=True,
 ALL_original_A_positions=32306054,ALL_new_A_coefficients=31839917,
 ALL_object_rows=4551571,ALL_robot_rows=8114,ALL_robot_terms=8798,ALL_source_factors=194,
 SOC4=900,SOC5=151,domains=38,namespace=1523,
 NEW_actual_API_contracts=1,no_old_controls_models_Ax_source_FK_native_or_solver_replayed=True,
 all_episode_inputs_primal_and_direction_UNASSIGNED=True))
