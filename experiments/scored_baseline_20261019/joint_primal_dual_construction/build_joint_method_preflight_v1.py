"""NEW fixed-method routing/metric construction; no episode pose evaluation."""
from joint_primal_dual_method_v1 import *
D=S/'e037-joint-method-preflight-v1';I=S/'e037-joint-executable-interface-v1'
contracts=S/'e037-joint-method-contracts-v1/report.json'
assert json.loads(contracts.read_text())['passed']
paths=[contracts,C/'joint_primal_dual_method_v1.py',I/'NEW-object-retained-A-RETURN.npz',I/'NEW-robot-linear-A-RETURN.npz',
 I/'NEW-interface-constants-RETURN.npz',I/'NEW-embedded-objective-P-RETURN.npz',I/'NEW-interface-input-bindings-RETURN.json',
 O/'NEW-complete-column-and-pair-incidence-RETURN.npz',O/'NEW-static-native-tree-RETURN.json',
 M/'continuous-angle-variables.json',M/'all-direct-source-factor-proposals.json',M/'retained-halfangle-domains-and-unknown-source-evidence.json']
save(D/'NEW-method-input-authority-RETURN.json',{str(p):sha(p) for p in paths})
A=sparse.load_npz(I/'NEW-object-retained-A-RETURN.npz');ar,ac=positive_sum_outer(A)
record_A=dict(shape=A.shape,nonzeros=A.nnz);del A
B=sparse.load_npz(I/'NEW-robot-linear-A-RETURN.npz');br,bc=positive_sum_outer(B);del B
cols=np.zeros(NVAR);cols[:465]=bc;cols[465:1523]=ac
save_npz(D/'NEW-constant-ALL-row-column-metric-bounds-RETURN.npz',object_row=ar,robot_row=br,column=cols)
P=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz').tocoo()
seal_return(D,'NEW-original-objective-P-RETURN',dict(data=P.data,row=P.row,col=P.col,shape=P.shape))
assert np.all(P.row==P.col) and np.all(P.data>=0)
diag=np.zeros(NVAR);diag[P.row]=P.data
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:
 q=np.r_[z['q'],np.zeros(NVAR-1523)];nn=int(z['nonnegative']);r=float(z['radius']);scale=float(z['scale'])
cones=layout_cones(nn)
save_npz(D/'NEW-original-objective-and-cone-layout-RETURN.npz',Pdiag=diag,q=q,cones=np.asarray(cones,dtype=np.int64))
with np.load(O/'NEW-complete-column-and-pair-incidence-RETURN.npz',allow_pickle=False) as z:
 route=z['geom_robot_columns'];pairs=z['geom_pair'];bodies=z['geom_bodyid']
ptr=[0];columns=[]
for frame in range(151):
 for p,(a,b) in enumerate(pairs):
  group=frame*1653+p
  cc=set(int(v) for v in np.r_[route[frame,a],route[frame,b]] if v>=0)
  if bodies[a]==29 or bodies[b]==29:cc.update(465+7*frame+k for k in range(6))
  cc.update([1522,1523+2*group,1523+2*group+1]);columns.extend(sorted(cc));ptr.append(len(columns))
save_npz(D/'NEW-ALL-geometric-method-column-pattern-RETURN.npz',indptr=np.asarray(ptr,dtype=np.int64),
 columns=np.asarray(columns,dtype=np.int32),pair=pairs,all_frames=np.arange(151,dtype=np.int32))
variables=json.loads((M/'continuous-angle-variables.json').read_text());lookup={v['name']:v['column'] for v in variables}
source=json.loads((M/'all-direct-source-factor-proposals.json').read_text())
domains=json.loads((M/'retained-halfangle-domains-and-unknown-source-evidence.json').read_text())
source_routes=[dict(factor=r['id'],SOC_rows=[3*r['id']+j for j in range(3)],depth_row=NSOCSRC+r['id'],
 columns=[lookup[f"q_{r['frame']}_{n}"] for n in r['free_joint_names']],
 radius_definition='ORIGINAL_cap_px+ORIGINAL_image_tolerance_px',strict_depth_POST=True) for r in source]
domain_routes=[dict(column=lookup[r['angle_variable']],rows=[NSOCSRC+NDEPTH+2*j,NSOCSRC+NDEPTH+2*j+1]) for j,r in enumerate(domains['retained_domain_factors'])]
save(D/'NEW-ALL-source-domain-method-routes-RETURN.json',dict(source=source_routes,domains=domain_routes))
save(D/'NEW-complete-method-declaration-RETURN.json',dict(
 method='ONE fixed two-update whole-joint nonlinear primal-dual construction',updates=2,terminal_step_ONLY=2,unique_oracles_max=3,
 method_wall_seconds=150,outer_guard_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),
 variables=NVAR,joint_variables=1523,continuous_body_local_chart_variables=2*NPAIR,rows=NROW,
 original_object_rows=NOBJ,robot_rows=NROB,source_SOC3=194,strict_depth=194,half_domain_rows=76,pair_rows=NPAIR,
 initialization='F original free-angle bytes; object eta/u/sigma/phase zero; all continuous geometric chart coordinates zero; fixed root/middle SAME saved unqualified hypotheses',
 no_old_rejected_point_or_dual_initializer=True,
 stereographic_scope='one finite continuous witness chart, absent pole UNKNOWN not physically excluded; not wrist axis/profile or finite direction bank',
 signed_constraint='F=A*x-b; feasible F in -K; y_next=projection_K(y+sigma*F)',
 primal_update='prox_original_Pq_and_original_phase_interval(w-tau*J.T*y_next)',
 metric='tau=.5/column_L1_outer, sigma=.5/row_L1_outer; WHOLE cone common sigma via maximum component bound; zero sums ->1',
 all_tie_bound='mean SEARCH J only for update; every tied correlated slope retained; componentwise absolute envelope checked BEFORE update',
 metric_proof='Cauchy/Schur: if row/column bounds enclose sum|Jij|, ||sqrt(Sigma) J sqrt(Tau)||<=.5; local numeric linearization only, NO nonlinear convergence/stationarity claim',
 positive_sum_proof='Twice positive finite RN sum encloses exact saved-coefficient sum when count<2^32, no overflow; subnormal additions exact until normal; gamma_n<1e-6. Conditional arithmetic bound NOT candidate tolerance',
 original_matrices_objective_conditioning_unchanged=True,
 phase_prox='exact ALREADY original [0,1] domain prox during algorithm; no command clipping and no terminal repair',
 all_rows_vertices_ties_unknowns_RETAINED=True,no_cut_row_axis_pose_rank_bank=True,
 every_input_constructor_oracle_steps_dual_adjoint_primal_terminal_RETURN_lossless_durable_BEFOREfurtherwork=True,
 no_extension_restart_linefraction_warmstart_best_intermediate=True,
 terminal_closed_model_AND_exact_zero_phase_AND_actual6motion_REQUIRED_before_any_actual_training_query=True,
 actual_full_training_integercells_NN_rank_median_quality_fulltriangles_hardfilter_NOT_replaced=True,
 original_fullpixel_heldout_POST_only_and_whole_source_URDF_motion_native_self_swept_motor_grasp_REQUIRED=True,
 no_episode_point_or_solver_evaluation_yet=True))
save(D/'report.json',dict(schema='v2d.NEW-joint-primal-dual-method-preflight.v1',
 rows=NROW,variables=NVAR,all_pair_patterns=NPAIR,geometric_pattern_entries=len(columns),
 whole_cones=len(cones),constant_object_operator=record_A,no_episode_pose_source_native_solver_evaluation=True))
