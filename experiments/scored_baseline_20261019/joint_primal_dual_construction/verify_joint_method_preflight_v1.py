"""Independent NEW signed method/complete routing/metric bindings."""
from joint_primal_dual_method_v1 import *
from fractions import Fraction as Q
D=S/'e037-joint-method-binding-v1';P=S/'e037-joint-method-preflight-v1';I=S/'e037-joint-executable-interface-v1'
authority=json.loads((P/'NEW-method-input-authority-RETURN.json').read_text())
sha_checks={p:sha(p)==h for p,h in authority.items()};save(D/'NEW-method-input-SHA-bindings-RETURN.json',sha_checks);assert all(sha_checks.values())
# NEW exact two-update algebra control on an unrelated one-variable toy.
x=np.zeros(NVAR);diag=np.zeros(NVAR);diag[0]=1.;tau=np.ones(NVAR)*.5;y=0.;states=[]
for k in range(2):
 y=max(0.,y+.5*(1-x[0]));raw=x.copy();raw[0]+=.5*y;x,prox=primal_prox(raw,tau,diag);states.append(dict(x=float(x[0]),y=y,prox=prox))
seal_return(D,'NEW-two-update-rational-control-RETURN',dict(states=states,exact_terminal='5/12'))
assert abs(states[0]['x']-float(Q(1,6)))<1e-15 and abs(states[1]['x']-float(Q(5,12)))<1e-15
with np.load(P/'NEW-constant-ALL-row-column-metric-bounds-RETURN.npz',allow_pickle=False) as z:bounds={k:z[k] for k in z.files}
checks=[]
for label,name,key,colslice in [('object','NEW-object-retained-A-RETURN.npz','object_row',slice(465,1523)),('robot','NEW-robot-linear-A-RETURN.npz','robot_row',slice(0,465))]:
 A=sparse.load_npz(I/name).tocoo();absdata=np.abs(A.data)
 # Independent incidence accumulation; no original model/solver/Ax replay.
 rows=np.bincount(A.row,weights=absdata,minlength=A.shape[0]);cols=np.bincount(A.col,weights=absdata,minlength=A.shape[1])
 returned=dict(operator=label,entries=A.nnz,
  ALL_rows_enclose_independent_positive_sum=bool(np.all(bounds[key]>=1.5*rows)),
  ALL_columns_enclose_independent_positive_sum=bool(np.all(bounds['column'][colslice]>=1.5*cols)),
  zero_rows_exact=bool(np.all(bounds[key][rows==0]==0)),finite=bool(np.isfinite(bounds[key]).all() and np.isfinite(bounds['column'][colslice]).all()),
  count_below_2pow32=A.nnz<2**32,
  scope='conditional finite RN additions/no overflow; 1.5 independent sum dominates exact coefficient sum under gamma_n<1e-6; not candidate/source tolerance')
 seal_return(D,'NEW-'+label+'-ALL-metric-bound-RETURN',dict(binding=returned,independent_row_sum=rows,independent_column_sum=cols))
 checks.append(returned);assert all(returned[k] for k in ['ALL_rows_enclose_independent_positive_sum','ALL_columns_enclose_independent_positive_sum','zero_rows_exact','finite','count_below_2pow32'])
 del A,absdata,rows,cols
with np.load(P/'NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:obj={k:z[k] for k in z.files}
oldP=sparse.load_npz(I/'NEW-embedded-objective-P-RETURN.npz')
with np.load(I/'NEW-interface-constants-RETURN.npz',allow_pickle=False) as z:oldq=z['q'];nn=int(z['nonnegative'])
objcheck=dict(Pdiagonal_exact=np.array_equal(obj['Pdiag'][:1523],oldP.diagonal()),
 no_direction_penalty=bool(np.all(obj['Pdiag'][1523:]==0)),q_exact=np.array_equal(obj['q'][:1523],oldq),no_direction_q=bool(np.all(obj['q'][1523:]==0)),
 all_object_and_source_whole_cones=np.array_equal(obj['cones'],np.asarray(layout_cones(nn))))
save(D/'NEW-original-objective-whole-cone-bindings-RETURN.json',objcheck);assert all(objcheck.values())
with np.load(P/'NEW-ALL-geometric-method-column-pattern-RETURN.npz',allow_pickle=False) as z:pat={k:z[k] for k in z.files}
with np.load(O/'NEW-complete-column-and-pair-incidence-RETURN.npz',allow_pickle=False) as z:old={k:z[k] for k in z.files}
pattern_checks=[]
for frame in range(151):
 okay=True
 for pair,(a,b) in enumerate(old['geom_pair']):
  group=1653*frame+pair;lo,hi=map(int,pat['indptr'][group:group+2]);actual=pat['columns'][lo:hi].tolist()
  expected=set(np.r_[old['geom_robot_columns'][frame,a],old['geom_robot_columns'][frame,b]].tolist());expected.discard(-1)
  if 29 in [old['geom_bodyid'][a],old['geom_bodyid'][b]]:expected.update(range(465+7*frame,465+7*frame+6))
  expected.update([1522,1523+2*group,1524+2*group]);okay=okay and actual==sorted(expected)
 pattern_checks.append(dict(frame=frame,ALL1653pair_column_sets_exact=okay))
save(D/'NEW-ALL249603-method-pair-incidence-binding-RETURN.json',pattern_checks)
assert all(r['ALL1653pair_column_sets_exact'] for r in pattern_checks)
variables=json.loads((M/'continuous-angle-variables.json').read_text());lookup={v['name']:v['column'] for v in variables}
source=json.loads((M/'all-direct-source-factor-proposals.json').read_text());domains=json.loads((M/'retained-halfangle-domains-and-unknown-source-evidence.json').read_text())
routes=json.loads((P/'NEW-ALL-source-domain-method-routes-RETURN.json').read_text());source_check=[]
for route,rec in zip(routes['source'],source,strict=True):
 expected=[lookup[f"q_{rec['frame']}_{n}"] for n in rec['free_joint_names']]
 source_check.append(dict(factor=rec['id'],columns=route['columns']==expected,
 SOC=route['SOC_rows']==[3*rec['id']+k for k in range(3)],depth=route['depth_row']==582+rec['id']))
for j,(route,rec) in enumerate(zip(routes['domains'],domains['retained_domain_factors'],strict=True)):
 source_check.append(dict(domain=j,column=route['column']==lookup[rec['angle_variable']],rows=route['rows']==[776+2*j,777+2*j]))
save(D/'NEW-ALL-source-domain-signed-route-bindings-RETURN.json',source_check)
assert all(all(v for k,v in r.items() if k not in ['factor','domain']) for r in source_check)
save(D/'report.json',dict(schema='v2d.NEW-joint-primal-dual-method-independent-preflight.v1',passed=True,
 NEW_exact_update_controls=1,all_metric_coefficients=31839917+8798,all_original_object_and_robot_rows=4559685,
 ALL249603pair_patterns=True,ALL194source38domainroutes=True,whole_cones=1245,rows=NROW,variables=NVAR,
 no_old_model_solver_controls_source_native_replayed=True,no_episode_point_evaluated=True))
