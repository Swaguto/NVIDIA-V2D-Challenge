"""Ten NEW paired-corner support/map contracts; no prior controls replayed."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,permutations
import datetime,hashlib,json,os,time
import cube_joint_support_v1 as h
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-cube-support-contract-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'NEW-contract-proposals-and-returns.jsonl').open('w');emit=lambda x:g.journal(J,x);controls=[]
def save(name,v):
 with(O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def begin(name,v):emit(dict(kind='BEFORE_NEW_contract',name=name,proposal=v))
def check(name,v,ok):emit(dict(kind='AFTER_NEW_contract_RETURN',name=name,record=v,passed=bool(ok)));assert ok,name;controls.append(name)
expected=[{(0,0,0):1,(1,0,0):-1,(0,1,0):-1,(0,0,1):-1,(1,1,0):1,(1,0,1):1,(0,1,1):1},{(0,0,1):1,(1,0,1):-1,(0,1,1):-1,(1,1,0):1},{(0,1,0):1,(1,1,0):-1,(0,1,1):-1,(1,0,1):1},{(1,0,0):1,(1,1,0):-1,(1,0,1):-1,(0,1,1):1}]
begin('complementary_product_cubic_cancellation',dict(representatives=h.REPRESENTATIVES));polys=[]
for v in h.REPRESENTATIVES:
 p=h.paired(v);emit(dict(kind='AFTER_pair_expansion_RETURN',representative=v,polynomial=h.polynomial_record(p)));polys.append(p)
check('complementary_product_cubic_cancellation',dict(polynomials=[h.polynomial_record(p)for p in polys]),all(p==e and max(map(sum,p))==2 for p,e in zip(polys,expected,strict=True)))
begin('nonnegative_multiaffine_corner_coefficients',dict(cube_vertices_are_proof_controls_NOT_robot_or_source_anchors=True));rows=[[h.evaluate(p,v)for v in product((0,1),repeat=3)]for p in polys];check('nonnegative_multiaffine_corner_coefficients',dict(values=[[c.rec(v)for v in row]for row in rows],positive_on_entire_cube_by_multiaffine_interpolation=True),all(sorted(row)==[0]*6+[1]*2 for row in rows))
begin('partition_of_unit_polynomial',{});total={}
for p in polys:total=h.plus(total,p)
check('partition_of_unit_polynomial',dict(sum=h.polynomial_record(total)),total=={(0,0,0):1})
begin('correlated_continuous_pushforward_moments',dict(s_uniform_unit_interval=True,coordinates=['s','1-s','(1+s)/3'],no_finite_anchor_approximation=True));M=[[F(0)]*4 for _ in range(4)];M[0][0]=1
for i in range(1,4):M[0][i]=M[i][0]=F(1,2)
for(i,j),v in {(1,2):F(1,6),(1,3):F(5,18),(2,3):F(2,9)}.items():M[i][j]=M[j][i]=v
vals=[c.dec(h.expectation(h.matrix_map(p,(1,2,3)),M)['value'])for p in polys];check('correlated_continuous_pushforward_moments',dict(moments=[[c.rec(v)for v in row]for row in M],expectations=list(map(c.rec,vals)),analytic_integrals=list(map(c.rec,[F(1,6),F(1,6),F(7,18),F(5,18)]))),vals==[F(1,6),F(1,6),F(7,18),F(5,18)])
begin('positive_unnormalized_mass',dict(scale=c.rec(F(7,3))));scaled=[[F(7,3)*v for v in row]for row in M];rv=[c.dec(h.expectation(h.matrix_map(p,(1,2,3)),scaled)['value'])for p in polys];check('positive_unnormalized_mass',dict(expectations=list(map(c.rec,rv)),classification=h.mass_case(scaled[0][0],rv),no_division=True),rv==[F(7,3)*v for v in vals]and sum(rv)==F(7,3))
begin('zero_signed_mass_NO_normalization',{});cases=[h.mass_case(F(0),[F(0)]*4),h.mass_case(F(0),[F(1),F(-1)]),h.mass_case(F(-2),[F(-1)])];check('zero_signed_mass_NO_normalization',dict(cases=cases),cases==['ZERO_positive_measure_ONLY','ZERO_mass_nonzero_moments_NOT_positive_measure','SIGNED_mass_NOT_positive_measure'])
begin('arbitrarily_small_exact_negative_is_retained',dict(synthetic_signed_functional_NOT_supported_measure=True));tiny=[[F(0)]*4 for _ in range(4)];tiny[0][0]=1;tiny[1][2]=tiny[2][1]=-F(1,2**180);r=h.expectation(h.matrix_map(polys[1],(1,2,3)),tiny);check('arbitrarily_small_exact_negative_is_retained',r,c.dec(r['value'])==-F(1,2**180)and r['classification'].startswith('NEGATIVE'))
begin('axis_permutation_and_complement_pair_invariance',{});ok=True;n=0
for perm in permutations(range(3)):
 for v,p in zip(h.REPRESENTATIVES,polys,strict=True):
  transformed={tuple(k[perm[i]]for i in range(3)):z for k,z in p.items()};vp=tuple(v[perm[i]]for i in range(3));out=h.paired(vp);ok=ok and transformed==out;n+=1;emit(dict(kind='AFTER_permutation_RETURN',permutation=perm,representative=v,polynomial=h.polynomial_record(out)))
check('axis_permutation_and_complement_pair_invariance',dict(identities=n),ok and n==24)
begin('upper_matrix_map_Frobenius_adjoint_and_mass_partition',{});checks=[]
for p in polys:
 mp=h.matrix_map(p,(1,2,3));A=[[F(0)]*4 for _ in range(4)]
 for(i,j),v in mp.items():A[i][j]=v if i==j else v/2;A[j][i]=A[i][j]
 lhs=sum(A[i][j]*M[i][j]for i,j in product(range(4),repeat=2));rhs=c.dec(h.expectation(mp,M)['value']);checks.append(lhs==rhs)
check('upper_matrix_map_Frobenius_adjoint_and_mass_partition',dict(identities=len(checks),partition_sum=c.rec(sum(vals)),mass=c.rec(M[0][0])),all(checks)and sum(vals)==M[0][0])
begin('missing_cross_moment_UNRESOLVED_NOT_product_of_means',{});missing=[list(row)for row in M];missing[1][3]=missing[3][1]=None;r=h.expectation(h.matrix_map(polys[1],(1,2,3)),missing);check('missing_cross_moment_UNRESOLVED_NOT_product_of_means',r,r['classification']=='UNRESOLVED_absent_moments'and r['missing_entries']==[[1,3]]and missing[1][3]is None)
J.close();report=dict(schema='v2d.cube-support-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_controls_passed=True,new_analytic_controls=len(controls),control_names=controls,all_old_controls_NOT_repeated=True,new_solvers=0,new_source_URDF_native_points=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',report);print(json.dumps(report))
