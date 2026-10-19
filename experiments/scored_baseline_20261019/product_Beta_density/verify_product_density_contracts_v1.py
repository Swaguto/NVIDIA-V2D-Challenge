"""Twelve NEW exact continuous-density/finite-moment/positivity contracts."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations,product,permutations
import datetime,hashlib,json,os,time
import product_beta_density_v1 as h
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-product-density-contract-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic();J=(O/'NEW-density-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v);controls=[]
def save(name,v):
 with(O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def begin(n,v):emit(dict(kind='BEFORE_NEW_contract',name=n,proposal=v))
def check(n,v,ok):emit(dict(kind='AFTER_NEW_contract_RETURN',name=n,record=v,passed=bool(ok)));assert ok,n;controls.append(n)
def known_matrix(ab,lam):
 n=len(ab);M=[[F(0)]*(n+1)for _ in range(n+1)];M[0][0]=1
 for i,(a,b)in enumerate(ab,1):M[0][i]=M[i][0]=F(a,a+b);M[i][i]=F(a*(a+1),(a+b)*(a+b+1))
 for i,j in combinations(range(n),2):
  vi=M[i+1][i+1]-M[0][i+1]**2;vj=M[j+1][j+1]-M[0][j+1]**2;M[i+1][j+1]=M[j+1][i+1]=M[0][i+1]*M[0][j+1]+lam.get((i,j),F(0))*vi*vj
 return M
begin('Beta_positive_integral_recurrence',dict(a=2,b=3,proof='Integrate derivative t^(a+r)*(1-t)^b; boundaries vanish for a+r,b>0. This gives (a+r)*B(a+r,b)=(a+b+r)*B(a+r+1,b).'));values=[h.beta_moment(F(2),F(3),r)for r in range(5)];check('Beta_positive_integral_recurrence',dict(values=list(map(c.rec,values))),values==[F(1),F(2,5),F(1,5),F(4,35),F(1,14)])
begin('inverse_Beta_parameter_and_variance_binding',{});M=known_matrix([(F(2),F(3))],{});pr=h.parameters(M,emit);emit(dict(kind='AFTER_NEW_inverse_parameter_RETURN',RETURN=dict(pr,parameters=h.recparams(pr['parameters']))));p=pr['parameters'][0];check('inverse_Beta_parameter_and_variance_binding',{k:c.rec(v)for k,v in p.items()},p['mu']==F(2,5)and p['variance']==F(1,25)and p['kappa']==5 and p['a']==2 and p['b']==3)
begin('continuous_two_variable_positive_correlated_density',dict(base='Uniform product',lambda_cross=c.rec(F(1))));M2=known_matrix([(F(1),F(1))]*2,{(0,1):F(1)});ps2=h.parameters(M2,emit)['parameters'];P2,cs2=h.density(M2,ps2,emit);emit(dict(kind='AFTER_NEW_two_variable_density_RETURN',polynomial=h.recpoly(P2),corrections=cs2));corners=[h.evaluate(P2,b)for b in product((F(0),F(1)),repeat=2)];check('continuous_two_variable_positive_correlated_density',dict(corner_values=list(map(c.rec,corners)),cross=c.rec(M2[1][2])),corners==[F(5,4),F(3,4),F(3,4),F(5,4)]and M2[1][2]==F(37,144))
begin('asymmetric_continuous_three_Beta_density',dict(a_b=[[2,3],[3,2],[1,4]],fixed_lambdas=['1/7','-1/11','1/13']));M3=known_matrix([(F(2),F(3)),(F(3),F(2)),(F(1),F(4))],{(0,1):F(1,7),(0,2):F(-1,11),(1,2):F(1,13)});ps3=h.parameters(M3,emit)['parameters'];P3,cs3=h.density(M3,ps3,emit);emit(dict(kind='AFTER_NEW_asymmetric_density_RETURN',polynomial=h.recpoly(P3),corrections=cs3));cv=[h.evaluate(P3,b)for b in product((F(0),F(1)),repeat=3)];check('asymmetric_continuous_three_Beta_density',dict(parameters=h.recparams(ps3),corner_values=list(map(c.rec,cv))),min(cv)>0 and [c.dec(r['coefficient'])for r in cs3]==[F(1,7),F(-1,11),F(1,13)])
begin('normalization_and_all_first_second_cross_identities',{});moments=[]
for i in range(4):
 for j in range(i,4):
  powers=tuple(int(i==k)+int(j==k)for k in range(1,4));value=h.weighted_moment(ps3,P3,powers);emit(dict(kind='AFTER_NEW_moment_RETURN',indices=[i,j],value=c.rec(value),target=c.rec(M3[i][j])));moments.append(value==M3[i][j])
check('normalization_and_all_first_second_cross_identities',dict(identities=len(moments)),all(moments)and len(moments)==10)
begin('centered_factor_orthogonality_and_pair_isolation',{});records=[]
for p in ps3:
 es=[h.beta_moment(p['a'],p['b'],k+1)-p['mu']*h.beta_moment(p['a'],p['b'],k)for k in range(3)];records.append(dict(centered_moments=list(map(c.rec,es)),variance=c.rec(p['variance'])))
check('centered_factor_orthogonality_and_pair_isolation',dict(records=records),all(c.dec(r['centered_moments'][0])==0 and c.dec(r['centered_moments'][1])==c.dec(r['variance'])for r in records))
begin('multiaffine_interpolation_entire_cube_not_anchor_selection',{});identities=[]
for t in product((F(1,4),F(2,5),F(3,4)),repeat=3):
 value=F(0)
 for b in product((0,1),repeat=3):
  weight=F(1)
  for x,bit in zip(t,b,strict=True):weight*=x if bit else 1-x
  value+=weight*h.evaluate(P3,b)
 identities.append(value==h.evaluate(P3,t)and value>0)
check('multiaffine_interpolation_entire_cube_not_anchor_selection',dict(exact_interior_identities=len(identities),positive_corner_minimum=c.rec(min(cv))),all(identities)and len(identities)==27)
begin('signed_zero_and_nonunit_mass_UNRESOLVED_no_normalization',{});returns=[]
for mass in [F(-1),F(0),F(2)]:
 Z=[list(row)for row in M2];Z[0][0]=mass;ret=h.parameters(Z,emit);emit(dict(kind='AFTER_NEW_mass_domain_RETURN',RETURN=ret));returns.append(ret['classification'])
check('signed_zero_and_nonunit_mass_UNRESOLVED_no_normalization',dict(returns=returns),all(v.startswith('UNRESOLVED')for v in returns))
begin('boundary_zero_and_invalid_variance_UNRESOLVED',{});returns=[]
for mu,second in [(F(0),F(0)),(F(1),F(1)),(F(1,2),F(1,4)),(F(1,2),F(1,2)),(F(1,2),F(3,4))]:
 Z=[[F(1),mu],[mu,second]];ret=h.parameters(Z,emit);emit(dict(kind='AFTER_NEW_variance_domain_RETURN',RETURN=ret));returns.append(ret['classification'])
check('boundary_zero_and_invalid_variance_UNRESOLVED',dict(returns=returns),len(returns)==5 and all(v=='UNRESOLVED_invalid_parameter_domains'for v in returns))
begin('absent_cross_NOT_filled_from_mean_products',{});Z=[list(row)for row in M2];Z[1][2]=Z[2][1]=None;ret=h.parameters(Z,emit);check('absent_cross_NOT_filled_from_mean_products',ret,ret['classification']=='UNRESOLVED_missing_moments_NO_mean_product_fill'and Z[1][2]is None)
begin('strict_tiny_negative_density_rejects_ansatz_ONLY',dict(lambda_cross=c.rec(F(4)+F(1,2**120))));N=known_matrix([(F(1),F(1))]*2,{(0,1):F(4)+F(1,2**120)});np=h.parameters(N,emit)['parameters'];negative_poly,cs=h.density(N,np,emit);emit(dict(kind='AFTER_NEW_negative_density_RETURN',polynomial=h.recpoly(negative_poly),corrections=cs));value=h.evaluate(negative_poly,(F(0),F(1)));check('strict_tiny_negative_density_rejects_ansatz_ONLY',dict(value=c.rec(value),classification='THIS_FIXED_density_ansatz_negative_NOT_matrix_or_family_exclusion'),value==-F(1,2**122))
begin('asymmetric_axis_order_invariance_without_parameter_sweep',{});ok=[]
for perm in permutations(range(3)):
 order=(0,)+tuple(i+1 for i in perm);Z=[[M3[i][j]for j in order]for i in order];zp=h.parameters(Z,emit)['parameters'];poly,cs=h.density(Z,zp,emit);emit(dict(kind='AFTER_NEW_permutation_density_RETURN',permutation=perm,polynomial=h.recpoly(poly)));expected={tuple(k[perm[i]]for i in range(3)):v for k,v in P3.items()};ok.append(poly==expected)
check('asymmetric_axis_order_invariance_without_parameter_sweep',dict(identities=len(ok)),all(ok)and len(ok)==6)
J.close();out=dict(schema='v2d.product-density-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_NEW_controls_passed=True,new_analytic_controls=len(controls),control_names=controls,all_old_controls_NOT_repeated=True,new_solver_calls=0,new_episode_evaluations=0,new_source_URDF_native_points=0,new_GPU_jobs=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
