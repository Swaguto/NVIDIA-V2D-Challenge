"""New arrow image-cone support contracts, not old packing/source controls."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-matrix-image-contract-v1';sys.path.insert(0,str(S/'compute'))
import matrix_image_support_v1 as a
import pointwise_support_moments_v1 as p
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda q:hashlib.sha256(Path(q).read_bytes()).hexdigest();j=(O/'all-NEW-arrow-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic();controls=[]
def check(name,record,valid):
 emit(dict(kind='AFTER_NEW_arrow_contract_RETURN',name=name,record=record,passed=bool(valid)));assert valid;controls.append(name)
def cert(M):return m.congruence_certificate(M,emit)
basis=[(0,0),(0,1),(1,0),(1,1)];entries={k:i for i,k in enumerate(product(range(5),repeat=2))}
emit(dict(kind='BEFORE_NEW_arrow_cone_equivalence_contract'))
for name,t,x,y,expected in [('inside',F(5),F(3),F(0),True),('boundary',F(5),F(3),F(4),True),('outside',F(4),F(3),F(4),False),('negative_depth',F(-5),F(0),F(0),False),('zero_apex',F(0),F(0),F(0),True),('zero_offapex',F(0),F(1),F(0),False)]:
 M=[[t,x,y],[x,t,0],[y,0,t]];r=cert(M);check('arrow_equivalence_'+name,dict(t=c.rec(t),x=c.rec(x),y=c.rec(y),certificate=r), (r['classification']=='exact_mixed_matrix_PSD')==expected and (t>=0 and t*t>=x*x+y*y)==expected)
# Positive supported/unsupported discrete measures exercise matrix polynomial support.
polys=[{(0,0):F(1)},{(1,0):F(4),(0,0):F(-2)},{}]
def weights_from_atoms(atoms):
 from math import comb
 return [sum(w*F(comb(4,k))*t**k*(1-t)**(4-k)*F(comb(4,l))*u**l*(1-u)**(4-l)for w,t,u in atoms)for k,l in product(range(5),repeat=2)]
for name,atoms,expected in [('supported',[(F(1,2),F(3,8),F(1,3)),(F(1,2),F(5,8),F(2,3))],True),('outside_cancelled_mean',[(F(1,2),F(0),F(1,3)),(F(1,2),F(1),F(2,3))],False),('zero_positive_measure',[],True),('signed_zero_mass',[(F(1),F(1,2),F(1,2)),(F(-1),F(0),F(1,2))],False)]:
 emit(dict(kind='BEFORE_NEW_matrix_measure_contract',name=name,atoms=[[c.rec(z)for z in atom]for atom in atoms]));weights=weights_from_atoms(atoms);M,maps,rows=a.localizer(entries,polys,basis,weights,emit);r=cert(M);passed=(r['classification']=='exact_mixed_matrix_PSD')==expected
 if name=='outside_cancelled_mean':
  scalar=cert([[M[i*4][k*4]for k in range(3)]for i in range(3)]);v=[F(-2),0,F(4),0,F(-1),0,0,0,0,0,0,0];co=m.polynomial_coefficients(entries,a.weighted_square(polys,basis,v));q=sum(w*weights[k]for k,w in co.items());passed=passed and scalar['classification']=='exact_mixed_matrix_PSD'and q==-3
  emit(dict(kind='AFTER_NEW_positive_mean_negative_support_RETURN',scalar=scalar,exact_weighted_expectation=c.rec(q),vector=list(map(c.rec,v))))
 check('matrix_supported_measure_'+name,dict(certificate=r,zero_signed_mass_NEVER_normalized=True),passed)
# Explicit source-model coefficient discrepancy and depth lower-bound guard.
emit(dict(kind='BEFORE_NEW_necessary_arrow_error_contract'))
source=dict(K=[[1,0,0],[0,1,0],[0,0,1]],pixel=[0,0],ORIGINAL_cap_px=1.,ORIGINAL_image_tolerance_px=0.,polynomial=dict(camera_numerators=[[[1.]],[[0.]],[[1.]]],denominator=[[1.,0.,1.],[0.,0.,0.],[1.,0.,1.]],positive_denominator_for_finite_chart=True,positive_depth_inequality_separate=True,minimum_depth=.125,source_scale=1.,depth_scale=1.,normalized_source_polynomial=[[-.125]],normalized_depth_polynomial=[[.875,0,-.125],[0,0,0],[-.125,0,-.125]]))
polys2,binding=a.construct(source,[(F(0),F(1)),(F(0),F(1))],emit);delta=c.dec(binding['diagonal_guard']);eps=c.dec(binding['uniform_source_discrepancy']);tl=c.dec(binding['t_lower']);check('necessary_guard_from_saved_model_error',binding,delta>0 and 2*tl*delta+delta*delta>=eps and c.dec(binding['Nz_lower'])>0)
# Unavailable powers/tiny terms/axis signs remain explicit, never rounded away.
emit(dict(kind='BEFORE_NEW_matrix_tensor_capacity_contract'))
tiny=F(1,2**300);validpolys=[{(2,2):tiny},{(1,0):F(1)},{}];_,maps,_=a.localizer(entries,validpolys,basis,[F(1,25)]*25,emit);reject=False
try:a.localizer(entries,[{(3,0):tiny},{},{}],basis,[F(1,25)]*25,emit)
except ValueError as e:emit(dict(kind='AFTER_NEW_unavailable_matrix_power_RETURN',error=str(e)));reject='UNRESOLVED'in str(e)
reordered=p.reorder({(2,0):tiny,(0,1):F(-3)},['u','v'],['v','u']);check('tensor_capacity_tiny_terms_and_axis_permutation',dict(available_shiftmaps=len(maps),tiny=c.rec(tiny),unavailable_power_retained=reject,reordered=p.polyrec(reordered)),reject and any(any(c.dec(v['value'])for v in x['coefficients'])for x in maps)and reordered=={(0,2):tiny,(1,0):F(-3)})
j.close();out=dict(schema='v2d.matrix-image-contracts.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/matrix_image_support_v1.py'),all_controls_passed=True,new_analytic_controls=len(controls),controls=controls,old_controls_polynomials_audits_or_solver_NOT_repeated=True,no_solver=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={q.name:sha(q)for q in O.iterdir()if q.is_file()and q.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with (O/'report.json').open('w')as f:f.write(json.dumps(out,allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k not in['controls','all_outputs_sha256']}))
