"""Independent canonical functional/map/adjoint and analytic embedding checks.
No old numerical inventory/copy/PSD audit; no actual moment values proposed.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
from itertools import combinations_with_replacement
from math import comb
import datetime,hashlib,json,os,time
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-global-functional-independent-v1';D=S/'e037-global-functional-model-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'independent-functional-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(n,v):
 with(O/n).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def mono(e,t):
 value=F(1)
 for r,x in zip(e,t,strict=True):value*=x**r
 return value
report=read(D/'report.json');bindings=report['saved_input_bindings'];emit(dict(kind='BEFORE_NEW_independent_functional_inputs',bindings=bindings));passed=all(sha(p)==v for p,v in bindings.items());emit(dict(kind='AFTER_NEW_independent_functional_inputs_RETURN',passed=passed));assert passed
symbolic=read(D/'symbolic-functional-model.json');variables=symbolic['basis_variables'];pair_defs=symbolic['pair_definitions'];descriptions=[()]+[(v,)for v in variables]+[tuple(p)for p in pair_defs];B=[tuple(Counter(names).get(v,0)for v in variables)for names in descriptions];assert B==[tuple(e)for e in symbolic['basis_powers']];expected={}
for i,j in combinations_with_replacement(range(24),2):
 counts=Counter(descriptions[i]+descriptions[j]);e=tuple(counts.get(v,0)for v in variables);expected.setdefault(e,[]).append((i,j))
assert len(expected)==261;classmaps={};known=free=knowncell=freecell=0;certs=[];aux=read(D/'FREE-symbolic-namespace-declarations.json');auxbycol={r['column']:r for r in aux};original=read(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json');normal=read(D/'original-prefix-and-authority-BY-HASH.json');massrow=original['equalities'][normal['normalization_row']];assert c.dec(massrow['bound'])==1
for number,p in enumerate(sorted(D.glob('monomial-class-*.json'))):
 r=read(p);e=tuple(r['powers']);emit(dict(kind='BEFORE_NEW_independent_functional_class',index=number,powers=e,matrix_entries=r['matrix_entries']));co={z['column']:c.dec(z['value'])for z in r['linear_map']};passed=set(map(tuple,r['matrix_entries']))==set(expected[e])and r['requested']=={v:k for v,k in zip(variables,e,strict=True)if k}
 if r['classification']=='FREE_shared_existential_symbol':
  col=r['symbolic_column'];declaration=auxbycol[col];passed=passed and co=={col:F(1)}and tuple(declaration['powers'])==e and declaration['no_value_assigned']and declaration['bounds_under_ORIGINAL_positive_exact_unit_mass_cube_model']==[0,1]and col>=1125;free+=1;freecell+=len(r['matrix_entries'])
 else:
  passed=passed and r['classification']=='KNOWN_original1125_column_linear_map'and all(0<=col<1125 for col in co)
  for provenance in r['all_known_alias_provenance']:
   source=Path(provenance['source']);passed=passed and sha(source)==provenance['source_sha256']and sha(provenance['prior_independent_certificate'])==provenance['prior_independent_certificate_sha256'];cached=read(source)
   if not any(e):saved={z['column']:c.dec(z['value'])for z in massrow['coefficients']}
   elif'anchor_coefficients'in cached:saved={z['column']:c.dec(z['value'])for z in cached['anchor_coefficients']}
   else:saved={z['column']:c.dec(z['value'])for z in cached['copies'][0]['coefficients']}
   passed=passed and saved==co=={z['column']:c.dec(z['value'])for z in provenance['linear_map']}
  passed=passed and all(z['coefficient_difference_zero']and z['additional_original_row_combination']==[]for z in r['canonical_alias_map_equivalences']);known+=1;knowncell+=len(r['matrix_entries'])
 classmaps[e]=co;ret=dict(index=number,powers=list(e),matrix_entries=r['matrix_entries'],linear_map_coefficients=len(co),classification=r['classification'],passed=passed);emit(dict(kind='AFTER_NEW_independent_functional_class_RETURN',RETURN=ret));certs.append(ret);assert passed
assert known==91 and free==170 and knowncell==110 and freecell==190 and set(auxbycol)==set(range(1125,1295));upper=read(D/'symbolic-global24-upper-linear-maps.json');matrix={};entrycerts=[]
for row in upper:
 i,j=row['matrix_entry'];e=tuple(a+b for a,b in zip(B[i],B[j],strict=True));co={z['column']:c.dec(z['value'])for z in row['linear_map']};passed=e==tuple(row['powers'])and co==classmaps[e];emit(dict(kind='AFTER_NEW_independent_entry_class_identity_RETURN',matrix_entry=[i,j],powers=e,passed=passed));entrycerts.append(dict(matrix_entry=[i,j],powers=list(e),passed=passed));assert passed;matrix[i,j]=co
# Full matrix and upper triangle implementations of the exact Frobenius adjoint.
emit(dict(kind='BEFORE_NEW_independent_symbolic_adjoint',dimension=24,PSD_packing_NOT_performed=True));upperadj={};fulladj={}
for(i,j),co in matrix.items():
 D_ij=F((i+1)*(j+1),37)
 for col,z in co.items():upperadj[col]=upperadj.get(col,F(0))+(1 if i==j else 2)*D_ij*z
for i in range(24):
 for j in range(24):
  co=matrix[min(i,j),max(i,j)]
  for col,z in co.items():fulladj[col]=fulladj.get(col,F(0))+F((i+1)*(j+1),37)*z
emit(dict(kind='AFTER_NEW_independent_symbolic_adjoint_RETURN',coefficients=len(upperadj),passed=upperadj==fulladj));assert upperadj==fulladj
# NEW analytic positive unit-mass deterministic embedding, NOT a robot/source state.
t=[F(i+1,i+13)for i in range(10)];emit(dict(kind='BEFORE_NEW_analytic_deterministic_embedding',normalized_tensor_proof_point=list(map(c.rec,t)),NOT_episode_pose_source_anchor_or_trajectory_candidate=True));tdict=dict(zip(variables,t));prefix=[]
for row in original['columns']:
 w=F(1)
 for name,k in zip(row['scope'],row['index'],strict=True):w*=comb(4,k)*tdict[name]**k*(1-tdict[name])**(4-k)
 prefix.append(w)
values=list(prefix)+[F(0)]*170
for col,r in auxbycol.items():values[col]=mono(tuple(r['powers']),t)
save('NEW-analytic-embedding-constructor-RETURN.json',dict(values=list(map(c.rec,values)),NOT_actual_original_RETURN_or_robot_candidate=True));emit(dict(kind='AFTER_NEW_analytic_embedding_constructor_RETURN',original_synthetic_Bernstein_columns=1125,synthetic_monomial_symbols=170,all_values_are_PROOF_control_ONLY=True));basisvalues=[mono(e,t)for e in B];embedding=[]
for(i,j),co in sorted(matrix.items()):
 value=sum(z*values[col]for col,z in co.items());target=basisvalues[i]*basisvalues[j];passed=value==target;ret=dict(matrix_entry=[i,j],left=c.rec(value),right=c.rec(target),passed=passed);emit(dict(kind='AFTER_NEW_analytic_embedding_entry_RETURN',RETURN=ret));embedding.append(ret);assert passed
for p,d in normal['original_source_time_prefix_files'].items():assert sha(p)==d
assert len(matrix)==300 and all(sha(p)==v for p,v in bindings.items());save('independent-261-class-identity-certificates.json',certs);save('independent-300-entry-map-certificates.json',entrycerts);save('independent-300-positive-measure-embedding-certificates.json',embedding);save('independent-exact-functional-adjoint.json',[dict(column=k,value=c.rec(v))for k,v in sorted(upperadj.items())]);J.close();out=dict(schema='v2d.global-functional-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_independent_checks_passed=True,basis_dimension=24,canonical_classes=261,known_classes=known,FREE_existential_classes=free,known_cell_bindings=knowncell,FREE_cell_bindings=freecell,symbolic_columns=1295,original_prefix_columns_UNCHANGED=1125,entry_map_identities=300,known_linear_alias_bindings=110,independent_exact_adjoint_coefficients=len(upperadj),NEW_analytic_positive_measure_embedding_entries=300,old_numeric_blocks_copy_audits_packing_controls_NOT_replayed=True,no_actual_numerical_moments_or_reference_proposed=True,all190unknown_cells_remain_unobserved_with_shared_symbols=True,classification=report['classification'],no_PSD_feasibility_common_source_time_measure_source_truth_or_trajectory_qualified=True,new_solver_calls=0,new_PSD_audits_or_completions=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
