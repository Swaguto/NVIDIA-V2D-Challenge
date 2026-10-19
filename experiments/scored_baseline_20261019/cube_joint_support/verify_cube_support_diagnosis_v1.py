"""Independent Boolean interpolation / direct moment identities for NEW480results.
No old Gram/PSD/scalar audit, solver, completion or source polynomial job.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product,combinations
from math import comb
import datetime,hashlib,json,os,time
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
S=Path(__file__).resolve().parents[1];O=S/'e037-cube-support-independent-v1';D=S/'e037-cube-support-diagnosis-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();J=(O/'independent-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(J,v)
def save(name,v):
 with(O/name).open('w')as f:f.write(json.dumps(g.plain(v),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
report=read(D/'report.json');bindings=report['saved_input_bindings'];emit(dict(kind='BEFORE_NEW_independent_input_binding',bindings=bindings));checked={p:sha(p)==v for p,v in bindings.items()};emit(dict(kind='AFTER_NEW_independent_input_binding_RETURN',checks=checked));assert all(checked.values());matrixpath=S/'e037-global-completion-evaluation-v1/separate-existential-completion-matrix.json';M=[[c.dec(v)for v in row]for row in read(matrixpath)['matrix']];rows=read(D/'all-480-exact-expectation-results.json');results=[];negative=[];groups={};coefficient_count=map_count=corner_count=0
for row in rows:
 triple=tuple(row['indices']);v=tuple(row['representative']);emit(dict(kind='BEFORE_NEW_independent_paired_corner',key=row['key'],triple=triple,representative=v));truth={b:int(b==v or b==tuple(1-x for x in v))for b in product((0,1),repeat=3)};power={}
 # Independent Boolean-lattice Moebius interpolation, no producer multiplication.
 for alpha in product((0,1),repeat=3):
  value=sum((-1)**(sum(alpha)-sum(beta))*truth[beta]for beta in product((0,1),repeat=3)if all(b<=a for a,b in zip(alpha,beta,strict=True)))
  if value:power[alpha]=F(value)
 emit(dict(kind='AFTER_NEW_Boolean_interpolation_RETURN',key=row['key'],power_coefficients=[dict(powers=a,value=c.rec(z))for a,z in sorted(power.items())]));saved={tuple(z['powers']):c.dec(z['coefficient'])for z in row['polynomial']};ok=power==saved and (1,1,1)not in power;coefficient_count+=8
 for b in product((0,1),repeat=3):
  val=sum(z*F(b[0])**a[0]*F(b[1])**a[1]*F(b[2])**a[2]for a,z in power.items());ok=ok and val==truth[b]and val>=0;corner_count+=1
 a,b,d=triple;m=M[0][0];x,y,z=[M[0][i]for i in triple];xy,xz,yz=M[a][b],M[a][d],M[b][d]
 # Direct four signed scalar formulas independent of generic map constructor.
 choices={(0,0,0):m-x-y-z+xy+xz+yz,(0,0,1):z-xz-yz+xy,(0,1,0):y-xy-yz+xz,(0,1,1):x-xy-xz+yz};value=choices[v];mp={}
 for powers,co in power.items():
  indices=[triple[i]for i in range(3)if powers[i]];entry=(0,0)if not indices else(0,indices[0])if len(indices)==1 else tuple(sorted(indices));mp[entry]=co
 savedmap={tuple(z['matrix_entry']):c.dec(z['coefficient'])for z in row['expectation_map']};mapvalue=sum(co*M[i][j]for(i,j),co in mp.items());ok=ok and mp==savedmap and value==mapvalue==c.dec(row['value']);map_count+=len(mp);result=dict(key=row['key'],polynomial_map_and_direct_expectation_verified=ok,exact_expectation=c.rec(value),negative=value<0,scope='ONLY_THIS_ONE_separate_DIAGNOSTIC_matrix');emit(dict(kind='AFTER_NEW_independent_paired_corner_RETURN',RETURN=result));results.append(result)
 if value<0:negative.append(result)
 groups.setdefault(triple,[]).append(value);assert ok
partitions=[dict(indices=triple,entries=len(values),sum=c.rec(sum(values)),equals_mass=sum(values)==M[0][0])for triple,values in sorted(groups.items())];emit(dict(kind='AFTER_NEW_independent_all_partitions_RETURN',partitions=partitions));assert set(groups)==set(combinations(range(1,11),3))and all(z['entries']==4 and z['equals_mass']for z in partitions);saved_negative=read(D/'all-negative-weighted-paired-corner-witnesses.json');assert {z['key']for z in negative}=={z['key']for z in saved_negative};assert len(rows)==480 and len(negative)==report['exact_negative_expectations'];events=[json.loads(line)for line in(D/'ONE-fixed-family-proposals-and-returns.jsonl').read_text().splitlines()];pairs=[z for z in events if z['kind']in['BEFORE_paired_corner_expectation','AFTER_paired_corner_expectation_RETURN']];journalok=len(pairs)==960 and all(pairs[2*i]['kind']=='BEFORE_paired_corner_expectation'and pairs[2*i+1]['kind']=='AFTER_paired_corner_expectation_RETURN'and pairs[2*i]['proposal']['key']==pairs[2*i+1]['key']for i in range(480));emit(dict(kind='AFTER_NEW_independent_journal_and_negative_witness_RETURN',ordered480durable_before_after_pairs=journalok,negative_witnesses=len(negative)));assert journalok and all(sha(p)==v for p,v in bindings.items())
save('all-independent-polynomial-map-expectation-certificates.json',results);save('all-independent-negative-witness-certificates.json',negative);save('all-independent-partition-certificates.json',partitions);J.close();out=dict(schema='v2d.cube-support-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_independent_checks_passed=True,independent_Boolean_coefficients=coefficient_count,corner_interpolation_nonnegativity_checks=corner_count,exact_expectation_map_coefficients=map_count,independent_direct_expectations=len(results),independent_mass_partition_checks=len(partitions),negative_witnesses_verified=len(negative),negative_triples=len({tuple(r['indices'])for r in saved_negative}),ordered_durable_before_after_pairs_verified=480,minimum_expectation=report['minimum_expectation'],classification=report['classification'],all_inputs_entries_and_raw_returns_UNCHANGED=True,no_other_completion_envelope_family_or_task_exclusion=True,new_solver_calls=0,new_source_URDF_native_points=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
