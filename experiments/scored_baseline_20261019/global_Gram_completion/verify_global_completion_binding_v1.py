"""Independent new66-variable completion/interval/support/canonical bindings."""
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
import datetime,hashlib,json,math,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-completion-binding-v1';P=S/'e037-global-completion-model-v1';sys.path.insert(0,str(S/'compute'))
import numpy as np
from scipy import sparse
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-NEW-global-completion-independent-bindings.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic();binding=read(P/'completion-model-binding.json');model=read(P/'separate-global-completion-model.json');assert sha(P/'separate-global-completion-model.json')==binding['separate_model_sha256'];assert sha(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json')==binding['original_source_model_sha256']and sha(S/'e037-image-conic-evaluation-v1/single-image-conic-solver-RETURN.json')==binding['original1125_RETURN_sha256'];layout=[(i,k)for k in range(11)for i in range(k+1)];index={p:i for i,p in enumerate(layout)};assert [tuple(v['matrix_entry'])for v in model['columns']]==layout;known={}
for r in model['known_intervals']:
 pair=tuple(r['entry']);origin=r['origin'];assert sha(origin['path'])==origin['sha256'];saved=read(origin['path']);assert r['interval']==saved['diagnostic_interval']and r['anchor']==saved['anchor_value'];requested=saved['requested'];variables=model['variables']
 if len(requested)==1:
  name,power=next(iter(requested.items()));i=variables.index(name)+1;expected=(0,i)if power==1 else(i,i)
 else:expected=tuple(sorted(variables.index(v)+1 for v in requested))
 assert pair==expected;known[pair]=list(map(c.dec,r['interval']));emit(dict(kind='AFTER_NEW_independent_frozen_interval_binding',pair=pair,origin=origin,interval=r['interval'],anchor=r['anchor']))
assert len(known)==39 and model['known_upper_entries']==40 and model['free_unknown_entries']==26 and model['variable_bounds']==[0,1];expected_rows={}
def add(name,co,b):expected_rows[name]=(co,F(b))
for pair,(lo,hi)in known.items():add('known-upper:'+str(pair),{index[pair]:F(1)},hi);add('known-lower:'+str(pair),{index[pair]:F(-1)},-lo)
for i in range(1,11):add('first<=mass:'+str(i),{index[0,i]:F(1),index[0,0]:F(-1)},0);add('second<=first:'+str(i),{index[i,i]:F(1),index[0,i]:F(-1)},0)
for i,k in combinations(range(1,11),2):add('cross<=first-i:'+str((i,k)),{index[i,k]:F(1),index[0,i]:F(-1)},0);add('cross<=first-j:'+str((i,k)),{index[i,k]:F(1),index[0,k]:F(-1)},0);add('Frechet-lower:'+str((i,k)),{index[0,i]:F(1),index[0,k]:F(1),index[0,0]:F(-1),index[i,k]:F(-1)},0)
def co(v):return {z['column']:c.dec(z['value'])for z in v}
assert len(expected_rows)==len(model['inequalities'])==233 and set(expected_rows)=={r['name']for r in model['inequalities']};assert len(model['equalities'])==1 and co(model['equalities'][0]['coefficients'])=={index[0,0]:F(1)}and c.dec(model['equalities'][0]['bound'])==1
A=sparse.load_npz(P/'completion-canonical-A.npz');b=np.load(P/'completion-canonical-b.npy');packed=read(P/'completion-necessary-packing.json');assert sha(P/'completion-canonical-A.npz')==binding['A_sha256']and sha(P/'completion-canonical-b.npy')==binding['b_sha256']and A.shape==(432,66)and len(b)==432;assert packed['nonnegative_rows']==365 and packed['PSD_packing']['row_start']==366
assert {int(k):F(float(v))for k,v in zip(A.getrow(0).indices,A.getrow(0).data)}=={0:F(1)}and b[0]==1;linear_coeffs=0;max_rhs=F(0)
for rownum,r in enumerate(model['inequalities'],1):
 exact,bound=expected_rows[r['name']];assert co(r['coefficients'])==exact and c.dec(r['bound'])==bound;actual={int(k):float(v)for k,v in zip(A.getrow(rownum).indices,A.getrow(rownum).data)};assert actual=={k:float(z)for k,z in exact.items()};error=sum(abs(F(actual[k])-z)for k,z in exact.items());guard=F(float(b[rownum]))-bound;assert guard>=error;max_rhs=max(max_rhs,guard);linear_coeffs+=len(exact)
for sign,rhs,startrow in [(-1,0,234),(1,1,300)]:
 for k in range(66):assert A.getrow(startrow+k).nnz==1 and A[startrow+k,k]==sign and b[startrow+k]==rhs
block=read(P/'separate-global-Gram-map.json');dictionary={k:co(v)for k,v in block['map_dictionary'].items()};maps=[[dictionary[k]for k in row]for row in block['entry_map_sha256']];assert len(maps)==11 and all(maps[i][k]=={index[min(i,k),max(i,k)]:F(1)}for i in range(11)for k in range(11));psd=packed['PSD_packing'];lo,hi=list(map(c.dec,psd['sqrt_two_interval']));assert lo*lo<=2<=hi*hi;errors=[[F(0)for _ in range(11)]for _ in range(11)]
for offset,(i,k)in enumerate(layout):
 rownum=366+offset;row=psd['rows'][offset];value=1. if i==k else math.sqrt(2);column=index[i,k];assert A.getrow(rownum).nnz==1 and A[rownum,column]==-value and b[rownum]==(psd['necessary_diagonal_guard']if i==k else 0);error=F(0)if i==k else max(abs(F(value)/lo-1),abs(F(value)/hi-1));errors[i][k]=errors[k][i]=error;assert error==c.dec(row['unit_box_unpacked_error'])
assert errors==[[c.dec(z)for z in r]for r in psd['entry_errors']]and max(map(sum,errors))==c.dec(psd['spectral_error'])and F(psd['necessary_diagonal_guard'])>=max(map(sum,errors));emit(dict(kind='AFTER_NEW_independent_global_completion_canonical_RETURN',known_intervals=39,unit_mass_equalities=1,necessary_inequalities=233,linear_coefficients=linear_coeffs,unitbox_rows=132,PSD_rows=66,unpacked_PSD_spectral_error=psd['spectral_error'],maximum_necessary_RHS_guard=c.rec(max_rhs),original1125table_NOT_changed=True));j.close();out=dict(schema='v2d.global-completion-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_bindings_passed=True,columns=66,known_entries=40,free_entries=26,rows=432,equalities=1,inequalities=233,nonnegative_rows=365,PSD_dimension=11,known_interval_bindings=39,linear_coefficients_checked=linear_coeffs,Gram_entry_maps_checked=121,PSD_packing_rows_checked=66,maximum_necessary_RHS_guard=c.rec(max_rhs),source_model_original1125weights_and_raw_residuals_UNCHANGED=True,unknowns_existential_NOT_observations_or_truth=True,model_binding_sha256=sha(P/'completion-model-binding.json'),A_sha256=sha(P/'completion-canonical-A.npz'),b_sha256=sha(P/'completion-canonical-b.npy'),new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with (O/'report.json').open('w')as f:f.write(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
