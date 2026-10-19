"""Independent NEW canonical bindings before the sole support-aware solve."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-supported-model-independent-v1';P=S/'e037-supported-conic-model-v1';sys.path.insert(0,str(S/'compute'))
import numpy as np
from scipy import sparse
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'all-independent-new-weighted-map-and-packing-bindings.jsonl').open('w');emit=lambda v:g.journal(j,v)
def decode(v):return {x['column']:c.dec(x['value'])for x in v}
def subtract(a,b):
    out=dict(a)
    for col,z in b.items():out[col]=out.get(col,F(0))-z
    return {col:v for col,v in out.items()if v}
report=read(P/'report.json')
for name,digest in report['all_outputs_sha256'].items():assert sha(P/name)==digest
binding=read(P/'supported-model-binding.json');baseparent=read(S/'e037-direct-conditional-SDP-v1/report.json')
for name,key in [('base_canonical_A','base_canonical_A_sha256'),('base_canonical_b','base_canonical_b_sha256'),('base_packing','base_packing_sha256')]:assert sha(binding[name])==binding[key]==baseparent['all_outputs_sha256'][Path(binding[name]).name]
base=sparse.load_npz(binding['base_canonical_A']);baseb=np.load(binding['base_canonical_b']);A=sparse.load_npz(P/'supported-canonical-A.npz').tocsr();b=np.load(P/'supported-canonical-b.npy');assert A.shape==(15840,1125)and(base-A[:14099]).nnz==0 and np.array_equal(baseb,b[:14099]);assert sha(binding['source_unavailable_record'])==binding['source_unavailable_record_sha256']
packed=read(P/'additional-weighted-PSD-packing.json');offset=14099;blocks_checked=0;entries_checked=0;exact_coefficients=0;packed_coefficients=0;origins=set()
for blockfile,packing in zip(binding['additional_block_files'],packed,strict=True):
    path=P/blockfile['file'];assert sha(path)==blockfile['sha256'];record=read(path);assert record['id']==packing['id']==blockfile['id'];dim=record['dimension'];assert dim==packing['dimension'];origin=record['origin'];assert sha(origin['path'])==origin['sha256'];original=read(origin['path']);origins.add(origin['path']);maps={}
    for key,value in record['map_dictionary'].items():assert hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest()==key;maps[key]=decode(value)
    M=[[maps[key]for key in row]for row in record['entry_map_sha256']];emit(dict(kind='BEFORE_NEW_independent_supported_block_binding',id=record['id'],dimension=dim,origin=origin))
    if record['kind'].startswith('joint:'):
        matrix=next(v for v in original['matrices']if record['kind']=='joint:'+v['name']);moments={tuple(v['power']):decode(v['coefficients'])for v in original['moment_maps']};basis=list(map(tuple,matrix['basis']));expected=[]
        for a in basis:
            row=[]
            for bb in basis:
                power=tuple(i+k for i,k in zip(a,bb));axis=matrix['axis']
                if axis is None:co=moments[power]
                else:
                    one=list(power);two=list(power);one[axis]+=1;two[axis]+=2;co=subtract(moments[tuple(one)],moments[tuple(two)])
                row.append(co)
            expected.append(row)
    else:
        byshift={tuple(v['shift']):decode(v['coefficients'])for v in original['unique_shift_maps']};basis=list(map(tuple,original['multiplier_basis']));expected=[[byshift[tuple(i+k for i,k in zip(a,bb))]for bb in basis]for a in basis]
    assert M==expected and len(M)==dim and all(M[i][k]==M[k][i]for i in range(dim)for k in range(dim));lo,hi=map(c.dec,packing['sqrt_two_interval']);assert lo>0 and lo*lo<2<hi*hi;errors=[[F(0)for _ in range(dim)]for _ in range(dim)];layout=[(i,k)for k in range(dim)for i in range(k+1)];assert len(packing['rows'])==len(layout)and packing['row_start']==offset
    for row,(i,k)in zip(packing['rows'],layout,strict=True):
        assert row['i']==i and row['j']==k;exact=M[i][k];numeric={v['column']:F(v['value'])for v in row['packed_coefficients']};error=F(0)
        for col,z in exact.items():
            value=numeric.get(col,F(0));error+=abs(value-z)if i==k else max(abs(value/lo-z),abs(value/hi-z))
        assert set(numeric)<=set(exact)and error==c.dec(row['unpacked_unit_box_error']);errors[i][k]=errors[k][i]=error;actual=A.getrow(offset);actual={int(col):F(float(z))for col,z in zip(actual.indices,actual.data)if z};assert actual=={col:-z for col,z in numeric.items()if z};expectedb=packing['outward_diagonal_guard']if i==k else 0.;assert F(float(b[offset]))==F(expectedb);entries_checked+=1;exact_coefficients+=len(exact);packed_coefficients+=len(numeric);offset+=1
        emit(dict(kind='AFTER_NEW_independent_weighted_row_RETURN',block_id=record['id'],canonical_row=offset-1,exact_map_coefficients=len(exact),unpacked_error=c.rec(error),sign='minus_M plus PSD_slack equals diagonal_error_guard'))
    radius=max(map(sum,errors));assert errors==[[c.dec(v)for v in row]for row in packing['entry_error_bounds']]and radius==c.dec(packing['spectral_error_bound'])and F(packing['outward_diagonal_guard'])>=radius and packing['row_end']==offset;blocks_checked+=1;emit(dict(kind='AFTER_NEW_independent_supported_block_RETURN',id=record['id'],rows=len(layout),spectral_error=c.rec(radius),diagonal_guard=packing['outward_diagonal_guard'],necessary_only=True))
assert offset==15840 and blocks_checked==112 and entries_checked==1741 and len(origins)==68;j.close();out=dict(schema='v2d.supported-model-independent-binding.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_bindings_passed=True,canonical_base_A_sha256=binding['base_canonical_A_sha256'],canonical_base_b_sha256=binding['base_canonical_b_sha256'],canonical_prefix_rows_reused_and_unchanged=14099,new_exact_blocks_checked=blocks_checked,saved_origins_checked=len(origins),new_packed_rows_checked=entries_checked,exact_upper_entry_coefficients_checked=exact_coefficients,float_packed_coefficients_checked=packed_coefficients,original_columns=1125,total_rows=15840,original_equalities_inequalities_boxes_and_PSD_prefix_unchanged=True,no_source_time_generation_old_solver_or_old_matrix_audit_reexecuted=True,all112new_spectral_error_bounds_are_necessary_NOT_candidate_tolerances=True,unavailable_source_nonconstant_factors_retained=5,model_binding_sha256=sha(P/'supported-model-binding.json'),combined_A_sha256=sha(P/'supported-canonical-A.npz'),combined_b_sha256=sha(P/'supported-canonical-b.npy'),new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='all_outputs_sha256'}))
