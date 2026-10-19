"""Generic saved weighted-matrix conic maps, packing and exact dual residuals.

All original canonical rows are reused, not rebuilt. New matrices use existing
table columns only. Float error bounds are necessary outer-model guards.
"""
from fractions import Fraction as F
import hashlib,json,math
import conditional_moment_cones_v1 as c
import direct_conditional_moment_sdp_v1 as old

def add(a,b,scale=F(1)):
    out=dict(a)
    for col,v in b.items():out[col]=out.get(col,F(0))+scale*v
    return {col:v for col,v in out.items()if v}
def co_rec(co):return [dict(column=k,value=c.rec(v))for k,v in sorted(co.items())if v]
def co_dec(co):return {v['column']:c.dec(v['value'])for v in co}
def block_record(block):
    maps={};references=[]
    for row in block['maps']:
        refs=[]
        for co in row:
            value=co_rec(co);key=hashlib.sha256(json.dumps(value,separators=(',',':')).encode()).hexdigest();maps.setdefault(key,value);refs.append(key)
        references.append(refs)
    return dict(id=block['id'],dimension=block['dimension'],kind=block['kind'],scope=block.get('scope'),origin=block.get('origin'),map_dictionary=maps,entry_map_sha256=references)
def block_decode(record):
    maps={k:co_dec(v)for k,v in record['map_dictionary'].items()};out={k:v for k,v in record.items()if k not in['map_dictionary','entry_map_sha256']};out['maps']=[[maps[key]for key in row]for row in record['entry_map_sha256']];return out
def base_blocks(packing):
    result=[]
    for i,b in enumerate(packing['PSD_blocks']):
        columns=b['group']['columns'];maps=[[{col:c.dec(v)for col,v in zip(columns,row)if c.dec(v)}for row in M]for M in b['exact_entry_maps']]
        result.append(dict(id='saved_base:'+str(i),dimension=b['dimension'],kind='saved_conditional:'+b['cone'],scope=b['group']['scope'],maps=maps,row_start=b['row_start'],row_end=b['row_end']))
    return result
def joint_blocks(record,origin):
    moments={tuple(v['power']):co_dec(v['coefficients'])for v in record['moment_maps']};result=[]
    for matrix in record['matrices']:
        basis=list(map(tuple,matrix['basis']));axis=matrix['axis'];maps=[]
        for a in basis:
            row=[]
            for b in basis:
                power=tuple(i+k for i,k in zip(a,b))
                if axis is None:co=moments[power]
                else:
                    one=list(power);two=list(power);one[axis]+=1;two[axis]+=2;co=add(moments[tuple(one)],moments[tuple(two)],F(-1))
                row.append(co)
            maps.append(row)
        result.append(dict(id='joint:'+str(record['scope_id'])+':'+matrix['name'],dimension=len(basis),kind='joint:'+matrix['name'],scope=record['scope'],origin=origin,maps=maps))
    return result
def support_block(record,origin):
    maps={tuple(v['shift']):co_dec(v['coefficients'])for v in record['unique_shift_maps']};basis=list(map(tuple,record['multiplier_basis']));M=[[maps[tuple(i+k for i,k in zip(a,b))]for b in basis]for a in basis]
    return dict(id='support:'+record['case']['id'],dimension=len(basis),kind='support:'+record['case']['kind'],scope=record['table_scope'],origin=origin,maps=M)
def pack_block(block):
    n=block['dimension'];maps=block['maps']
    if n not in [3,4,6,8,10]or len(maps)!=n or any(len(row)!=n for row in maps):raise ValueError('new declared cone dimensions3/4/6/8/10 only')
    if any(maps[i][k]!=maps[k][i]for i in range(n)for k in range(n)):raise ValueError('asymmetric linear maps cannot be averaged')
    lo,hi=old.root_two_interval();rows=[];errors=[[F(0)for _ in range(n)]for _ in range(n)]
    for i,k in old.layout(n):
        exact=maps[i][k];packed={col:float(v)*(1 if i==k else math.sqrt(2))for col,v in exact.items()};error=sum(abs(F(packed[col])-v)if i==k else max(abs(F(packed[col])/lo-v),abs(F(packed[col])/hi-v))for col,v in exact.items());errors[i][k]=errors[k][i]=error
        rows.append(dict(i=i,j=k,packed_coefficients=[dict(column=col,value=z)for col,z in sorted(packed.items())if z],unpacked_unit_box_error=c.rec(error)))
    radius=max(map(sum,errors));guard=old.up(radius)
    return dict(id=block['id'],dimension=n,rows=rows,sqrt_two_interval=[c.rec(lo),c.rec(hi)],entry_error_bounds=[[c.rec(v)for v in row]for row in errors],spectral_error_bound=c.rec(radius),outward_diagonal_guard=guard,unit_box_assumption=[0,1],necessary_guard_NOT_original_candidate_tolerance=True)
def append_blocks(base_A,base_b,blocks,emit):
    import numpy as np
    from scipy import sparse
    rr=[];cc=[];vv=[];b=[];packed=[];offset=base_A.shape[0];n=base_A.shape[1]
    for block in blocks:
        emit(dict(kind='BEFORE_NEW_weighted_PSD_packing',id=block['id'],dimension=block['dimension']))
        value=pack_block(block);value['row_start']=offset+len(b)
        for row in value['rows']:
            index=len(b)
            for co in row['packed_coefficients']:rr.append(index);cc.append(co['column']);vv.append(-co['value'])
            b.append(value['outward_diagonal_guard']if row['i']==row['j']else 0.)
        value['row_end']=offset+len(b);packed.append(value);block['row_start']=value['row_start'];block['row_end']=value['row_end'];emit(dict(kind='AFTER_NEW_weighted_PSD_packing_RETURN',record=value))
    extra=sparse.csc_matrix((vv,(rr,cc)),shape=(len(b),n));combined=sparse.vstack([base_A,extra],format='csc');combined.sort_indices()
    return combined,np.concatenate([base_b,np.array(b)]),packed
def dual_certificate(model,blocks,z,emit):
    eq=len(model['equalities']);ni=len(model['inequalities']);n=len(model['columns']);beta=list(map(F,z[:eq]));alpha=list(map(F,z[eq:eq+ni]));bad=[i for i,v in enumerate(alpha)if v<0];emit(dict(kind='AFTER_NEW_supported_scalar_dual_sign_RETURN',negative_inequality_duals=bad,actual_solver_returns_NOT_changed=True))
    if bad:return dict(classification='UNRESOLVED_negative_scalar_dual',indices=bad)
    residual=[F(0)]*n;rhs=F(0)
    for rows,values in [(model['equalities'],beta),(model['inequalities'],alpha)]:
        for row,w in zip(rows,values,strict=True):
            rhs+=w*c.dec(row['bound'])
            for v in row['coefficients']:residual[v['column']]+=w*c.dec(v['value'])
    records=[]
    for block in blocks:
        emit(dict(kind='BEFORE_NEW_supported_conic_Gram_certificate',id=block['id'],dimension=block['dimension']))
        raw=old.smat(z[block['row_start']:block['row_end']],block['dimension']);G,B=old.rational_gram(raw,lambda v:emit(dict(block_id=block['id'],**v)));d=block['dimension']
        for i in range(d):
            for k in range(i,d):
                weight=G[i][k]*(1 if i==k else 2)
                for col,v in block['maps'][i][k].items():residual[col]-=weight*v
        record=dict(id=block['id'],dimension=d,kind=block['kind'],raw_dual_matrix=[[c.rec(v)for v in row]for row in raw],Gram_factor=[[c.rec(v)for v in row]for row in B],certificate_matrix=[[c.rec(v)for v in row]for row in G]);records.append(record);emit(dict(kind='AFTER_NEW_supported_conic_Gram_RETURN',record=record))
    minimum=sum(min(F(0),v)for v in residual);gap=minimum-rhs
    return dict(classification='EXACT_frozen_supported_model_exclusion'if gap>0 else'UNRESOLVED_no_strict_supported_conic_gap',alpha=list(map(c.rec,alpha)),beta=list(map(c.rec,beta)),PSD_Gram_certificates=records,exact_residual=list(map(c.rec,residual)),finite_unit_box_minimum=c.rec(minimum),RHS=c.rec(rhs),strict_gap=c.rec(gap),original_unrounded_rational_rows_and_maps_used=True,actual_weights_and_raw_duals_NOT_clipped_or_normalized=True,certificate_Gram_products_are_NOT_robot_commands=True,scope='Necessary frozen region2/fixed unqualified wrist/coefficient model only; no source truth/global task impossibility')
