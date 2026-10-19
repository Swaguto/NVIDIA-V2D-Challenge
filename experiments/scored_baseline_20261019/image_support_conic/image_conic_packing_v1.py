"""NEW dimension12 packing of SAVED matrix-image supports; no source generation."""
from fractions import Fraction as F
import math
import conditional_moment_cones_v1 as c
import supported_conic_maps_v1 as h
import direct_conditional_moment_sdp_v1 as old

def saved_image_block(record,origin):
 maps={(v['family'],tuple(v['shift'])):h.co_dec(v['coefficients'])for v in record['unique_shift_maps']};rows=[[None]*12 for _ in range(12)]
 for r in record['matrix_entry_records']:
  rows[r['row']][r['column']]={}if r['family']is None else maps[r['family'],tuple(r['shift'])]
 if any(v is None for row in rows for v in row):raise ValueError('incomplete saved image map')
 return dict(id='image:'+str(record['frame']),dimension=12,kind='image:camera_cone',scope=record['table_scope'],origin=origin,maps=rows)
def pack12(block):
 n=block['dimension'];maps=block['maps']
 if n!=12 or len(maps)!=n or any(len(row)!=n for row in maps):raise ValueError('NEW declared image dimension12 only')
 if any(maps[i][k]!=maps[k][i]for i in range(n)for k in range(n)):raise ValueError('asymmetric image maps retained, never averaged')
 lo,hi=old.root_two_interval();rows=[];errors=[[F(0)for _ in range(n)]for _ in range(n)]
 for i,k in old.layout(n):
  exact=maps[i][k];packed={col:float(v)*(1 if i==k else math.sqrt(2))for col,v in exact.items()};error=sum(abs(F(packed[col])-v)if i==k else max(abs(F(packed[col])/lo-v),abs(F(packed[col])/hi-v))for col,v in exact.items());errors[i][k]=errors[k][i]=error
  rows.append(dict(i=i,j=k,packed_coefficients=[dict(column=col,value=z)for col,z in sorted(packed.items())if z],unpacked_unit_box_error=c.rec(error)))
 radius=max(map(sum,errors));guard=old.up(radius)
 return dict(id=block['id'],dimension=n,rows=rows,sqrt_two_interval=[c.rec(lo),c.rec(hi)],entry_error_bounds=[[c.rec(v)for v in row]for row in errors],spectral_error_bound=c.rec(radius),outward_diagonal_guard=guard,unit_box_assumption=[0,1],necessary_packing_error_NOT_candidate_tolerance=True)
def append12(base_A,base_b,blocks,emit):
 import numpy as np
 from scipy import sparse
 rr=[];cc=[];vv=[];b=[];packed=[];offset=base_A.shape[0]
 for block in blocks:
  emit(dict(kind='BEFORE_NEW_image12_PSD_packing',id=block['id']));v=pack12(block);v['row_start']=offset+len(b)
  for row in v['rows']:
   index=len(b)
   for z in row['packed_coefficients']:rr.append(index);cc.append(z['column']);vv.append(-z['value'])
   b.append(v['outward_diagonal_guard']if row['i']==row['j']else 0.)
  v['row_end']=offset+len(b);block.update(row_start=v['row_start'],row_end=v['row_end']);packed.append(v);emit(dict(kind='AFTER_NEW_image12_PSD_packing_RETURN',record=v))
 extra=sparse.csc_matrix((vv,(rr,cc)),shape=(len(b),base_A.shape[1]));combined=sparse.vstack([base_A,extra],format='csc');combined.sort_indices();return combined,np.concatenate([base_b,np.array(b)]),packed
