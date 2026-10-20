"""NEW24Dglobal packing and170auxiliary boxes; old prefix stays exact.
All rounding guards are necessary outer-model bounds,not candidate tolerances.
"""
from fractions import Fraction as F
from pathlib import Path
import hashlib,json,math
import conditional_moment_cones_v1 as c
import direct_conditional_moment_sdp_v1 as old
import supported_conic_maps_v1 as generic

def global_block(folder):
 folder=Path(folder);model=json.loads((folder/'symbolic-functional-model.json').read_text());rows=json.loads((folder/'symbolic-global24-upper-linear-maps.json').read_text());M=[[None]*24 for _ in range(24)]
 for row in rows:
  i,j=row['matrix_entry'];co=generic.co_dec(row['linear_map']);M[i][j]=M[j][i]=co
 if len(rows)!=300 or any(v is None for row in M for v in row):raise ValueError('incomplete SAVED global24maps')
 return dict(id='global:common_monomial_functional',dimension=24,kind='global:monomial_functional',scope=model['basis_variables'],origin=str(folder),maps=M)

def pack24(block):
 n=block['dimension'];M=block['maps']
 if n!=24 or len(M)!=24 or any(len(row)!=24 for row in M):raise ValueError('NEW declared dimension24only')
 if any(M[i][j]!=M[j][i]for i in range(n)for j in range(n)):raise ValueError('asymmetric24maps_NOT_averaged')
 lo,hi=old.root_two_interval();errors=[[F(0)]*n for _ in range(n)];rows=[]
 for i,j in old.layout(n):
  co=M[i][j];packed={col:float(v)*(1 if i==j else math.sqrt(2))for col,v in co.items()};err=sum(abs(F(packed[k])-v)if i==j else max(abs(F(packed[k])/lo-v),abs(F(packed[k])/hi-v))for k,v in co.items());errors[i][j]=errors[j][i]=err;rows.append(dict(i=i,j=j,packed_coefficients=[dict(column=k,value=v)for k,v in sorted(packed.items())if v],unpacked_unit_box_error=c.rec(err)))
 radius=max(map(sum,errors));guard=old.up(radius)
 return dict(id=block['id'],dimension=24,rows=rows,sqrt_two_interval=[c.rec(lo),c.rec(hi)],entry_error_bounds=[[c.rec(v)for v in row]for row in errors],spectral_error_bound=c.rec(radius),outward_diagonal_guard=guard,all_original_AND_auxiliary_variables_unit_box_assumption=True,packing_guard_NECESSARY_NOT_original_candidate_tolerance=True)

def append_global(base_A,base_b,block,total_columns,auxiliary_columns,emit):
 import numpy as np
 from scipy import sparse
 if total_columns<base_A.shape[1]:raise ValueError('original prefix columns cannot shrink')
 emit(dict(kind='BEFORE_NEW_global_prefix_extension',old_rows=base_A.shape[0],old_columns=base_A.shape[1],total_columns=total_columns,auxiliary_columns=auxiliary_columns))
 extension=sparse.hstack([base_A,sparse.csc_matrix((base_A.shape[0],total_columns-base_A.shape[1]))],format='csc');extension.sort_indices();emit(dict(kind='AFTER_NEW_global_prefix_extension_RETURN',shape=list(extension.shape),original_prefix_coefficients_unchanged=True))
 rr=[];cc=[];vv=[];box_b=[]
 for sign,rhs in[(-1.,0.),(1.,1.)]:
  for col in auxiliary_columns:rr.append(len(box_b));cc.append(col);vv.append(sign);box_b.append(rhs)
 boxes=sparse.csc_matrix((vv,(rr,cc)),shape=(len(box_b),total_columns));emit(dict(kind='AFTER_NEW_global_auxiliary_box_constructor_RETURN',rows=len(box_b),row_start=base_A.shape[0],columns=auxiliary_columns,lower=0,upper=1,positive_exact_original_unit_mass_necessity_ONLY=True))
 emit(dict(kind='BEFORE_NEW_global24packing',id=block['id']));packing=pack24(block);packing['row_start']=base_A.shape[0]+len(box_b);rr=[];cc=[];vv=[];psd_b=[]
 for row in packing['rows']:
  for r in row['packed_coefficients']:rr.append(len(psd_b));cc.append(r['column']);vv.append(-r['value'])
  psd_b.append(packing['outward_diagonal_guard']if row['i']==row['j']else 0.)
 packing['row_end']=packing['row_start']+len(psd_b);block.update(row_start=packing['row_start'],row_end=packing['row_end']);emit(dict(kind='AFTER_NEW_global24packing_RETURN',RETURN=packing));psd=sparse.csc_matrix((vv,(rr,cc)),shape=(len(psd_b),total_columns));A=sparse.vstack([extension,boxes,psd],format='csc');A.sort_indices();b=np.concatenate([base_b,np.array(box_b),np.array(psd_b)]);emit(dict(kind='AFTER_NEW_global_canonical_constructor_RETURN',rows=A.shape[0],columns=A.shape[1],new_box_rows=len(box_b),new_PSD_rows=len(psd_b)))
 return A,b,packing

def saved_base_blocks(image_folder):
 image_folder=Path(image_folder);read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();binding=read(image_folder/'image-model-binding.json');sb=read(binding['supported_base_binding']);SB=Path(binding['supported_base_binding']).parent
 if sha(binding['supported_base_binding'])!=binding['supported_base_binding_sha256']or sha(sb['base_packing'])!=sb['base_packing_sha256']:raise ValueError('saved prefix binding mismatch')
 blocks=generic.base_blocks(read(sb['base_packing']))
 for folder,files,packedfile in[(SB,sb['additional_block_files'],SB/'additional-weighted-PSD-packing.json'),(image_folder,binding['image_block_files'],image_folder/'image-five-PSD-packing.json')]:
  for f,packing in zip(files,read(packedfile),strict=True):
   p=folder/f['file']
   if sha(p)!=f['sha256']:raise ValueError('saved prefix block hash mismatch')
   block=generic.block_decode(read(p));block.update(row_start=packing['row_start'],row_end=packing['row_end']);blocks.append(block)
 if len(blocks)!=1297:raise ValueError('original1297cones required')
 return blocks,sb
