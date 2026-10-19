"""NEW existential first/second Gram completion, never an original table repair."""
from fractions import Fraction as F
from itertools import combinations
import math
import conditional_moment_cones_v1 as c
import direct_conditional_moment_sdp_v1 as old

def co(co):return [dict(column=k,value=c.rec(v))for k,v in sorted(co.items())if v]
def row(name,values,bound):return dict(name=name,coefficients=co(values),bound=c.rec(bound))
def formulate(variables,known):
 n=len(variables)+1
 if n!=11:raise ValueError('NEW declared global dimension11 only')
 entries=old.layout(n);index={p:i for i,p in enumerate(entries)};columns=[]
 for i,k in entries:
  role='original_unit_mass_target'if i==k==0 else'known_interval'if(i,k)in known else'free_existential_mixed_completion'
  if role=='free_existential_mixed_completion'and(i==0 or i==k):raise ValueError('first and diagonal entries require saved intervals')
  columns.append(dict(matrix_entry=[i,k],role=role,original1125table_column=False,coordinate_names=[variables[a-1]if a else'constant1'for a in [i,k]]))
 eq=[row('original_unit_mass_target',{index[0,0]:F(1)},F(1))];ineq=[]
 for pair,value in sorted(known.items()):
  lo,hi=value['interval'];anchor=value['anchor']
  if lo>hi or not lo<=anchor<=hi:raise ValueError('invalid frozen known interval/anchor')
  col=index[pair];ineq.extend([row('known-upper:'+str(pair),{col:F(1)},hi),row('known-lower:'+str(pair),{col:F(-1)},-lo)])
 for i in range(1,n):
  ineq.extend([row('first<=mass:'+str(i),{index[0,i]:F(1),index[0,0]:F(-1)},F(0)),row('second<=first:'+str(i),{index[i,i]:F(1),index[0,i]:F(-1)},F(0))])
 for i,k in combinations(range(1,n),2):
  col=index[i,k];ineq.extend([row('cross<=first-i:'+str((i,k)),{col:F(1),index[0,i]:F(-1)},F(0)),row('cross<=first-j:'+str((i,k)),{col:F(1),index[0,k]:F(-1)},F(0)),row('Frechet-lower:'+str((i,k)),{index[0,i]:F(1),index[0,k]:F(1),index[0,0]:F(-1),col:F(-1)},F(0))])
 maps=[[{index[min(i,k),max(i,k)]:F(1)}for k in range(n)]for i in range(n)]
 return dict(schema='v2d.existential-global-Gram-model.v1',variables=variables,columns=columns,equalities=eq,inequalities=ineq,variable_bounds=[0,1],known_intervals=[dict(entry=list(pair),interval=[c.rec(z)for z in val['interval']],anchor=c.rec(val['anchor']),origin=val.get('origin'))for pair,val in sorted(known.items())],known_upper_entries=1+len(known),free_unknown_entries=len(columns)-1-len(known),unit_mass_is_ORIGINAL_model_target_raw_masses_NOT_normalized=True,normalized_tensor_coordinates_NOT_robot_angles=True,original1125table_and_source_time_model_UNCHANGED=True,diagnostic_interval_envelope_NOT_all_corrections_or_family_bounds=True),dict(id='NEW_existential_global_Gram',dimension=n,kind='diagnostic:first_second_Gram',maps=maps)
def pack11(block):
 maps=block['maps'];n=block['dimension']
 if n!=11 or len(maps)!=n or any(len(v)!=n for v in maps):raise ValueError('dimension11 shape')
 if any(maps[i][k]!=maps[k][i]for i in range(n)for k in range(n)):raise ValueError('asymmetric map not averaged')
 lo,hi=old.root_two_interval();errors=[[F(0)for _ in range(n)]for _ in range(n)];rows=[]
 for i,k in old.layout(n):
  packed={col:float(v)*(1 if i==k else math.sqrt(2))for col,v in maps[i][k].items()};e=sum(abs(F(packed[col])-v)if i==k else max(abs(F(packed[col])/lo-v),abs(F(packed[col])/hi-v))for col,v in maps[i][k].items());errors[i][k]=errors[k][i]=e;rows.append(dict(i=i,j=k,packed_coefficients=[dict(column=col,value=z)for col,z in sorted(packed.items())if z],unit_box_unpacked_error=c.rec(e)))
 rad=max(map(sum,errors));return dict(dimension=11,rows=rows,entry_errors=[[c.rec(z)for z in v]for v in errors],sqrt_two_interval=[c.rec(lo),c.rec(hi)],spectral_error=c.rec(rad),necessary_diagonal_guard=old.up(rad),guard_NOT_original_candidate_tolerance=True)
def assemble(model,block,emit):
 import numpy as np
 from scipy import sparse
 rr=[];cc=[];vv=[];b=[];linear=[];n=len(model['columns'])
 for eq in model['equalities']:
  for v in eq['coefficients']:
   z=c.dec(v['value']);value=float(z)
   if F(value)!=z:raise ValueError('nonbinary equality not silently packed')
   rr.append(len(b));cc.append(v['column']);vv.append(value)
  z=c.dec(eq['bound']);value=float(z)
  if F(value)!=z:raise ValueError('nonbinary equality RHS')
  b.append(value)
 eqcount=len(b)
 for r in model['inequalities']:
  values,rhs,error=old.outer_linear(r);emit(dict(kind='AFTER_NEW_completion_linear_packing_RETURN',record=error))
  for col,z in values:rr.append(len(b));cc.append(col);vv.append(z)
  b.append(rhs);linear.append(error)
 for sign,rhs in [(-1.,0.),(1.,1.)]:
  for col in range(n):rr.append(len(b));cc.append(col);vv.append(sign);b.append(rhs)
 packing=pack11(block);packing['row_start']=len(b)
 for r in packing['rows']:
  for v in r['packed_coefficients']:rr.append(len(b));cc.append(v['column']);vv.append(-v['value'])
  b.append(packing['necessary_diagonal_guard']if r['i']==r['j']else 0.)
 packing['row_end']=len(b);block.update(row_start=packing['row_start'],row_end=packing['row_end']);emit(dict(kind='AFTER_NEW_completion_PSD_packing_RETURN',record=packing));A=sparse.csc_matrix((vv,(rr,cc)),shape=(len(b),n));A.sort_indices();return A,np.array(b),dict(equalities=eqcount,inequalities=len(model['inequalities']),nonnegative_rows=packing['row_start']-eqcount,linear_rounding=linear,PSD_packing=packing,columns=n,rows=len(b))
def matrix(block,values):return [[sum(v*F(values[col])for col,v in co.items())for co in row]for row in block['maps']]
def auxiliary_exact_proposal(model,raw_values):
 values=list(map(F,raw_values));index={tuple(v['matrix_entry']):i for i,v in enumerate(model['columns'])};values[index[0,0]]=F(1)
 for v in model['known_intervals']:values[index[tuple(v['entry'])]]=c.dec(v['anchor'])
 return dict(schema='v2d.separate-existential-completion-proposal.v1',values=list(map(c.rec,values)),construction='declared saved rational anchors and original mass1; free entries solely from one raw solver RETURN',actual_original1125weights_and_auxiliary_RETURN_NOT_modified=True,not_an_observed_moment_or_global_supported_measure_or_robot_trajectory=True)
