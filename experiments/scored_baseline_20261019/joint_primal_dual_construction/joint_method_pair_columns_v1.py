"""NEW method-specific lossless pair-frame columns; no operator or row changes."""
import numpy as np
from joint_lossless_returns_v1 import LosslessRecorder
SCALARS=('support_A','support_B','residual','unit_equality','phase_residual_J')
VECTORS=('unit_J','n_world','n_local','SEARCH_direction_residual_J')
KEYS={'status',*SCALARS,*VECTORS,'A_ties','B_ties','ALL_tie_factor_derivatives','SEARCH_command_residual_J','mean_tie_selection_NOT_universal_directional_derivative','ALL_tie_direction_derivatives'}

def pack_pair_frame(records):
    n=len(records);valid=[];unknown=[];a_ptr=[0];b_ptr=[0];cmd_ptr=[0];as_ptr=[0];bs_ptr=[0]
    at=[];bt=[];ad=[];bd=[];cols=[];means=[];asl=[];bsl=[]
    scalars=[];vectors=[];flags=[]
    for i,rec in enumerate(records):
        o=rec['ALL_support_RETURN']
        if o['status']!='RETURNED':unknown.append((i,o));continue
        if set(o)!=KEYS:raise ValueError('new raw support schema must retain every field')
        valid.append(i);scalars.append([o[k] for k in SCALARS]);vectors.append([o[k] for k in VECTORS]);flags.append(o['mean_tie_selection_NOT_universal_directional_derivative'])
        at.extend(o['A_ties']);bt.extend(o['B_ties']);a_ptr.append(len(at));b_ptr.append(len(bt))
        ad.extend(o['ALL_tie_direction_derivatives']['A']);bd.extend(o['ALL_tie_direction_derivatives']['B'])
        for c,sl in o['ALL_tie_factor_derivatives'].items():
            cols.append(int(c));means.append(o['SEARCH_command_residual_J'][c]);asl.extend(sl['A']);bsl.extend(sl['B']);as_ptr.append(len(asl));bs_ptr.append(len(bsl))
        cmd_ptr.append(len(cols))
    return dict(schema='v2d.NEW-lossless-pair-frame-columns.v1',pairs=np.asarray([r['pair'] for r in records],dtype=np.int64),
        chart=np.asarray([r['chart'] for r in records]),n=np.asarray([r['n'] for r in records]),DN=np.asarray([r['DN'] for r in records]),
        conditional_filter=[r['conditional_filter'] for r in records],valid=np.asarray(valid,dtype=np.int64),unknown_raw=unknown,
        scalar_fields=SCALARS,vector_fields=VECTORS,scalars=np.asarray(scalars,dtype=np.float64).reshape(-1,len(SCALARS)),
        vectors=np.asarray(vectors,dtype=np.float64).reshape(-1,len(VECTORS),3),mean_is_SEARCH_only=np.asarray(flags,dtype=np.bool_),
        A_ptr=np.asarray(a_ptr,dtype=np.int64),B_ptr=np.asarray(b_ptr,dtype=np.int64),A_ties=np.asarray(at,dtype=np.int64),B_ties=np.asarray(bt,dtype=np.int64),
        A_direction_slopes=np.asarray(ad,dtype=np.float64).reshape(-1,3),B_direction_slopes=np.asarray(bd,dtype=np.float64).reshape(-1,3),
        command_ptr=np.asarray(cmd_ptr,dtype=np.int64),columns=np.asarray(cols,dtype=np.int64),mean_command=np.asarray(means,dtype=np.float64),
        A_slope_ptr=np.asarray(as_ptr,dtype=np.int64),B_slope_ptr=np.asarray(bs_ptr,dtype=np.int64),A_slopes=np.asarray(asl,dtype=np.float64),B_slopes=np.asarray(bsl,dtype=np.float64))

def unpack_NEW_synthetic(p):
    """Only NEW serializer controls, never missing state reconstruction."""
    records=[dict(pair=int(k),chart=p['chart'][i],n=p['n'][i],DN=p['DN'][i],conditional_filter=p['conditional_filter'][i]) for i,k in enumerate(p['pairs'])]
    for j,i in enumerate(p['valid']):
        o=dict(status='RETURNED');o.update({k:float(p['scalars'][j,t]) for t,k in enumerate(SCALARS)});o.update({k:p['vectors'][j,t] for t,k in enumerate(VECTORS)})
        aa=slice(p['A_ptr'][j],p['A_ptr'][j+1]);bb=slice(p['B_ptr'][j],p['B_ptr'][j+1]);o.update(A_ties=p['A_ties'][aa],B_ties=p['B_ties'][bb],
          ALL_tie_direction_derivatives=dict(A=p['A_direction_slopes'][aa],B=p['B_direction_slopes'][bb]),
          mean_tie_selection_NOT_universal_directional_derivative=bool(p['mean_is_SEARCH_only'][j]))
        slopes={};means={}
        for k in range(p['command_ptr'][j],p['command_ptr'][j+1]):
            c=str(p['columns'][k]);slopes[c]=dict(A=p['A_slopes'][p['A_slope_ptr'][k]:p['A_slope_ptr'][k+1]],B=p['B_slopes'][p['B_slope_ptr'][k]:p['B_slope_ptr'][k+1]]);means[c]=float(p['mean_command'][k])
        o['ALL_tie_factor_derivatives']=slopes;o['SEARCH_command_residual_J']=means;records[i]['ALL_support_RETURN']=o
    for i,o in p['unknown_raw']:records[i]['ALL_support_RETURN']=o
    return records

class MethodRecorder(LosslessRecorder):
    def __call__(self,name,value):
        if name.endswith('-ALL-pair-RETURN'):value=pack_pair_frame(value)
        return super().__call__(name,value)
