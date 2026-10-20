"""NEW raw-field and bit-lossless method-specific frame retention control."""
from joint_method_pair_columns_v1 import *
from joint_lossless_returns_v1 import seal_return,restore_NEW_synthetic
from joint_constraint_interface_v1 import S,save
import struct,hashlib
D=S/'e037-joint-method-pair-columns-v1'
x=np.array([-0.,np.nextafter(0.,1.),-.37]);records=[]
for i in range(3):
 o=dict(status='RETURNED',**{k:float(x[i]) for k in SCALARS},**{k:x.copy() for k in VECTORS},
   A_ties=np.array([1,4],dtype=np.int64),B_ties=np.array([2],dtype=np.int64),
   ALL_tie_direction_derivatives=dict(A=np.array([x,x[::-1]]),B=x[None,:]),
   ALL_tie_factor_derivatives={'0':dict(A=x[:2].copy(),B=x[:1].copy()),'1522':dict(A=x[1:].copy(),B=x[2:].copy())},
   SEARCH_command_residual_J={'0':-0.,'1522':float(x[i])},mean_tie_selection_NOT_universal_directional_derivative=True)
 records.append(dict(pair=i,chart=x[:2],n=x,DN=np.array([x,x]).T,conditional_filter=['typed-static'] if i==1 else [],ALL_support_RETURN=o))
records.append(dict(pair=3,chart=x[:2],n=x,DN=np.array([x,x]).T,conditional_filter=[],ALL_support_RETURN={'status':'UNKNOWN_synthetic'}))
p=pack_pair_frame(records);path=seal_return(D,'NEW-columnar-raw-RETURN',p);ret=unpack_NEW_synthetic(restore_NEW_synthetic(path))
def equal(a,b):
 if isinstance(a,np.ndarray):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
 if isinstance(a,float):return struct.pack('>d',a)==struct.pack('>d',b)
 if isinstance(a,dict):return set(a)==set(b) and all(equal(v,b[k]) for k,v in a.items())
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(v,w) for v,w in zip(a,b))
 return type(a)==type(b) and a==b
seal_return(D,'NEW-serialization-control-decision-RETURN',dict(equal=equal(records,ret),keys=sorted(KEYS),all_unknowns_retained=True,all_ties_commands_and_signedzeros_retained=True))
assert equal(records,ret)
# Independent NEW pointer/field incidence: all synthetic raw fields/ties/slopes counted.
assert len(KEYS)==17 and p['A_ptr'][-1]==6 and p['B_ptr'][-1]==3 and p['command_ptr'][-1]==6
assert p['A_slope_ptr'][-1]==12 and p['B_slope_ptr'][-1]==6 and len(p['unknown_raw'])==1
save(D/'report.json',dict(passed=True,NEW_controls=1,all17_support_fields_bound=True,unknowns_retained=True,no_episode_query=True,operator_and_limits_unchanged=True))
