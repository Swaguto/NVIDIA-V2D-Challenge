"""NEW independent saved-certificate audit with fixed IEEE integer accumulators."""
from joint_saved_mean_bound_v1 import S,C,OLD,sha,save,typed,frame_files,dynamic_file
from joint_mean_independent_integer_v1 import *
from joint_lossless_returns_v1 import seal_return
import numpy as np,time,json,traceback
D=S/'e037-joint-mean-proof-independent-audit-v1';EVAL=S/'e037-joint-mean-proof-evaluation-v1';MAP=S/'e037-joint-mean-proof-maps-v1';start=time.monotonic();deadline=start+210
assert json.loads((S/'e037-joint-mean-proof-integer-controls-v1/report.json').read_text())['passed']
report=json.loads((EVAL/'report.json').read_text());mp=typed(MAP/'NEW-complete-proof-maps-RETURN.json');source=typed(dynamic_file());J=source['J'];E=source['absolute_tie_envelope'];ff=frame_files();visits=np.zeros(J.nnz,dtype=np.int8);terms=0;entry_count=0;blocks=0
proof_files=sorted(EVAL.glob('NEW-block-*-ALL-exact-proof-RETURN.json'))
save(D/'NEW-independent-audit-input-bindings-BEFORE.json',{str(p):sha(p) for p in [EVAL/'report.json',MAP/'NEW-complete-proof-maps-RETURN.json',MAP/'NEW-complete-proof-maps-RETURN.npz',C/'joint_mean_independent_integer_v1.py',*proof_files,*[p.with_suffix('.npz') for p in proof_files]]})
try:
 for file in proof_files:
  if time.monotonic()>deadline:raise TimeoutError('declared independent audit210sec cap')
  block=int(file.name.split('-')[2]);f=block-1;packet=typed(file);ids=packet['data_positions'];a=packet['numerator_ptr'];b=packet['denominator_ptr'];nb=packet['numerator_bytes'].tobytes();db=packet['denominator_bytes'].tobytes()
  assert packet['shape']==(len(ids),4) and len(a)==len(b)==4*len(ids)+1 and a[-1]==len(nb) and b[-1]==len(db)
  p=typed(ff[f]) if f>=0 else None
  if p is not None:
   af=[integer1074(x) for x in p['A_slopes']];bf=[integer1074(x) for x in p['B_slopes']]
   av=[tuple(integer1074(x) for x in row) for row in p['A_direction_slopes']];bv=[tuple(integer1074(x) for x in row) for row in p['B_direction_slopes']]
  equal=[];enclosed=[]
  for j,pos in enumerate(ids):
   if j%256==0 and time.monotonic()>deadline:raise TimeoutError('declared independent audit210sec cap')
   saved=[]
   for t in range(4):
    i=4*j+t;n=int.from_bytes(nb[a[i]:a[i+1]],'little',signed=True);d=int.from_bytes(db[b[i]:b[i+1]],'little');saved.append(Q(n,d));terms+=1
   actual=Q(integer1074(J.data[pos]))*T;kind=int(mp['kind'][pos]);local=int(mp['local'][pos]);k=int(mp['pair'][pos])
   if kind in [0,1]:expected=(actual,abs(actual),Q(0),abs(actual))
   elif kind==2:
    aa=af[p['A_slope_ptr'][local]:p['A_slope_ptr'][local+1]];bb=bf[p['B_slope_ptr'][local]:p['B_slope_ptr'][local+1]];expected=independent_bound(aa,bb,T)
   else:
    dn=[integer1074(x) for x in p['DN'][k,:,local]]
    aa,ea=exact_dot_side(av[p['A_ptr'][k]:p['A_ptr'][k+1]],dn);bb,eb=exact_dot_side(bv[p['B_ptr'][k]:p['B_ptr'][k+1]],dn);expected=independent_bound(aa,bb,S2,ea,eb)
   upper=Q((int(packet['outward_mantissa_hi'][j])<<64)|int(packet['outward_mantissa_lo'][j]))*Q(2)**int(packet['outward_exponent'][j])
   equal.append(tuple(saved)==expected);enclosed.append(abs(actual-saved[0])<=saved[2] and abs(actual)<=saved[3]<=upper and abs(saved[0])<=saved[1])
  seal_return(D,f'NEW-block-{block:03d}-independent-proof-RETURN',dict(data_positions=ids,ALL_four_exact_terms_equal=np.asarray(equal,dtype=np.bool_),ALL_inequalities=np.asarray(enclosed,dtype=np.bool_),
   original_J_E_bytes_equal=bool(packet['original_J'].tobytes()==J.data[ids].tobytes() and packet['original_E'].tobytes()==E.data[ids].tobytes())))
  assert all(equal) and all(enclosed) and packet['original_J'].tobytes()==J.data[ids].tobytes() and packet['original_E'].tobytes()==E.data[ids].tobytes()
  visits[ids]+=1;entry_count+=len(ids);blocks+=1
except BaseException as e:
 seal_return(D,'NEW-audit-actual-boundary-RETURN',dict(error=repr(e),traceback=traceback.format_exc(),completed_entries=entry_count,completed_blocks=blocks))
 save(D/'report.json',dict(passed=False,status='UNRESOLVED_independent_audit',completed_entries=entry_count,completed_blocks=blocks,wall_seconds=time.monotonic()-start,no_replay_no_repair=True))
else:
 all_complete=bool(np.all(visits==1));seal_return(D,'NEW-ALL-entry-coverage-RETURN',dict(visits=visits,ALL_once=all_complete,original_certificate_status=report['status']))
 expected_count=report.get('ALL933560_entries',report.get('completed_entries'));assert entry_count==expected_count
 save(D/'report.json',dict(passed=True,certificate_full=all_complete,ALL_verified_entries=entry_count,ALL_verified_exact_terms=terms,blocks=blocks,wall_seconds=time.monotonic()-start,
  independent_IEEE_2pow1074_and_2pow2148_integer_accumulators=True,no_producer_mean_dot_proof_replay=True,no_missing_intermediate_reconstruction=True,
  original_J_E_bytes_unchanged=True,no_source_geometry_method_native_or_candidate_query=True,no_trajectory_qualification=True))
