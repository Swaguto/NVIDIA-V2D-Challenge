"""SOLE fixed all-row saved-return rounding certificate. No candidate or query."""
from joint_saved_mean_bound_v1 import *
from joint_mean_proof_packets_v1 import rational_packet
from joint_lossless_returns_v1 import seal_return
import traceback
D=S/'e037-joint-mean-proof-evaluation-v1';MAP=S/'e037-joint-mean-proof-maps-v1';start=time.monotonic();deadline=start+180;completed=0;frames=0;failures=0;max_ratio=Q(0);max_gap=Q(0)
save(D/'NEW-fixed-certificate-declaration-BEFORE.json',dict(input_sha256={str(p):sha(p) for p in [MAP/'NEW-fixed-proof-declaration-RETURN.json',MAP/'NEW-complete-proof-maps-RETURN.json',MAP/'NEW-complete-proof-maps-RETURN.npz',S/'e037-joint-mean-proof-binding-v1/report.json',S/'e037-joint-mean-proof-packet-control-v1/report.json',C/'joint_saved_mean_bound_v1.py',C/'joint_mean_proof_packets_v1.py']},
 certificate_wall_seconds=180,outer_guard_seconds=240,RSS_bytes=2*1024**3,ALL933560_entries_REQUIRED=True,no_retry_or_gate_tolerance_change=True,no_episode_method_source_geometry_native_query=True))
assert json.loads((S/'e037-joint-mean-proof-binding-v1/report.json').read_text())['passed'] and json.loads((S/'e037-joint-mean-proof-packet-control-v1/report.json').read_text())['passed']
mp=typed(MAP/'NEW-complete-proof-maps-RETURN.json');old=typed(dynamic_file());J=old['J'];E=old['absolute_tie_envelope'];ff=frame_files()
try:
 for f in range(-1,151):
  if time.monotonic()>deadline:raise TimeoutError('declared certificate180sec cap')
  ids=np.flatnonzero(mp['kind']==0) if f<0 else np.flatnonzero(mp['frame']==f)
  p=typed(ff[f]) if f>=0 else None;proof=[];hi=[];lo=[];exp=[]
  for ix,pos in enumerate(ids):
   if ix%256==0 and time.monotonic()>deadline:raise TimeoutError('declared certificate180sec cap')
   kind=int(mp['kind'][pos]);local=int(mp['local'][pos]);k=int(mp['pair'][pos]);actual=q(J.data[pos])
   if kind in [0,1]:v=(actual,abs(actual),Q(0),abs(actual))
   elif kind==2:
    aa=p['A_slopes'][p['A_slope_ptr'][local]:p['A_slope_ptr'][local+1]];bb=p['B_slopes'][p['B_slope_ptr'][local]:p['B_slope_ptr'][local+1]]
    v=combine_proof([q(x) for x in aa],[q(x) for x in bb])
   else:
    aa=p['A_direction_slopes'][p['A_ptr'][k]:p['A_ptr'][k+1]];bb=p['B_direction_slopes'][p['B_ptr'][k]:p['B_ptr'][k+1]];dn=p['DN'][k,:,local]
    av=[];bv=[];ae=[];be=[]
    for a in aa:i,e=dot_proof(a,dn);av.append(i);ae.append(e)
    for b in bb:i,e=dot_proof(b,dn);bv.append(i);be.append(e)
    v=combine_proof(av,bv,ae,be)
   proof.append(v);m,e=dyadic96_upper(v[3])
   if m.bit_length()>96:
    if m!=2**96:raise ValueError('UNKNOWN_outward_significand')
    m//=2;e+=1 # exact same dyadic value, lossless serialization normalization
   hi.append(m>>64);lo.append(m&((1<<64)-1));exp.append(e)
  packet=rational_packet(proof);packet.update(data_positions=ids,original_J=J.data[ids],original_E=E.data[ids],
   outward_mantissa_hi=np.asarray(hi,dtype=np.uint64),outward_mantissa_lo=np.asarray(lo,dtype=np.uint64),outward_exponent=np.asarray(exp,dtype=np.int16))
  seal_return(D,f'NEW-block-{f+1:03d}-ALL-exact-proof-RETURN',packet)
  failed=[]
  for j,(pos,v) in enumerate(zip(ids,proof)):
   actual=q(J.data[pos]);gap=abs(actual-v[0]);upper=dyadic_value((hi[j]<<64)|lo[j],exp[j]);ok=(abs(v[0])<=v[1] and gap<=v[2] and abs(actual)<=v[3]<=upper)
   if not ok:failed.append(int(pos))
   max_gap=max(max_gap,gap)
   if v[2]:max_ratio=max(max_ratio,gap/v[2])
  seal_return(D,f'NEW-block-{f+1:03d}-proof-decision-RETURN',dict(ALL_entries=len(ids),failures=failed,all_exact_inequalities_pass=not failed))
  failures+=len(failed);completed+=len(ids);frames+=int(f>=0)
  if failed:raise ValueError('UNRESOLVED_conditional_RN_bound_failure')
except BaseException as e:
 seal_return(D,'NEW-actual-certificate-boundary-RETURN',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),completed_entries=completed,completed_frames=frames,unsaved_partial_proof_state_UNAVAILABLE=True))
 save(D/'report.json',dict(status='UNRESOLVED_fixed_certificate_limit' if isinstance(e,TimeoutError) else 'UNRESOLVED_actual_certificate_exception',passed=False,completed_entries=completed,completed_frames=frames,failures=failures,wall_seconds=time.monotonic()-start,no_retry_no_old_J_E_repair=True,no_qualification=True))
else:
 save(D/'report.json',dict(status='PASS_conditional_saved_binary_RN_enclosure_ONLY',passed=True,ALL933560_entries=completed,ALL151_frames=frames,failures=failures,
 exact_max_observed_mean_product_gap=str(max_gap),exact_max_fraction_of_RN_budget=str(max_ratio),max_observed_gap=float(max_gap),
 wall_seconds=time.monotonic()-start,ALL_raw_J_E_unchanged=True,old30_oneULP_defects_RETAINED=True,no_unsaved_numeric_intermediate_reconstruction=True,
 conditional_RN_operation_count_no_overflow_scope_ONLY=True,no_global_binary_source_native_identity=True,no_candidate_gate_tolerance_or_method_repair=True,
 no_old_method_restart_or_new_candidate_selection=True,no_episode_source_geometry_native_optimizer_physics_GPU_PPO_query=True,no_trajectory_qualification=True))
