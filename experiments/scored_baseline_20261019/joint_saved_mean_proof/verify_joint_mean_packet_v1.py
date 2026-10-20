from joint_saved_mean_bound_v1 import *
from joint_mean_proof_packets_v1 import *
from joint_lossless_returns_v1 import seal_return
D=S/'e037-joint-mean-proof-packet-control-v1';rows=[[Q(-2**300+1,3),Q(1,2**2148),Q(0),Q(2**1000+1,7)],[Q(-128),Q(127),Q(128),Q(-129)]]
p=rational_packet(rows);path=seal_return(D,'NEW-exact-rational-packet-RETURN',p);read=typed(path);ret=restore_NEW_packet_control(read)
seal_return(D,'NEW-packet-control-decision-RETURN',dict(equal=ret==rows,signed_integer_boundary_exact=True,positive_denominator_and_ALL_term_pointers=True))
assert ret==rows and len(p['numerator_ptr'])==9 and np.all(np.diff(p['numerator_ptr'])>0) and np.all(np.diff(p['denominator_ptr'])>0)
save(D/'report.json',dict(passed=True,NEW_controls=1,no_old_control_or_numeric_operator_replay=True))
