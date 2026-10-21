"""ONE concrete NEW zero-slice general-RHS solve-return contract."""
from joint_zero_conic_v1 import *
from joint_zero_return_guard_v1 import linear_return_guard,Q
from joint_saved_mean_bound_v1 import save
D=S/'e037-joint-zero-return-contract-v1';record=MethodRecorder(D)
normal=np.array([[2.,.5],[.5,2.]]);answer=np.array([.5,-.5]);rhs=np.array([.75,-.75])
record('NEW-dyadic-general-RHS-input-BEFORE',dict(normal=normal,solution=answer,rhs=rhs))
g=linear_return_guard(normal,answer,rhs,record,'NEW-dyadic-general-RHS')
exact=[sum(Q.from_float(float(normal[i,j]))*Q.from_float(float(answer[j])) for j in range(2))-Q.from_float(float(rhs[i])) for i in range(2)]
record('NEW-exact-dyadic-general-RHS-witness-RETURN',dict(exact_residual_numerators=[a.numerator for a in exact],exact_residual_denominators=[a.denominator for a in exact],all_guard_entries=g['all_within']))
assert exact==[Q(0),Q(0)] and g['all_within'] and Q.from_float(g['gamma_outward_float'])>=Q(g['gamma_rational_numerator'],g['gamma_rational_denominator'])
save(D/'report.json',dict(passed=True,NEW_controls=1,concrete_zero_slice_general_RHS_return_guard=True,no_old_control_solver_or_episode_query_replay=True))
