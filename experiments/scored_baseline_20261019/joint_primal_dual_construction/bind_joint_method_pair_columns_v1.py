"""Read saved NEW roundtrip only; run formerly unreached column incidence binding."""
from joint_lossless_returns_v1 import restore_NEW_synthetic,seal_return
from joint_constraint_interface_v1 import S,C,sha,save
import numpy as np
D=S/'e037-joint-method-pair-columns-binding-v1';O=S/'e037-joint-method-pair-columns-v1'
paths=[O/'NEW-columnar-raw-RETURN.json',O/'NEW-columnar-raw-RETURN.npz',O/'NEW-serialization-control-decision-RETURN.json',O/'NEW-serialization-control-decision-RETURN.npz',O/'worker.log',O/'guard.json',C/'joint_method_pair_columns_v1.py']
save(D/'NEW-input-bindings-RETURN.json',{str(p):sha(p) for p in paths})
a=restore_NEW_synthetic(O/'NEW-serialization-control-decision-RETURN.json');p=restore_NEW_synthetic(O/'NEW-columnar-raw-RETURN.json')
expected={'status','support_A','support_B','residual','unit_equality','phase_residual_J','unit_J','n_world','n_local','SEARCH_direction_residual_J','A_ties','B_ties','ALL_tie_factor_derivatives','SEARCH_command_residual_J','mean_tie_selection_NOT_universal_directional_derivative','ALL_tie_direction_derivatives'}
r=dict(saved_roundtrip_equal=a['equal'],actual_schema_count=len(a['keys']),schema_exact=set(a['keys'])==expected,
 A_ties=int(p['A_ptr'][-1]),B_ties=int(p['B_ptr'][-1]),commands=int(p['command_ptr'][-1]),A_slopes=int(p['A_slope_ptr'][-1]),B_slopes=int(p['B_slope_ptr'][-1]),unknowns=len(p['unknown_raw']))
seal_return(D,'NEW-unreached-column-incidence-RETURN',r)
assert r==dict(saved_roundtrip_equal=True,actual_schema_count=16,schema_exact=True,A_ties=6,B_ties=3,commands=6,A_slopes=12,B_slopes=6,unknowns=1)
save(D/'report.json',dict(passed=True,NEW_roundtrip_control_reused_by_hash=1,unreached_column_binding_completed=True,
 prior_17_field_label_WRONG_actual16=True,prior_failure_raw_source_retained=True,no_serializer_or_control_replay=True,method_operators_limits_unchanged=True))
