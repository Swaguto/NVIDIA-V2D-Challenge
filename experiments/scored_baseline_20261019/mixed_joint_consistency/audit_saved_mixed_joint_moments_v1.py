"""One new 19-scope mixed joint/time diagnosis of the SAVED direct SDP table.

Fixed scope: 13pairs,6triples,63total-degree2 moment/interval localizer matrices.
No solver call, source regeneration, degree sweep, or actual weight repair.
"""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-saved-mixed-joint-moments-v1';sys.path.insert(0,str(S/'compute'))
import mixed_joint_moments_v1 as m
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());j=(O/'all-mixed-joint-maps-states-congruences-and-witnesses.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic()
contract=read(S/'e037-mixed-joint-contract-v1/report.json');assert contract['all_controls_passed']and contract['helper_sha256']==sha(S/'compute/mixed_joint_moments_v1.py')
modelpath=S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json';returnpath=S/'e037-direct-conditional-SDP-v1/single-direct-SDP-solver-RETURN.json';parent=read(S/'e037-direct-conditional-SDP-v1/report.json');assert sha(modelpath)==parent['base_model_sha256']and sha(returnpath)==parent['all_outputs_sha256'][returnpath.name]
model=read(modelpath);ret=read(returnpath);weights=list(map(F,ret['x']));alltables=m.tables(model['columns']);selected=[(scope,entries)for scope,entries in alltables.items()if len(scope)>1];assert Counter(map(lambda v:len(v[0]),selected))=={2:13,3:6}
domains=S/'e037-common-moment-model-v1/common-degree-factor-representations.json';domain=read(domains)['domains'];binding=dict(model_sha256=sha(modelpath),actual_solver_RETURN_sha256=sha(returnpath),domains_file_sha256=sha(domains),normalized_coordinates='t in[0,1] maps to the ORIGINAL saved restricted half-angle domains',domains=domain,full_region2_and_unknown109_bands_unchanged=True,conditional_fixed_unqualified_wrist_scope=True)
(O/'existing-model-table-domain-bindings.json').write_text(json.dumps(g.plain(binding),allow_nan=False,indent=2)+'\n')
emit(dict(kind='BEFORE_NEW_bounded_mixed_joint_table_diagnosis',scopes=19,pair_scopes=13,triple_scopes=6,matrices=63,total_degree_moment_basis=2,total_degree_localizer_basis=1,existing_tensor_degree=4,actual_weights_NOT_changed=True,old_univariate_audit_reused_by_hash_NOT_reexecuted=sha(S/'e037-direct-SDP-independent-v1/report.json')))
summaries=[];records=[];moments_count=0;map_coefficients=0
for index,(scope,entries)in enumerate(selected):
    dim=len(scope);emit(dict(kind='BEFORE_NEW_mixed_scope_proposal',scope_id=index,scope=scope,entries=[dict(index=list(a),column=col,actual_weight=c.rec(weights[col]))for a,col in entries.items()]))
    y={};maps=[]
    for power in m.basis(dim,4):
        emit(dict(kind='BEFORE_NEW_mixed_monomial_map',scope_id=index,power=power));co=m.monomial_coefficients(entries,power);value=sum(a*weights[col]for col,a in co.items());rec=dict(power=list(power),coefficients=[dict(column=col,value=c.rec(a))for col,a in co.items()],actual_moment=c.rec(value));emit(dict(kind='AFTER_NEW_mixed_monomial_map_RETURN',scope_id=index,record=rec));y[power]=value;maps.append(rec);moments_count+=1;map_coefficients+=len(co)
    scope_record=dict(scope_id=index,scope=list(scope),columns=[dict(index=list(a),column=col)for a,col in entries.items()],actual_total_mass=c.rec(y[(0,)*dim]),moment_maps=maps,matrices=[])
    for axis in [None,*range(dim)]:
        basis=m.basis(dim,2 if axis is None else 1);name='moment'if axis is None else'interval_localizer:'+str(axis);emit(dict(kind='BEFORE_NEW_mixed_matrix_proposal',scope_id=index,name=name,axis=axis,basis=basis))
        M=m.matrix_from_moments(basis,y,axis);emit(dict(kind='AFTER_NEW_mixed_matrix_map_RETURN',scope_id=index,name=name,matrix=[[c.rec(v)for v in row]for row in M]))
        cert=m.congruence_certificate(M,lambda v:emit(dict(scope_id=index,matrix_name=name,**v)));record=dict(name=name,axis=axis,basis=[list(a)for a in basis],matrix=[[c.rec(v)for v in row]for row in M],certificate=cert)
        if cert['classification']=='exact_negative_mixed_square':
            emit(dict(kind='BEFORE_NEW_mixed_polynomial_square_witness',scope_id=index,name=name));witness=m.square_witness(entries,basis,axis,list(map(c.dec,cert['vector'])),weights);emit(dict(kind='AFTER_NEW_mixed_polynomial_square_witness_RETURN',scope_id=index,name=name,witness=witness));record['negative_square_witness']=witness;assert c.dec(witness['actual_exact_expectation'])<0
        scope_record['matrices'].append(record);summaries.append(dict(scope_id=index,scope=list(scope),name=name,dimension=len(M),classification=cert['classification'],rank=cert.get('rank'),negative_expectation=record.get('negative_square_witness',{}).get('actual_exact_expectation'),necessary_l1_change=record.get('negative_square_witness',{}).get('necessary_l1_weight_change_to_remove_THIS_witness')))
    records.append(scope_record)
    with (O/f'scope-{index:02d}-exact-joint-square-record.json').open('w')as f:f.write(json.dumps(g.plain(scope_record),allow_nan=False,indent=2)+'\n');f.flush();__import__('os').fsync(f.fileno())
assert len(summaries)==63;j.close();bad=[v for v in summaries if v['classification']=='exact_negative_mixed_square'];output=dict(schema='v2d.saved-mixed-joint-moment-diagnosis.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/mixed_joint_moments_v1.py'),binding=binding,scopes=19,pair_scopes=13,triple_scopes=6,new_mixed_monomial_maps=moments_count,new_map_coefficients=map_coefficients,new_joint_matrices=63,negative_joint_matrices=len(bad),PSD_joint_matrices=63-len(bad),negative_by_type=dict(Counter('moment'if v['name']=='moment'else'localizer'for v in bad)),matrix_summaries=summaries,classification='saved_table_rejected_by_NEW_mixed_square_necessities'if bad else'new_mixed_PSD_necessities_clear_NOT_global_common_measure_or_path',independent_witness_and_map_verification_PENDING=True,old_univariate_PSD_return_and_nonzero_equality_residual_preserved=True,no_actual_weight_clipping_normalization_or_repair=True,no_family_exclusion_or_deterministic_candidate_claim=True,no_old_producer_audit_or_solver_reexecuted=True,no_source_polynomial_degree_or_authority_change=True,current_verified_reference_and_middle_component_unchanged=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(output),allow_nan=False,indent=2)+'\n');print(json.dumps({k:v for k,v in output.items()if k not in['binding','matrix_summaries','all_outputs_sha256']}))
