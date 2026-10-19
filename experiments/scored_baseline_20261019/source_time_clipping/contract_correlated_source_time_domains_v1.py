"""NEW five-clock continuous contraction, no branches/point seeds/path optimizer."""
from pathlib import Path
from fractions import Fraction as F
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-correlated-source-time-clipping-v1';sys.path.insert(0,str(S/'compute'))
import correlated_bezier_clipping_v1 as b
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();P=S/'e037-source-time-clipping-binding-v1/correlated-source-time-clipping-inputs.json';report=read(P.parent/'report.json');assert report['all_controls_passed']and sha(P)==report['all_outputs_sha256'][P.name]and report['helper_sha256']==sha(S/'compute/correlated_bezier_clipping_v1.py');data=read(P)
initial={k:[b.dec(v)for v in r]for k,r in data['initial_domains'].items()};domains={k:r.copy()for k,r in initial.items()};states={f['id']:dict(grid=b.decode(f['lower_grid']),bounds=[[b.dec(v)for v in r]for r in f['initial_bounds']])for f in data['factors']};j=(O/'all-correlated-domain-proposals-coefficients-clips-and-outcomes.jsonl').open('w');steps=[];empty=False;empty_factor=None;last_pass=None
def encbox(x):return {k:[b.rec(v)for v in r]for k,r in x.items()}
g.journal(j,dict(kind='BEFORE_NEW_correlated_continuous_contraction',input_sha256=sha(P),initial_domains=encbox(domains),maximum_passes=24,source_interiors_and_unknown_bands_retained=True,no_isolated_state_fixed=True))
for sweep in range(data['maximum_passes']):
    changed=False;last_pass=sweep
    for factor in data['factors']:
        names=factor['variables'];degree=tuple(factor['degree']);state=states[factor['id']];current=[domains[k].copy()for k in names];previous=state['bounds'];before=encbox(domains)
        g.journal(j,dict(kind='BEFORE_NEW_correlated_factor_contraction',pass_index=sweep,factor=factor['id'],domains=before))
        if any(a==z for a,z in current):
            g.journal(j,dict(kind='AFTER_NEW_correlated_factor_contraction',pass_index=sweep,factor=factor['id'],classification='UNRESOLVED_degenerate_chart',domains=before));continue
        ranges=[((a-c)/(d-c),(z-c)/(d-c))for(a,z),(c,d)in zip(current,previous,strict=True)];grid=b.lower_round(b.restrict(state['grid'],degree,ranges));state.update(grid=grid,bounds=current)
        clips=[b.clip_axis(grid,degree,axis)for axis in range(len(names))]
        outcome=dict(kind='AFTER_NEW_correlated_factor_contraction',pass_index=sweep,factor=factor['id'],lower_grid=b.encode(grid),grid_bounds=[[b.rec(v)for v in r]for r in current],axis_clips=clips,classification='necessary_outer_restriction')
        if any(r['empty']for r in clips):
            empty=True;empty_factor=factor['id'];outcome.update(classification='EMPTY_necessary_sublevel_in_current_correlated_domains',domains=before);g.journal(j,outcome);steps.append(outcome);break
        for axis,name in enumerate(names):
            lo,hi=current[axis];clip=clips[axis];raw=[lo+(hi-lo)*b.dec(clip[k])for k in['lower','upper']];proposal=[max(domains[name][0],b.outer_quantize(raw[0],initial[name],'lower')),min(domains[name][1],b.outer_quantize(raw[1],initial[name],'upper'))]
            if proposal[0]>proposal[1]:raise ArithmeticError('outward quantization inverted a necessary domain')
            changed|=proposal!=domains[name];domains[name]=proposal
        outcome['domains']=encbox(domains);g.journal(j,outcome);steps.append(outcome)
    g.journal(j,dict(kind='AFTER_NEW_correlated_contraction_pass',pass_index=sweep,empty=empty,changed=changed,domains=encbox(domains)))
    if empty or not changed:break
j.close();classification='conditional_five_clock_outer_domain_EMPTY'if empty else('UNRESOLVED_fixed_point_outer_domain'if not changed else'UNRESOLVED_declared_24pass_budget')
result=dict(schema='v2d.correlated-source-time-domain-contraction.v1',classification=classification,empty_factor=empty_factor,passes=last_pass+1,factor_steps=len(steps),initial_domains=encbox(initial),final_domains=encbox(domains),
    normalized_width_ratios={k:float((r[1]-r[0])/(initial[k][1]-initial[k][0]))for k,r in domains.items()},
    continuous_five_clock_domain_NOT_finite_nodes=True,full_filled_source_interiors_and_three_unknown_event_bands_preserved=True,
    scope='Necessary constraints within inherited region2/fixed source-wrist robot hypothesis; no global task/root/source-truth conclusion',
    no_point_candidate_selected=True,no_root_or_reference_mutation=True,source_accuracy_global_task_and_native_motor_grasp_NOT_established=True)
(O/'correlated-domain-contraction-result.json').write_text(json.dumps(g.plain(result),allow_nan=False,indent=2)+'\n')
out=dict(schema='v2d.correlated-source-time-clipping-report.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/correlated_bezier_clipping_v1.py'),
    classification=classification,empty_factor=empty_factor,passes=last_pass+1,factor_steps=len(steps),source_factors=5,temporal_factors=44,variables=10,
    no_splits_or_point_banks_LP_graphs_or_optimizers=True,all_proposals_returns_coefficients_and_unknown_states_saved=True,
    conditional_scope_only_and_current_components_unchanged=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,
    all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
