"""NEW closed/singular/nonelliptic 2R source contracts; no episode work."""
from pathlib import Path
import datetime,hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-closed-2R-source-contract-v1';start=time.monotonic();sys.path.insert(0,str(S/'compute'))
import numpy as np
import thumb_conic_geometry_v1 as g
from closed_2R_source_domains_v1 import closed_2R_source_chart
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-new-analytic-closed-source-contracts.jsonl').open('w');controls=[];K=np.diag([100.,100.,1.]);H=np.eye(3)
def check(name,args,expect):
    result=closed_2R_source_chart(*args);passed=result['classification'] in expect;row=dict(name=name,result=result,passed=passed);g.journal(j,row);controls.append(row);assert passed,name;return result
point=lambda a,b:np.array([.0458*np.cos(a)+.035*np.cos(a+b),.0458*np.sin(a)+.035*np.sin(a+b)])
check('straight_distal_singleton_source_point',(H,K,100*point(.2,0),.001,[.2,.2],[0.,0.]),['constructive_closed_2R_source_witness'])
check('single_configuration_outside_source',(H,K,[40.,40.],.001,[.2,.2],[0.,0.]),['empty_single_configuration'])
check('closed_straight_boundary_continuous_arc',(H,K,100*point(.2,0),.1,[.1,.3],[0.,.4]),['constructive_closed_2R_source_witness'])
check('whole_workspace_inside_source_disk',(H,K,[0.,0.],100.,[.1,.3],[0.,.4]),['constructive_closed_2R_source_witness'])
check('zero_width_proximal_distal_arc',(H,K,100*point(.2,.2),.1,[.2,.2],[0.,.4]),['constructive_closed_2R_source_witness'])
badH=H.copy();badH[2,2]=-1;check('negative_depth_never_promoted',(badH,K,[0.,0.],100.,[.1,.3],[0.,.4]),['unresolved_no_verified_boundary_or_stationary_witness'])
hyper=H.copy();hyper[2,0]=20;result=check('indefinite_conic_with_constructive_source_point',(hyper,K,[0.,0.],10.,[.1,.3],[0.,.4]),['constructive_closed_2R_source_witness']);assert min(result['conic_eigenvalues'])<0
invalid=False
try:closed_2R_source_chart(H,K,[0.,0.],1.,[.1,.3],[-.1,.4])
except ValueError:invalid=True
row=dict(name='negative_distal_chart_not_silently_relabelled',passed=invalid);g.journal(j,row);controls.append(row);assert invalid
j.close();out=dict(schema='v2d.closed-2R-source-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/closed_2R_source_domains_v1.py'),new_analytic_contracts=len(controls),all_controls_passed=all(v['passed'] for v in controls),no_old_controls_episode_source_native_or_physics_repeated=True,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p) for p in O.iterdir() if p.is_file() and p.name not in ['guard.json','worker.log']},wall_seconds=time.monotonic()-start);(O/'report.json').write_text(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');print(json.dumps(g.plain(out)))
