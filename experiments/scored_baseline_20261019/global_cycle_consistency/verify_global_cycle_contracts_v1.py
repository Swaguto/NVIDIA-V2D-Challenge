"""Eleven NEW continuous/even-cycle/overlap/availability contracts; no solver."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-global-cycle-contract-v1';sys.path.insert(0,str(S/'compute'))
import global_cycle_moments_v1 as h
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();j=(O/'all-NEW-global-cycle-contract-proposals-and-returns.jsonl').open('w');emit=lambda v:g.journal(j,v);start=time.monotonic();controls=[]
def check(name,record,ok):emit(dict(kind='AFTER_NEW_global_cycle_contract_RETURN',name=name,record=record,passed=bool(ok)));assert ok;controls.append(name)
v=F(1,300);point=lambda x:[F(x),F(x)]
for name,edges,expected in [('incompatible_even_cycle',[3*v/5+3*v/10,3*v/5+3*v/10,3*v/5+3*v/10,-3*v/5-3*v/10],'EXCLUDED_cycle_covariance_envelope'),('common_continuous_rank1',[v,-v,v,-v],'NECESSARY_cycle_bounds_NOT_EXCLUDED'),('common_continuous_two_uniforms',[3*v/5,4*v/5,-3*v/5,4*v/5],'NECESSARY_cycle_bounds_NOT_EXCLUDED')]:
 emit(dict(kind='BEFORE_NEW_even_cycle_contract',name=name,variance=c.rec(v),edges=list(map(c.rec,edges)),bounded_continuous_uniform_realization_per_edge=True));r=h.cycle4([point(v)]*4,list(map(point,edges)));ok=r['classification']==expected
 if name=='incompatible_even_cycle':ok=ok and all(v*v-z*z>0 for z in edges)and c.dec(r['strict_separation'])>0
 if name=='common_continuous_rank1':ok=ok and c.dec(r['intersection'][0])<=-v<=c.dec(r['intersection'][1])
 if name=='common_continuous_two_uniforms':ok=ok and c.dec(r['intersection'][0])<=0<=c.dec(r['intersection'][1])
 check(name,r,ok)
for name,p,expected in [('zero_pivot_free_diagonal',point(0),'NECESSARY_missing_covariance_interval'),('zero_pivot_nonzero_cross',point(F(1,20)),'EXCLUDED_zero_pivot_nonzero_cross')]:
 emit(dict(kind='BEFORE_NEW_zero_pivot_contract',name=name,P=h.recI(p)));r=h.triangle_interval(point(1),point(0),point(1),p,point(0));check(name,r,r['classification']==expected)
tiny=F(1,2**120);emit(dict(kind='BEFORE_NEW_pivot_boundary_contract',tiny=c.rec(tiny)));r=h.triangle_interval(point(1),[-tiny,tiny],point(1),point(0),point(0));check('unresolved_zero_crossing_pivot',r,r['classification']=='UNRESOLVED_pivot_interval_meets_zero')
emit(dict(kind='BEFORE_NEW_positive_zero_signed_mass_contract'));first=[point(1)]*4;second=[point(2*(F(1,4)+v))]*4;cross=[point(2*(F(1,4)+v))]*4;positive=h.covariance_envelope(point(2),first,second,cross);zero=h.covariance_envelope(point(0),[point(0)]*4,[point(0)]*4,[point(0)]*4);badzero=h.covariance_envelope(point(0),[point(1)]*4,[point(0)]*4,[point(0)]*4);signed=h.covariance_envelope(point(-1),first,second,cross);check('mass_not_normalized_or_relabelled',dict(positive=positive,zero=zero,badzero=badzero,signed=signed),positive['variance_intervals']==[h.recI(point(4*v))]*4 and zero['classification']=='ZERO_positive_measure_ONLY'and badzero['classification'].startswith('EXCLUDED')and signed['classification'].startswith('UNRESOLVED_signed'))
emit(dict(kind='BEFORE_NEW_fixed96_sqrt_contract'));roots=[]
for x in [F(0),F(2),F(49,121),F(1,2**250)]:
 lo,hi=h.sqrt96(x);roots.append(dict(value=c.rec(x),bounds=h.recI([lo,hi])));assert lo*lo<=x<=hi*hi and hi-lo<=F(1,2**96)
check('fixed96_outward_square_root',dict(roots=roots),True)
# New residual-path identity with a tiny deliberately mismatched raw copy.
cols=[dict(scope=['u'],index=[k])for k in range(5)]+[dict(scope=['u','v'],index=[i,k])for i,k in product(range(5),repeat=2)];tables=h.tables(cols);eq=[]
for i in range(5):
 co={i:F(-1)};co.update({tables['u','v'][i,k]:F(1)for k in range(5)});eq.append(dict(coefficients=[dict(column=k,value=c.rec(z))for k,z in sorted(co.items())],bound=c.rec(F(0))))
model=dict(columns=cols,equalities=eq);weights=[F(1,5)]*5+[F(1,25)]*25;weights[tables['u','v'][4,0]]+=tiny;emit(dict(kind='BEFORE_NEW_overlap_residual_contract',tiny=c.rec(tiny),raw_weights=list(map(c.rec,weights))));links=h.marginal_links(model,tables);path=h.projection_path(links,('u','v'),('u',));combine=h.path_combination(links,path,{'u':1});actual=h.sparse_add(h.mono_map(('u','v'),tables['u','v'],{'u':1}),h.mono_map(('u',),tables['u',],{'u':1}),F(-1));derived={};rhs=F(0);value=F(0)
for row,z in combine.items():
 co={v['column']:c.dec(v['value'])for v in eq[row]['coefficients']};derived=h.sparse_add(derived,co,z);value+=z*sum(a*weights[k]for k,a in co.items())
check('tiny_shared_row_residual_identity',dict(combination=[dict(row=k,value=c.rec(z))for k,z in combine.items()],exact_difference=c.rec(value),raw_weights_NOT_repaired=True),derived==actual and value==tiny)
# Exhaustively compare the NEW four-clique maximum tree weight with all16trees.
emit(dict(kind='BEFORE_NEW_running_intersection_contract'));scopes={('a','b'):{},('b','c'):{},('c','d'):{},('a','d'):{}};r=h.scope_structure(scopes,['a','b','c','d']);weights_tree=[]
for code in product(range(4),repeat=2):
 degree=[1]*4
 for z in code:degree[z]+=1
 edges=[]
 for z in code:
  leaf=next(i for i in range(4)if degree[i]==1);edges.append((leaf,z));degree[leaf]-=1;degree[z]-=1
 left=[i for i in range(4)if degree[i]==1];edges.append(tuple(left));weights_tree.append(sum(len(set(r['maximal_scopes'][i])&set(r['maximal_scopes'][k]))for i,k in edges))
chain=h.scope_structure({('a','b'):{},('b','c'):{},('c','d'):{}},['a','b','c','d']);check('running_intersection_max_tree_and_gap',dict(cycle=r,chain=chain,all16weights=weights_tree),max(weights_tree)==r['maximum_intersection_weight']==3 and r['running_intersection_required_weight']==4 and not r['standard_junction_tree_exists']and chain['standard_junction_tree_exists'])
check('missing_products_remain_free_unknown',dict(unknown_variable_pairs=r['unknown_variable_pairs'],cycles=r['induced_four_variable_cycles']),r['unknown_variable_pairs']==[['a','c'],['b','d']]and len(r['induced_four_variable_cycles'])==1)
j.close();out=dict(schema='v2d.global-cycle-contract.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),helper_sha256=sha(S/'compute/global_cycle_moments_v1.py'),all_controls_passed=True,new_analytic_controls=len(controls),controls=controls,old_odd_cycle_and_mixed_matrix_controls_NOT_repeated=True,no_solver_no_source_generation_no_degree_change=True,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,all_outputs_sha256={p.name:sha(p)for p in O.iterdir()if p.is_file()and p.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start)
with (O/'report.json').open('w')as f:f.write(json.dumps(g.plain(out),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in out.items()if k not in['all_outputs_sha256','controls']}))
