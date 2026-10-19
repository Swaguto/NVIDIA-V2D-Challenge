"""Independent exact NEW image-matrix identities and saved square certificates.

No producer/solver/congruence algorithm is rerun. Affine substitutions use
repeated products; moments use falling factorials; saved witnesses/congruences
are checked directly against unaltered saved binary weights and scalar models.
"""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import datetime,hashlib,json,os,sys,time
S=Path(__file__).resolve().parents[1];O=S/'e037-matrix-image-independent-v1';sys.path.insert(0,str(S/'compute'))
import conditional_moment_cones_v1 as c
import thumb_conic_geometry_v1 as g
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());start=time.monotonic();j=(O/'independent-new-image-matrix-necessity-map-and-square-certificates.jsonl').open('w');emit=lambda v:g.journal(j,v)
def save(name,value):
 with (O/name).open('w')as f:f.write(json.dumps(g.plain(value),allow_nan=False,indent=2)+'\n');f.flush();os.fsync(f.fileno())
def poly(v):return {tuple(k['power']):c.dec(k['value'])for k in v}
def arr(v):return {(i,k):F(z)for i,row in enumerate(v)for k,z in enumerate(row)if z}
def add(*terms):
 q={}
 for p,z in terms:
  for k,v in p.items():q[k]=q.get(k,F(0))+v*z
 return {k:v for k,v in q.items()if v}
def mul(p,q):
 r={}
 for k,v in p.items():
  for l,z in q.items():
   e=tuple(a+b for a,b in zip(k,l,strict=True));r[e]=r.get(e,F(0))+v*z
 return {k:v for k,v in r.items()if v}
def uniform(p,bounds):
 result=F(0);terms=[]
 for k,v in sorted(p.items()):
  factor=F(1)
  for n,(lo,hi)in zip(k,bounds,strict=True):factor*=max(abs(lo),abs(hi))**n
  result+=abs(v)*factor;terms.append(dict(power=k,coefficient=c.rec(v),monomial_bound=c.rec(factor)))
 return result,terms
def affine(p,bounds):
 out={}
 for k,v in p.items():
  term={(0,0):v}
  for axis,degree in enumerate(k):
   lo,hi=bounds[axis];e=[0,0];e[axis]=1;factor={(0,0):lo,tuple(e):hi-lo}
   for _ in range(degree):term=mul(term,factor)
  out=add((out,F(1)),(term,F(1)))
 return out
def fall(k,r):
 a=b=1
 for n in range(r):a*=k-n;b*=4-n
 return F(a,b)
def coefs(polynomial,table):
 out={}
 for idx,col in table.items():
  value=sum(v*fall(idx[0],k[0])*fall(idx[1],k[1])for k,v in polynomial.items())
  if value:out[col]=value
 return out
def qform(M,v):return sum(a*M[i][k]*b for i,a in enumerate(v)for k,b in enumerate(v))
parent=S/'e037-matrix-image-support-v1';report=read(parent/'report.json')
for n,d in report['all_outputs_sha256'].items():assert sha(parent/n)==d
system=read(S/'e037-index-source-time-system-v1/coupled-source-time-polynomial-system.json');sources={z['frame']:z['source_record_COPIED_not_regenerated']for z in system['source_constraints']};model=read(S/'e037-bounded-moment-cone-v1/update-05-necessary-LP-model.json');tables={}
for col,v in enumerate(model['columns']):tables.setdefault(tuple(v['scope']),{})[tuple(v['index'])]=col
weights=list(map(F,read(S/'e037-supported-conic-evaluation-v1/single-supported-solver-RETURN.json')['x']));counts=dict(necessary_error_identities=0,uniform_error_terms=0,exact_affine_polynomials=0,exact_shiftmaps=0,exact_nonzero_coefficients=0,matrix_entry_identities=0,PSD_congruence_entries=0,negative_square_certificates=0);records=[]
for summary in report['summaries']:
 frame=summary['frame'];r=read(parent/f'frame-{frame}-matrix-source-support.json');emit(dict(kind='BEFORE_NEW_independent_image_matrix_certificate',frame=frame,source_record_sha256=sha(parent/f'frame-{frame}-matrix-source-support.json')))
 if r['classification']=='UNRESOLVED_necessary_matrix_conversion':records.append(dict(frame=frame,classification=r['classification']));continue
 src=sources[frame];assert r['source_record_COPIED']==src;sp=src['polynomial'];N=list(map(arr,sp['camera_numerators']));D=arr(sp['denominator']);assert D=={(0,0):1,(0,2):1,(2,0):1,(2,2):1};bounds=[[c.dec(z)for z in box]for box in r['half_angle_bounds']];R=F(src['ORIGINAL_cap_px'])+F(src['ORIGINAL_image_tolerance_px']);depth=F(sp['minimum_depth']);K=src['K'];pixel=src['pixel'];T=add((N[2],R));X=add((N[0],F(K[0][0])),(N[2],F(K[0][2])-F(pixel[0])));Y=add((N[1],F(K[1][1])),(N[2],F(K[1][2])-F(pixel[1])));e=add((mul(X,X),F(1)),(mul(Y,Y),F(1)),(mul(T,T),F(-1)),(arr(sp['normalized_source_polynomial']),-F(sp['source_scale'])));de=add((N[2],F(1)),(D,-depth),(arr(sp['normalized_depth_polynomial']),-F(sp['depth_scale'])));eps,terms=uniform(e,bounds);eta,dterms=uniform(de,bounds);nz=depth-eta;tmin=R*nz;delta=eps/(2*tmin);slack=2*tmin*delta+delta*delta-eps;er=r['necessary_arrow_error_binding'];expected=[eps,eta,nz,tmin,delta,slack];keys=['uniform_source_discrepancy','uniform_depth_discrepancy','Nz_lower','t_lower','diagonal_guard','guard_squared_slack'];emit(dict(kind='AFTER_NEW_independent_necessary_error_RETURN',frame=frame,values={k:c.rec(v)for k,v in zip(keys,expected)},source_terms=terms,depth_terms=dterms));assert all(v==c.dec(er[k])for k,v in zip(keys,expected))and nz>0 and tmin>0 and slack>=0 and F(sp['source_scale'])>0 and F(sp['depth_scale'])>0;assert e==poly(er['source_discrepancy_power'])and de==poly(er['depth_discrepancy_power']);counts['necessary_error_identities']+=6;counts['uniform_error_terms']+=len(terms)+len(dterms)
 half=[add((T,F(1)),({(0,0):delta},F(1))),X,Y];assert half==list(map(poly,r['half_angle_arrow_polynomials']));scope=r['table_scope'];order=r['source_order'];normal=[]
 for z in half:
  q=affine(z,bounds);q={tuple(k[order.index(v)]for v in scope):x for k,x in q.items()};normal.append(q)
 assert normal==list(map(poly,r['normalized_arrow_polynomials']));counts['exact_affine_polynomials']+=3;basis=[tuple(k)for k in r['multilinear_basis']];assert basis==[(0,0),(0,1),(1,0),(1,1)];table=tables[tuple(scope)];assert set(table)==set(product(range(5),repeat=2));cache={}
 for record in r['unique_shift_maps']:
  family=record['family'];shift=tuple(record['shift']);moved={tuple(a+b for a,b in zip(k,shift)):v for k,v in normal[family].items()};assert all(max(k)<=4 for k in moved);co=coefs(moved,table);actual_co={x['column']:c.dec(x['value'])for x in record['coefficients']};actual_value=sum(v*weights[k]for k,v in co.items());emit(dict(kind='AFTER_NEW_independent_falling_map_RETURN',frame=frame,family=family,shift=shift,actual_value=c.rec(actual_value),coefficients=[dict(column=k,value=c.rec(v))for k,v in sorted(co.items())]));assert moved==poly(record['polynomial'])and co==actual_co and actual_value==c.dec(record['actual_value']);cache[family,shift]=(co,actual_value);counts['exact_shiftmaps']+=1;counts['exact_nonzero_coefficients']+=len(co)
 M=[[c.dec(z)for z in row]for row in r['matrix']];assert len(M)==12;entry_by={(v['row'],v['column']):v for v in r['matrix_entry_records']}
 for i,k in product(range(12),repeat=2):
  a,b=i//4,k//4;shift=tuple(z+w for z,w in zip(basis[i%4],basis[k%4]));family=0 if a==b else 1 if {a,b}=={0,1}else 2 if {a,b}=={0,2}else None;value=cache[family,shift][1]if family is not None else F(0);rec=entry_by[i,k];assert M[i][k]==value==c.dec(rec['actual_entry'])and rec['family']==family and tuple(rec['shift'])==shift;counts['matrix_entry_identities']+=1
 certificate=r['certificate'];assert certificate['classification']==r['classification'];result=dict(frame=frame,classification=r['classification'],uniform_source_error=c.rec(eps),uniform_depth_error=c.rec(eta),necessary_diagonal_guard=c.rec(delta))
 if r['classification']=='exact_mixed_matrix_PSD':
  B=[list(map(c.dec,v))for v in certificate['basis_columns']];diagonal=list(map(c.dec,certificate['diagonal']));assert len(B)==len(diagonal)==12 and all(v>=0 for v in diagonal)
  for col,v in enumerate(B):assert v[col]==1 and all(z==0 for z in v[col+1:])
  for i,k in product(range(12),repeat=2):
   actual=sum(B[i][row]*M[row][col]*B[k][col]for row,col in product(range(12),repeat=2));assert actual==(diagonal[i]if i==k else 0);counts['PSD_congruence_entries']+=1
  emit(dict(kind='AFTER_NEW_independent_positive_congruence_RETURN',frame=frame,columns=certificate['basis_columns'],diagonal=certificate['diagonal'],rank=sum(v>0 for v in diagonal)));result['rank']=sum(v>0 for v in diagonal)
 else:
  raw=list(map(c.dec,certificate['vector']));rawq=qform(M,raw);assert rawq<0 and rawq==c.dec(certificate['expectation']);w=r['negative_source_support_witness'];v=list(map(c.dec,w['component_major_vector']));assert v==[z/max(map(abs,raw))for z in raw];q=[{k:z for k,z in zip(basis,v[i*4:(i+1)*4])if z}for i in range(3)];C=[[normal[0],normal[1],normal[2]],[normal[1],normal[0],{}],[normal[2],{},normal[0]]];wp={}
  for i,k in product(range(3),repeat=2):wp=add((wp,F(1)),(mul(C[i][k],mul(q[i],q[k])),F(1)))
  co=coefs(wp,table);expectation=sum(z*weights[col]for col,z in co.items());emit(dict(kind='AFTER_NEW_independent_negative_source_square_RETURN',frame=frame,polynomial=[dict(power=k,value=c.rec(z))for k,z in sorted(wp.items())],coefficient_map=[dict(column=k,value=c.rec(z))for k,z in sorted(co.items())],expectation=c.rec(expectation),raw_witness_expectation=c.rec(rawq)));assert wp==poly(w['weighted_polynomial'])and co=={z['column']:c.dec(z['value'])for z in w['linear_coefficients']}and expectation==c.dec(w['actual_expectation'])==qform(M,v)<0;counts['negative_square_certificates']+=1;result['negative_expectation']=c.rec(expectation)
 records.append(result)
j.close();bad=[z for z in records if z['classification']=='exact_negative_mixed_square'];out=dict(schema='v2d.independent-matrix-image-support.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(__file__),all_independent_checks_passed=True,new_matrix_factors=len(records),negative_matrix_factors=len(bad),negative_frames=[z['frame']for z in bad],PSD_factors=sum(z['classification']=='exact_mixed_matrix_PSD'for z in records),unresolved_factors=sum(z['classification']=='UNRESOLVED_necessary_matrix_conversion'for z in records),counts=counts,results=records,saved_raw_weights_equalities_and_original_gates_UNCHANGED=True,scalar_source_nonconstant_unavailability_preserved=True,necessary_matrix_guards_do_NOT_change_original_candidate_tolerances=True,new_solver_calls=0,new_native_position_checks=0,new_physics_steps=0,new_GPU_jobs=0,new_PPO_steps=0,classification='saved_AlmostSolved_RETURN_rejected_as_source_supported_measure_NOT_family_exclusion'if bad else'local_matrix_source_support_clear_NOT_realization',all_outputs_sha256={q.name:sha(q)for q in O.iterdir()if q.is_file()and q.name not in['guard.json','worker.log']},wall_seconds=time.monotonic()-start);save('report.json',out);print(json.dumps({k:v for k,v in out.items()if k not in['results','all_outputs_sha256']}))
