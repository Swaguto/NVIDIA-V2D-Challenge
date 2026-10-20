"""Independent per-operand dyadic and affine-polynomial source-map audit.

No builder/helper, source metric, raster, KD query, pose or old audit is rerun.
"""
from pathlib import Path
from fractions import Fraction as Q
import ast,datetime,hashlib,json,os
import numpy as np,trimesh

S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-source-numerical-independent-v1'
B=S/'e037-source-numerical-bindings-v1';D=S/'e037-object-dynamic-runtime-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def unpack(v):
 if isinstance(v,dict)and set(v)=={'numerator','denominator'}:return Q(int(v['numerator']),int(v['denominator']))
 if isinstance(v,dict):return {k:unpack(x)for k,x in v.items()}
 if isinstance(v,list):return [unpack(x)for x in v]
 return v
def pack(v):
 if isinstance(v,Q):return dict(numerator=str(v.numerator),denominator=str(v.denominator))
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {str(k):pack(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [pack(x)for x in v]
 return v
def save(n,v):
 p=O/n;p.write_text(json.dumps(pack(v),indent=2,allow_nan=False)+'\n');p.chmod(0o600)
 with p.open('rb')as f:os.fsync(f.fileno())
 return dict(path=str(p),sha256=sha(p))
def pair(x):
 n,d=float(x).as_integer_ratio();return n,d.bit_length()-1
def scale(p,E):return p[0]<<(E-p[1])
def plus(a,b):return [x+y for x,y in zip(a,b,strict=True)]
def times(a,b):return [a*x for x in b]

with(O/'normalized-BEFORE-AFTER-journal.jsonl').open('w')as log:
 def emit(v):log.write(json.dumps(pack(v),allow_nan=False)+'\n');log.flush();os.fsync(log.fileno())
 scope=dict(NEW_cache_alias_and_parametric_error_coefficients_ONLY=True,
  arithmetic='independent per-operand dyadic exponent shifts, not producer common-array matmul',
  parameter_bound='independent affine-polynomial error propagation in free Hx,Hy,Hz',
  all_old_source_NN_raster_controls_audits_NOT_replayed=True,no_pose_solver_native_physics_GPU=True)
 save('ONE-independent-scope-BEFORE.json',scope);emit(dict(kind='BEFORE_NEW_independent_binding',scope=scope))
 inputs=read(B/'NEW-input-provenance-BY-HASH.json')
 for p in inputs.values():assert sha(p['path'])==p['sha256']
 Fdir=S/'e037-thumb-ray-independent-validation-v1/frozen';manifest=read(Fdir/'manifest.json')
 V=np.asarray(trimesh.load(Fdir/manifest['objects'][0]['mesh'],force='mesh',process=False).vertices,float)
 Vpair=[[pair(x)for x in v]for v in V]
 with np.load(D/'dynamic-source-data.npz')as z:C0=z['C0'];poses=z['original_camera_from_object'];W=z['camera_from_world_linear'];times0=z['timestamps_s'];indices=z['frame_indices']
 records=unpack(read(B/'all151-NEW-source-numerical-bindings-RETURN.json'))
 # Original verified dot proof constants are reused by hash; their controls and
 # operation-order audit are not reexecuted. Only NEW affine propagation follows.
 u=Q(1,2**53);eta=Q(1,2**1074);g=5*u/(1-5*u);tau=5*eta/(1-5*u)
 totals=0;nonzero=0;maximum=Q(0);allret=[]
 for f,r in enumerate(records):
  emit(dict(kind='BEFORE_NEW_frame_certificate',frame=f,producer_alias=r['exact_signed_defects']['sha256']))
  assert sha(r['exact_signed_defects']['path'])==r['exact_signed_defects']['sha256']
  with np.load(r['exact_signed_defects']['path'])as z:given=z['signed_integer_numerators'];E=int(z['denominator_exponent'])
  Rp=[[pair(x)for x in row]for row in poses[f,:3,:3]];tp=[pair(x)for x in poses[f,:3,3]]
  expected=[];maxnums=[0,0,0];witness=[0,0,0];bad=0;local_nonzero=0
  for p,v in enumerate(Vpair):
   row=[]
   for i in range(3):
    # Independent scalar expansion: each input/product has its own denominator.
    n=scale(pair(C0[f,p,i]),E)-scale(tp[i],E)
    for j in range(3):n-=v[j][0]*Rp[i][j][0]<<(E-v[j][1]-Rp[i][j][1])
    row.append(str(n));bad+=int(n!=int(given[p,i]));local_nonzero+=int(n!=0)
    if abs(n)>maxnums[i]:maxnums[i]=abs(n);witness[i]=p
   expected.append(row)
  width=max(len(x)for row in expected for x in row);p=O/f'frame-{f:03d}-NEW-independent-alias-numerators-RETURN.npz'
  np.savez_compressed(p,signed_integer_numerators=np.array(expected,dtype=f'U{width}'),denominator_exponent=np.array(E,np.int64));p.chmod(0o600)
  with p.open('rb')as fp:os.fsync(fp.fileno())
  emit(dict(kind='AFTER_NEW_independent_all_component_RETURN',frame=f,path=str(p),sha256=sha(p),nonmatching_components=bad))
  maxima=[Q(n,1<<E)for n in maxnums]
  # Independent polynomial construction: e_dot + e_add + exact alias error.
  Cabs=[Q(float(x))for x in np.max(abs(C0[f]),axis=0)];coeff=[]
  for i in range(3):
   Dpoly=[Q(0),*[abs(Q(float(x)))for x in W[f,i]]]
   Edot=plus(times(g,Dpoly),[tau,Q(0),Q(0),Q(0)])
   Aexact=plus([Cabs[i],Q(0),Q(0),Q(0)],Dpoly)
   Eadd=plus(times(u,plus(Aexact,Edot)),[eta,Q(0),Q(0),Q(0)])
   coeff.append(plus(plus(Edot,Eadd),[maxima[i],Q(0),Q(0),Q(0)]))
  ret=dict(frame=f,exact_components_checked=len(V)*3,nonmatching_components=bad,
   nonzero_exact_components=local_nonzero,maximum_exact_component_defects=maxima,
   witness_indices=witness,NEW_independent_error_coefficients=coeff,
   all12coefficient_differences_zero=coeff==r['NEW_conditional_camera_error_coefficients'],
   raw_C0_and_source_geometry_NOT_recomputed_by_floating_operator=True,
   no_actual_delta_halfwidth_or_pose_assigned=True)
  ref=save(f'frame-{f:03d}-NEW-independent-numerical-binding-RETURN.json',ret);emit(dict(kind='AFTER_NEW_independent_frame_RETURN',frame=f,actual_RETURN=ref))
  assert bad==0 and coeff==r['NEW_conditional_camera_error_coefficients']
  assert maxima==r['max_absolute_alias_by_component']and witness==r['max_witness_vertex_indices']
  assert Cabs==r['cached_C0_component_absolute']and local_nonzero==r['nonzero_exact_alias_components']
  assert r['frame']==f and r['source_frame_index']==int(indices[f])and r['timestamp_s']==float(times0[f])
  assert r['world_component_halfwidth_symbols_UNASSIGNED']==[f'H_{f}_{j}'for j in range(3)]
  assert all(x>=0 for row in coeff for x in row)
  totals+=len(V)*3;nonzero+=local_nonzero;maximum=max(maximum,*maxima);allret.append(ret)
 # NEW operator-scope AST identities; no source function is executed.
 ops=read(B/'NEW-actual-operator-scope-bindings-RETURN.json')
 for label in ['original_fullpixel','EVEN_training']:
  ref=ops[label]['source'];assert sha(ref['path'])==ref['sha256']
  fn='observed_fit'if label=='original_fullpixel'else'vertex_median_semantics'
  tree=ast.parse(Path(ref['path']).read_text());fun=next(x for x in ast.walk(tree)if isinstance(x,ast.FunctionDef)and x.name==fn)
  query=next(x for x in ast.walk(fun)if isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='query')
  kw={a.arg:ast.literal_eval(a.value)for a in query.keywords}
  if label=='original_fullpixel':assert 'k'not in kw and kw['workers']==1
  else:assert kw['k']==2 and kw['workers']==1
 public=(S/'e037-source-numerical-reference-v1/_ckdtree.pyx').read_text()
 assert 'object k=1' in public
 # Verify NEW journal pairs only, with constructors saved before the map return.
 j=[json.loads(x)for x in (B/'normalized-BEFORE-AFTER-journal.jsonl').read_text().splitlines()];pairs=[]
 for f in range(151):
  position={k:next(i for i,v in enumerate(j)if v.get('kind')==k and v.get('frame')==f)for k in ['BEFORE_NEW_cache_alias','AFTER_NEW_exact_alias_constructor_RETURN','AFTER_NEW_conditional_error_map_RETURN']}
  assert position['BEFORE_NEW_cache_alias']<position['AFTER_NEW_exact_alias_constructor_RETURN']<position['AFTER_NEW_conditional_error_map_RETURN'];pairs.append(dict(frame=f,positions=position))
 save('all151-NEW-independent-numerical-bindings.json',allret);save('all151-NEW-ordered-durable-pairs.json',pairs)
 report=dict(schema='v2d.NEW-source-numerical-independent.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  frames=151,NEW_exact_cache_alias_components=totals,nonzero_exact_components=nonzero,
  maximum_exact_absolute_alias=maximum,maximum_absolute_alias_m_float=float(maximum),
  NEW_exact_conditional_error_coefficients=151*12,all_NEW_differences_zero=True,
  NEW_distinct_original_k1_training_k2_operator_bindings=2,NEW_durable_frame_pairs=151,
  no_old_source_NN_raster_control_producer_or_audit_replayed=True,
  original_binary_search_sqrt_raster_build_source_accuracy_and_candidate_qualification_NOT_proved=True,
  no_numeric_episode_pose_solver_native_physics_GPU_or_original_gate_change=True)
 save('report.json',report);emit(dict(kind='AFTER_NEW_independent_report_RETURN',actual_RETURN=report));print(json.dumps(pack(report)))
