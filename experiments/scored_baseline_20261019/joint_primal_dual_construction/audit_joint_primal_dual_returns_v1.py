"""Independent SAVED NEW return audit only. No method/map/source/FK/native replay."""
from pathlib import Path
import json,hashlib,os,re
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';O=S/'e037-joint-primal-dual-evaluation-v1';D=S/'e037-joint-primal-dual-return-audit-v1'
from joint_lossless_returns_v1 import seal_return

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(p):
 doc=json.loads(p.read_text())
 with np.load(p.parent/doc['array_file'],allow_pickle=False) as z:
  def u(t):
   k=t['type']
   if k=='array':
    a=z[t['key']];assert a.dtype.str==t['dtype'] and list(a.shape)==t['shape'] and hashlib.sha256(a.tobytes()).hexdigest()==t['bytes_sha256'];return a
   if k=='mapping':return {u(k):u(v) for k,v in t['items']}
   if k in ['list','tuple']:
    a=[u(v) for v in t['items']];return tuple(a) if k=='tuple' else a
   if k=='numpy_scalar':return np.frombuffer(bytes.fromhex(t['bytes']),dtype=t['dtype'])[0]
   if k=='float':return float.fromhex(t['hex'])
   if k=='int':return int(t['value'])
   if k in ['str','bool']:return t['value']
   if k=='none':return None
   if k=='path':return Path(t['value'])
   if k=='sparse':return (sparse.csc_matrix if t['format']=='csc' else sparse.csr_matrix)((u(t['data']),u(t['indices']),u(t['indptr'])),shape=t['shape'])
   raise ValueError(k)
  return u(doc['tree'])
report=json.loads((O/'report.json').read_text());guard=json.loads((O/'guard.json').read_text());files=sorted(O.iterdir());bindings={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in files if p.is_file()}
seal_return(D,'NEW-ALL-actual-return-file-bindings-RETURN',bindings)
patterns=np.load(S/'e037-joint-method-preflight-v1/NEW-ALL-geometric-method-column-pattern-RETURN.npz',allow_pickle=False);ptr=patterns['indptr'];cols=patterns['columns'];patterns.close()
counts={};all_pairs=0;unknown=0;commands=0;slopes=0;maxres=0.;completeframes={};branches=0
for p in files:
 name=p.name
 if name.endswith('-ALL-pair-RETURN.json'):
  a=read(p);m=re.search(r'NEW-step-(\d+)-frame-(\d+)',name)
  if m:tag=m.group(1);frame=int(m.group(2))
  else:tag='terminal';frame=int(re.search(r'frame-(\d+)',name).group(1))
  assert np.array_equal(a['pairs'],np.arange(1653,dtype=np.int64));completeframes.setdefault(tag,[]).append(frame)
  assert a['chart'].shape==(1653,2) and a['n'].shape==(1653,3) and a['DN'].shape==(1653,3,2)
  assert len(a['valid'])+len(a['unknown_raw'])==1653;unknown+=len(a['unknown_raw']);all_pairs+=1653
  assert np.array_equal(a['valid'],np.arange(1653)) # actual finite returns; unknowns fail honestly if present
  for label in ['A','B']:
   pt=a[label+'_ptr'];assert len(pt)==1654 and np.all(np.diff(pt)>0) and pt[-1]==len(a[label+'_ties'])==len(a[label+'_direction_slopes'])
  cp=a['command_ptr'];assert len(cp)==1654 and cp[-1]==len(a['columns'])==len(a['mean_command'])
  for side in ['A','B']:
   sp=a[side+'_slope_ptr'];assert len(sp)==len(a['columns'])+1 and sp[-1]==len(a[side+'_slopes']);slopes+=int(sp[-1])
  for i in range(1653):
   c=a['columns'][cp[i]:cp[i+1]];commands+=len(c);g=frame*1653+i
   allowed=cols[ptr[g]:ptr[g+1]]
   assert all(k in allowed for k in c)
   for side in ['A','B']:
    n=a[side+'_ptr'][i+1]-a[side+'_ptr'][i]
    assert np.all(np.diff(a[side+'_slope_ptr'][cp[i]:cp[i+1]+1])==n)
  assert a['mean_is_SEARCH_only'].all()
  # Residual arithmetic is audited using SAVED support scalars only, not geometry.
  # Phase zero at first oracle; later raw phase is bound via saved oracle input below.
  wpath=next(O.glob(f'*-NEW-step-{tag}-NEW-input-BEFORE.json')) if tag!='terminal' else next(O.glob('*-NEW-terminal-step-02-NEW-input-BEFORE.json'))
  if tag not in counts:
   w=read(wpath)['w'];counts[tag]=dict(phase=float(w[1522]),w_hash=hashlib.sha256(w.tobytes()).hexdigest())
  r=.11684069766135205;expected=.001+r*counts[tag]['phase']-a['scalars'][:,0]-a['scalars'][:,1]
  maxres=max(maxres,float(np.max(np.abs(expected-a['scalars'][:,2]))))
 elif name.endswith('-whole-product-projection-RETURN.json'):
  a=read(p);assert len(a['ALL_cone_branches'])==1245;assert np.isfinite(a['value']).all();branches+=1245
for tag,ff in completeframes.items():assert ff==list(range(len(ff)))
# Verify completed raw update identities from SAVED full inputs/returns, no Ax/J/projection replay.
with np.load(S/'e037-joint-method-preflight-v1/NEW-original-objective-and-cone-layout-RETURN.npz',allow_pickle=False) as z:q=z['q'];pd=z['Pdiag']
update_checks=[]
for k in range(1,report['completed_updates']+1):
 tag=f'NEW-step-{k:02d}'
 get=lambda suffix:read(next(O.glob('*-'+tag+'-'+suffix+'.json')))
 w=get('NEW-input-BEFORE')['w'];m=get('raw-metrics-RETURN');grad=get('ALL-adjoint-RETURN')['gradient'];raw=get('primal-prox-input-RETURN')['value'];ret=get('primal-prox-RETURN')
 e1=np.max(np.abs(w-m['tau']*(grad+q)-raw));before=raw/(1+m['tau']*pd);phase=before[1522];before[1522]=ret['w'][1522]
 e2=np.max(np.abs(before-ret['w']));assert e1==0 and e2==0 and min(1,max(0,phase))==ret['w'][1522]
 update_checks.append(dict(step=k,raw_update_identity=float(e1),original_prox_identity=float(e2)))
boundary=[p.name for p in files if p.name.endswith('-actual-boundary-exception-RETURN.json')]
result=dict(passed=True,status=report['status'],completed_updates=report['completed_updates'],completed_full_oracles=report['completed_full_oracles'],
 ALL_saved_pair_instances=all_pairs,ALL_saved_pair_command_incidences=commands,ALL_saved_tied_slopes=slopes,unknown_support_returns=unknown,
 complete_frame_prefixes={k:len(v) for k,v in completeframes.items()},ALL_saved_projection_cone_branches=branches,
 max_saved_support_residual_identity=maxres,update_checks=update_checks,boundary_files=boundary,
 partial_or_unavailable_future_returns_NOT_reconstructed=True,no_method_Ax_adjoint_projector_geometry_source_FK_native_replay=True,
 no_feasibility_stationarity_infeasibility_or_physical_qualification=True)
seal_return(D,'NEW-independent-saved-return-audit-RETURN',result)
assert maxres==0
with (D/'report.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
