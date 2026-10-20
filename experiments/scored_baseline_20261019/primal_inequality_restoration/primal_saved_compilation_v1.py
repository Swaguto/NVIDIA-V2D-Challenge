"""SHA-checked hydration of cached primal arrays; no polynomial/map rebuild."""
from pathlib import Path
import hashlib,json
import numpy as np
import conditional_moment_cones_v1 as c
import primal_inequality_restoration_v1 as p

def hydrate(folder,emit):
 folder=Path(folder);read=lambda f:json.loads(Path(f).read_text());sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest();binding=read(folder/'compiled-primal-binding.json')
 for filename,digest in binding['compiled_file_sha256'].items():
  if sha(folder/filename)!=digest:raise ValueError('saved compilation SHA mismatch')
 if p.CONFIG!=binding['algorithm_configuration']:raise ValueError('declared algorithm configuration changed')
 s=p.PolynomialSystem.__new__(p.PolynomialSystem);s.names=binding['variables'];s.dim=len(s.names);s.rows=[];source=read(folder/'exact-primal-row-bindings.json')
 for r in source:
  s.rows.append(dict(id=r['id'],scope=r['scope'],index=r['index'],terms=[(tuple(t['power']),c.dec(t['value']))for t in r['terms']],sign=r['sign'],strict=r['strict']))
 s.derivative_maps=read(folder/'derivative-index-maps.json');s.count=len(s.rows)
 with np.load(folder/'compiled-primal-system.npz',allow_pickle=False)as z:
  for name in['rowids','powers','coeff','drows','dcols','dpowers','dcoeff','scales','signs','strict']:
   value=z[name].copy();value.setflags(write=False);setattr(s,name,value)
 if s.powers.shape!=(len(s.coeff),s.dim)or s.dpowers.shape!=(len(s.dcoeff),s.dim)or len(s.scales)!=s.count or len(s.derivative_maps)!=s.count:raise ValueError('compiled shapes inconsistent')
 emit(dict(kind='AFTER_NEW_saved_primal_hydration_RETURN',rows=s.count,variables=s.dim,arrays_readonly=True,no_polynomial_derivative_or_normalizer_recomputed=True,compiled_binding_sha256=sha(folder/'compiled-primal-binding.json')));return s
