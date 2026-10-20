"""NEW conditional RN proof on SAVED tied slopes; no episode operators.

All products below are exact rational proof expressions on saved binary inputs.
They are not reconstructions of unsaved numeric dot/mean intermediate returns.
No old J/E, command, source, model, objective or qualification gate is changed.
"""
from pathlib import Path
from fractions import Fraction as Q
import json,hashlib,os,time
import numpy as np
from scipy import sparse
S=Path(__file__).resolve().parents[1];C=S/'compute';OLD=S/'e037-joint-primal-dual-evaluation-v1'
U=Q(1,2**53);TINY=Q(1,2**1074);BIT_CAP=8192;NORMAL_GUARD=Q(2**1022)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 with Path(p).open('x') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def typed(p):
 p=Path(p);d=json.loads(p.read_text())
 with np.load(p.parent/d['array_file'],allow_pickle=False) as z:
  def r(t):
   k=t['type']
   if k=='array':
    a=z[t['key']];assert a.dtype.str==t['dtype'] and list(a.shape)==t['shape'] and hashlib.sha256(a.tobytes()).hexdigest()==t['bytes_sha256'];return a
   if k=='mapping':return {r(k):r(v) for k,v in t['items']}
   if k in ['list','tuple']:
    a=[r(v) for v in t['items']];return tuple(a) if k=='tuple' else a
   if k=='numpy_scalar':return np.frombuffer(bytes.fromhex(t['bytes']),dtype=t['dtype'])[0]
   if k=='float':return float.fromhex(t['hex'])
   if k=='int':return int(t['value'])
   if k in ['str','bool']:return t['value']
   if k=='none':return None
   if k=='path':return Path(t['value'])
   if k=='sparse':return (sparse.csc_matrix if t['format']=='csc' else sparse.csr_matrix)((r(t['data']),r(t['indices']),r(t['indptr'])),shape=t['shape'])
   raise ValueError(k)
  return r(d['tree'])
def q(v):return Q(float(v))
def gamma(k):
 if k<1 or k*U>=1:raise ValueError('UNKNOWN_RN_count_scope')
 return k*U/(1-k*U)
def dot_proof(a,b):
 terms=[q(x)*q(y) for x,y in zip(a,b,strict=True)];total=sum(terms,Q(0));m=2*len(terms)+16
 # gamma_m covers each RN product/addition in ANY reduction association with
 # <=m such operations; absolute subnormal allowances remain separate.
 error=gamma(m)*sum(map(abs,terms),Q(0))+m*TINY/(1-m*U)
 return total,error
def means_proof(values,input_errors=None):
 n=len(values)
 if n==0:raise ValueError('UNKNOWN_empty_tie')
 m=2*n+16;avg=sum(values,Q(0))/n;inerr=sum(input_errors,Q(0))/n if input_errors is not None else Q(0)
 error=inerr+gamma(m)*(sum(map(abs,values),Q(0))/n+inerr)+m*TINY/(1-m*U)
 return avg,error
def combine_proof(a,b,ea=None,eb=None):
 ma,da=means_proof(a,ea);mb,db=means_proof(b,eb)
 ideal=ma+mb;tie=max(map(abs,a))+max(map(abs,b))
 budget=da+db+U*(abs(ma)+abs(mb)+da+db)+TINY
 bound=tie+budget
 for v in [ideal,tie,budget,bound]:
  if max(v.numerator.bit_length(),v.denominator.bit_length())>BIT_CAP:raise ValueError('UNKNOWN_exact_arithmetic8192bit_cap')
 if bound>=NORMAL_GUARD:raise ValueError('UNKNOWN_finite_RN_range')
 return ideal,tie,budget,bound
def dyadic96_upper(x):
 if x<0:raise ValueError('positive envelope required')
 if x==0:return (0,0)
 e=x.numerator.bit_length()-x.denominator.bit_length()-95
 ratio=x/Q(2)**e;mant=(ratio.numerator+ratio.denominator-1)//ratio.denominator
 return int(mant),int(e)
def dyadic_value(m,e):return Q(m)*Q(2)**e

def frame_files():return sorted(OLD.glob('*-NEW-step-01-frame-*-ALL-pair-RETURN.json'))
def dynamic_file():return next(OLD.glob('*-NEW-step-01-ALL-dynamic-F-J-envelope-RETURN.json'))
