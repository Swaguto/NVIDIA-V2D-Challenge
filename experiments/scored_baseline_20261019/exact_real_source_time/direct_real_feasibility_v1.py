"""NEW exact SMT-LIB encoder and bounded real-algebraic RETURN handling.
No saved source polynomial, numerical tolerance or robot command is changed.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json,os,re,subprocess,time
import conditional_moment_cones_v1 as c

def rational(v):
 v=F(v);a=str(abs(v.numerator))
 if v.denominator!=1:a='(/ '+a+' '+str(v.denominator)+')'
 return '(- '+a+')'if v<0 else a

def parse(text):
 tokens=re.findall(r'\(|\)|"(?:\\.|[^"\\])*"|[^\s()]+',text);stack=[];out=[]
 for t in tokens:
  if t=='(':stack.append([])
  elif t==')':
   if not stack:raise ValueError('unbalanced RETURN')
   v=stack.pop();(stack[-1]if stack else out).append(v)
  else:(stack[-1]if stack else out).append(t)
 if stack:raise ValueError('incomplete RETURN')
 return out

def exact(expr,variables=None):
 variables={}if variables is None else variables
 if isinstance(expr,str):return F(variables[expr])if expr in variables else F(expr)
 op,*args=expr;v=[exact(x,variables)for x in args]
 if op=='+':return sum(v,F(0))
 if op=='*':
  result=F(1)
  for x in v:result*=x
  return result
 if op=='-':return -v[0]if len(v)==1 else v[0]-sum(v[1:],F(0))
 if op=='/'and len(v)==2:return v[0]/v[1]
 if op=='^'and len(v)==2 and v[1].denominator==1 and v[1]>=0:return v[0]**int(v[1])
 raise ValueError('unsupported exact expression')

def polynomial(coefficients,names):
 if len(set(names))!=len(names)or any(not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*',n)for n in names):raise ValueError('unaliased explicit variable scope required')
 terms=[]
 for power,value in sorted(coefficients.items()):
  if len(power)!=len(names)or any(type(k)is not int or k<0 for k in power):raise ValueError('unavailable or invalid power/scope')
  value=F(value)
  if not value:continue
  factors=[rational(value)]+[name if k==1 else '(^ '+name+' '+str(k)+')'for name,k in zip(names,power,strict=True)if k]
  terms.append(factors[0]if len(factors)==1 else '(* '+' '.join(factors)+')')
 return '0'if not terms else terms[0]if len(terms)==1 else '(+ '+' '.join(terms)+')'

def query(names,constraints,values=True):
 if len(set(names))!=len(names):raise ValueError('duplicate variable declaration')
 return '(set-logic QF_NRA)\n(set-option :produce-models true)\n'+''.join('(declare-fun '+n+' () Real)\n'for n in names)+''.join('(assert '+co+')\n'for co in constraints)+'(check-sat)\n'+('(get-value ('+' '.join(names)+'))\n'if values else '')+'(exit)\n'

def classification(stdout):
 forms=parse(stdout)
 if not forms or forms[0]not in['sat','unsat','unknown']:return dict(status='UNRESOLVED_nonstandard_RETURN',forms=forms)
 return dict(status=forms[0],forms=forms,solver_status_NOT_candidate_or_exclusion=True)

def algebraic_interval(expr,emit,bits=96):
 try:
  v=exact(expr);out=dict(kind='exact_rational',lower=c.rec(v),upper=c.rec(v),exact_expression=expr);emit(out);return out
 except (ValueError,TypeError):pass
 if isinstance(expr,list)and len(expr)==2 and expr[0]=='-':
  inner=algebraic_interval(expr[1],emit,bits);out=dict(kind='exact_negated_algebraic_interval',lower=c.rec(-c.dec(inner['upper'])),upper=c.rec(-c.dec(inner['lower'])),exact_expression=expr,interval_midpoint_NOT_accepted_candidate=True);emit(out);return out
 if not isinstance(expr,list)or len(expr)!=3 or expr[0]!='root-obj':raise ValueError('UNRESOLVED unsupported algebraic RETURN')
 import sympy as sp
 x=sp.Symbol('x')
 def symbolic(e):
  if isinstance(e,str):return x if e=='x'else sp.Rational(e)
  op,*args=e;v=[symbolic(a)for a in args]
  if op=='+':return sum(v)
  if op=='*':
   z=sp.Integer(1)
   for a in v:z*=a
   return z
  if op=='-':return -v[0]if len(v)==1 else v[0]-sum(v[1:])
  if op=='/'and len(v)==2:return v[0]/v[1]
  if op=='^'and len(v)==2 and v[1].is_Integer and v[1]>=0:return v[0]**v[1]
  raise ValueError('UNRESOLVED algebraic syntax')
 P=sp.Poly(symbolic(expr[1]),x,domain=sp.QQ);coeff=P.all_coeffs();index=int(expr[2]);degree=P.degree()
 if degree<1 or degree>32 or any(max(abs(int(a.p)).bit_length(),int(a.q).bit_length())>4096 for a in coeff):raise ValueError('UNRESOLVED NEW algebraic RETURN32degree/4096bit parse cap')
 emit(dict(kind='BEFORE_NEW_algebraic_RETURN_isolation',degree=degree,root_index=index,coefficients=[c.rec(F(int(v.p),int(v.q)))for v in coeff],fixed_bits=bits,NOT_old_source_root_chart=True))
 roots=P.intervals(eps=sp.Rational(1,1<<bits));returned=[dict(lower=c.rec(F(int(a.p),int(a.q))),upper=c.rec(F(int(b.p),int(b.q))),multiplicity=int(k))for(a,b),k in roots];emit(dict(kind='AFTER_NEW_algebraic_RETURN_isolation_RETURN',all_real_intervals=returned,complex_root_count=int(degree-sum(k for _,k in roots))))
 if index<1 or index>len(roots):raise ValueError('UNRESOLVED algebraic root index')
 (a,b),mult=roots[index-1];out=dict(kind='exact_algebraic_isolating_interval',lower=c.rec(F(int(a.p),int(a.q))),upper=c.rec(F(int(b.p),int(b.q))),root_index=index,multiplicity=int(mult),exact_expression=expr,interval_midpoint_NOT_accepted_candidate=True);emit(out);return out

def run_once(executable,query_path,output,seconds,emit):
 """One subprocess; memory/RSS/disk monitored including solver child."""
 import shutil
 output=Path(output);stdout_path=output/'solver-stdout.txt';stderr_path=output/'solver-stderr.txt';start=time.monotonic();maximum=0;event=None;cmd=[str(executable),'-smt2','-T:'+str(seconds),'-memory:1536','smt.threads=1','parallel.enable=false',str(query_path)]
 emit(dict(kind='BEFORE_NEW_exact_real_subprocess',command=cmd,timeout_seconds=seconds,child_plus_worker_RSS_limit=2*1024**3,disk_reserve=int(1.5*1024**3)))
 with stdout_path.open('w')as sout,stderr_path.open('w')as serr:
  proc=subprocess.Popen(cmd,stdout=sout,stderr=serr,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'));emit(dict(kind='AFTER_NEW_subprocess_constructor_RETURN',pid=proc.pid))
  while proc.poll()is None:
   rss=subprocess.run(['ps','-p',str(proc.pid)+','+str(os.getpid()),'-o','rss='],capture_output=True,text=True);total=sum(int(v)*1024 for v in rss.stdout.split());maximum=max(maximum,total)
   if total>2*1024**3:event='combined_child_worker_RSS_guard'
   elif shutil.disk_usage(output).free<int(1.5*1024**3):event='disk_reserve_guard'
   elif time.monotonic()-start>seconds+2:event='independent_subprocess_timeout_guard'
   if event:
    proc.terminate()
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=2)
    break
   time.sleep(.1)
  proc.wait();sout.flush();os.fsync(sout.fileno());serr.flush();os.fsync(serr.fileno())
 result=dict(exit_code=proc.returncode,wall_seconds=time.monotonic()-start,maximum_combined_child_worker_RSS_bytes=maximum,guard_event=event,stdout=stdout_path.read_text(),stderr=stderr_path.read_text(),one_subprocess_call_NO_retry=True);emit(dict(kind='AFTER_NEW_exact_real_SUBPROCESS_RETURN',RETURN=result));return result
