"""NEW concrete all-tie rounding proof controls; no old controls or query replay."""
from joint_saved_mean_bound_v1 import *
from joint_lossless_returns_v1 import seal_return
D=S/'e037-joint-mean-proof-contracts-v1';contracts=[]
def check(name,raw,passed):
 seal_return(D,name,raw);contracts.append(dict(name=name,passed=bool(passed)));assert passed,name
v=np.array([.1,.1,.1]);actual=float(v.mean());out=combine_proof([q(x) for x in v],[Q(0)])
check('NEW-observed-mean-rounding-and-negative-old-envelope',dict(actual=actual,old_max=float(max(v)),proof=[str(x) for x in out]),actual>max(v) and abs(q(actual)-out[0])<=out[2] and q(actual)<=out[3])
a=[Q(1),Q(-1),Q(0)];b=[Q(-2),Q(2)];out=combine_proof(a,b)
check('NEW-correlated-ALL-tie-cancellation',dict(proof=[str(x) for x in out]),out[0]==0 and out[1]==3 and all(abs(x+y)<=out[1] for x in a for y in b))
z=np.nextafter(0.,1.);ideal,err=dot_proof([z],[z]);observed=float(z*z)
check('NEW-subnormal-product-not-lost',dict(ideal=str(ideal),budget=str(err),observed=observed),ideal==Q(1,2**2148) and abs(q(observed)-ideal)<=err)
a=np.array([.13,-.21,.31]);b=np.array([.7,-.2,.11]);ideal,err=dot_proof(a,b);actual=float(a@b)
check('NEW-three-product-RN-scope',dict(ideal=str(ideal),budget=str(err),actual=actual),abs(q(actual)-ideal)<=err)
ideals=[];errs=[]
for v in [[.1,-.2,.3],[-.1,.2,-.3]]:
 i,e=dot_proof(v,b);ideals.append(i);errs.append(e)
out=combine_proof(ideals,[Q(0)],errs,[Q(0)])
check('NEW-product-then-ALL-mean-bound',dict(proof=[str(x) for x in out]),abs(out[0])<=out[1] and out[2]>sum(errs,Q(0))/2)
values=[Q(0),Q(1,3),Q(2**600+1,2**700),Q(1,2**2148)];pairs=[dyadic96_upper(v) for v in values]
check('NEW-96bit-outward-construction',dict(values=[str(x) for x in values],encoded=pairs),all(dyadic_value(*p)>=x for p,x in zip(pairs,values)) and pairs[0]==(0,0))
unknown=False
try:means_proof([])
except ValueError:unknown=True
check('NEW-empty-tie-UNKNOWN-not-deleted',dict(unknown=unknown),unknown)
# NEW independently decoded bit payload and typed ragged proof schema.
raw=np.array([-0.,np.nextafter(0.,1.),.1]);p=seal_return(D,'NEW-typed-proof-input-RETURN',dict(a=raw,proof=['0','1/3'],keys=(1,'pair')));r=typed(p)
check('NEW-proof-reader-bit-binding',dict(equal=r['a'].tobytes()==raw.tobytes(),proof=r['proof'],keys=r['keys']),r['a'].tobytes()==raw.tobytes() and r['proof']==['0','1/3'] and r['keys']==(1,'pair'))
save(D/'report.json',dict(passed=True,NEW_contracts=len(contracts),contracts=contracts,no_episode_method_source_geometry_native_or_old_control_query=True,
 conditional_saved_binary_RN_scope_only=True,no_repair_of_old_J_E_or_candidate_tolerance=True))
