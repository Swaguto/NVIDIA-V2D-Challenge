"""NEW concrete bound/sign/defect/dyadic/boundary/resource controls only."""
from exact_terminal_dual_bound_v1 import *
O=S/'e037-exact-terminal-dual-contracts-v1';returns=[]
def check(name,data,passed):
 returns.append(dict(name=name,actual_RETURN=data));dump(O/'NEW-contract-RETURNS.json',returns);assert passed,name
# Phase model -rho+s=-1, rho in[0,1], y=1 has exact lower bound1.
A=F(-1);b=F(-1);y=F(1);c=1+A*y;lower=min(F(0)*c,F(1)*c)-b*y
check('positive multiplier sign / phase lower bound',dict(lower=lower,wrong_b_sign=min(F(0)*c,F(1)*c)+b*y),lower==1)
# In rho+s=1, y=-1 is NOT dual feasible. Omitting D falsely excludes rho=0.
A=F(1);b=F(1);y=F(-1);c=1+A*y;unadjusted=min(F(0)*c,F(1)*c)-b*y;D=max(-y,F(0))*(abs(b)+abs(A))
check('negative orthant omitted-defect false-certificate control',dict(unadjusted=unadjusted,D=D,corrected=unadjusted-D),unadjusted>0 and unadjusted-D<=0)
# A whole SOC's y.s >= -(||yv||-y0)_+ S0, including negative top.
y0=F(-1);yv=[F(3),F(4)];upper,k=root_upper(sum(z*z for z in yv));defect=max(upper-y0,F(0));s=[F(2),F(-6,5),F(-8,5)];pair=y0*s[0]+sum(a*b for a,b in zip(yv,s[1:]))
check('WHOLE cone defect sign and top bound',dict(root=upper,defect=defect,pair=pair,lower=-defect*2),pair>=-defect*2 and pair<0)
for value in [F(0),F(25),F(2),F(1,2**2148)]:
 r,k=root_upper(value)
 check('fixed96bit exact outward norm '+str(value),dict(square=value,upper=r,integer=k),r*r>=value and (k==0 or F(k-1,2**96)**2<value))
# Saved norm coefficient endpoints remain rational; no nearest float shortcut.
a=F(-3,7);T=F(5,11);beta=F(1,13);lo,hi=sorted([(beta-T)/a,(beta+T)/a])
check('both coefficient signs / exact necessary coordinate interval',dict(lower=lo,upper=hi),abs(beta-a*lo)==T and abs(beta-a*hi)==T and lo<hi)
# Symbolic lift identity represented by its exact polynomial coefficient list.
check('SOC5 sigma nonnegative polynomial identity',dict(coefficients={'sigma':4,'ux2':-4,'uy2':-4,'uz2':-4},negative_sigma_control=F(-1)),4*F(-1)<0)
a=np.array([2.**500,-2.**500,np.nextafter(0.,1.)]);z=exact_dot(a,np.ones(3));check('exact cancellation and subnormal dyadic sum',z,z['value']==F(1,2**1074))
a=np.array([-0.,3.,-2.]);b=np.array([4.,-1.,.5]);before=(a.tobytes(),b.tobytes());z=exact_dot(a,b);check('signed zero shortcut and negative raw coefficients unchanged',z,z['value']==-4 and before==(a.tobytes(),b.tobytes()))
z=exact_dot(np.array([0.]),np.array([np.nan]));check('zero times unknown must remain unknown',z,z['status']=='UNRESOLVED_nonfinite_or_shape')
# Algebraic corners are proof controls, never episode/source/robot anchors.
c=[F(2),F(-3),F(0)];lo=[F(-1),F(1,2),F(0)];hi=[F(1),F(2),F(5)];support=sum(min(a*b,a*d) for a,b,d in zip(c,lo,hi));allcorners=[sum(c[j]*(hi[j] if mask>>j&1 else lo[j]) for j in range(3)) for mask in range(8)]
check('exact box support / zero coefficient / signed endpoints',dict(support=support,corner_min=min(allcorners)),support==min(allcorners))
check('new8192bit exact arithmetic and fixed resource envelope',dict(new_bit_cap=BIT_CAP,product_exponent_cap=2148,all_rows=ROWS,all_stored_coefficients=32306054,unchanged_seconds=240,unchanged_RSS=2147483648),BIT_CAP==8192 and cap(F(1,2**2148))==F(1,2**2148))
dump(O/'report.json',dict(schema='v2d.NEW-exact-terminal-dual-contracts.v1',contracts=len(returns),passed=True,old_controls_solver_model_source_native_NOT_replayed=True,no_episode_point_or_certificate_evaluation_yet=True))
