from joint_mean_independent_integer_v1 import *
from joint_saved_mean_bound_v1 import S,save
from joint_lossless_returns_v1 import seal_return
import numpy as np
D=S/'e037-joint-mean-proof-integer-controls-v1'
values=[0.,-0.,.1,-.3,np.nextafter(0.,1.),np.finfo(float).max]
returned=[integer1074(x) for x in values];expected=[Q(float(x))/T for x in values]
seal_return(D,'NEW-bit-decoder-control-RETURN',dict(equal=all(Q(x)==y for x,y in zip(returned,expected)),signed_zero_raw_payloads=[struct.pack('>d',x).hex() for x in values[:2]],cases=len(values)))
assert all(Q(x)==y for x,y in zip(returned,expected))
a=[integer1074(x) for x in [.1,-.2,.3]];b=[integer1074(x) for x in [.5,-.7,.2]];v,errors=exact_dot_side([a],b)
qsum=sum((Q(float(x))*Q(float(y)) for x,y in zip([.1,-.2,.3],[.5,-.7,.2])),Q(0))
seal_return(D,'NEW-independent-product-accumulator-control-RETURN',dict(exact=str(v[0]*S2),expected=str(qsum),error=str(errors[0])))
assert v[0]*S2==qsum and errors[0]>0
save(D/'report.json',dict(passed=True,NEW_controls=2,no_old_controls_mean_matmul_or_certificate_replay=True))
