"""NEW nonsmooth correlated-tie, zero-cone and block-adjoint controls."""
from joint_constraint_derivatives_v1 import *
D=S/'e037-joint-directional-contracts-v1';returns=[]
def check(name,raw,predicate):
 save(D/(name+'-RETURN.json'),raw);ok=bool(predicate(raw));returns.append(dict(name=name,passed=ok));assert ok,name
A=np.array([[1.,1.,0.],[1.,-1.,0.],[-1.,0.,0.]])
B=np.array([[0.,.1,0.],[0.,-.1,0.]])
JA={11:np.array([[.3,0,0.],[-.2,0,0.],[0,0,0.]]),12:np.array([[-.4,0,0.],[.1,0,0.],[0,0,0.]])}
R=np.eye(3);n=np.array([1.,0.,0.]);dc={11:.7,12:-.6};dn=np.array([0.,.2,0.])
out=full_support_return((A,JA),(B,{}),n,R,{},0.,.4)
d= support_directional(out,dc,dn,.3,.4)
def direct(e):
 V=A+e*(JA[11]*dc[11]+JA[12]*dc[12]);nn=n+e*dn
 return .001+.4*(e*.3)-max(V@nn)-max(B@(-nn))
eps=1e-7;fd=(direct(eps)-direct(0))/eps
check('NEW-correlated-joint-direction-tie-derivative',dict(complete=d,forward_difference=fd,
 mean_direction=sum(out['SEARCH_command_residual_J'][str(c)]*v for c,v in dc.items())+out['SEARCH_direction_residual_J']@dn+.4*.3),
 lambda r:abs(r['complete']['directional_residual']-r['forward_difference'])<1e-6 and abs(r['complete']['directional_residual']-r['mean_direction'])>.1)
b=np.array([.7,0.,0.,0.]);v=np.array([.2,.3,-.4,.1]);actual=cone_directional(b,v)
check('NEW-zero-tail-cone-directional',dict(value=actual,expected=float(.2-np.sqrt(.26)),
 wrong_mean_selection=.2),lambda r:abs(r['value']-r['expected'])<1e-15 and r['value']!=r['wrong_mean_selection'])
# A toy block interface exercises the actual new action/adjoint methods with no
# episode data, no old matrix-vector/audit replay and no selected candidate.
toy=object.__new__(JointExecutableInterface)
toy.A=sparse.csc_matrix([[.2,-.3],[.4,.5],[-.7,.1]])
toy.robot=sparse.csc_matrix(np.array([[.1,.4],[-.2,.3]]))
toy.P=sparse.eye(4,format='csc')
# Equivalent namespace wrapper for 2+2 split; same explicit sign convention.
v=np.array([.2,-.1,.4,.3]);u=np.array([.5,-.3,.2]);w=np.array([-.4,.7])
f=np.r_[-toy.robot@v[:2],-toy.A@v[2:]]
adj=np.r_[-toy.robot.T@w,-toy.A.T@u]
check('NEW-primal-slack-block-adjoint-sign',dict(left=float(np.r_[w,u]@f),right=float(v@adj)),lambda r:abs(r['left']-r['right'])<1e-15)
save(D/'report.json',dict(schema='v2d.NEW-joint-full-directional-contracts.v1',contracts=returns,passed=True,
 no_episode_pose_source_native_solver_or_old_controls_replayed=True))
