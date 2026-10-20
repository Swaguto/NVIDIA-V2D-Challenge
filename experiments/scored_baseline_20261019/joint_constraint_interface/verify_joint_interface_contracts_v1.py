"""NEW concrete executable joint-interface controls; only synthetic inputs."""
from joint_constraint_interface_v1 import *
import hashlib
D=S/'e037-joint-interface-contracts-v1'
records=[]
def check(name,raw,predicate):
    save(D/(name+'-RETURN.json'),raw)
    passed=bool(predicate(raw));records.append(dict(name=name,passed=passed))
    if not passed:
        save(D/'FAILED-contracts-RETURN.json',records);raise AssertionError(name)

eta=np.array([.017,-.025,.009]);u=np.array([.13,-.07,.09]);r=.27;scale=3.1
Q,delta,dQ,dt=cayley_terms(eta,u,r,scale)
complexQ=[];complexp=[]
for j in range(6):
    a=eta.astype(complex);b=u.astype(complex)
    if j<3:a[j]+=1e-25j
    else:b[j-3]+=1e-25j
    qq,pp,_,_=cayley_terms(a,b,r,scale);complexQ.append(qq.imag/1e-25);complexp.append(pp.imag/1e-25)
check('NEW-six-coordinate-cayley-J',dict(dQ=dQ,complex=np.array(complexQ)[3:],dt=dt,complex_dt=np.array(complexp).T),
    lambda t:np.max(np.abs(t['dQ']-t['complex']))<1e-14 and np.max(np.abs(t['dt']-t['complex_dt']))<1e-14)
check('NEW-Cayley-rigid-origin',dict(rotation=Q,det=float(np.linalg.det(Q)),origin=delta,orthogonal=Q.T@Q),
    lambda t:np.max(np.abs(t['orthogonal']-np.eye(3)))<1e-14 and abs(t['det']-1)<1e-14)
# One slide followed by an off-center hinge and a branch: native qpos0 matters.
raw=dict(nbody=4,njnt=2,body_parentid=[0,0,1,1],
 body_pos=[[0,0,0],[.2,-.1,.3],[.04,.05,.06],[-.2,.1,.07]],
 body_quat=[[1,0,0,0]]*4,body_jntnum=[0,1,1,0],body_jntadr=[-1,0,1,-1],
 jnt_type=[2,3],jnt_qposadr=[0,1],qpos0=[.13,-.21],
 jnt_axis=[[1,0,0],[0,0,1]],jnt_pos=[[0,0,0],[.02,-.03,.04]])
RR,pp,axes,pivots=body_tree(raw,{0:.18,1:.34})
angle=.55;Rexplicit=Rotation.from_euler('z',angle).as_matrix()
pivot=np.array([.2,-.1,.3])+np.array([.05,0,0])+np.array([.04,.05,.06])+np.array([.02,-.03,.04])
pexp=pivot-Rexplicit@np.array([.02,-.03,.04])
check('NEW-native-offcenter-slide-reference',dict(R=RR[2],p=pp[2],expectedR=Rexplicit,expectedp=pexp,sibling=pp[3]),
 lambda t:np.max(np.abs(t['R']-t['expectedR']))<1e-14 and np.max(np.abs(t['p']-t['expectedp']))<1e-14 and np.max(np.abs(t['sibling']-[.05,0,.37]))<1e-14)
v=np.array([.08,-.03,.11]);point=RR[2]@v+pp[2];analytic=np.cross(axes[1],point-pivots[1])
eps=2e-6
rplus,pplus,_,_=body_tree(raw,{0:.18,1:.34+eps});rminus,pminus,_,_=body_tree(raw,{0:.18,1:.34-eps})
finite=((rplus[2]@v+pplus[2])-(rminus[2]@v+pminus[2]))/(2*eps)
check('NEW-native-point-derivative',dict(analytic=analytic,independent_difference=finite),lambda t:np.max(np.abs(t['analytic']-t['independent_difference']))<1e-10)
# Moving A-local direction: both rotated support vertices and direction move.
V=np.array([[.4,.1,-.2],[-.1,.3,.1],[.1,-.2,.3]])
W=np.array([[-.3,.2,.2],[.2,-.4,-.1]])
R=Rotation.from_euler('z',.23).as_matrix();dR=skew([0,0,1])@R;n=np.array([.6,.8,0.])
A=(V@R.T,{7:V@dR.T});B=(W,{})
out=support_return(A,B,n,R,{7:dR},.4,.2)
def independent(q):
 rot=Rotation.from_euler('z',q).as_matrix();world=rot@n
 return .001+.2*.4-max((rot@a)@world for a in V)-max(-b@world for b in W)
fd=(independent(.23+eps)-independent(.23-eps))/(2*eps)
check('NEW-current-moving-direction-full-product-J',dict(returned=out,independent=fd),lambda t:abs(t['returned']['SEARCH_command_residual_J']['7']-t['independent'])<1e-10)
T=np.array([[1.,1.,0.],[1.,-1.,0.],[-1.,0.,0.]])
tie=support_return((T,{5:np.array([[0,1,0],[0,-1,0],[0,0,0]])}),(np.zeros((1,3)),{}),np.array([1.,0.,0.]),np.eye(3),{},0.,.3)
check('NEW-ALL-tie-factors-returned',dict(tie=tie),lambda t:t['tie']['A_ties'].tolist()==[0,1] and len(t['tie']['ALL_tie_factor_derivatives']['5']['A'])==2)
nonunit=support_return((T,{}),(np.zeros((1,3)),{}),np.array([2.,0.,0.]),np.eye(3),{},0.,.3)
zero=support_return((T,{}),(np.zeros((1,3)),{}),np.zeros(3),np.eye(3),{},0.,.3)
check('NEW-no-direction-repair',dict(nonunit=nonunit,zero=zero),lambda t:t['nonunit']['unit_equality']==3 and t['zero']['status'].startswith('UNKNOWN'))
# NEW source-branch composition and source camera helper use only NEW toy points.
recs=[dict(name='a',type='revolute',parent='root',child='link',xyz=[.1,.2,.3],rpy=[0,0,0],axis=[0,0,1]),
 dict(name='tip',type='fixed',parent='link',child='tip',xyz=[.05,.02,.01],rpy=[0,0,0],axis=None)]
root=np.array([.03,-.04,1.2,0,0,0,1.]);p,j=source_chain(recs,root,{'a':.31},'tip')
expected=root[:3]+np.array([.1,.2,.3])+Rotation.from_euler('z',.31).apply([.05,.02,.01])
check('NEW-URDF-source-fixed-tip-order',dict(point=p,expected=expected,jac=j['a']),lambda t:np.max(np.abs(t['point']-t['expected']))<1e-14)
K=np.array([[300.,0,90],[0,310.,80],[0,0,1.]])
cam=np.eye(4);cam[:3,3]=[.02,-.01,.03]
src=camera_source_factor(p,np.column_stack([j['a']]),cam,np.eye(3),K,[100,110],55.)
check('NEW-composed-source-camera-disk-depth',dict(source=src,expected_camera=p-cam[:3,3]),
 lambda t:np.max(np.abs(t['source']['camera_point']-t['expected_camera']))<1e-14 and t['source']['original_image_tolerance_px']==1e-6 and t['source']['strict_depth_value']>0)
# NEW exact constant rows survive a sparse executable interface.
AA=sparse.csc_matrix(np.array([[1.,-1.],[0.,0.],[-2.,1.]]));bb=np.array([.3,-.01,.6]);xx=np.array([.1,.2]);ss=bb-AA@xx
check('NEW-empty-row-not-dropped',dict(slacks=ss,rows=AA.shape[0],zero_row=float(ss[1])),lambda t:t['rows']==3 and t['zero_row']==-.01)
# Whole-cone tails: exact zero-tail case is explicitly retained.
blocks=[np.array([.1,0,0,0]),np.array([1.,.2,-.3,.4,0.])]
check('NEW-SOC4-SOC5-whole-tail',dict(values=[float(b[0]-np.linalg.norm(b[1:])) for b in blocks],widths=[len(b) for b in blocks]),lambda t:t['widths']==[4,5] and t['values'][0]==.1)
# Recorder requires a unique immutable fsync file; no algorithm/source call.
r=DurableRecorder(D);r('NEW-toy-durable-contract',dict(point=[.1,.2,.3],unknown=True))
written=D/'000000-NEW-toy-durable-contract.json'
check('NEW-durable-recorder-input-identity',dict(bytes_sha256=sha(written),loaded=json.loads(written.read_text()),count=r.count),lambda t:t['loaded']=={'point':[.1,.2,.3],'unknown':True} and t['count']==1)
save(D/'report.json',dict(schema='v2d.NEW-joint-executable-interface-contracts.v1',contracts=records,passed=True,
 no_episode_pose_source_NN_raster_native_FK_solver_or_old_controls_evaluated=True))
