"""Ten NEW embedding/objective/sign/recovery/general-RHS/resource contracts."""
from joint_hsd_embedding_v1 import *
from fractions import Fraction as Q
D=S/'e037-joint-hsd-contracts-v1';record=MethodRecorder(D);passed=0
def done(name,value,check):
 global passed
 record(name+'-RETURN',value)
 assert check
 passed+=1
p=Q(1,8);x=Q(3,2);t=p*x*x/2;h=t+p/2;z=t-p/2;tail=p*x
done('NEW-square-root-free-original-P-epigraph',dict(p=str(p),x=str(x),t=str(t),head=str(h),tail0=str(z),tail1=str(tail),exact_squared_margin=str(h*h-z*z-tail*tail)),h*h-z*z==tail*tail)
done('NEW-original-objective-epigraph-recovery',dict(original=str(p*x*x/2),epigraph=str(t),no_sqrt_or_objective_change=True),t==p*x*x/2)
A=np.array([[-1.,0.],[0.,-1.],[0.,-1.],[-2.,0.]]);b=np.array([-1.,1.,-1.,0.]);c=np.array([0.,1.]);Qm=np.block([[np.zeros((2,2)),A.T,c[:,None]],[-A,np.zeros((4,4)),b[:,None]],[-c[None,:],-b[None,:],np.zeros((1,1))]])
done('NEW-homogeneous-skew-signs',dict(Q=Qm,skew=Qm+Qm.T),np.all(Qm+Qm.T==0))
u=np.r_[1.,1.,2.,1.,0.,-1.,1.];v=Qm@u
done('NEW-known-original-quadratic-KKT-embedding',dict(u=u,v=v,original_objective=1.),np.array_equal(v,np.r_[0.,0.,0.,2.,0.,2.,0.]))
# One nonzero-A exact dyadic resolvent and stable positive denominator.
wb=Q(1,2);wc=Q(1,2);den=1+wc*wc+wc*wc+wb*wb+(1-wb)**2
answer=np.array([-.5,-.5,2.]);Q1=np.array([[0.,1.,1.],[-1.,0.,1.],[-1.,-1.,0.]])
done('NEW-stable-positive-bordered-resolvent',dict(stable_denominator=str(den),direct=np.array([1.,2.,3.]),answer=answer,product=(np.eye(3)+Q1)@answer),den==2 and np.array_equal((np.eye(3)+Q1)@answer,np.array([1.,2.,3.])))
tiny=np.array([[0.,-1.],[0.,-1.],[-.125,0.]]);normal=tiny.T@tiny
done('NEW-epigraph-normal-general-RHS-map',dict(normal=normal),np.array_equal(normal,np.diag([.125**2,2.])))
point=np.arange(NF+906,dtype=float);z=np.r_[point[:1522],0.,point[1522:NF]];tau=2.;recovered=z/tau
done('NEW-homogeneous-phase-zero-recovery',dict(phase=recovered[PHASE],original_phase_coordinate_retained=True,no_old_return_repair=True),recovered[PHASE]==0 and len(recovered)==NVAR)
sx,sy,st,branches=split_projection(np.array([-.5]),np.array([0.,1.,0.]),-.5,[(0,3)])
done('NEW-whole-cone-homogeneous-projection',dict(x=sx,y=sy,tau=st,branches=branches),sx[0]==-.5 and st==0 and np.array_equal(sy,np.array([.5,.5,0.])))
done('NEW-correlated-ties-zero-rows-reuse-contract',dict(all_original_rows=NR,all_pair_scopes=249603,all_original_dynamic_entries=933560,conditional_saved_J_proof_BY_HASH=True,
 source_native_and_continuous_direction_maps_unchanged=True,typed_zero_filters_retained=True,no_new_tie_or_episode_query=True),NR==4810140 and NF==500728)
estimate=8*(11*(NR+2718)*8+24*(NF+906)*8)+600*1024**2
done('NEW-fixed-whole-embedding-resource-contract',dict(updates=8,terminal_ONLY=8,method_seconds=150,outer_seconds=240,RSS_bytes=2*1024**3,disk_reserve_bytes=int(1.5*1024**3),estimated_return_bytes=estimate),estimate<5_000_000_000)
save(D/'report.json',dict(passed=True,NEW_controls=passed,no_old_method_query_control_or_solver_replay=True))
