"""NEW step proposal model for an actual hard-source-filtered construction.

Median gradients are SEARCH predictors (including chosen branches at ties),
never source feasibility certificates. IoU has no invented derivative here.
All normalized matrices only propose steps; actual source/motion still rule.
"""
import numpy as np
from scipy import sparse

CONFIG=dict(world_box_halfwidth_m=.02,step_halfwidth_m=.001,phase_slack_upper_m=.05,
 quadratic_step_regularization=1e-3,maximum_major_steps=8,maximum_trials=8,
 maximum_backtracks_per_major=3,trial_fractions=[1.,.5,.25],method_seconds=160.,
 QP_max_iterations=100,QP_seconds=10.,threads=1,default_QP_tolerances=True,
 original_native_proxy_threshold_m=.001,original_object_motion_comparison=1e-9)

def build_direction(state,data,config=CONFIG):
    delta=np.asarray(state['world_delta'],float);nf=len(delta);nv=3*nf;n=nv+1
    M=data['motion_M'].tocoo();caps=np.asarray(data['motion_caps'],float)
    nm=len(caps);frames=np.asarray(data['pair_frames'],int);ng=len(frames)
    grad=np.asarray(state['median_gradient'],float)
    upper=np.minimum(config['step_halfwidth_m'],config['world_box_halfwidth_m']-delta.ravel())
    lower=np.maximum(-config['step_halfwidth_m'],-config['world_box_halfwidth_m']-delta.ravel())
    rr=[np.arange(n),n+np.arange(n)];cc=[np.arange(n),np.arange(n)];vv=[np.ones(n),-np.ones(n)]
    rhs=[np.r_[upper,config['phase_slack_upper_m']],np.r_[-lower,0.]]
    row=2*n;blocks=dict(box=[0,2*n])
    rr.extend([row+np.repeat(np.arange(ng),3),row+np.arange(ng)])
    cc.extend([(3*frames[:,None]+np.arange(3)).ravel(),np.full(ng,nv)])
    vv.extend([-np.asarray(state['active_normals']).ravel(),-np.ones(ng)])
    rhs.append(np.asarray(state['gap_proxy'])+config['original_native_proxy_threshold_m']);blocks['geometry']=[row,row+ng];row+=ng
    rr.append(row+np.repeat(np.arange(nf),3));cc.append(np.arange(nv));vv.append(grad.ravel())
    rhs.append(np.asarray(data['baseline_median'])-np.asarray(state['median']));blocks['median_predictor']=[row,row+nf];row+=nf
    Rz=np.asarray(data['Rcw'])[:,2,:]
    depth=np.asarray(data['minimum_C0_z'])+np.einsum('ij,ij->i',Rz,delta,optimize=False)
    rr.append(row+np.repeat(np.arange(nf),3));cc.append(np.arange(nv));vv.append(-Rz.ravel())
    rhs.append(depth-.01);blocks['strict_depth_predictor']=[row,row+nf];row+=nf
    nn=row;blocks['motion_SOC4']=[row,row+4*nm]
    r=4*(M.row//3)+1+M.row%3
    rr.append(row+r);cc.append(M.col);vv.append(-M.data)
    current=np.asarray(data['motion_base']).ravel()+data['motion_M']@delta.ravel()
    rhs.append(np.column_stack([caps,current.reshape(nm,3)]).ravel());row+=4*nm
    raw_A=sparse.coo_matrix((np.concatenate(vv),(np.concatenate(rr),np.concatenate(cc))),shape=(row,n)).tocsc()
    raw_b=np.concatenate(rhs)
    # Positive conditioning factors scale a WHOLE SOC together. They are not
    # source weights, authority, tolerances or a normalization of raw returns.
    maxA=np.asarray(abs(raw_A).max(axis=1).toarray()).ravel()
    scales=np.maximum(1.,np.maximum(maxA,np.abs(raw_b)))
    for k in range(nm):
        a=nn+4*k;scales[a:a+4]=np.max(scales[a:a+4])
    A=sparse.diags(1./scales)@raw_A;b=raw_b/scales
    P=sparse.diags(np.r_[np.full(nv,config['quadratic_step_regularization']),0.],format='csc')
    q=np.r_[np.zeros(nv),1.]
    return dict(A=A.tocsc(),b=b,P=P,q=q,raw_A=raw_A,raw_b=raw_b,scales=scales,
     rows=row,columns=n,nonnegative_rows=nn,SOC4=nm,blocks=blocks,
     delta=delta,upper_step=upper,lower_step=lower,
     median_predictor_NOT_source_certificate=True,strict_depth_actual_POSTcheck_REQUIRED=True)

def actual_trial_decision(current_phase,trial_phase,source_returns,motion_checks,box_ok,
                          expected_frames,expected_proposal_id,nonzero_frames):
    # Each actual trial is sealed separately. Raw solver directions are neither
    # clipped nor repaired; a predeclared line fraction creates a NEW trial.
    ids=[x.get('frame')for x in source_returns]
    complete=(len(ids)==len(expected_frames)and len(set(ids))==len(ids)and set(ids)==set(expected_frames))
    bound=complete and all(x.get('proposal_id')==expected_proposal_id and
      x.get('status')==('NEW_training_source_predicate_ONLY'if x['frame']in nonzero_frames
                       else'CACHED_F_training_values_BY_HASH')for x in source_returns)
    source=bound and all(x.get('passes_TRAINING_construction')is True for x in source_returns)
    motion=(len(motion_checks)==6 and all(x is True for x in motion_checks.values()))
    finite=bool(np.isfinite([current_phase,trial_phase]).all())
    descend=finite and trial_phase<current_phase
    return dict(accepted=bool(source and motion and box_ok and descend),
     complete_actual_training_source=bool(complete),all_actual_training_source=bool(source),
     sealed_proposal_and_nonzero_cache_binding=bool(bound),
     all_six_original_motion=bool(motion),algorithmic_box_ok=bool(box_ok),strict_proxy_descent=bool(descend),
     source_numeric_returns_NOT_global_source_truth=True,
     original_fullpixel_native_self_swept_motor_grasp_NOT_qualified=True)
