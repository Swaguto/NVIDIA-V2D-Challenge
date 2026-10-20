"""Independent scalar row builder for EACH new step model before its solver."""
import numpy as np
from scipy import sparse

def independently_bind(problem,state,data,config):
    delta=np.asarray(state['world_delta']);nf=len(delta);nv=3*nf;n=nv+1
    rows=[];cols=[];values=[];b=[]
    def add(terms,rhs):
        row=len(b)
        for j,v in terms:
            if v!=0:rows.append(row);cols.append(j);values.append(float(v))
        b.append(float(rhs))
    H=config['world_box_halfwidth_m'];h=config['step_halfwidth_m']
    for j in range(n):add([(j,1.)],min(h,H-delta.ravel()[j])if j<nv else config['phase_slack_upper_m'])
    for j in range(n):add([(j,-1.)],min(h,H+delta.ravel()[j])if j<nv else 0.)
    for k,f in enumerate(data['pair_frames']):
        add([(3*int(f)+j,-state['active_normals'][k,j])for j in range(3)]+[(nv,-1.)],state['gap_proxy'][k]+config['original_native_proxy_threshold_m'])
    for f in range(nf):add([(3*f+j,state['median_gradient'][f,j])for j in range(3)],data['baseline_median'][f]-state['median'][f])
    for f in range(nf):
        rz=data['Rcw'][f,2]
        depth=float(data['minimum_C0_z'][f])+sum(float(rz[j])*float(delta[f,j])for j in range(3))
        add([(3*f+j,-rz[j])for j in range(3)],depth-.01)
    M=data['motion_M'].tocsr();base=np.asarray(data['motion_base']).ravel();motion_start=len(b)
    for k,cap in enumerate(data['motion_caps']):
        add([],cap)
        for axis in range(3):
            r=3*k+axis;sl=slice(M.indptr[r],M.indptr[r+1]);js=M.indices[sl];vs=M.data[sl]
            cur=base[r]+sum(float(v)*float(delta.ravel()[j])for j,v in zip(js,vs,strict=True))
            add([(int(j),-v)for j,v in zip(js,vs,strict=True)],cur)
    expected=sparse.coo_matrix((values,(rows,cols)),shape=(len(b),n)).tocsc();eb=np.array(b)
    coef=(problem['raw_A']-expected).tocoo();rhs=problem['raw_b']-eb
    scaled=(problem['A']-sparse.diags(1./problem['scales'])@expected).tocoo()
    er=np.asarray(abs(expected).max(axis=1).toarray()).ravel();scale=np.maximum(1.,np.maximum(er,abs(eb)))
    for k in range(len(data['motion_caps'])):
        p=motion_start+4*k;scale[p:p+4]=max(scale[p:p+4])
    ds=problem['scales']-scale
    ep=np.r_[np.full(nv,config['quadratic_step_regularization']),0.]
    obj=np.r_[problem['P'].diagonal()-ep,problem['q']-np.r_[np.zeros(nv),1.]]
    diagnostics=dict(rows=len(b),columns=n,raw_coefficients=expected.nnz,
     raw_A_error=float(max(abs(coef.data),default=0)),raw_b_error=float(max(abs(rhs),default=0)),
     scaled_A_error=float(max(abs(scaled.data),default=0)),scale_error=float(max(abs(ds),default=0)),
     objective_error=float(max(abs(obj),default=0)),minimum_positive_scale=float(min(problem['scales'])),
     rows_and_columns_match=(problem['raw_A'].shape==expected.shape),
     diagnostic_bound_NOT_candidate_tolerance=1e-12,
     residuals=dict(raw_b=rhs,raw_A_row=coef.row,raw_A_col=coef.col,raw_A_value=coef.data,
      scaled_A_row=scaled.row,scaled_A_col=scaled.col,scaled_A_value=scaled.data,scale=ds,objective=obj))
    diagnostics['passed']=bool(diagnostics['rows_and_columns_match']and diagnostics['minimum_positive_scale']>=1. and all(diagnostics[k]<=1e-12 for k in ['raw_A_error','raw_b_error','scaled_A_error','scale_error','objective_error']))
    return diagnostics
