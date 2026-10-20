"""NEW derivatives of ORIGINAL world-relative angular clock functional."""
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation
from object_full_pose_maps_v1 import right_jacobian


def angular_clock_maps(base_rotations,timestamps):
 R=np.asarray(base_rotations,float);t=np.asarray(timestamps,float);dt=np.diff(t)
 if R.shape!=(len(t),3,3)or not np.isfinite(R).all()or not np.isfinite(t).all()or np.any(dt<=0):raise ValueError('all original finite ordered clocks required')
 rotations=Rotation.from_matrix(R);phi=(rotations[1:]*rotations[:-1].inv()).as_rotvec();omega=phi/dt[:,None];mid=(dt[:-1]+dt[1:])/2;acc=np.diff(omega,axis=0)/mid[:,None]
 rows=[];cols=[];vals=[];blocks=[];pi=[]
 for i,p in enumerate(phi):
  # R' from Rotation is the ACTUAL from_matrix interpretation used by the
  # original angular operator. Bases are rotation matrices to diagnostic error;
  # this derivative is a SEARCH map, never an original gate certificate.
  Ri=rotations[i+1].as_matrix();B=np.linalg.solve(right_jacobian(-p),Ri)/dt[i];blocks.append(B);pi.append(abs(np.linalg.norm(p)-np.pi)<=32*np.finfo(float).eps)
  for j in range(3):
   for k in range(3):
    rows.extend([3*i+j,3*i+j]);cols.extend([3*i+k,3*(i+1)+k]);vals.extend([-B[j,k],B[j,k]])
 J=sparse.coo_matrix((vals,(rows,cols)),shape=(3*len(dt),3*len(t))).tocsr()
 D=sparse.diags([-1/mid,1/mid],[0,1],shape=(len(mid),len(dt)));AJ=sparse.kron(D,sparse.eye(3),format='csr')@J
 return dict(principal_relative_rotvec=phi,angular_velocity=omega,angular_acceleration=acc,angular_velocity_J=J,angular_acceleration_J=AJ,clock_blocks=np.array(blocks),pi_branch_unknown=np.array(pi,bool),original_p95_must_use_actual_changed_rotations=True)
