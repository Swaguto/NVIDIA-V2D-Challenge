"""NEW numerical precondition for full product projection; no gate change."""
from whole_row_conic_split_v1 import *
def guarded_projection(v,nn,cones):
 v=np.asarray(v,float)
 if not np.isfinite(v).all():return dict(status='UNRESOLVED_nonfinite',value=v.copy(),branches=[])
 # Every SOC has <=5 coordinates. Four squared components <=2^1002,
 # comfortably finite in binary64. This is a stop guard, not an acceptance cap.
 if np.max(abs(v),initial=0)>float(2**500):return dict(status='UNRESOLVED_projection_norm_range',value=v.copy(),branches=[])
 return project_product(v,nn,cones)
