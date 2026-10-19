"""Compare production surface semantics against frozen private rays; no fit or label changes."""
from pathlib import Path
import datetime,hashlib,json,sys
S=Path(__file__).resolve().parents[2];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
import numpy as np
from source_hand_depth import compare_estimated_surface
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();O=S/'e037-source-surface-operator-verification-v1';O.mkdir()
fit=S/'e037-native-RGB-landmark-adapter-fit-v1/source-hand-candidate.npz';B=S/'scored-cohort-14-results/reconstruction-e037/bundles/e037/episode.npz'
with np.load(B) as z:camera=z['camera_c2w'][50].copy();K=z['intrinsics'].copy();faces=z['hand_right_faces'].copy()
with np.load(fit) as z:v=np.einsum('ni,ij->nj',z['vertices_world'][0]-camera[:3,3],camera[:3,:3]);pixels=z['heldout_pixels'].copy();depth=z['heldout_estimated_source_depth_m'].copy();old=z['heldout_front_depth_m'].copy()
result=compare_estimated_surface(v,faces,pixels,K,depth,np.ones(len(depth),bool),evidence_kind='estimated_visible_surface_depth');finite=np.isfinite(old);assert np.array_equal(result.model_hit,finite);error=float(abs(result.predicted_depth_m[finite]-old[finite]).max());assert error<1e-12;assert np.isinf(result.residual_m[~result.model_hit]).all()
report=dict(schema='v2d.source-surface-operator-private-verification.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_sha256=sha(Path(__file__)),operator_sha256=sha(R/'scripts/track3/source_hand_depth.py'),source_candidate_sha256=sha(fit),bundle_sha256=sha(B),source_observation_kind='estimated_visible_surface_depth',old_independent_ray_operator_closure_max_m=error,expected_rays=len(pixels),unchanged_model_hits=int(result.model_hit.sum()),miss_residuals_retained_as_infinite=True,complete_coverage_passed=bool(result.model_hit.all()),old_heldout_failures_or_flags_changed=False,old_failed_candidate_promoted=False,native_reference_contact_gate_m=.001,new_CPU_fit_or_GPU_inference=False,physical_task_accepted=False)
(O/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
