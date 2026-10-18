"""Bind existing private crop requests to all canonical hand source hashes.

No decoding, detector, fitter, renderer, GPU job or model inference is repeated.
"""
from pathlib import Path
import hashlib,json,sys,time
S=Path(__file__).resolve().parents[1];R=Path('/Users/rohithiruku/v2d_hand_continuity_fix');sys.path.insert(0,str(R/'scripts/track3'))
from recover_source_hand import validate_request,canonical_sources,digest
O=S/'e037-fresh-source-hand-request-v2';O.mkdir();old=S/'e037-fresh-source-hand-request-v1/request.json';data=json.load(open(old));B=Path(data['canonical_bundle']);manifest=B.parent/'manifest.json';data.update(canonical_manifest_sha256=digest(manifest),parent_request_sha256=digest(old),request_revision=2)
p=O/'request.json';p.write_text(json.dumps(data,indent=2)+'\n');checked=validate_request(p);sources=canonical_sources(checked)
report=dict(schema='v2d.bound-source-hand-recovery-request.v2',producer_sha256=digest(Path(__file__)),recovery_code_sha256=digest(R/'scripts/track3/recover_source_hand.py'),merge_code_sha256=digest(R/'scripts/track3/merge_recovered_source_hand.py'),request_sha256=digest(p),parent_request_sha256=digest(old),canonical_manifest_sha256=digest(manifest),canonical_hand_source_count=len(sources),selected_frames=[r['frame_idx'] for r in data['frames']],canonical_hand_source_sha256={str(v[0].relative_to(S)):digest(v[0]) for v in sources.values()},original_request_or_source_files_changed=False,source_images_reused_without_decoding=True,GPU_inference=False,reference_accepted=False,training_approved=False,required_reference_contact_tolerance_m=.001)
(O/'binding-receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='canonical_hand_source_sha256'},indent=2))
