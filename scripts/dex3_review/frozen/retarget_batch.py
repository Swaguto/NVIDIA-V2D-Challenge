"""Bounded private Dex3 preview batch; each source bundle remains immutable."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

R = Path('/workspace/hand-continuity-repair/dex3-isaac-review')
PY = R / 'ik-runtime/bin/python'
CODE = R / 'retarget-code/robotic_grounding'
EPISODES = [2, 11, 12, 13, 18, 21, 23, 24, 31, 40, 41, 42]
deadline = datetime.datetime.fromisoformat('2026-10-15T05:45:00+00:00').timestamp()
env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
status = {}
while len(status) < len(EPISODES) and time.time() < deadline:
    progress = False
    for episode in EPISODES:
        label = f'e{episode:03d}'
        if label in status or not (R / 'episodes' / label / 'bundle/validation.json').is_file():
            continue
        import_result = subprocess.run([str(PY), '-c', 'import pinocchio,pink,viser,torch;print("IK_RUNTIME_READY")'], env=env, capture_output=True, text=True)
        if import_result.returncode:
            (R / 'ik-import-check.log').write_text(import_result.stdout + import_result.stderr)
            time.sleep(5)
            break
        out = R / 'episodes' / label / 'retarget'
        print('RETARGET_START', label, flush=True)
        with (R / 'episodes' / label / 'retarget.log').open('w') as log:
            try:
                result = subprocess.run([str(PY), str(CODE / 'scripts/retarget/track3_to_dex3.py'), '--bundle', str(R / 'episodes' / label / 'bundle'), '--out', str(out), '--max-iter', '100'], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time()))
                rc = result.returncode
            except subprocess.TimeoutExpired:
                rc = 124
        status[label] = {'exit_code': rc, 'retarget': str(out)}
        (R / 'retarget-status.json').write_text(json.dumps(status, indent=2))
        print('RETARGET_FINISH', label, rc, flush=True)
        progress = True
        if rc:
            raise RuntimeError('Retarget failed; preserve diagnostics and inspect ' + label)
    if not progress:
        time.sleep(5)
print('RETARGET_BATCH_END', len(status), flush=True)
