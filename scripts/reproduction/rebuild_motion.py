"""Export and retarget a freshly refined episode using the frozen replay sources."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

def main(args):
    kit=args.kit.resolve();trial=args.refinement.resolve();out=args.out.resolve()
    if out.exists():raise FileExistsError(out)
    receipt=json.loads((trial/'run-receipt.json').read_text())
    if receipt['status']!='completed':raise ValueError('Refinement is incomplete')
    with (trial/'scene.npz').open('rb') as f:scene_digest=hashlib.file_digest(f,'sha256').hexdigest()
    if scene_digest!=receipt['scene_sha256']:raise ValueError('Refined scene differs from its completion receipt')
    inputs_root=args.inputs.resolve() if args.inputs else kit/'perception-checkpoint'
    expected_lock=receipt.get('experimental_input_lock_sha256')
    if expected_lock:
        if args.inputs is None:raise ValueError('This experiment requires its --inputs directory')
        lock=inputs_root/'input-files.json'
        with lock.open('rb') as f:actual_lock=hashlib.file_digest(f,'sha256').hexdigest()
        if actual_lock!=expected_lock:raise ValueError('Refinement input lock differs')
        for name,digest in json.loads(lock.read_text()).items():
            with (inputs_root/name).open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=digest:raise ValueError('Changed experiment input: '+name)
    elif args.inputs is not None:raise ValueError('Refinement has no experimental input lock')
    inputs=json.loads((inputs_root/'inputs.json').read_text())
    if inputs['episode']!=receipt['episode']:raise ValueError('Episode mismatch')
    code=kit/'source/robotic_grounding'
    reference=json.loads((kit/'bundle/manifest.json').read_text())
    commands=[
        [code/'source/robotic_grounding/robotic_grounding/retarget/track3_bundle.py',
         '--scene',trial/'scene.npz','--trial',trial,'--inputs',inputs_root/'inputs.json',
         '--out',out/'bundle','--source-commit',reference['source_commit'],'--episode',receipt['episode']],
        [code/'scripts/retarget/track3_to_dex3.py','--bundle',out/'bundle','--out',out/'retarget',
         '--max-iter','100','--mano-to-robot-scale','1.0'],
        [code/'scripts/retarget/check_track3_penetration.py','--bundle',out/'bundle',
         '--retarget',out/'retarget','--out',out/'gate']]
    out.mkdir(parents=True)
    report={'episode':receipt['episode'],'status':'running','commands':[]}
    try:
        for i,command in enumerate(commands):
            command=[sys.executable,*map(str,command)]
            report['commands'].append(command)
            with (out/f'stage-{i}.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            print(f'Stage {i} completed',flush=True)
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:(out/'rebuild.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,help='Hash-locked inputs used by an experimental refinement')
    for name in ('kit','refinement','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
