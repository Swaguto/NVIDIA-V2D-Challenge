"""Repeat frozen refinement using relocatable inputs and separately supplied MANO."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main(args):
    kit=args.kit.resolve();out=args.out.resolve();source=args.source.resolve()
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((kit/'artifact-manifest.json').read_text())
    for name,record in manifest['files'].items():
        if sha(kit/name)!=record['sha256']:raise ValueError(f'Changed kit file: {name}')
    reference=json.loads((kit/'refinement-reference/run-receipt.json').read_text())
    inputs=args.inputs.resolve() if args.inputs else kit/'perception-checkpoint'
    if args.inputs:
        for name,digest in json.loads((inputs/'input-files.json').read_text()).items():
            if sha(inputs/name)!=digest:raise ValueError('Changed experiment input: '+name)
        if json.loads((inputs/'inputs.json').read_text())['episode']!=reference['episode']:
            raise ValueError('Experiment episode differs')
    module=source/'v2d/gsplat_refinement/lib'
    for name,digest in reference['source_sha256'].items():
        if sha(module/name)!=digest:raise ValueError(f'Changed optimizer source: {name}')
    source_manifest=json.loads(args.source_manifest.read_text(encoding='utf-8-sig'))
    for name,digest in source_manifest.items():
        if sha(source/name)!=digest:raise ValueError(f'Changed dependency source: {name}')
    asset_manifest=json.loads(args.asset_manifest.read_text(encoding='utf-8-sig'))
    for name,record in asset_manifest.items():
        if sha(args.mano_assets/name)!=record['sha256']:raise ValueError(f'MANO asset differs: {name}')
    kwargs=reference['arguments'].copy()
    old_inputs=str(Path(kwargs['frames_dir']).parent)
    old_out=str(Path(kwargs['scene_export_path']).parent)
    for key,value in kwargs.items():
        if isinstance(value,str):
            if value.startswith(old_inputs+'/'):kwargs[key]=str(inputs/value[len(old_inputs)+1:])
            elif value.startswith(old_out+'/'):kwargs[key]=str(out/value[len(old_out)+1:])
    kwargs['mano_assets_root']=str(args.mano_assets.resolve())
    if not args.mano_assets.is_dir():raise FileNotFoundError(args.mano_assets)
    if args.check_only:
        print(json.dumps({'status':'inputs_and_sources_verified','episode':reference['episode'],
            'kit_files':len(manifest['files']),'source_files':len(source_manifest),
            'model_files':len(asset_manifest)}));return
    # Preserve the recorded configuration, including seed and full frame sequence.
    sys.path.insert(0,str(source))
    from v2d.gsplat_refinement.lib.refine_simple import refine_simple
    import numpy as np
    import torch
    import gsplat
    out.mkdir(parents=True)
    receipt={'status':'running','episode':reference['episode'],'arguments':kwargs,
        'experimental_inputs':str(inputs) if args.inputs else None,
        'experimental_input_lock_sha256':sha(inputs/'input-files.json') if args.inputs else None,
        'source_sha256':{str(p.relative_to(source)):sha(p) for p in source.rglob('*.py')},
        'asset_sha256':{name:record['sha256'] for name,record in asset_manifest.items()},
        'torch':torch.__version__,'gsplat':gsplat.__version__,
        'gpu':torch.cuda.get_device_name(0),'fresh_perception_inference':False,
        'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def save():(out/'run-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    save()
    try:
        refine_simple(**kwargs)
        with np.load(out/'scene.npz') as actual,np.load(kit/'refinement-reference/scene.npz') as expected:
            if not np.array_equal(actual['frame_indices'],expected['frame_indices']):
                raise ValueError('Refinement frame coverage differs')
        receipt.update(status='completed',scene_sha256=sha(out/'scene.npz'))
    except Exception as exc:
        receipt.update(status='failed',error=repr(exc));raise
    finally:
        receipt['ended_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--inputs',type=Path,help='Separate hash-locked experimental inputs; creates a new experiment')
    for name in ('kit','out','source','mano-assets','source-manifest','asset-manifest'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
