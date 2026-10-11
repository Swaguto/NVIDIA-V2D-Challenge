"""Run the same automatic method across episodes, loading each GPU model once.

The model cache changes loading cost only. Each episode retains independent
wearer association, frame checks, receipts and quality outputs.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import automatic_hands
import automatic_hand_quality


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    for name in ('cache-root','out','detector','hamer-source','hamer-weights','sam-checkpoint'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('automatic_hands.json'))
    p.add_argument('--freeze',type=Path)
    a=p.parse_args();policy=json.loads(a.config.read_text())
    if len(a.episodes)!=len(set(a.episodes)):raise ValueError('Duplicate episode')
    if set(a.episodes)&set(policy['holdout_episodes']) and a.freeze is None:
        raise ValueError('Holdouts require a frozen method before any data is read')
    if a.freeze and json.loads(a.freeze.read_text())['method_files']!=automatic_hands.method_hash(a.config):
        raise ValueError('Method differs from freeze')
    a.out.mkdir(parents=True,exist_ok=False)
    hashes=automatic_hands.method_hash(a.config)
    receipt={'episodes':a.episodes,'method_files':hashes,'freeze_sha256':automatic_hands.sha(a.freeze) if a.freeze else None,
             'status':'running','completed':[],'quality_accepted':False}

    def save():automatic_hands.write(a.out/'suite-receipt.json',receipt)

    sources={};depths={};arguments={}
    for episode in a.episodes:
        label=f'e{episode:03d}';root=a.cache_root/label
        sources[episode]=root/'issue6-full-inputs'/label
        depths[episode]=root/'issue6-full-runs/all20-final-v3'/label/'depth'
        arguments[episode]=['--source',str(sources[episode]),'--depth-root',str(depths[episode]),
            '--out',str(a.out/label/'proposals'),'--detector',str(a.detector),
            '--hamer-source',str(a.hamer_source),'--hamer-weights',str(a.hamer_weights),
            '--sam-checkpoint',str(a.sam_checkpoint),'--config',str(a.config)]
        if a.freeze:arguments[episode]+=['--freeze',str(a.freeze)]
    save()
    try:
        import torch
        from ultralytics import YOLO
        detector=YOLO(str(a.detector)).to('cuda')
        for episode in a.episodes:
            print(json.dumps({'episode':episode,'stage':'detect','status':'running'}),flush=True)
            automatic_hands.main(arguments[episode]+['--stage','detect'],{'detector':detector})
        del detector;gc.collect();torch.cuda.empty_cache()
        sys.path.insert(0,str(a.hamer_source))
        from hamer.models import load_hamer
        model,cfg=load_hamer(str(a.hamer_weights));model=model.to('cuda').eval()
        for episode in a.episodes:
            print(json.dumps({'episode':episode,'stage':'fit','status':'running'}),flush=True)
            automatic_hands.main(arguments[episode]+['--stage','fit'],{'hamer':(model,cfg)})
        del model,cfg;gc.collect();torch.cuda.empty_cache()
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_l.yaml',str(a.sam_checkpoint),device='cuda'))
        for episode in a.episodes:
            print(json.dumps({'episode':episode,'stage':'quality','status':'running'}),flush=True)
            root=a.out/f'e{episode:03d}'
            args=SimpleNamespace(source=sources[episode],depth_root=depths[episode],proposals=root/'proposals',
                                 out=root/'quality',sam_checkpoint=a.sam_checkpoint,config=a.config)
            automatic_hand_quality.run(args,predictor)
            receipt['completed'].append(episode);save()
        receipt['status']='completed'
        if hashes!=automatic_hands.method_hash(a.config):raise ValueError('Method changed during suite')
    except Exception as exc:
        receipt.update(status='failed',error=str(exc));raise
    finally:save()


if __name__=='__main__':main()
