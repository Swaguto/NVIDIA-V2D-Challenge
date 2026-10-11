"""Freeze tested development sources and model hashes before opening holdouts."""
import argparse
import datetime
import json
from pathlib import Path
from automatic_hands import method_hash,sha,write


def freeze(config,receipts,quality_reports,out,runtime_inventory=None,input_inventory=None):
    if out.exists():raise FileExistsError(out)
    policy=json.loads(config.read_text()); hashes=method_hash(config)
    if len(receipts)!=len(quality_reports) or not receipts:raise ValueError('Pair development runs with quality evidence')
    models=None; episodes=[]; evidence=[]
    keys=('detector_sha256','hamer_checkpoint_sha256','sam_checkpoint_sha256','mano_right_sha256',
          'hamer_config_sha256','mano_mean_params_sha256','hamer_source_commit')
    for path,quality_path in zip(receipts,quality_reports,strict=True):
        receipt=json.loads(path.read_text()); quality=json.loads(quality_path.read_text())
        episode=receipt['episode']
        if episode in policy['holdout_episodes']:raise ValueError('Cannot tune/freeze from held-out evidence')
        if receipt['method_files']!=hashes:raise ValueError('Development did not run current method')
        if receipt['status']!='completed_proposals_require_accuracy_checks':raise ValueError('Development run incomplete or source changed')
        if quality['episode']!=episode or quality['config_sha256']!=sha(config):raise ValueError('Quality evidence differs')
        if quality.get('method_files')!=hashes or quality.get('method_unchanged_during_run') is not True:
            raise ValueError('Quality source differs from tested method')
        if quality['source_proposal_receipt_sha256']!=sha(path):raise ValueError('Quality report refers to another proposal run')
        actual={k:receipt[k] for k in keys}
        if models is not None and actual!=models:raise ValueError('Development model profiles differ')
        models=actual;episodes.append(episode)
        evidence.append({'episode':episode,'proposal_receipt_sha256':sha(path),'quality_report_sha256':sha(quality_path)})
    write(out,{'schema':'v2d.automatic-hand-method-freeze.v1','frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'method_files':hashes,'models':models,'development_episodes':sorted(set(episodes)),
               'holdout_episodes':policy['holdout_episodes'],'evidence':evidence,
               'runtime_inventory_sha256':sha(runtime_inventory) if runtime_inventory else None,
               'input_inventory_sha256':sha(input_inventory) if input_inventory else None,
               'development_runtimes':[json.loads(path.read_text()).get('runtime') for path in receipts],
               'quality_accepted':False,'scope':'frozen candidate for evaluation, not accepted accuracy or generalization'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('automatic_hands.json'))
    p.add_argument('--receipts',type=Path,nargs='+',required=True)
    p.add_argument('--quality-reports',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--runtime-inventory',type=Path)
    p.add_argument('--input-inventory',type=Path)
    a=p.parse_args();freeze(a.config,a.receipts,a.quality_reports,a.out,a.runtime_inventory,a.input_inventory)
