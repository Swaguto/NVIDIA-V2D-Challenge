"""The reproduction check must reject dropped support and misplaced geometry."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest


@pytest.mark.parametrize('difference,expected', [('none',0),('support',1),('position',1),('missing_geometry',1),('nonfinite_geometry',1)])
def test_reproduction_rejects_real_differences(tmp_path,difference,expected):
    script=Path(__file__).resolve().parents[3]/'validation/issue6/compare_automatic_reproduction.py'
    for name in ('reference','candidate'):
        root=tmp_path/name;(root/'proposals').mkdir(parents=True)
        receipt={k:'same' for k in ('method_files','episode','source_manifest_sha256','detector_sha256',
            'hamer_checkpoint_sha256','sam_checkpoint_sha256','mano_right_sha256','hamer_config_sha256',
            'mano_mean_params_sha256','hamer_source_commit')}
        (root/'proposals/receipt.json').write_text(json.dumps(receipt))
        quality=root/'quality';(quality/'geometry/left').mkdir(parents=True)
        supported=not(name=='candidate' and difference=='support')
        (quality/'frame-evidence.json').write_text(json.dumps([{'frame':frame,'hands':{'left':{'accepted_by_consistency_proxies':supported}}} for frame in (0,1)]))
        value=.02 if name=='candidate' and difference=='position' else 0
        vertices=np.zeros((778,3));vertices[:,0]=value
        joints=np.zeros((21,3));joints[:,0]=value
        np.savez(quality/'geometry/left/000000.npz',vertices=vertices,joints=joints,raw_vertices=vertices,raw_joints=joints)
        if name=='candidate' and difference=='missing_geometry':continue
        if name=='candidate' and difference=='nonfinite_geometry':vertices[:]=np.nan
        np.savez(quality/'geometry/left/000001.npz',vertices=vertices,joints=joints,raw_vertices=vertices,raw_joints=joints)
    result=subprocess.run([sys.executable,str(script),'--reference',str(tmp_path/'reference'),
        '--candidate',str(tmp_path/'candidate'),'--out',str(tmp_path/'report.json')],capture_output=True,text=True)
    assert result.returncode==expected,result.stdout+result.stderr
    assert json.loads((tmp_path/'report.json').read_text())['passed']==(expected==0)
