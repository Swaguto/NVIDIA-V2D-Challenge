"""Portable container entry point for the frozen Dex3 preview workflow."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
IK = Path(os.environ.get('V2D_IK_PYTHON', '/opt/ik/bin/python'))
SIM = Path(os.environ.get('V2D_SIM_PYTHON', '/opt/sim/bin/python'))
LAB = Path(os.environ.get('V2D_LAB', '/opt/IsaacLab'))

def run(cmd, **kwargs):
    subprocess.run([str(x) for x in cmd], check=True, **kwargs)

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    d=s.add_parser('doctor');d.add_argument('--runtime',choices=['ik','sim','all'],default='all');d.add_argument('--gpu',action='store_true')
    d=s.add_parser('retarget');d.add_argument('--bundle',type=Path,required=True);d.add_argument('--out',type=Path,required=True);d.add_argument('--assets',type=Path,required=True)
    d=s.add_parser('render');d.add_argument('--bundle',type=Path,required=True);d.add_argument('--retarget',type=Path,required=True);d.add_argument('--out',type=Path,required=True);d.add_argument('--assets',type=Path,required=True)
    d=s.add_parser('package');d.add_argument('--bundle',type=Path,required=True);d.add_argument('--retarget',type=Path,required=True);d.add_argument('--sim',type=Path,required=True);d.add_argument('--human',type=Path,required=True);d.add_argument('--out',type=Path,required=True)
    d=s.add_parser('compare');d.add_argument('--reference',type=Path,required=True);d.add_argument('--candidate',type=Path,required=True)
    a=p.parse_args()
    if a.command=='doctor':
        if a.runtime in ['ik','all']:
            run([IK,ROOT/'scripts/dex3_review/check_runtime.py','--runtime','ik','--lock','/opt/locks/ik-runtime-lock.txt'])
        if a.runtime in ['sim','all']:
            run([SIM,'/opt/locks/setup_sim6.py','--check','--lock','/opt/locks/sim6-runtime-lock.txt','--lab',LAB])
        if a.gpu:run(['nvidia-smi'])
        return
    if a.command in ['retarget','render']:
        run([IK,ROOT/'scripts/dex3_review/restore_assets.py','--source',a.assets,'--destination',ROOT/'robotic_grounding/source/robotic_grounding/robotic_grounding/assets'])
    if a.command=='retarget':
        run([IK,ROOT/'robotic_grounding/scripts/retarget/track3_to_dex3.py','--bundle',a.bundle,'--out',a.out,'--max-iter','100'])
    elif a.command=='render':
        if os.environ.get('OMNI_KIT_ACCEPT_EULA')!='YES':raise RuntimeError('Read the NVIDIA license and set ACCEPT_EULA=Y in your Compose environment before rendering.')
        env=dict(os.environ);env.pop('PYTHONPATH',None)
        run([SIM,ROOT/'scripts/dex3_review/frozen/replay_continuity_dex3_sim6.py','--source',ROOT/'robotic_grounding','--bundle',a.bundle,'--retarget',a.retarget,'--out',a.out,'--width','1280','--height','800','--headless','--enable_cameras','--device','cpu','--rendering_mode','performance'],cwd=LAB,env=env)
    elif a.command=='compare':
        run([IK,ROOT/'scripts/dex3_review/compare_retarget.py','--reference',a.reference,'--candidate',a.candidate])
    else:
        run([SIM,ROOT/'scripts/dex3_review/package_portable.py' ,'--bundle',a.bundle,'--retarget',a.retarget,'--sim',a.sim,'--human',a.human,'--out',a.out])

if __name__=='__main__':main()
