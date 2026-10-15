"""Install or check the measured Linux Sim 6 replay environment from a pack lock.

This restores the minimal replay closure, not all optional Isaac Lab features.
NVIDIA license acceptance and host graphics libraries are prerequisites.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import venv

COMMIT='28a37cecdd433c22d9eabd6a5954add9f13a8951'

def pins(path):
    result=[]
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        if line.startswith('-e git+https://github.com/isaac-sim/IsaacLab.git@'+COMMIT):continue
        if not re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+',line):
            raise ValueError('Unexpected lock entry: '+line)
        result.append(line)
    return result

def run(command):
    subprocess.run(list(map(str,command)),check=True)

def main(args):
    if sys.platform!='linux' or sys.version_info[:2]!=(3,12):
        raise RuntimeError('Use Linux Python 3.12; measured patch version is 3.12.15')
    locked=pins(args.lock)
    if args.check:
        mismatches=[]
        for pin in locked:
            name,expected=pin.split('==')
            try:actual=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:actual=None
            if actual!=expected:mismatches.append({'name':name,'expected':expected,'actual':actual})
        actual_commit=subprocess.check_output(['git','-C',str(args.lab),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(args.lab),'status','--porcelain','--untracked-files=no'],text=True).strip()
        import isaaclab
        imported=Path(isaaclab.__file__).resolve()
        if actual_commit!=COMMIT or dirty or not imported.is_relative_to(args.lab.resolve()):
            raise ValueError('Isaac Lab checkout/import does not match the pinned source')
        report={'status':'passed' if not mismatches else 'failed','python':sys.version,
                'checked_packages':len(locked),'lab_commit':actual_commit,'mismatches':mismatches,
                'scope':'installed package versions and tracked Lab source; no renderer or optional training dependency claim'}
        print(json.dumps(report,indent=2))
        if mismatches:raise SystemExit(1)
        return
    if args.prefix is None:raise ValueError('--prefix is required for installation')
    prefix=args.prefix.resolve();lab=args.lab.resolve()
    if prefix.exists() or lab.exists():raise FileExistsError('Use new environment and Lab directories')
    venv.EnvBuilder(with_pip=True).create(prefix)
    requirements=prefix/'replay-requirements.txt'
    requirements.write_text('\n'.join(locked)+'\n')
    python=prefix/'bin/python'
    # A pinned installer supports resuming multi-gigabyte NVIDIA wheel downloads.
    run([python,'-m','pip','install','pip==25.3'])
    # All transitive versions are already recorded; do not resolve optional Lab extras.
    run([python,'-m','pip','install','--no-compile','--no-deps','--extra-index-url',
         'https://pypi.nvidia.com','-r',requirements])
    run(['git','clone','https://github.com/isaac-sim/IsaacLab.git',lab])
    run(['git','-C',lab,'checkout','--detach',COMMIT])
    run([python,'-m','pip','install','--no-compile','--no-deps','-e',lab/'source/isaaclab'])
    run([python,Path(__file__).resolve(),'--check','--lock',args.lock.resolve(),'--lab',lab])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lock',type=Path,required=True)
    p.add_argument('--lab',type=Path,required=True)
    p.add_argument('--prefix',type=Path)
    p.add_argument('--check',action='store_true')
    main(p.parse_args())
