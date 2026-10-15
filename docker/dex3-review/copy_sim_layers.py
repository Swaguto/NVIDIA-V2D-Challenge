"""Copy an installed simulator prefix into bounded registry layers, preserving bytes."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil
import stat

CACHE = Path('lib/python3.12/site-packages/isaacsim/extscache')
PARTS = 4
LIMIT = 8 * 1024**3

def members(root):
    return [p for p in sorted(root.rglob('*')) if p.is_file() or p.is_symlink()]

def selected(source, part):
    if part == 'root':
        return [p for p in members(source) if not p.relative_to(source).is_relative_to(CACHE)]
    children=sorted((source/CACHE).iterdir())
    return [p for i,child in enumerate(children) if i % PARTS == int(part)
            for p in ([child] if not child.is_dir() or child.is_symlink() else members(child))]

def copy_part(source, destination, part, limit=LIMIT):
    paths=selected(source,part)
    size=sum(p.stat().st_size for p in paths if not p.is_symlink())
    if size > limit:
        raise ValueError(f'Layer exceeds {limit} byte payload bound: {size}')
    if part == 'root':
        def ignore(directory,names):
            return ['extscache'] if Path(directory)==source/CACHE.parent else []
        shutil.copytree(source,destination,symlinks=True,dirs_exist_ok=True,ignore=ignore)
    else:
        target=destination/CACHE;target.mkdir(parents=True,exist_ok=True)
        for i,child in enumerate(sorted((source/CACHE).iterdir())):
            if i % PARTS != int(part):continue
            if child.is_dir() and not child.is_symlink():
                shutil.copytree(child,target/child.name,symlinks=True,dirs_exist_ok=True)
            else:shutil.copy2(child,target/child.name,follow_symlinks=False)
    print('SIM_LAYER_COPIED',part,len(paths),size,flush=True)

def inventory(root):
    result={}
    for p in members(root):
        relative=p.relative_to(root).as_posix()
        if p.is_symlink():result[relative]=('symlink',p.readlink().as_posix())
        else:
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            result[relative]=('file',stat.S_IMODE(p.stat().st_mode),p.stat().st_size,h.hexdigest())
    return result

def verify(source,destination):
    expected=inventory(source);actual=inventory(destination)
    if expected!=actual:
        bad=[p for p in expected.keys()|actual.keys() if expected.get(p)!=actual.get(p)]
        raise ValueError(f'Simulator copy differs: {bad[:20]}')
    print('SIM_PREFIX_BYTES_VERIFIED',len(expected),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--part',choices=['root','0','1','2','3'])
    group.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:verify(args.source,args.destination)
    else:copy_part(args.source,args.destination,args.part)
