"""Hash inputs including validity masks, without reading reconstruction results."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-root',type=Path,required=True)
    p.add_argument('--episodes',type=int,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--compare',type=Path)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    files={}
    for episode in a.episodes:
        label=f'e{episode:03d}';root=a.cache_root/label
        source=root/'issue6-full-inputs'/label
        paths=[source/'manifest.json']
        for folder,pattern in [(source/'color','*.png'),(source/'valid','*.png'),
                               (root/'issue6-full-runs/all20-final-v3'/label/'depth','*.npy')]:
            paths.extend(sorted(folder.glob(pattern)))
        for path in paths:files[path.relative_to(a.cache_root).as_posix()]=sha(path)
    result={'episodes':a.episodes,'files':files}
    if a.compare:
        reference=json.loads(a.compare.read_text())
        result['matches_reference']=reference['files']==files
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'files':len(files),'matches_reference':result.get('matches_reference')}))
    if a.compare and not result['matches_reference']:raise SystemExit('Input mismatch')
