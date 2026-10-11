"""Record effective package versions and inference source files, without downloads."""
import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import sys

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.out.exists():raise FileExistsError(a.out)
packages={}
for dist in metadata.distributions():
    name=dist.metadata['Name']
    if name in packages:continue
    direct=dist.read_text('direct_url.json')
    packages[name]={'version':dist.version,'source':json.loads(direct) if direct else None}
sources={}
for name in ('sam2','ultralytics','smplx','manotorch','chumpy'):
    root=Path(importlib.util.find_spec(name).origin).parent
    files={}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in ('.py','.yaml','.yml','.json'):
            files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    sources[name]=files
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'python':sys.version,'platform':platform.platform(),
    'packages':packages,'inference_source_files':sources,'clean_bootstrap_validated':False},indent=2)+'\n')
print(json.dumps({'packages':len(packages),'source_files':sum(len(v) for v in sources.values())}))
