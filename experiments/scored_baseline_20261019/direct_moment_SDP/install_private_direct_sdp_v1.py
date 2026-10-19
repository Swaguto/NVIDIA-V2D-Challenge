"""Install three SHA-pinned binary wheels into a private target, never globally.

Exec pip in this worker PID so the existing guard observes its RSS and time.
"""
from pathlib import Path
import hashlib,json,os,sys,urllib.request
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-SDP-runtime-v1'
pins=json.loads((S/'compute/direct-SDP-pinned-wheels-v1.json').read_text())
with (O/'installation-journal.jsonl').open('w') as j:
    def emit(v):j.write(json.dumps(v,allow_nan=False)+'\n');j.flush();os.fsync(j.fileno())
    wheels=[]
    for p in pins:
        emit(dict(kind='BEFORE_private_binary_download',**p))
        data=urllib.request.urlopen(p['url'],timeout=20).read()
        digest=hashlib.sha256(data).hexdigest()
        emit(dict(kind='AFTER_private_binary_download',filename=p['filename'],actual_sha256=digest,bytes=len(data)))
        assert digest==p['sha256'] and len(data)==p['size']
        path=O/p['filename'];path.write_bytes(data);os.chmod(path,0o600);wheels.append(str(path))
    target=O/'site-packages'
    args=[sys.executable,'-m','pip','install','--no-index','--no-deps','--no-compile','--disable-pip-version-check','--target',str(target),'--report',str(O/'pip-install-RETURN.json'),*wheels]
    emit(dict(kind='BEFORE_exec_private_binary_install',arguments=args,target=str(target),global_environment_unchanged=True))
os.execv(sys.executable,args)
