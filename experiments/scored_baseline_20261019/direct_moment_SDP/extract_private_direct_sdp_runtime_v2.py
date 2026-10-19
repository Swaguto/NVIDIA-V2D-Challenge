"""Extract already SHA-verified binary wheels; private Python has no pip.

No download or global environment modification. Wheel RECORD is also checked.
"""
from pathlib import Path,PurePosixPath
import base64,csv,hashlib,io,json,os,sys,zipfile,importlib.metadata
S=Path(__file__).resolve().parents[1];O=S/'e037-direct-SDP-runtime-v2';source=S/'e037-direct-SDP-runtime-v1';target=O/'site-packages';target.mkdir(mode=0o700)
pins=json.loads((S/'compute/direct-SDP-pinned-wheels-v1.json').read_text());j=(O/'runtime-extraction-journal.jsonl').open('w')
def emit(v):j.write(json.dumps(v,allow_nan=False)+'\n');j.flush();os.fsync(j.fileno())
inventory=[]
for pin in pins:
    wheel=source/pin['filename'];emit(dict(kind='BEFORE_saved_wheel_extraction',pin=pin))
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==pin['sha256']
    with zipfile.ZipFile(wheel) as z:
        for name in z.namelist():
            path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts and '.data' not in path.parts
        records=[n for n in z.namelist()if n.endswith('.dist-info/RECORD')];assert len(records)==1
        count=0
        for name,digest,size in csv.reader(io.StringIO(z.read(records[0]).decode())):
            data=z.read(name)
            if digest:
                algo,value=digest.split('=',1);assert algo=='sha256'
                assert base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()==value
            if size:assert len(data)==int(size)
            out=target/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data);os.chmod(out,0o600);count+=1
    emit(dict(kind='AFTER_saved_wheel_extraction',name=pin['name'],version=pin['version'],RECORD_files_verified=count))
sys.path.insert(0,str(target));import clarabel,cffi,pycparser
emit(dict(kind='AFTER_private_runtime_import',clarabel=clarabel.__version__,cffi=cffi.__version__,pycparser=pycparser.__version__,PSD_API=repr(clarabel.PSDTriangleConeT(3))))
settings=clarabel.DefaultSettings()
data=dict(schema='v2d.private-direct-SDP-runtime.v2',pins=pins,isolated_target=str(target),no_global_or_production_environment_change=True,old_pip_process_failure_retained=True,defaults=repr(settings),API=dict(PSD=repr(clarabel.PSDTriangleConeT(3)),Zero=repr(clarabel.ZeroConeT(1))),files_sha256={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()for p in target.rglob('*')if p.is_file()})
(O/'runtime-inventory.json').write_text(json.dumps(data,indent=2)+'\n');j.close();print(json.dumps(dict(clarabel=clarabel.__version__,private_target=str(target),files=len(data['files_sha256']))))
