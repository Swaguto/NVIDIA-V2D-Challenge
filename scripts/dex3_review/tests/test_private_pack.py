"""Controls for the migration archive's trust and write boundaries."""
import hashlib,importlib.util,io,json,tarfile
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def module(name):
 s=importlib.util.spec_from_file_location(name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
restore=module('restore_pack').restore
assets=module('restore_assets').restore

def archive(tmp_path,change=None):
 data=b'preserved input';entry={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
 manifest={'files':{'inputs/one':entry},'episodes':[{'episode':'e011'}]}
 members=[('dex3-review/inputs/one',data),('dex3-review/pack-manifest.json',json.dumps(manifest).encode())]
 if change:members=change(members)
 p=tmp_path/'pack.tar'
 with tarfile.open(p,'w') as t:
  for name,b in members:
   i=tarfile.TarInfo(name);i.size=len(b);t.addfile(i,io.BytesIO(b))
 return p,hashlib.sha256(p.read_bytes()).hexdigest()

def test_restores_verified_bytes(tmp_path):
 p,h=archive(tmp_path);out=tmp_path/'out';restore(p,h,out);assert (out/'inputs/one').read_bytes()==b'preserved input'

def test_rejects_wrong_outer_hash_before_writing(tmp_path):
 p,h=archive(tmp_path)
 with pytest.raises(ValueError,match='Archive SHA256'):restore(p,'0'*64,tmp_path/'out')
 assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('name',['dex3-review/../../escape','/absolute','outside/file'])
def test_rejects_path_escape_before_writing(tmp_path,name):
 p,h=archive(tmp_path,lambda m:m+[(name,b'x')])
 with pytest.raises(ValueError,match='Unsafe member'):restore(p,h,tmp_path/'out')
 assert not (tmp_path/'out').exists()

def test_rejects_duplicate_member(tmp_path):
 p,h=archive(tmp_path,lambda m:m+[m[0]])
 with pytest.raises(ValueError,match='Duplicate'):restore(p,h,tmp_path/'out')
 assert not (tmp_path/'out').exists()

def test_rejects_corrupt_member_with_correct_outer_hash(tmp_path):
 p,h=archive(tmp_path,lambda m:[(m[0][0],b'corrupt'),m[1]])
 with pytest.raises(ValueError,match='Changed input'):restore(p,h,tmp_path/'out')
 assert not (tmp_path/'out').exists()

def test_rejects_existing_output(tmp_path):
 p,h=archive(tmp_path);out=tmp_path/'out';out.mkdir();(out/'sentinel').write_text('keep')
 with pytest.raises(FileExistsError):restore(p,h,out)
 assert (out/'sentinel').read_text()=='keep'

def test_rejects_links(tmp_path):
 p=tmp_path/'bad.tar'
 with tarfile.open(p,'w') as t:
  m=tarfile.TarInfo('dex3-review/link');m.type=tarfile.SYMTYPE;m.linkname='/tmp/escape';t.addfile(m)
 h=hashlib.sha256(p.read_bytes()).hexdigest()
 with pytest.raises(ValueError,match='Unsafe member'):restore(p,h,tmp_path/'out')

def test_all_assets_checked_before_destination_changes(tmp_path):
 src=tmp_path/'robot-assets';(src/'meshes').mkdir(parents=True);(src/'meshes/a').write_bytes(b'ok');(src/'meshes/b').write_bytes(b'changed')
 (tmp_path/'pack-manifest.json').write_text(json.dumps({'robot_assets':{'files_sha256':{'meshes/a':hashlib.sha256(b'ok').hexdigest(),'meshes/b':hashlib.sha256(b'expected').hexdigest()}}}))
 with pytest.raises(ValueError,match='Changed asset'):assets(src,tmp_path/'dst')
 assert not (tmp_path/'dst').exists()
