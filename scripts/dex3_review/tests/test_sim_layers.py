import importlib.util
from pathlib import Path
import pytest

path=Path(__file__).resolve().parents[3]/'docker/dex3-review/copy_sim_layers.py'
spec=importlib.util.spec_from_file_location('sim_layers',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def prefix(tmp_path):
 source=tmp_path/'source';source.mkdir()
 (source/'bin').mkdir();(source/'bin/python').write_bytes(b'executable');(source/'bin/python').chmod(0o755)
 (source/'lib64').symlink_to('lib')
 for i in range(9):
  root=source/module.CACHE/f'extension-{i}';root.mkdir(parents=True)
  (root/'payload').write_bytes(bytes([i])*37)
  (root/'alias').symlink_to('payload')
 return source

def test_parts_preserve_every_file_and_symlink(tmp_path):
 source=prefix(tmp_path);destination=tmp_path/'destination'
 all_paths=[]
 for part in ['root','0','1','2','3']:
  all_paths.extend(module.selected(source,part));module.copy_part(source,destination,part)
 assert len(all_paths)==len(set(all_paths))==len(module.members(source))
 module.verify(source,destination)
 (destination/'bin/python').write_bytes(b'changed')
 with pytest.raises(ValueError,match='differs'):module.verify(source,destination)

def test_layer_bound_refuses_before_copying(tmp_path):
 source=prefix(tmp_path);destination=tmp_path/'destination'
 with pytest.raises(ValueError,match='payload bound'):module.copy_part(source,destination,'root',limit=1)
 assert not destination.exists()
