"""The migration branch must preserve the executed sources and original receipts."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
def test_executed_snapshot_bytes_are_unchanged():
 d=ROOT/'docs/team/dex3-isaac-review'
 hashes=json.loads((d/'snapshot-hashes.json').read_text())
 for row in json.loads((d/'orchestration-snapshot.json').read_text()):hashes[row['path']]=row['sha256']
 for rel,expected in hashes.items():
  p=(ROOT/rel).resolve()
  assert p.is_relative_to(ROOT)
  assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,rel
