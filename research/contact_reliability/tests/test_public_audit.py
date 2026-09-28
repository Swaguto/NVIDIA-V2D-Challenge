import json
import pytest
pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq
from v2d_reliability.track3 import audit_public, PROXY_EPISODES


def test_public_membership_is_not_storage_order(tmp_path):
    root = tmp_path / "data/chunk-000"; root.mkdir(parents=True)
    for ep in (*PROXY_EPISODES, 2):
        row = {"episode_index": ep, "frame_index": 0, "capture_time": .05,
               "observation.objects": [{"name": "b", "pose": [0, 0, 0, 1, 0, 0, 0], "visible": True},
                                       {"name": "a", "pose": [0]*7, "visible": False}]}
        table = pa.Table.from_pylist([row]).replace_schema_metadata(
            {b"pose_convention": b"world_T_object", b"objects": json.dumps(["a", "b"]).encode()})
        pq.write_table(table, root / f"episode_{ep:06d}.parquet")
    report = audit_public(tmp_path)
    assert report["total_frames"] == 6
    assert all(r["source_to_canonical_indices"] == [1, 0] for r in report["episodes"])
