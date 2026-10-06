"""Audit actual contact artifacts, independently of the production gap filler."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'completion-review')
args = parser.parse_args()
root = args.root
audits = []
for episode in (2, 13, 22):
    bundle = root / f'e{episode:03d}'
    run = bundle / f'issue6-full-runs/calibrated-temporal/e{episode:03d}'
    source = run / 'contacts/contacts.json'
    if not source.exists():
        audits.append({'episode': episode, 'status': 'pending'})
        continue
    rows = json.loads(source.read_text())
    settings = json.loads((run / 'contacts/settings.json').read_text())
    report = json.loads((run / 'report/report.json').read_text())
    assert settings['threshold_m'] == .02 and settings['bridge_frames'] == 15
    assert sorted(set(r['frame_id'] for r in rows)) == [f['id'] for f in report['frame_axis']]
    times = {f['id']:f['timestamp_s'] for f in report['frame_axis']}
    groups = defaultdict(list)
    valid = raw = bridged = unknown = 0
    for row in rows:
        assert all(row[key] == 'predicted' for key in ('hand_provenance','object_provenance','camera_provenance'))
        assert math.isfinite(row['timestamp_s'])
        assert row['timestamp_s'] == times[row['frame_id']]
        groups[(row['hand'], row['fingertip'], row['object_id'])].append(row)
        if row['valid']:
            valid += 1
            distance = row['surface_distance_m']
            assert math.isfinite(distance) and distance >= 0
            assert row['raw_contact'] == (distance < .02)
            assert row['window_contact'] == (row['raw_contact'] or row['bridged'])
            raw += int(row['raw_contact'])
            bridged += int(row['bridged'])
        else:
            unknown += 1
            assert row['raw_contact'] is None and row['window_contact'] is None
            assert not row['bridged'] and row['reason']
    for group in groups.values():
        group.sort(key=lambda r:r['frame_id'])
        assert len({r['frame_id'] for r in group}) == len(group)
        for index, row in enumerate(group):
            if not row['bridged']:
                continue
            left = index - 1
            while left >= 0 and group[left]['bridged']:
                left -= 1
            right = index + 1
            while right < len(group) and group[right]['bridged']:
                right += 1
            assert left >= 0 and right < len(group)
            assert group[left]['raw_contact'] and group[right]['raw_contact']
            assert right-left-1 <= 15
            window = group[left:right+1]
            assert all(r['valid'] and r['hand_track_id'] == row['hand_track_id'] for r in window)
            assert [r['frame_id'] for r in window] == list(range(window[0]['frame_id'],window[-1]['frame_id']+1))
    assert (valid, unknown, raw, bridged) == tuple(report[k] for k in ('contact_valid_count','contact_unknown_count','raw_contact_count','bridged_count'))
    audits.append({'episode':episode,'status':'passed','rows':len(rows),'valid':valid,
                   'unknown':unknown,'raw_contacts':raw,'bridged':bridged,
                   'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                   'checks':['full original frame axis','predicted provenance','strict 2 cm threshold',
                             'unknown observations remain unknown','bridges have two raw-contact endpoints, at most 15 consecutive valid same-track gap frames','report counts match raw artifact']})
output = root / 'contact-artifact-audit.json'
output.write_text(json.dumps(audits,indent=2)+'\n')
print(json.dumps(audits,indent=2))
