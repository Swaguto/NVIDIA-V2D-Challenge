"""Audit actual contact artifacts, independently of the production gap filler."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import argparse

def check(condition):
    if not condition:
        raise ValueError("Contact artifact integrity check failed")


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).parent / 'completion-review')
parser.add_argument('--episodes', default='2,13,22', help='comma-separated required episode IDs')
parser.add_argument('--direct-run-root', action='store_true', help='root contains eNNN pipeline runs directly')
parser.add_argument('--require-complete', action='store_true', help='return failure if any required episode is pending')
parser.add_argument('--output', type=Path, help='separate receipt path for concurrent episode subsets')
args = parser.parse_args()
root = args.root
audits = []
for episode in map(int,args.episodes.split(',')):
    bundle = root / f'e{episode:03d}'
    run = bundle if args.direct_run_root else bundle / f'issue6-full-runs/calibrated-temporal/e{episode:03d}'
    source = run / 'contacts/contacts.json'
    if not source.exists():
        audits.append({'episode': episode, 'status': 'pending'})
        continue
    rows = json.loads(source.read_text())
    settings = json.loads((run / 'contacts/settings.json').read_text())
    report = json.loads((run / 'report/report.json').read_text())
    lock_source = run / 'inputs.lock.json'
    manifest = json.loads(lock_source.read_text())['manifest']
    maximum = manifest.get('settings', {}).get('bridge_frames', 15)
    check(type(maximum) is int and 0 <= maximum <= 15)
    check(type(settings['bridge_frames']) is int)
    check(manifest.get('settings', {}).get('contact_threshold_m', .02) == .02)
    check(settings['threshold_m'] == .02 and settings['bridge_frames'] == maximum)
    check(sorted(set(r['frame_id'] for r in rows)) == [f['id'] for f in report['frame_axis']])
    check([f['id'] for f in report['frame_axis']] == [f['id'] for f in manifest['frames']])
    times = {f['id']:f['timestamp_s'] for f in report['frame_axis']}
    check(times == {f['id']:f['timestamp_s'] for f in manifest['frames']})
    # A file can contain the whole time axis while silently omitting a hand,
    # fingertip, or object. Require one observation (including explicit unknowns)
    # for every manifest-declared combination.
    expected = {
        (frame_id, side, tip, obj['id'])
        for frame_id in times
        for side in ('left', 'right')
        for tip in ('thumb', 'index', 'middle', 'ring', 'little')
        for obj in manifest['objects']
    }
    actual = [(r['frame_id'], r['hand'], r['fingertip'], r['object_id']) for r in rows]
    check(len(actual) == len(expected) and set(actual) == expected)
    groups = defaultdict(list)
    valid = raw = bridged = unknown = 0
    for row in rows:
        check(type(row['valid']) is bool and type(row['bridged']) is bool)
        check(all(row[key] == 'predicted' for key in ('hand_provenance','object_provenance','camera_provenance')))
        check(math.isfinite(row['timestamp_s']))
        check(row['timestamp_s'] == times[row['frame_id']])
        groups[(row['hand'], row['fingertip'], row['object_id'])].append(row)
        if row['valid']:
            check(type(row['raw_contact']) is bool and type(row['window_contact']) is bool)
            check(not row['bridged'] or row['raw_contact'] is False)
            valid += 1
            distance = row['surface_distance_m']
            check(math.isfinite(distance) and distance >= 0)
            check(row['raw_contact'] == (distance < .02))
            check(row['window_contact'] == (row['raw_contact'] or row['bridged']))
            raw += int(row['raw_contact'])
            bridged += int(row['bridged'])
        else:
            unknown += 1
            check(row['raw_contact'] is None and row['window_contact'] is None)
            check(not row['bridged'] and row['reason'])
    for group in groups.values():
        group.sort(key=lambda r:r['frame_id'])
        check(len({r['frame_id'] for r in group}) == len(group))
        for index, row in enumerate(group):
            if not row['bridged']:
                continue
            left = index - 1
            while left >= 0 and group[left]['bridged']:
                left -= 1
            right = index + 1
            while right < len(group) and group[right]['bridged']:
                right += 1
            check(left >= 0 and right < len(group))
            check(group[left]['raw_contact'] and group[right]['raw_contact'])
            check(right-left-1 <= maximum)
            window = group[left:right+1]
            check(all(r['valid'] and r['hand_track_id'] == row['hand_track_id'] for r in window))
            check([r['frame_id'] for r in window] == list(range(window[0]['frame_id'],window[-1]['frame_id']+1)))
    check((valid, unknown, raw, bridged) == tuple(report[k] for k in ('contact_valid_count','contact_unknown_count','raw_contact_count','bridged_count')))
    audits.append({'episode':episode,'status':'passed','rows':len(rows),'valid':valid,
                   'unknown':unknown,'raw_contacts':raw,'bridged':bridged,
                   'bridge_frames':maximum,
                   'input_lock_sha256':hashlib.sha256(lock_source.read_bytes()).hexdigest(),
                   'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                   'checks':['full original frame axis and timestamps match the input lock',
                             'every declared hand/fingertip/object observation is present',
                             'predicted provenance','strict 2 cm threshold',
                             'unknown observations remain unknown',
                             f'bridges have two raw-contact endpoints and at most {maximum} consecutive valid same-track gap frames',
                             'report counts match raw artifact']})
output = args.output or root / 'contact-artifact-audit.json'
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(audits,indent=2)+'\n')
print(json.dumps(audits,indent=2))
if args.require_complete and any(audit['status'] != 'passed' for audit in audits):
    raise SystemExit('Required contact artifacts are incomplete')
