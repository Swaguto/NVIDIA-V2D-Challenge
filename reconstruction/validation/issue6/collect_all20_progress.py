"""Report completed artifacts; acceptance stays bound to the reviewed configuration."""
import datetime
import hashlib
import json
import os
from pathlib import Path

EPISODES = [0, 2, 11, 12, 13, 18, 21, 22, 23, 24, 26, 27, 31, 39, 40, 41, 42, 43, 44, 45]
STAGES = ('validate', 'hands', 'depth', 'align', 'contacts', 'report')


def read(path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    rows = []
    for episode in EPISODES:
        name = f'e{episode:03d}'
        hand = Path('/workspace/temporal-hands') / name
        run = Path('/workspace/issue6-full-runs/all20-final') / name
        review = Path('/workspace/pr32-evidence/all20-final-review') / name
        lock = read(run / 'inputs.lock.json')
        config = hashlib.sha256(json.dumps(lock, sort_keys=True, allow_nan=False).encode()).hexdigest() if lock else None
        receipts = {stage: read(run / stage / 'receipt.json', {}) for stage in STAGES}
        stages = {stage: bool(config) and receipt.get('status') == 'complete'
                  and receipt.get('config_sha256') == config for stage, receipt in receipts.items()}
        audits = read(review / 'stage-integrity-audit.json', {}).get('episodes', [])
        audit = next((item for item in audits if item.get('episode') == episode), {})
        integrity = audit.get('status') == 'passed' and audit.get('config_sha256') == config
        if integrity:
            integrity = set(audit['stages']) == set(STAGES) and all(
                stages[stage] and audit['stages'][stage]['receipt_sha256'] == digest(run / stage / 'receipt.json')
                for stage in STAGES)
        contacts = read(review / 'contact-artifact-audit.json', [])
        contact = next((item for item in contacts if item.get('episode') == episode), {})
        contact_file = run / 'contacts/contacts.json'
        audited = (contact.get('status') == 'passed' and contact_file.is_file()
                   and contact.get('sha256') == digest(contact_file))
        geometry = read(hand / 'dynhamr_camera_conversion_probe.json', {})
        checks = geometry.get('checks', [])
        axis = read(hand / 'validation_receipt.json', {}).get('selected_frame_axis', [])
        geometry_ok = (geometry.get('converted_records') is not None and bool(axis)
                       and len(checks) == 2 * len(axis)
                       and all(item['max_vertex_error_m'] < 1e-5 for item in checks))
        fit = read(hand / 'fit_receipt.json', {})
        fixed = (fit.get('fixed_shape_exactly_preserved') is True
                 and fit.get('root_iterations') == 50 and fit.get('smooth_iterations') == 300)
        visual = read(review / 'visual-review.json', {})
        accepted = (all(stages.values()) and audited and integrity and geometry_ok and fixed
                    and visual.get('status') == 'passed' and visual.get('configuration_sha256') == config)
        rows.append({'episode': episode,
                     'calibrated_inputs': (Path('/workspace/issue6-full-inputs') / name / 'manifest.json').is_file(),
                     'wearer_detections': (hand / 'wearer_candidates.json').is_file(),
                     'hand_predictions': (hand / 'predictions.pkl').is_file(),
                     'fit': fixed, 'stages': stages, 'contact_audit': audited,
                     'stage_integrity': integrity, 'all_frame_geometry': geometry_ok, 'accepted': accepted})
    fits = sum(row['fit'] for row in rows)
    complete = sum(all(row['stages'].values()) for row in rows)
    progress = {'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'detections_complete': sum(row['wearer_detections'] for row in rows),
                'accepted': sum(row['accepted'] for row in rows),
                'current': f'{fits} verified fixed-shape hand fits; {complete} final six-stage reconstructions. '
                           'Final acceptance requires reviewed videos.', 'episodes': rows}
    destination = Path('/workspace/pr32-evidence/all20-status.json')
    temporary = destination.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(progress, indent=2) + '\n')
    temporary.replace(destination)
    print(json.dumps({key: value for key, value in progress.items() if key != 'episodes'}))


if __name__ == '__main__':
    main()
