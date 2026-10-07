"""Run three disjoint fit queues and reconstruct whichever verified fit is ready.

This replay entry point uses the same native settings as the serial scripts.
It requires prepared inputs, saved detections, CAD tracking and native runtimes.
Visual acceptance and verified private checkpoint transfer remain separate.
"""
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

DEFAULT_EPISODES = '0,2,11,12,13,18,21,22,23,24,26,27,31,39,40,41,42,43,44,45'


def fit_ready(episode):
    path = Path(f'/workspace/temporal-hands/e{episode:03d}/fit_receipt.json')
    if not path.is_file():
        return False
    receipt = json.loads(path.read_text())
    if not (receipt['fixed_shape_exactly_preserved'] and receipt['root_iterations'] == 50
            and receipt['smooth_iterations'] == 300):
        raise ValueError(f'Episode {episode} does not match the frozen fit settings')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', default=DEFAULT_EPISODES)
    parser.add_argument('--workers', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--review-workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    episodes = [int(value) for value in args.episodes.split(',')]
    if not episodes or len(set(episodes)) != len(episodes):
        parser.error('Episode list must be nonempty and contain no duplicates')
    base = Path('/workspace/pr32-evidence')
    base.mkdir(exist_ok=True)
    helpers = Path(__file__).resolve().parent
    # The serial recipe and this entry point exclude each other. Independent
    # producer queues additionally hold their own queue locks.
    lock = (base / 'all20-hand-preparation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for pointer in ('all20-hand-preparation.pid', 'all20-split-review.pid'):
        path = base / pointer
        if path.is_file():
            try:
                pid = int(path.read_text())
                state = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[0]
            except (ValueError, FileNotFoundError):
                continue
            if state != 'Z':
                raise RuntimeError(f'Existing native scheduler {pid} is still running')
    lengths = {}
    for episode in episodes:
        manifest = json.loads(Path(f'/workspace/issue6-full-inputs/e{episode:03d}/manifest.json').read_text())
        lengths[episode] = len(manifest['frames'])
    queues = [[] for _ in range(args.workers)]
    loads = [0] * args.workers
    for episode in sorted(episodes, key=lambda value: (-lengths[value], value)):
        queue = min(range(args.workers), key=loads.__getitem__)
        queues[queue].append(episode)
        loads[queue] += lengths[episode]
    receipt = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'episode_queues': queues, 'source_frame_loads': loads, 'completed': [],
               'failed_reviews': [], 'settings_changed': False, 'status': 'running'}
    receipt_path = base / 'parallel-replay-scheduling.json'

    def save():
        temporary = receipt_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(receipt, indent=2) + '\n')
        temporary.replace(receipt_path)

    producers = []
    reviews = {}
    logs = []
    try:
        for index, queue in enumerate(queues):
            if not queue:
                continue
            log = (base / f'parallel-replay-fit-{index}.log').open('ab')
            logs.append(log)
            producers.append(subprocess.Popen(['bash', str(helpers / 'prepare_episode_hands_parallel.sh'),
                                              f'replay-{index}', ','.join(map(str, queue))],
                                              stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True))
        for pointer in ('all20-hand-preparation.pid', 'all20-split-review.pid'):
            (base / pointer).write_text(str(os.getpid()) + '\n')
        remaining = list(episodes)
        save()
        while remaining:
            for episode, (process, started) in list(reviews.items()):
                if process.poll() is None:
                    continue
                remaining.remove(episode)
                del reviews[episode]
                result = {'episode': episode, 'elapsed_s': time.monotonic() - started}
                if process.returncode != 0:
                    result['returncode'] = process.returncode
                    receipt['failed_reviews'].append(result)
                else:
                    receipt['completed'].append(result)
                save()
            failed = [process.pid for process in producers if process.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Native preparation failed: {failed}; inspect queue logs')
            ready = [episode for episode in remaining if episode not in reviews and fit_ready(episode)]
            if not ready and remaining and not reviews and all(process.poll() is not None for process in producers):
                raise RuntimeError('Preparation ended before all required fits completed')
            for episode in ready[:max(0, args.review_workers - len(reviews))]:
                log = (base / f'parallel-replay-review-e{episode:03d}.log').open('ab')
                logs.append(log)
                process = subprocess.Popen(['bash', str(helpers / 'review_fitted_episodes.sh'), str(episode)],
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                reviews[episode] = (process, time.monotonic())
            receipt.update(active_episodes=sorted(reviews), ready_episodes=ready)
            save()
            if remaining:
                time.sleep(5)
        for process in producers + [job[0] for job in reviews.values()]:
            if process.wait() != 0:
                raise RuntimeError(f'Preparation process {process.pid} failed')
        receipt.update(status=('failed_reviews_require_repair' if receipt['failed_reviews']
                               else 'machine_checks_complete_visual_review_required'), active_episode=None,
                       finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        save()
        if receipt['failed_reviews']:
            raise SystemExit('Failed reviews remain unaccepted; inspect their episode logs before replay')
    finally:
        for process in producers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
        for log in logs:
            log.close()
        lock.close()


if __name__ == '__main__':
    main()
