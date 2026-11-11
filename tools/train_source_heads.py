"""Run declared source-only head recipes on at most two physical GPUs.

No target evaluation, checkpoint search, or pretrained AML head initialization.
The output directory must be new; existing experiment artifacts are preserved.
"""

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def timestamp():
    return datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--recipes', nargs='+', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dinov2-repo', type=Path, required=True)
    parser.add_argument('--pretrained-weights-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.gpus) <= 2 or len(set(args.gpus)) != len(args.gpus):
        parser.error('Use one or two distinct physical GPU indices, never more than two')
    gpu_lines = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    gpu_ids = {int(line.split(',')[0]): line.split(',')[1].strip()
               for line in gpu_lines.splitlines()}
    if any(gpu not in gpu_ids for gpu in args.gpus):
        parser.error('Requested GPU index is not present')
    jobs = queue.Queue()
    names = set()
    declarations = []
    for path in args.recipes:
        config = json.loads(path.read_text())
        if (config['mode'] != 'train' or config.get('eval_frequency') != 0
                or config.get('checkpoint') or config.get('eval_shots')
                or config.get('eval_seeds')):
            parser.error(f'Recipe must start from scratch and disable target evaluation: {path}')
        if path.stem in names:
            parser.error(f'Duplicate recipe name: {path.stem}')
        names.add(path.stem)
        declarations.append({'recipe': str(path), 'configuration': config})
        jobs.put(path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    status = {'started_at': timestamp(), 'gpus': args.gpus,
              'gpu_uuids': {gpu: gpu_ids[gpu] for gpu in args.gpus},
              'max_active_gpus': len(args.gpus), 'recipes': declarations, 'jobs': {}}
    lock = threading.Lock()

    def record(name, info):
        with lock:
            status['jobs'][name] = info
            pending = args.output_dir / 'status.json.tmp'
            pending.write_text(json.dumps(status, indent=2) + '\n')
            pending.replace(args.output_dir / 'status.json')

    def worker(gpu):
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=gpu_ids[gpu], PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', XFORMERS_DISABLED='1', PYTHONUNBUFFERED='1')
        failures = 0
        while True:
            try:
                recipe = jobs.get_nowait()
            except queue.Empty:
                return failures
            command = [sys.executable, '-u', str(ROOT / 'run.py'), '--config', str(recipe),
                       '--data-root', str(args.data_root), '--device', 'cuda:0',
                       '--dinov2-repo', str(args.dinov2_repo),
                       '--pretrained-weights-dir', str(args.pretrained_weights_dir),
                       '--output-dir', str(args.output_dir / recipe.stem)]
            started = time.monotonic()
            info = {'status': 'running', 'physical_gpu': gpu, 'command': command,
                    'started_at': timestamp()}
            with (args.output_dir / (recipe.stem + '.log')).open('x') as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                info['pid'] = process.pid
                record(recipe.stem, dict(info))
                print(f'Start {recipe.stem}: GPU {gpu}, pid={process.pid}', flush=True)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            info.update(status='complete' if code == 0 else 'failed', returncode=code,
                        completed_at=timestamp(), elapsed_seconds=time.monotonic() - started)
            record(recipe.stem, info)
            print(f'{recipe.stem}: {info["status"]}, seconds={info["elapsed_seconds"]:.1f}', flush=True)
            failures += int(code != 0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        failures = sum(pool.map(worker, args.gpus))
    raise SystemExit(int(failures != 0))


if __name__ == '__main__':
    main()
