"""Evaluate a fixed profile over declared seeds, using at most two GPUs."""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tools.check_release import file_digest, verify_manifest
from tools.train_source_heads import timestamp

ROOT = Path(__file__).resolve().parents[1]


def seed_lanes(seeds, gpus, jobs_per_gpu=1):
    """Keep a strict physical-card limit while optionally overlapping CPU metrics."""
    if (not seeds or len(set(seeds)) != len(seeds)
            or not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus)
            or jobs_per_gpu not in (1, 2)):
        raise ValueError('Require distinct seeds, one/two distinct GPUs, and one/two jobs per GPU')
    lanes = [(gpu, slot) for slot in range(jobs_per_gpu) for gpu in gpus]
    return [{'gpu': gpu,
             'key': str(gpu) if jobs_per_gpu == 1 else f'{gpu}_slot{slot}',
             'seeds': seeds[index::len(lanes)]}
            for index, (gpu, slot) in enumerate(lanes) if seeds[index::len(lanes)]]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--jobs-per-gpu', type=int, choices=[1, 2], default=1,
                        help='Overlap CPU metric work on the selected cards; 2 needs RAM for four feature caches.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--dinov2-repo', type=Path, required=True)
    parser.add_argument('--pretrained-weights-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    cli = parser.parse_args()
    if not 1 <= len(cli.gpus) <= 2 or len(set(cli.gpus)) != len(cli.gpus):
        parser.error('Select one or two distinct physical GPUs')
    config = json.loads(cli.config.read_text())
    seeds = config.get('eval_seeds')
    if (config['mode'] != 'eval' or config['eval_protocol'] != 'exhaustive'
            or not seeds or len(set(seeds)) != len(seeds) or config.get('image_only')):
        parser.error('Require a complete-metric exhaustive profile with distinct declared seeds')
    release = json.loads(cli.manifest.read_text())
    if release['config_sha256'] != file_digest(cli.config):
        parser.error('Manifest belongs to a different evaluation profile')
    verify_manifest(cli.manifest)
    gpu_lines = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader'], text=True)
    gpu_ids = {int(line.split(',')[0]): line.split(',')[1].strip()
               for line in gpu_lines.splitlines()}
    if any(gpu not in gpu_ids for gpu in cli.gpus):
        parser.error('Requested GPU is absent')
    cli.output_dir.mkdir(parents=True, exist_ok=False)
    status = {'started_at': timestamp(), 'declared_seeds': seeds,
              'shots': config['eval_shots'], 'profile_sha256': file_digest(cli.config),
              'release_manifest_sha256': file_digest(cli.manifest),
              'physical_gpus': cli.gpus, 'max_active_gpus': len(cli.gpus),
              'jobs_per_gpu': cli.jobs_per_gpu, 'jobs': {}}
    lock = threading.Lock()

    def record(key, info):
        with lock:
            status['jobs'][key] = dict(info)
            temporary = cli.output_dir / 'status.json.tmp'
            temporary.write_text(json.dumps(status, indent=2) + '\n')
            temporary.replace(cli.output_dir / 'status.json')

    def worker(item):
        gpu, key, selected = item['gpu'], item['key'], item['seeds']
        env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   CUDA_VISIBLE_DEVICES=gpu_ids[gpu], PYTHONDONTWRITEBYTECODE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   NUMEXPR_NUM_THREADS='2', XFORMERS_DISABLED='1', PYTHONUNBUFFERED='1')
        command = [sys.executable, '-u', str(ROOT / 'run.py'), '--config', str(cli.config),
                   '--data-root', str(cli.data_root), '--device', 'cuda:0',
                   '--dinov2-repo', str(cli.dinov2_repo),
                   '--pretrained-weights-dir', str(cli.pretrained_weights_dir),
                   '--eval-seeds', *map(str, selected),
                   '--output-dir', str(cli.output_dir / f'gpu{key}')]
        info = {'status': 'running', 'seeds': selected, 'physical_gpu': gpu,
                'uuid': gpu_ids[gpu], 'started_at': timestamp(), 'command': command}
        started = time.monotonic()
        with (cli.output_dir / f'gpu{key}.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT)
            info['pid'] = process.pid
            record(key, info)
            print(f'Evaluating seeds {selected} on GPU {gpu}: pid={process.pid}', flush=True)
            try:
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        info.update(status='complete' if code == 0 else 'failed', returncode=code,
                    completed_at=timestamp(), elapsed_seconds=time.monotonic() - started)
        record(key, info)
        print(f'GPU {gpu}: {info["status"]}, seconds={info["elapsed_seconds"]:.1f}', flush=True)
        return code

    lanes = seed_lanes(seeds, cli.gpus, cli.jobs_per_gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        codes = list(pool.map(worker, lanes))
    raise SystemExit(int(any(codes)))


if __name__ == '__main__':
    main()
