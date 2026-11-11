"""Verify exported model tensors and write a checksum manifest."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import torch


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(path):
    """Check distributable files without requiring private training artifacts."""
    manifest = json.loads(path.read_text())
    config = Path(manifest['config'])
    if file_digest(config) != manifest['config_sha256']:
        raise ValueError(f'Release configuration checksum differs: {config}')
    for record in manifest['weights']:
        weight = Path(record['path'])
        if weight.stat().st_size != record['size_bytes'] or file_digest(weight) != record['sha256']:
            raise ValueError(f'Release checkpoint checksum differs: {weight}')
    return len(manifest['weights'])


def python_source_fingerprint(root):
    root = Path(root)
    files = sorted(root.rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    return {'path': str(root.resolve()), 'number_of_python_files': len(files),
            'python_source_sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify-manifest', type=Path,
                        help='Verify published config/weight checksums without original training files.')
    parser.add_argument('--dinov2-repo', type=Path,
                        help='Record the local DINOv2 Python source fingerprint for reproducibility.')
    cli = parser.parse_args()
    if cli.verify_manifest:
        count = verify_manifest(cli.verify_manifest)
        print(f'Verified {count} release checkpoint checksums and configuration; original training files not required')
        return
    if cli.config is None or cli.output is None:
        parser.error('Export audit requires --config and --output; download verification uses --verify-manifest')
    config = json.loads(cli.config.read_text())
    paths = {config['checkpoint'], *config['image_experts']}
    for shot in config['eval_shots']:
        paths.update(pattern.format(shot=shot) for pattern in config['pixel_experts'])
    records = []
    for filename in sorted(paths):
        path = Path(filename)
        state = torch.load(path, map_location='cpu', weights_only=False)
        source_path = Path(state['source_checkpoint'])
        if file_digest(source_path) != state['source_sha256']:
            raise ValueError(f'Source checkpoint changed: {source_path}')
        original = torch.load(source_path, map_location='cpu', weights_only=False)
        weights = original.get('model', original)
        if set(weights) != set(state['model']):
            raise ValueError(f'Exported parameter names differ: {path}')
        for name, value in weights.items():
            if not torch.equal(value, state['model'][name]):
                raise ValueError(f'Exported parameter changed: {path}: {name}')
        records.append({'path': str(path), 'size_bytes': path.stat().st_size,
                        'sha256': file_digest(path),
                        'frozen_encoder_signature': state['frozen_encoder_signature'],
                        'source_checkpoint': str(source_path),
                        'source_sha256': state['source_sha256'],
                        'training_configuration_known': state['training_configuration_known'],
                        'exact_parameter_copy_verified': True})
    result = {
        'config': str(cli.config), 'config_sha256': file_digest(cli.config), 'weights': records,
        'software': {name: version(name) for name in (
            'torch', 'torchvision', 'timm', 'numpy', 'scipy', 'scikit-learn', 'Pillow', 'einops'
        )},
        'cuda_runtime': torch.version.cuda,
    }
    repo = cli.dinov2_repo or Path(torch.hub.get_dir()) / 'facebookresearch_dinov2_main'
    if repo.is_dir():
        result['recorded_dinov2_source'] = python_source_fingerprint(repo)
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    cli.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Verified {len(records)} exact model exports; manifest={cli.output}')


if __name__ == '__main__':
    main()
