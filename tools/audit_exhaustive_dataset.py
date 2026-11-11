"""Audit complete MVTec/VisA coverage against independent split definitions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from utils.dataset import ExhaustiveEvalDataset


def visa_split_records(root):
    """Read the official CSV, not a possibly incomplete generated meta.json."""
    records = {'train': {}, 'test': {}}
    seen = set()
    with (root / 'split_csv/1cls.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row['split'] not in records or row['label'] not in ('normal', 'anomaly'):
                raise ValueError(f'Unsupported VisA split/label: {row}')
            path = (root / row['image']).resolve()
            if not path.is_relative_to(root.resolve()):
                raise ValueError(f'VisA image lies outside the dataset root: {path}')
            if path in seen:
                raise ValueError(f'Duplicate VisA CSV image or train/test overlap: {path}')
            seen.add(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            anomaly = int(row['label'] == 'anomaly')
            if anomaly and (not row['mask'] or not (root / row['mask']).is_file()):
                raise FileNotFoundError(f'Missing VisA anomaly mask for {path}')
            records[row['split']][path] = {
                'product': row['object'], 'anomaly': anomaly,
                'mask_path': (root / row['mask']).resolve() if anomaly else None,
            }
    return records


def audit_dataset(data_root, seeds, fold=0):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Declare distinct support seeds')
    if fold not in (0, 1):
        raise ValueError('fold must be 0 (MVTec target) or 1 (VisA target)')
    target_domain = 'mvtec' if fold == 0 else 'visa'
    target_root = data_root / target_domain
    official = visa_split_records(target_root) if fold == 1 else None
    if official is not None:
        test_records = official['test']
        meta = json.loads((target_root / 'meta.json').read_text())
        metadata_normals = {(target_root / record['img_path']).resolve()
                            for samples in meta['train'].values() for record in samples
                            if not record['anomaly']}
        official_normals = {path for path, record in official['train'].items()
                            if not record['anomaly']}
        if metadata_normals != official_normals:
            raise ValueError('VisA normal support pool differs from the official CSV')
        for samples in meta['test'].values():
            for sample in samples:
                image_path = (target_root / sample['img_path']).resolve()
                if sample['anomaly'] and image_path in test_records:
                    if ((target_root / sample['mask_path']).resolve()
                            != test_records[image_path]['mask_path']):
                        raise ValueError(f'Official VisA anomaly mask mismatch: {image_path}')
    else:
        image_suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
        test_records = {
            path.resolve(): {'product': path.relative_to(target_root).parts[0],
                             'anomaly': int(path.parent.name != 'good')}
            for path in target_root.glob('*/test/*/*')
            if path.is_file() and path.suffix.lower() in image_suffixes
        }
    if not test_records:
        raise ValueError('No official test images found')
    entries = []
    common_queries = None
    for seed in seeds:
        by_shot = {}
        for shot in (1, 2, 4):
            dataset = ExhaustiveEvalDataset(data_root=str(data_root), fold=fold,
                                            shot=(shot, 0), seed=seed)
            manifest = dataset.evaluation_manifest
            queries = {Path(record['path']) for record in manifest['queries']}
            if queries != set(test_records):
                raise ValueError(f'Test-file coverage differs: omitted={len(set(test_records) - queries)}, '
                                 f'extra={len(queries - set(test_records))}')
            for record in manifest['queries']:
                expected = {key: test_records[Path(record['path'])][key]
                            for key in ('product', 'anomaly')}
                if expected != {key: record[key] for key in ('product', 'anomaly')}:
                    raise ValueError(f'Official class/label mismatch: {record["path"]}')
            if common_queries is not None and common_queries != queries:
                raise ValueError('Test queries differ across shot counts or seeds')
            common_queries = queries
            by_shot[shot] = manifest['support_paths']
            entries.append({'seed': seed, 'shot': shot,
                            'number_of_unique_queries': len(queries),
                            'query_support_overlap': manifest['query_support_overlap'],
                            'query_paths_sha256': manifest['query_paths_sha256'],
                            'support_paths': manifest['support_paths']})
        for product, paths in by_shot[4].items():
            if paths[:1] != by_shot[1][product] or paths[:2] != by_shot[2][product]:
                raise ValueError(f'Non-nested support sets: {seed}: {product}')
    metadata = {}
    for domain in ('mvtec', 'visa'):
        path = data_root / domain / 'meta.json'
        metadata[domain] = {'path': str(path.resolve()),
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    result = {'target': 'MVTec AD' if fold == 0 else 'VisA', 'fold': fold,
              'metadata': metadata, 'declared_seeds': seeds,
              'disk_test_images': len(test_records), 'omitted_test_images': 0,
              'normal_queries': sum(not r['anomaly'] for r in test_records.values()),
              'anomalous_queries': sum(r['anomaly'] for r in test_records.values()),
              'nested_support_sets_verified': True, 'runs': entries}
    if official is not None:
        split_path = target_root / 'split_csv/1cls.csv'
        result['official_split_sha256'] = hashlib.sha256(split_path.read_bytes()).hexdigest()
        result['official_anomaly_masks_verified'] = True
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1], default=0)
    parser.add_argument('--seeds', nargs='+', type=int, default=[111, 222, 333])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit_dataset(args.data_root, args.seeds, args.fold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'All {result["disk_test_images"]} {result["target"]} test files covered exactly once; '
          f'1/2/4-shot nested references verified across {len(args.seeds)} seeds')


if __name__ == '__main__':
    main()
