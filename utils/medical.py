"""Content-disjoint evaluation of the supplied two-dimensional BraTS conversion."""

import hashlib
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image

from utils.dataset import ExhaustiveEvalDataset


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Medical input lies outside the data root: {relative}')
    return path


def prepare_split(data_root, seeds, maximum_shot=4):
    """Reserve the union of declared supports before seeing model predictions."""
    if not seeds or len(set(seeds)) != len(seeds) or maximum_shot != 4:
        raise ValueError('Declare distinct support seeds and maximum shot four')
    root = Path(data_root).resolve()
    meta_path = root / 'brats/meta.json'
    meta = json.loads(meta_path.read_text())
    if set(meta['train']) != {'brain'} or set(meta['test']) != {'brain'}:
        raise ValueError('Expected the supplied single-category BraTS conversion')
    images, masks, rows = {}, {}, {}
    for split in ('train', 'test'):
        rows[split] = []
        for entry in meta[split]['brain']:
            relative = 'brats/' + entry['img_path']
            if relative in images:
                raise ValueError('Duplicate metadata image path')
            path = resolve(root, relative)
            with Image.open(path) as opened:
                image = opened.convert('RGB')
                size = image.size
                pixel_hash = hashlib.sha256(str(size).encode() + image.tobytes()).hexdigest()
            images[relative] = {'sha256': digest(path), 'pixel_sha256': pixel_hash}
            mask = 'brats/' + entry['mask_path'] if entry['anomaly'] else ''
            if mask:
                mask_path = resolve(root, mask)
                expected = root / 'brats/brain/ground_truth' / path.parent.name / (path.stem + '_mask.png')
                if mask_path != expected.resolve():
                    raise ValueError(f'Medical image/mask filename mismatch: {relative}')
                with Image.open(mask_path) as opened:
                    if opened.size != size or not np.asarray(opened.convert('L')).any():
                        raise ValueError(f'Empty or incorrectly sized lesion mask: {mask}')
                masks[mask] = digest(mask_path)
            rows[split].append({'product': 'brain', 'img_path': relative,
                                'mask_path': mask, 'anomaly': int(entry['anomaly'])})
        disk = {p.relative_to(root).as_posix()
                for p in (root / f'brats/brain/{split}').glob('*/*')
                if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')}
        if disk != {r['img_path'] for r in rows[split]}:
            raise ValueError(f'Medical metadata/disk coverage differs: {split}')
        hashes = [images[r['img_path']]['pixel_sha256'] for r in rows[split]]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f'Duplicate image content within {split}')
    if any(r['anomaly'] for r in rows['train']):
        raise ValueError('The medical reference pool must contain only normals')
    pool = sorted(rows['train'], key=lambda r: r['img_path'])
    if len(pool) < maximum_shot:
        raise ValueError('Insufficient distinct normal references')
    supports = {}
    for seed in seeds:
        shuffled = list(pool)
        random.Random(f'{seed}:brain').shuffle(shuffled)
        supports[str(seed)] = {'brain': [r['img_path'] for r in shuffled[:maximum_shot]]}
    reserved_hashes = {images[path]['pixel_sha256']
                       for selected in supports.values() for path in selected['brain']}
    ordered = sorted(rows['test'], key=lambda r: (r['anomaly'], r['img_path']))
    queries = [r for r in ordered if images[r['img_path']]['pixel_sha256'] not in reserved_hashes]
    excluded = [r['img_path'] for r in ordered
                if images[r['img_path']]['pixel_sha256'] in reserved_hashes]
    if {r['anomaly'] for r in queries} != {0, 1}:
        raise ValueError('Both query labels must remain after support isolation')
    train_hashes = {images[r['img_path']]['pixel_sha256'] for r in rows['train']}
    test_hashes = {images[r['img_path']]['pixel_sha256'] for r in rows['test']}
    return {
        'format_version': 1, 'protocol': 'brats_2d_content_disjoint',
        'declared_seeds': seeds, 'maximum_shot': maximum_shot,
        'selection': 'seeded_full_pool_shuffle_prefix',
        'query_policy': 'all_test_images_except_content_copies_of_union_of_declared_supports',
        'metadata_path': 'brats/meta.json', 'metadata_sha256': digest(meta_path),
        'original_test_images': len(rows['test']),
        'original_train_test_content_overlap': len(train_hashes & test_hashes),
        'number_of_queries': len(queries),
        'normal_queries': sum(not r['anomaly'] for r in queries),
        'anomalous_queries': sum(r['anomaly'] for r in queries),
        'queries': queries, 'supports': supports, 'excluded_queries': excluded,
        'images': images, 'masks': masks,
        'patient_identity_available': False,
    }


def validate_split(plan, data_root):
    root = Path(data_root).resolve()
    if digest(resolve(root, plan['metadata_path'])) != plan['metadata_sha256']:
        raise ValueError('Medical metadata changed after split declaration')
    if plan['protocol'] != 'brats_2d_content_disjoint':
        raise ValueError('Unsupported medical split protocol')
    for relative, record in plan['images'].items():
        if digest(resolve(root, relative)) != record['sha256']:
            raise ValueError(f'Medical image changed: {relative}')
    for relative, sha in plan['masks'].items():
        if digest(resolve(root, relative)) != sha:
            raise ValueError(f'Medical mask changed: {relative}')
    # Rebuild independently from the unchanged on-disk metadata and images.
    # This also rejects altered exclusions, support draws, labels or pixel hashes.
    if plan != prepare_split(root, plan['declared_seeds'], plan['maximum_shot']):
        raise ValueError('Medical split differs from deterministic content audit')


class MedicalEvalDataset(ExhaustiveEvalDataset):
    """Reuse the existing tensors/preprocessing and fixed-reference evaluator."""

    def __init__(self, *, data_root, medical_split, shot, seed, transform,
                 cache_supports=False, **kwargs):
        if kwargs.get('split', 'eval') != 'eval' or kwargs.get('fold', 1) != 1:
            raise ValueError('Medical evaluation requires source-MVTec fold 1')
        self.data_root = str(Path(data_root).resolve())
        self.data_mode, self.split = 'mvtec_brats', 'val'
        self.n_shot, self.a_shot = shot
        self.seed, self.transform = seed, transform
        mask_size = kwargs.get('mask_size')
        self.mask_size = (mask_size, mask_size) if mask_size else None
        self.cache_supports, self._transformed_supports = cache_supports, {}
        plan = json.loads(Path(medical_split).read_text())
        validate_split(plan, Path(self.data_root))
        if seed not in plan['declared_seeds'] or self.a_shot or self.n_shot not in (1, 2, 4):
            raise ValueError('Undeclared medical seed or support budget')
        self.products, self.all_product, self.class_ids = ['brain'], ['brain'], [0]
        def entry(record):
            return {**record,
                    'img_path': str(resolve(Path(self.data_root), record['img_path'])),
                    'mask_path': str(resolve(Path(self.data_root), record['mask_path']))
                    if record['mask_path'] else ''}
        self.query_samples = [(r['product'], entry(r)) for r in plan['queries']]
        self.fixed_supports = {'brain': [entry({'img_path': path, 'mask_path': '', 'anomaly': 0})
                                       for path in plan['supports'][str(seed)]['brain'][:self.n_shot]]}
        self.evaluation_manifest = self.audit_manifest()
        self.evaluation_manifest.update(
            medical_split_sha256=digest(Path(medical_split)),
            content_query_support_overlap=0,
            original_test_images=plan['original_test_images'],
            excluded_reference_copies=len(plan['excluded_queries']))
