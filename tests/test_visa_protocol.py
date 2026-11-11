"""Reverse-domain coverage and training isolation regression tests."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from PIL import Image
from torchvision import transforms

from run import validate_checkpoint_fold
from tools.audit_exhaustive_dataset import audit_dataset, visa_split_records
from utils.dataset import ExhaustiveEvalDataset, FSDataset


class VisaProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        csv_rows = []
        for domain, count in (('mvtec', 15), ('visa', 12)):
            metadata = {'train': {}, 'test': {}}
            base = self.root / domain
            for index in range(count):
                product = f'{domain}_{index:02d}'
                for split in ('train', 'test'):
                    rows = []
                    for item in range(5 if split == 'train' else 2):
                        anomaly = int(split == 'test' and item == 1)
                        kind = 'bad' if anomaly else 'good'
                        path = f'{product}/{split}/{kind}/{item}.png'
                        mask = f'{product}/ground_truth/bad/{item}_mask.png' if anomaly else ''
                        for filename in (path, mask):
                            if filename:
                                asset = base / filename
                                asset.parent.mkdir(parents=True, exist_ok=True)
                                Image.new('L' if filename == mask else 'RGB', (28, 28),
                                          color=255 if anomaly else 0).save(asset)
                        rows.append({'img_path': path, 'mask_path': mask,
                                     'anomaly': anomaly, 'cls_name': product, 'specie_name': kind})
                        if domain == 'visa':
                            csv_rows.append({'object': product, 'split': split,
                                             'label': 'anomaly' if anomaly else 'normal',
                                             'image': path, 'mask': mask})
                    metadata[split][product] = rows
            (base / 'meta.json').write_text(json.dumps(metadata))
        csv_path = self.root / 'visa/split_csv/1cls.csv'
        csv_path.parent.mkdir()
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['object', 'split', 'label', 'image', 'mask'])
            writer.writeheader()
            writer.writerows(csv_rows)

    def test_reverse_fold_uses_only_mvtec_source_and_visa_target(self):
        with mock.patch('builtins.print'):
            source = FSDataset(data_root=str(self.root), fold=1, split='train', shot=(1, 0))
            target = FSDataset(data_root=str(self.root), fold=1, split='eval', shot=(1, 0))
        self.assertEqual(len(source.products), 15)
        self.assertEqual(len(target.products), 12)
        self.assertTrue(all(name.startswith('mvtec_') for name in source.products))
        self.assertTrue(all(name.startswith('visa_') for name in target.products))
        self.assertFalse(set(source.products) & set(target.products))
        with self.assertRaisesRegex(ValueError, 'source training categories'):
            validate_checkpoint_fold({'args': {'fold': 0}}, 1)

    def test_official_csv_full_coverage_and_nested_supports(self):
        with mock.patch('builtins.print'):
            result = audit_dataset(self.root, [111, 222], fold=1)
        self.assertEqual(result['target'], 'VisA')
        self.assertEqual(result['disk_test_images'], 24)
        self.assertEqual(result['normal_queries'], 12)
        self.assertEqual(result['anomalous_queries'], 12)
        self.assertEqual(len(result['runs']), 6)
        self.assertTrue(result['nested_support_sets_verified'])
        self.assertIn('official_split_sha256', result)
        self.assertTrue(result['official_anomaly_masks_verified'])

    def test_cached_support_inputs_equal_uncached_inputs(self):
        arguments = dict(data_root=str(self.root), fold=1, shot=(4, 0), seed=111,
                         transform=transforms.ToTensor())
        with mock.patch('builtins.print'):
            plain = ExhaustiveEvalDataset(**arguments)
            cached = ExhaustiveEvalDataset(**arguments, cache_supports=True)
        with mock.patch.object(cached, 'read_data', wraps=cached.read_data) as reader:
            actual = [cached[index] for index in (0, 1)]
            self.assertEqual(reader.call_count, 3)  # two queries, one support-set read
        for index, result in enumerate(actual):
            expected = plain[index]
            self.assertEqual(result['sample_product'], expected['sample_product'])
            self.assertTrue(torch.equal(result['image_level_label'], expected['image_level_label']))
            for key in ('query', 'support_normal'):
                for first, second in zip(result[key], expected[key]):
                    self.assertTrue(torch.equal(first, second))
        self.assertEqual(cached.evaluation_manifest, plain.evaluation_manifest)

    def test_forward_fold_audit_still_works(self):
        with mock.patch('builtins.print'):
            result = audit_dataset(self.root, [111], fold=0)
        self.assertEqual(result['target'], 'MVTec AD')
        self.assertEqual(result['disk_test_images'], 30)

    def test_omitted_test_image_in_metadata_is_rejected(self):
        path = self.root / 'visa/meta.json'
        metadata = json.loads(path.read_text())
        metadata['test']['visa_00'].pop()
        path.write_text(json.dumps(metadata))
        with mock.patch('builtins.print'), self.assertRaisesRegex(ValueError, 'omitted=1'):
            audit_dataset(self.root, [111], fold=1)

    def test_wrong_test_label_is_rejected(self):
        path = self.root / 'visa/meta.json'
        metadata = json.loads(path.read_text())
        metadata['test']['visa_00'][1]['anomaly'] = 0
        path.write_text(json.dumps(metadata))
        with mock.patch('builtins.print'), self.assertRaisesRegex(ValueError, 'class/label mismatch'):
            audit_dataset(self.root, [111], fold=1)

    def test_truncated_support_pool_is_rejected(self):
        path = self.root / 'visa/meta.json'
        metadata = json.loads(path.read_text())
        metadata['train']['visa_00'].pop()
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'support pool differs'):
            audit_dataset(self.root, [111], fold=1)

    def test_wrong_but_existing_anomaly_mask_is_rejected(self):
        path = self.root / 'visa/meta.json'
        metadata = json.loads(path.read_text())
        metadata['test']['visa_00'][1]['mask_path'] = metadata['test']['visa_01'][1]['mask_path']
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'anomaly mask mismatch'):
            audit_dataset(self.root, [111], fold=1)

    def test_duplicate_csv_image_is_rejected(self):
        path = self.root / 'visa/split_csv/1cls.csv'
        lines = path.read_text().splitlines()
        path.write_text('\n'.join(lines + [lines[1]]) + '\n')
        with self.assertRaisesRegex(ValueError, 'Duplicate VisA CSV'):
            visa_split_records(self.root / 'visa')

    def test_recipes_support_explicit_reverse_source_training(self):
        from run import parse_args
        recipes = list(Path('configs/train').glob('*.json'))
        self.assertEqual(len(recipes), 2)
        for path in recipes:
            with mock.patch('sys.argv', ['run.py', '--config', str(path), '--fold', '1']):
                args = parse_args()
            self.assertEqual(args.fold, 1)
            self.assertEqual(args.mode, 'train')
            self.assertEqual(args.eval_frequency, 0)
            self.assertIsNone(args.checkpoint)

    def test_training_queue_rejects_more_than_two_or_duplicate_gpus(self):
        from tools.train_source_heads import main
        for gpus in (['0', '1', '2'], ['0', '0']):
            argv = ['train_source_heads', '--recipes', 'recipe.json', '--gpus', *gpus,
                    '--data-root', 'data', '--dinov2-repo', 'dino',
                    '--pretrained-weights-dir', 'weights', '--output-dir', 'unused']
            with mock.patch('sys.argv', argv), mock.patch('sys.stderr'), \
                    mock.patch('tools.train_source_heads.subprocess.check_output') as launch:
                with self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                launch.assert_not_called()

    def test_evaluation_queue_enforces_the_same_gpu_limit(self):
        from tools.evaluate_seed_queue import main
        argv = ['evaluate_seed_queue', '--config', 'profile.json', '--manifest', 'manifest.json',
                '--gpus', '0', '1', '2', '--data-root', 'data', '--dinov2-repo', 'dino',
                '--pretrained-weights-dir', 'weights', '--output-dir', 'unused']
        with mock.patch('sys.argv', argv), mock.patch('sys.stderr'), \
                mock.patch('tools.evaluate_seed_queue.subprocess.check_output') as launch:
            with self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            launch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
