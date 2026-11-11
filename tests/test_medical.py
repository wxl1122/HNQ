"""Protect medical transfer from duplicate-content leakage and wrong source heads."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from run import validate_checkpoint_fold
from utils.medical import MedicalEvalDataset, prepare_split, validate_split
from utils.metrics import FewShotMetric


class MedicalTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        base = self.root / 'brats'
        metadata = {split: {'brain': []} for split in ('train', 'test')}
        for split in ('train', 'test'):
            for index in range(24 if split == 'test' else 20):
                anomaly = int(index >= 20)
                kind = 'lesion' if anomaly else 'good'
                name = f'brain/{split}/{kind}/{index:03d}.png'
                mask = f'brain/ground_truth/lesion/{index:03d}_mask.png' if anomaly else ''
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (28, 28), color=index * 10).save(path)
                if mask:
                    path = base / mask
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('L', (28, 28), color=255).save(path)
                metadata[split]['brain'].append({'img_path': name, 'mask_path': mask,
                                                 'anomaly': anomaly, 'specie_name': kind})
        (base / 'meta.json').write_text(json.dumps(metadata))
        self.plan = prepare_split(self.root, [111, 222])
        self.split = self.root / 'split.json'
        self.split.write_text(json.dumps(self.plan))

    def test_copies_are_excluded_by_content_across_all_shots_and_seeds(self):
        self.assertEqual(self.plan['original_train_test_content_overlap'], 20)
        query_hashes = {self.plan['images'][r['img_path']]['pixel_sha256'] for r in self.plan['queries']}
        common = None
        for seed in [111, 222]:
            draws = {}
            for shot in [1, 2, 4]:
                dataset = MedicalEvalDataset(data_root=self.root, medical_split=self.split,
                                             shot=(shot, 0), seed=seed, transform=transforms.ToTensor())
                manifest = dataset.evaluation_manifest
                if common is not None:
                    self.assertEqual(common, manifest['queries'])
                common = manifest['queries']
                draws[shot] = manifest['support_paths']['brain']
                for path in draws[shot]:
                    relative = Path(path).relative_to(self.root).as_posix()
                    self.assertNotIn(self.plan['images'][relative]['pixel_sha256'], query_hashes)
            self.assertEqual(draws[4][:1], draws[1])
            self.assertEqual(draws[4][:2], draws[2])

    def test_altered_split_and_modified_images_are_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan['queries'].pop()
        with self.assertRaisesRegex(ValueError, 'deterministic content audit'):
            validate_split(plan, self.root)
        path = self.root / self.plan['queries'][0]['img_path']
        Image.new('RGB', (28, 28), color=255).save(path)
        with self.assertRaisesRegex(ValueError, 'image changed'):
            validate_split(self.plan, self.root)

    def test_medical_transfer_requires_known_mvtec_source(self):
        validate_checkpoint_fold({'args': {'fold': 1, 'data_mode': 'mvtec_visa'}}, 1, 'mvtec_brats')
        for state in ({}, {'args': {'fold': 0, 'data_mode': 'mvtec_visa'}},
                      {'args': {'fold': 1, 'data_mode': 'realiad'}}):
            with self.assertRaisesRegex(ValueError, 'MVTec-trained'):
                validate_checkpoint_fold(state, 1, 'mvtec_brats')

    def test_existing_but_misassigned_lesion_mask_is_rejected(self):
        path = self.root / 'brats/meta.json'
        metadata = json.loads(path.read_text())
        metadata['test']['brain'][20]['mask_path'] = metadata['test']['brain'][21]['mask_path']
        path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'image/mask filename mismatch'):
            prepare_split(self.root, [111, 222])

    def test_auroc_only_equals_full_metrics_exactly(self):
        rng = np.random.default_rng(22)
        full, reduced = FewShotMetric(['brain']), FewShotMetric(['brain'], metric_set='auroc')
        labels = np.array([0, 0, 1, 1])
        masks = np.zeros((4, 12, 12), dtype=np.float32)
        masks[2:, 3:7, 4:8] = 1
        for metric in (full, reduced):
            metric.stat_buffer['brain'] = {
                'i_score': [0.1, 0.3, 0.7, 0.9], 'i_gt': list(labels),
                'p_score_map': list(rng.random((4, 12, 12))), 'p_gt': list(masks)}
        reduced.stat_buffer = copy.deepcopy(full.stat_buffer)
        first, second = full.get_scores(), reduced.get_scores()
        self.assertEqual((first[0], first[3]), second)

    def test_cached_medical_supports_preserve_inputs(self):
        args = dict(data_root=self.root, medical_split=self.split, shot=(4, 0),
                    seed=111, transform=transforms.ToTensor())
        plain = MedicalEvalDataset(**args)
        cached = MedicalEvalDataset(**args, cache_supports=True)
        for i in [0, 1, len(plain)-1]:
            for key in ('query', 'support_normal'):
                for a, b in zip(plain[i][key], cached[i][key]):
                    self.assertTrue(torch.equal(a, b))


if __name__ == '__main__':
    unittest.main()
