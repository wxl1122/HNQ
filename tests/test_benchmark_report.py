"""A table-ready benchmark must be complete and match its frozen protocol."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.check_release import file_digest
from tools.summarize_benchmark import attach_paired_baseline, build_report, compare_paper_tolerance
from tools.summarize_full_evaluation import METRICS


class BenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = self.root = Path(self.temporary.name)
        self.profile = root / 'profile.json'
        profile = {'mode': 'eval', 'fold': 1, 'n_shot': 1, 'eval_seeds': [111, 222],
                   'eval_shots': [1, 2, 4], 'eval_protocol': 'exhaustive',
                   'checkpoint': 'head.pth', 'image_experts': [], 'pixel_experts': [],
                   'image_size': 448, 'image_proxy_weight': 0.35}
        self.profile.write_text(json.dumps(profile))
        metadata = root / 'meta.json'
        metadata.write_text('{}')
        self.manifest = root / 'manifest.json'
        self.manifest.write_text(json.dumps({
            'config_sha256': file_digest(self.profile), 'weights': [{'path': 'head.pth'}],
            'software': {}, 'cuda_runtime': '12.8'}))
        queries = [{'path': str(root / 'test/normal.png'), 'product': 'widget', 'anomaly': 0},
                   {'path': str(root / 'test/anomaly.png'), 'product': 'widget', 'anomaly': 1}]
        self.records = {}
        audits = []
        for shot in (1, 2, 4):
            for seed in (111, 222):
                supports = {'widget': [str(root / f'train/{seed}_{i}.png') for i in range(shot)]}
                audits.append({'seed': seed, 'shot': shot, 'query_paths_sha256': 'fixed',
                               'support_paths': supports})
                values = {name: 0.8 + (0.02 if seed == 222 else 0) for name in METRICS}
                self.records[1, shot, seed] = {
                    'config': {**profile, 'seed': seed, 'n_shot': shot, 'data_root': str(root)},
                    'manifest': {'queries': queries, 'number_of_queries': 2,
                                 'query_paths_sha256': 'fixed', 'support_paths': supports},
                    'metrics': {**values, 'per_category': {'widget': values}},
                }
        self.audit = root / 'audit.json'
        self.audit.write_text(json.dumps({
            'fold': 1, 'target': 'VisA', 'declared_seeds': [111, 222], 'omitted_test_images': 0,
            'nested_support_sets_verified': True, 'disk_test_images': 2,
            'normal_queries': 1, 'anomalous_queries': 1, 'runs': audits,
            'metadata': {'visa': {'path': str(metadata), 'sha256': file_digest(metadata)}}}))
        self.head = {'args': {'fold': 1, 'data_mode': 'mvtec_visa', 'mode': 'train',
                              'eval_frequency': 0, 'checkpoint': None, 'epochs': 2},
                     'epoch': 1, 'training_configuration_known': True}

    def report(self, records=None, head=None):
        with mock.patch('tools.summarize_benchmark.verify_manifest'), \
                mock.patch('tools.summarize_benchmark.torch.load', return_value=head or self.head):
            return build_report(self.records if records is None else records,
                                self.profile, self.audit, self.manifest)

    def test_complete_means_and_sample_standard_deviation(self):
        report = self.report()
        self.assertTrue(report['complete'])
        self.assertEqual(report['number_of_evaluations'], 12)
        self.assertEqual(len(report['runs']), 6)
        value = report['summary'][0]['statistics']['i_auroc']
        self.assertAlmostEqual(value['mean_percent'], 81)
        self.assertAlmostEqual(value['std_pp'], 2 ** 0.5)

    def test_missing_seed_cannot_be_published(self):
        records = dict(self.records)
        records.pop((1, 4, 222))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.report(records)

    def test_tolerance_distinguishes_exceeding_from_absolute_reproduction(self):
        from run import PAPER_TARGETS
        report = self.report()
        for row in report['summary']:
            for metric, expected in PAPER_TARGETS[1, row['shot']].items():
                row['statistics'][metric]['mean_percent'] = 100 * expected + 1
        comparison = compare_paper_tolerance(report)
        self.assertTrue(comparison['all_mean_metrics_meet_performance_floor'])
        self.assertFalse(comparison['all_mean_metrics_within_absolute_tolerance'])
        self.assertFalse(comparison['all_individual_runs_meet_performance_floor'])
        self.assertTrue(comparison['individual_run_metrics_below_floor'])
        report['summary'][0]['statistics']['i_auroc']['mean_percent'] = 100 * PAPER_TARGETS[1, 1]['i_auroc'] - 0.5001
        self.assertFalse(compare_paper_tolerance(report)['all_mean_metrics_meet_performance_floor'])
        report['complete'] = False
        with self.assertRaisesRegex(ValueError, 'complete benchmark'):
            compare_paper_tolerance(report)

    def test_forward_head_cannot_fill_reverse_table(self):
        head = copy.deepcopy(self.head)
        head['args']['fold'] = 0
        with self.assertRaisesRegex(ValueError, 'source-only'):
            self.report(head=head)

    def test_changed_score_weight_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[1, 1, 111]['config']['image_proxy_weight'] = 0.7
        with self.assertRaisesRegex(ValueError, 'profile differs'):
            self.report(records)

    def test_nan_score_is_rejected(self):
        records = copy.deepcopy(self.records)
        records[1, 1, 111]['metrics']['i_auroc'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid metric'):
            self.report(records)

    def test_metric_roundoff_is_preserved_but_real_overflow_is_rejected(self):
        records = copy.deepcopy(self.records)
        for row in records.values():
            row['metrics']['i_ap'] = 1.0000000000000002
            row['metrics']['per_category']['widget']['i_ap'] = 1.0000000000000002
        report = self.report(records)
        self.assertGreater(report['runs'][0]['metrics_percent']['i_ap'], 100)
        records[1, 1, 111]['metrics']['i_ap'] = 1.001
        with self.assertRaisesRegex(ValueError, 'Invalid metric'):
            self.report(records)

    def test_changed_supports_are_rejected(self):
        records = copy.deepcopy(self.records)
        records[1, 1, 111]['manifest']['support_paths'] = {'widget': ['other.png']}
        with self.assertRaisesRegex(ValueError, 'independently audited'):
            self.report(records)

    def test_paired_baseline_preserves_verified_measurements(self):
        enhanced, baseline = self.report(), self.report()
        result = attach_paired_baseline(enhanced, baseline)
        self.assertEqual(result['paired_baseline']['number_of_pairs'], 6)
        self.assertEqual(result['runs'][0]['baseline_percent'], baseline['runs'][0]['metrics_percent'])
        self.assertTrue(all(value == 0 for value in result['runs'][0]['enhanced_minus_baseline_pp'].values()))

    def test_unpaired_baseline_cannot_be_attached(self):
        enhanced, baseline = self.report(), self.report()
        baseline['runs'][0]['supports']['widget'] = ['different.png']
        with self.assertRaisesRegex(ValueError, 'not paired'):
            attach_paired_baseline(enhanced, baseline)


if __name__ == '__main__':
    unittest.main()
