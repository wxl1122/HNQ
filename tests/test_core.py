import random
import json
import tempfile
from pathlib import Path
import types
import unittest
from unittest import mock

import torch
from torch import nn

from models.model import AML
from models.ensemble import PixelExpertEnsemble, fuse_pixel_experts
from run import configure_trainable_parameters, load_checkpoint, validate_checkpoint_fold
from utils.dataset import ExhaustiveEvalDataset, FSDataset
from utils.metrics import FewShotMetric
from utils.checkpoint import frozen_encoder_signature
from utils.support_views import normal_reference_views
from tools.summarize_full_evaluation import summarize, publication_report
from tools.check_release import file_digest, verify_manifest


class DummyDino(nn.Module):
    patch_size = 14
    blocks = (None, None)

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(1))

    def get_intermediate_layers(self, images, n=1, return_class_token=False):
        batch, _, height, width = images.shape
        tokens = (height // 14) * (width // 14)
        base = images.mean((1, 2, 3), keepdim=False)[:, None, None]
        positions = torch.arange(tokens, device=images.device)[None, :, None] / tokens
        feature = (base + positions).expand(batch, tokens, 384) * self.anchor
        features = (feature * 0.9, feature) if isinstance(n, list) else (feature,)
        if return_class_token:
            return tuple((tokens, tokens.mean(dim=1)) for tokens in features)
        return features


def model_args():
    return types.SimpleNamespace(
        backbone_name="dinov2_vits14",
        dinov2_repo=None,
        num_a_queries=25,
        n_shot=1,
        image_size=28,
        image_topk=4,
    )


class ModelTests(unittest.TestCase):
    def test_versioned_source_target_fold_cannot_be_silently_swapped(self):
        state = {'args': {'fold': 0, 'data_mode': 'mvtec_visa'}}
        validate_checkpoint_fold(state, 0)
        with self.assertRaisesRegex(ValueError, 'source training categories'):
            validate_checkpoint_fold(state, 1)
        validate_checkpoint_fold(state, 1, 'realiad')
        validate_checkpoint_fold({}, 1)  # Unversioned provenance remains unknown.

    def build_model(self):
        with mock.patch("torch.hub.load", return_value=DummyDino()):
            model = AML(model_args())
        configure_trainable_parameters(model)
        return model

    def test_forward_shapes_and_frozen_encoder_mode(self):
        model = self.build_model().train()
        self.assertFalse(model.vit_encoder.training)
        query = torch.rand(2, 1, 3, 28, 28)
        mask = torch.zeros(2, 28, 28)
        label = torch.tensor([0, 1])
        support = [torch.rand(2, 1, 3, 28, 28), torch.zeros(2, 1, 28, 28)]
        image_logits, pixel_logits, _, _, _ = model(
            model_args(), query, mask, label, support
        )
        self.assertEqual(tuple(image_logits.shape), (2, 2))
        self.assertEqual(tuple(pixel_logits.shape), (2, 2, 28, 28))
        outputs, (proxy, nearest) = model(
            model_args(), query, mask, label, support,
            return_score_components=True,
        )
        self.assertEqual(len(outputs), 5)
        self.assertEqual(tuple(proxy.shape), (2, 4))
        self.assertEqual(tuple(nearest.shape), (2, 4))

    def test_legacy_checkpoint_matches_25_proxy_model(self):
        source, model = self.build_model(), self.build_model()
        legacy = {name: value for name, value in source.state_dict().items()
                  if not name.startswith('vit_encoder.')}
        self.assertEqual(tuple(legacy["learnable_token.weight"].shape), (25, 384))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic_legacy.pth'
            torch.save(legacy, path)
            load_checkpoint(path, model)
        for name, value in legacy.items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    def test_checkpoint_rejects_a_different_frozen_encoder(self):
        source = self.build_model()
        state = {'model': {name: value for name, value in source.state_dict().items()
                           if not name.startswith('vit_encoder.')},
                 'frozen_encoder_signature': frozen_encoder_signature(source)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'inference.pth'
            torch.save(state, path)
            load_checkpoint(path, self.build_model())
            changed = self.build_model()
            with torch.no_grad():
                changed.vit_encoder.anchor.fill_(2)
            with self.assertRaisesRegex(RuntimeError, 'Frozen encoder'):
                load_checkpoint(path, changed)

    def test_frozen_feature_cache_preserves_outputs_and_evicts(self):
        model = self.build_model()
        image = torch.rand(2, 1, 3, 28, 28)
        expected = model.feature_forward(image)
        model.feature_cache_size = 1
        with mock.patch.object(model.vit_encoder, 'get_intermediate_layers',
                               wraps=model.vit_encoder.get_intermediate_layers) as encoder:
            actual = model.feature_forward(image)
            repeated = model.feature_forward(image[1:])
            self.assertEqual(encoder.call_count, 1)
        torch.testing.assert_close(expected, actual)
        torch.testing.assert_close(expected[1:], repeated)
        self.assertEqual(len(model._feature_cache), 1)

    def test_fixed_support_batch_deduplication_is_exact(self):
        model = self.build_model()
        model.feature_cache_size = 8
        image = torch.rand(1, 2, 3, 28, 28).expand(4, -1, -1, -1, -1)
        expected = model.feature_forward(image)
        model._feature_cache.clear()
        with mock.patch('models.model.hashlib.sha256', wraps=__import__('hashlib').sha256) as digest:
            actual = model.feature_forward(image, fixed_supports=True)
        self.assertEqual(digest.call_count, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_residual_matches_are_aligned_with_nls_values(self):
        model = self.build_model()
        query = torch.tensor([[[[0., 2.], [3., 0.]]]])
        support = torch.tensor([[[[4., 0.], [0., 5.]]]])
        residual, matched = model.get_res_feat(query, support, return_matched=True)
        torch.testing.assert_close(matched, support.flip(-2))
        torch.testing.assert_close(query, residual + matched)
        _, _, normality = model.get_res_feat(query, support, return_matched=True, return_similarity=True)
        torch.testing.assert_close(normality, model.get_pseudo_mask(query, support).squeeze(-1), rtol=0, atol=0)

    def test_matching_layers_reject_duplicate_indices(self):
        args = model_args()
        args.matching_layers = [1, 1]
        with self.assertRaisesRegex(ValueError, 'distinct ascending'):
            AML(args)

    def test_matching_layers_preserve_head_features_and_residual_geometry(self):
        args = model_args()
        args.matching_layers = [0, 1]
        with mock.patch('torch.hub.load', return_value=DummyDino()):
            model = AML(args)
        image = torch.rand(2, 1, 3, 28, 28)
        encoded = model.feature_forward(image)
        expected = model.vit_encoder.get_intermediate_layers(image[:, 0])[0]
        torch.testing.assert_close(encoded[:, 0, :, :384], expected)
        self.assertEqual(encoded.shape[-1], 3 * 384)
        model.feature_cache_size = 8
        cached = model.feature_forward(image)
        head_only = model.feature_forward(image, head_only=True)
        torch.testing.assert_close(head_only, cached[..., :384], rtol=0, atol=0)
        self.assertEqual(next(iter(model._feature_cache.values())).shape[-1], 3 * 384)
        query = torch.tensor([[[[1., 0.], [0., 2.]]]])
        reference = torch.tensor([[[[4., 0.], [0., 5.]]]])
        residual, matched = model.get_res_feat(
            query, reference, return_matched=True,
            query_matching=query.flip(-2), reference_matching=reference)
        torch.testing.assert_close(matched, reference.flip(-2))
        torch.testing.assert_close(residual + matched, query)

    def test_separate_image_readout_reuses_head_and_keeps_pixel_predictions(self):
        import copy
        args = model_args()
        args.matching_layers = [0, 1]
        with mock.patch('torch.hub.load', return_value=DummyDino()):
            model = AML(args).eval()
        query = torch.rand(2, 1, 3, 28, 28)
        support = [torch.rand(2, 1, 3, 28, 28), torch.zeros(2, 1, 28, 28)]
        mask, labels = torch.zeros(2, 28, 28), torch.tensor([0, 1])
        ordinary = model(args, query, mask, labels, support)
        image_args = copy.copy(args)
        image_args.support_rotations = [0, 90]
        image_args._final_layer_matching = True
        expected_image = model(image_args, query, mask, labels, support)[0]
        args.image_support_rotations = [0, 90]
        args.image_final_layer_matching = True
        combined = model(args, query, mask, labels, support)
        torch.testing.assert_close(combined[0], expected_image, rtol=0, atol=0)
        torch.testing.assert_close(combined[1], ordinary[1], rtol=0, atol=0)
        changed = model(args, query, 1-mask, labels.flip(0), support)
        torch.testing.assert_close(combined[0], changed[0], rtol=0, atol=0)
        torch.testing.assert_close(combined[1], changed[1], rtol=0, atol=0)

    def test_shared_encoder_keeps_different_descriptor_caches_separate(self):
        from run import parse_args, with_pixel_experts
        with mock.patch('sys.argv', ['run.py', '--device', 'cpu']):
            args = parse_args()
        args.feature_cache_size = 8
        primary = self.build_model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.pth'
            expert_args = {**vars(args), 'matching_layers': [0, 1]}
            torch.save({'args': expert_args, 'model': primary.state_dict()}, path)
            args.pixel_experts = [str(path)]
            with mock.patch('torch.hub.load', return_value=DummyDino()):
                ensemble = with_pixel_experts(args, primary, torch.device('cpu'), {})
            expert = ensemble.experts[0]
            self.assertIs(expert.vit_encoder, primary.vit_encoder)
            self.assertIsNot(expert._feature_cache, primary._feature_cache)
            image = torch.rand(1, 1, 3, 28, 28)
            torch.testing.assert_close(expert.feature_forward(image)[..., :384], primary.feature_forward(image))

    def test_image_head_reuses_rich_cache_and_declared_image_reference_views(self):
        from run import parse_args, with_pixel_experts
        with mock.patch('sys.argv', ['run.py', '--device', 'cpu', '--matching-layers', '0', '1']):
            args = parse_args()
        args.feature_cache_size = 8
        with mock.patch('torch.hub.load', return_value=DummyDino()):
            primary = AML(args).eval()
        configure_trainable_parameters(primary)
        source_args = {**vars(args), 'matching_layers': None}
        args.image_support_rotations = [0, 90]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image_head.pth'
            torch.save({'args': source_args, 'model': primary.state_dict()}, path)
            args.image_experts = [str(path)]
            with mock.patch('torch.hub.load', return_value=DummyDino()):
                ensemble = with_pixel_experts(args, primary, torch.device('cpu'), {})
            expert = ensemble.image_experts[0]
            self.assertIs(expert.vit_encoder, primary.vit_encoder)
            self.assertIs(expert._frozen_feature_provider, primary)
            self.assertEqual(ensemble.image_args[0].support_rotations, [0, 90])
            query = torch.rand(1, 1, 3, 28, 28)
            with mock.patch.object(primary.vit_encoder, 'get_intermediate_layers',
                                   wraps=primary.vit_encoder.get_intermediate_layers) as encoder:
                packed = primary.feature_forward(query)
                final = expert.feature_forward(query)
            self.assertEqual(encoder.call_count, 1)
            torch.testing.assert_close(final, packed[..., :384], rtol=0, atol=0)
            self.assertEqual(next(iter(primary._feature_cache.values())).shape[-1], 3 * 384)

    def test_attention_masks_stay_with_each_batch_item(self):
        model = self.build_model()
        features = torch.rand(2, 1, 4, 384)
        valid = torch.tensor([[[[1.], [0.], [1.], [0.]]],
                              [[[0.], [1.], [0.], [1.]]]])
        with mock.patch.object(model.first_cross_attention, 'forward',
                               wraps=model.first_cross_attention.forward) as cross:
            model.attention_forward(model.first_cross_attention, model.first_self_attention,
                                    model.learnable_token, features, features, valid)
        mask = cross.call_args.kwargs['memory_mask']
        for item in range(2):
            expected = -1e9 * (1 - valid[item, 0, :, 0])
            torch.testing.assert_close(mask[item * 4:(item + 1) * 4, 0], expected.expand(4, -1))

    def test_release_predictions_do_not_use_query_labels_or_masks(self):
        model = self.build_model().eval()
        query = torch.rand(2, 1, 3, 28, 28)
        support = [torch.rand(2, 2, 3, 28, 28), torch.zeros(2, 2, 28, 28)]
        for nls_values in ('matched', 'spatial_mean'):
            for orientation in ('legacy_anomaly', 'paper_normality'):
                args = model_args()
                args.n_shot = 2
                args.nls_values = nls_values
                args.proxy_orientation = orientation
                args.loss_mode = 'paper' if orientation == 'paper_normality' else 'legacy'
                args.support_rotations = [0, 90, 180, 270]
                first = model(args, query, torch.zeros(2, 28, 28), torch.tensor([0, 1]), support)
                second = model(args, query, torch.ones(2, 28, 28), torch.tensor([1, 0]), support)
                torch.testing.assert_close(first[0], second[0], rtol=0, atol=0)
                torch.testing.assert_close(first[1], second[1], rtol=0, atol=0)

    def test_release_config_loads_and_cli_overrides_defaults(self):
        from run import main, parse_args
        with mock.patch('sys.argv', ['run.py', '--config',
                                    'configs/hnq_base.json', '--seed', '222']):
            args = parse_args()
        self.assertEqual(args.seed, 222)
        self.assertEqual(args.eval_shots, [1, 2, 4])
        self.assertEqual(args.image_experts, ['checkpoints/normality.pth'])
        self.assertEqual(args.pixel_experts, [])
        for recipe in Path('configs/train').glob('*.json'):
            with mock.patch('sys.argv', ['run.py', '--config', str(recipe)]):
                training = parse_args()
            self.assertEqual(training.mode, 'train')
            self.assertEqual(training.eval_frequency, 0)
        args.device = 'cpu'
        with mock.patch('run.parse_args', return_value=args), \
                mock.patch('run.AML', return_value=self.build_model()), \
                mock.patch('run.load_checkpoint', return_value={'args': dict(vars(args))}), \
                mock.patch('run.build_loader'), \
                mock.patch('run.with_pixel_experts'), \
                mock.patch('run.evaluate_and_save') as evaluate, \
                mock.patch('builtins.print'):
            main()
        self.assertEqual(
            [(call.args[0].seed, call.args[0].n_shot) for call in evaluate.call_args_list],
            [(seed, shot) for seed in args.eval_seeds for shot in args.eval_shots],
        )

    def test_paper_loss_requires_normality_orientation(self):
        model = self.build_model()
        args = model_args()
        args.loss_mode = 'paper'
        query = torch.rand(2, 1, 3, 28, 28)
        mask = torch.zeros(2, 28, 28)
        mask[1, 7:14, 7:14] = 1
        support = [torch.rand(2, 1, 3, 28, 28), torch.zeros(2, 1, 28, 28)]
        with self.assertRaisesRegex(ValueError, 'paper loss requires'):
            model(args, query, mask, torch.tensor([0, 1]), support)
        args.proxy_orientation = 'paper_normality'
        outputs = model(args, query, mask, torch.tensor([0, 1]), support)
        loss = outputs[2] + outputs[3] + outputs[4]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()


class DatasetTests(unittest.TestCase):
    def test_preprocessing_cache_preserves_images_masks_and_evicts(self):
        from collections import OrderedDict
        import numpy as np
        from PIL import Image
        from run import image_transform
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path, mask_path = root / 'image.png', root / 'mask.png'
            pixels = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
            Image.fromarray(pixels).save(image_path)
            mask = np.zeros((7, 9), dtype=np.uint8)
            mask[2:5, 3:7] = 255
            Image.fromarray(mask).save(mask_path)
            dataset = FSDataset.__new__(FSDataset)
            dataset.transform = image_transform(28)
            dataset.image_cache_size = 0
            record = {'img_path': str(image_path), 'mask_path': str(mask_path), 'anomaly': 1}
            images, masks, labels = dataset.read_data([record])
            expected_image = dataset.image_list_transform(images)
            expected_mask = dataset.mask_list_transform(masks, (28, 28))
            dataset.image_cache_size = 1
            dataset._image_cache = OrderedDict()
            first = dataset.read_data([record])
            with mock.patch('utils.dataset.Image.open', side_effect=AssertionError('cache miss')):
                second = dataset.read_data([record])
            for actual in (first, second):
                torch.testing.assert_close(dataset.image_list_transform(actual[0]), expected_image, rtol=0, atol=0)
                torch.testing.assert_close(dataset.mask_list_transform(actual[1], (28, 28)), expected_mask, rtol=0, atol=0)
                self.assertEqual(actual[2], labels)
            dataset.read_data([{**record, 'anomaly': 0}])
            self.assertEqual(len(dataset._image_cache), 1)
            # Encoder resolution can change without changing evaluation labels.
            dataset.transform = image_transform(42)
            dataset.mask_size = (28, 28)
            dataset._image_cache.clear()
            resized = dataset.read_data([record])
            self.assertEqual(tuple(dataset.image_list_transform(resized[0]).shape), (1, 3, 42, 42))
            torch.testing.assert_close(dataset.mask_list_transform(resized[1], (28, 28)),
                                       expected_mask, rtol=0, atol=0)

    @staticmethod
    def sample(path, anomaly):
        return {"img_path": path, "mask_path": "", "anomaly": anomaly}

    def make_dataset(self):
        dataset = FSDataset.__new__(FSDataset)
        dataset.split = "val"
        dataset.eval_label_sampling = "balanced"
        dataset.seed = 7
        dataset.class_ids = [0]
        dataset.all_product = ["widget"]
        dataset.n_shot = 1
        dataset.a_shot = 0
        dataset.metadata = {
            "widget": {
                "normal": {"good": [self.sample("n0", 0), self.sample("n1", 0)]},
                "abnormal": {"bad": [self.sample("a0", 1), self.sample("a1", 1)]},
            }
        }
        dataset.read_data = lambda samples: samples
        return dataset

    def test_validation_episode_is_deterministic_and_has_no_leakage(self):
        dataset = self.make_dataset()
        first = dataset.load_frame(0)
        second = dataset.load_frame(0)
        self.assertEqual(first, second)
        query, normal_support, _, _ = first
        self.assertNotEqual(query[0]["img_path"], normal_support[0]["img_path"])

    def test_validation_alternates_labels(self):
        dataset = self.make_dataset()
        self.assertEqual(dataset.load_frame(0)[0][0]["anomaly"], 0)
        self.assertEqual(dataset.load_frame(1)[0][0]["anomaly"], 1)

    def test_train_reference_pool_does_not_change_query_pool(self):
        dataset = self.make_dataset()
        dataset.normal_support_split = 'train'
        dataset.train_normal_supports = {'widget': [self.sample('train/normal', 0)]}
        for index in range(10):
            query, support, _, _ = dataset.load_frame(index)
            self.assertIn(query[0]['img_path'], ('n0', 'n1', 'a0', 'a1'))
            self.assertEqual(support[0]['img_path'], 'train/normal')
            self.assertEqual(support[0]['anomaly'], 0)

    def test_natural_sampling_uses_test_image_frequency(self):
        dataset = self.make_dataset()
        dataset.eval_label_sampling = 'natural'
        dataset.metadata['widget']['abnormal']['bad'] = [
            self.sample(f'a{index}', 1) for index in range(8)
        ]
        first = dataset.load_frame(0)
        self.assertEqual(first, dataset.load_frame(0))
        labels = [dataset.load_frame(i)[0][0]['anomaly'] for i in range(200)]
        self.assertGreater(sum(labels), 140)
        self.assertLess(sum(labels), 190)

    def test_source_eval_selects_source_classes(self):
        dataset = FSDataset.__new__(FSDataset)
        dataset.data_mode = 'mvtec_visa'
        dataset.fold = 0
        dataset.nclass = 27
        dataset.all_product = [f'class_{index}' for index in range(27)]
        dataset.split = 'val'
        dataset.source_eval = True
        dataset.class_ids = dataset.build_class_ids()
        self.assertEqual(list(dataset.class_ids), list(range(15, 27)))
        self.assertEqual(dataset.products, dataset.all_product[15:])


class ExhaustiveDatasetTests(unittest.TestCase):
    def build_dataset(self, shot=1, seed=111, duplicate=False, overlap=False):
        def initialize(dataset, *args, **kwargs):
            dataset.data_mode = 'mvtec_visa'
            dataset.data_root = '/fixture'
            dataset.data_name_json = 'meta.json'
            dataset.n_shot, dataset.a_shot = shot, 0
            dataset.seed = seed
            dataset.class_ids = [0]
            dataset.all_product = ['widget']
            normal = {'img_path': '/fixture/mvtec/test/good.png',
                      'mask_path': '', 'anomaly': 0}
            abnormal = {'img_path': '/fixture/mvtec/test/bad.png',
                        'mask_path': '/fixture/mvtec/mask/bad.png', 'anomaly': 1}
            dataset.metadata = {'widget': {'normal': {'good': [normal]},
                                           'abnormal': {'bad': [abnormal]}}}
            if duplicate:
                dataset.metadata['widget']['normal']['good'].append(dict(normal))
        train = {'train': {'widget': [
            {'img_path': 'train/normal%d.png' % index, 'mask_path': '', 'anomaly': 0}
            for index in range(20)]}}
        if overlap:
            train['train']['widget'] = [
                {'img_path': '/fixture/mvtec/test/good.png', 'mask_path': '', 'anomaly': 0}]
        # The two dataset files are read independently, as in real metadata.
        with mock.patch.object(FSDataset, '__init__', initialize), \
             mock.patch('builtins.open', mock.mock_open(read_data=json.dumps(train))), \
             mock.patch('utils.dataset.os.path.isfile', return_value=True):
            return ExhaustiveEvalDataset(split='eval')

    def test_complete_unique_coverage_and_nested_deterministic_supports(self):
        one = self.build_dataset(1)
        two = self.build_dataset(2)
        four = self.build_dataset(4)
        repeat = self.build_dataset(4)
        self.assertEqual(len(four), 2)
        self.assertEqual(one.query_samples, four.query_samples)
        self.assertEqual(one.fixed_supports['widget'], four.fixed_supports['widget'][:1])
        self.assertEqual(two.fixed_supports['widget'], four.fixed_supports['widget'][:2])
        self.assertEqual(four.evaluation_manifest, repeat.evaluation_manifest)
        self.assertNotEqual(four.fixed_supports, self.build_dataset(4, seed=222).fixed_supports)
        manifest = four.evaluation_manifest
        self.assertEqual(manifest['number_of_queries'], 2)
        self.assertEqual(manifest['number_of_unique_queries'], 2)
        self.assertEqual(manifest['query_support_overlap'], 0)
        self.assertEqual(manifest['per_category_counts']['widget'],
                         {'total': 2, 'normal': 1, 'anomaly': 1})

    def test_duplicate_queries_and_leaked_supports_fail(self):
        with self.assertRaisesRegex(ValueError, 'duplicate query'):
            self.build_dataset(duplicate=True)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            self.build_dataset(overlap=True)

    def test_missing_input_fails(self):
        dataset = self.build_dataset()
        with mock.patch('utils.dataset.os.path.isfile', return_value=False):
            with self.assertRaises(FileNotFoundError):
                dataset.audit_manifest()


class SupportViewTests(unittest.TestCase):
    def test_identity_and_rotations_preserve_original_values(self):
        images = torch.arange(2 * 3 * 4 * 4).float().reshape(2, 1, 3, 4, 4)
        self.assertIs(normal_reference_views(images), images)
        views = normal_reference_views(images, [0, 90, 180, 270])
        self.assertEqual(tuple(views.shape), (2, 4, 3, 4, 4))
        for index, angle in enumerate((0, 90, 180, 270)):
            torch.testing.assert_close(views[:, index:index + 1],
                                       torch.rot90(images, angle // 90, dims=(-2, -1)))
        with self.assertRaises(ValueError):
            normal_reference_views(images, scales=[0.75])

class MetricTests(unittest.TestCase):
    def test_parallel_category_metrics_equal_serial_results(self):
        categories = ['first', 'second', 'third']
        serial = FewShotMetric(categories, gaussian_sigma=0, workers=1)
        parallel = FewShotMetric(categories, gaussian_sigma=0, workers=4)
        generator = torch.Generator().manual_seed(42)
        for product in categories:
            image = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])
            labels = torch.tensor([0, 1, 0])
            pixels = torch.rand(3, 2, 8, 8, generator=generator)
            masks = torch.zeros(3, 8, 8)
            masks[1, 2:5, 2:5] = 1
            for metric in (serial, parallel):
                metric.update(image, labels, pixels, masks, [product] * 3)
        self.assertEqual(serial.get_scores(), parallel.get_scores())
        self.assertTrue((serial.results == parallel.results).all())
        with self.assertRaises(ValueError):
            FewShotMetric(categories, workers=12)

    def test_release_download_verification_needs_no_training_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, weights, manifest = root / 'config.json', root / 'weights.bin', root / 'manifest.json'
            config.write_text('{}')
            weights.write_bytes(b'example serialized weights')
            manifest.write_text(json.dumps({
                'config': str(config), 'config_sha256': file_digest(config),
                'weights': [{'path': str(weights), 'size_bytes': weights.stat().st_size,
                             'sha256': file_digest(weights), 'source_checkpoint': 'not-distributed.pth'}]}))
            self.assertEqual(verify_manifest(manifest), 1)
            weights.write_bytes(b'changed bytes')
            with self.assertRaisesRegex(ValueError, 'checksum differs'):
                verify_manifest(manifest)

    def test_empty_exhaustive_summary_cannot_claim_success(self):
        summary = summarize({}, {}, [111, 222, 333])
        self.assertFalse(summary['complete'])
        self.assertFalse(summary['all_18_mean_metrics_exceed_paper'])
        self.assertEqual(summary['number_of_verified_pairs'], 0)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            publication_report(summary, {}, {}, None, None, None)
        with self.assertRaises(ValueError):
            summarize({}, {}, [111, 111])

    def test_pixel_consensus_fusion_is_label_free_and_finite(self):
        maps = [torch.tensor([0.2, 0., 0.8]), torch.tensor([0.8, 0.5, 0.2])]
        torch.testing.assert_close(fuse_pixel_experts(maps), torch.tensor([0.5, 0.25, 0.5]))
        torch.testing.assert_close(fuse_pixel_experts(maps, weights=[0.25, 0.75]),
                                   torch.tensor([0.65, 0.375, 0.35]))
        with self.assertRaises(ValueError):
            fuse_pixel_experts(maps, 'unknown')
        with self.assertRaises(ValueError):
            fuse_pixel_experts(maps, weights=[1])
        for weights in ([-1, 2], [0, 0], [float('nan'), 1], [float('inf'), 1]):
            with self.assertRaises(ValueError):
                fuse_pixel_experts(maps, weights=weights)

    def test_image_only_metrics_match_full_image_metrics_without_storing_pixels(self):
        complete = FewShotMetric(['widget'], gaussian_sigma=0)
        fast = FewShotMetric(['widget'], image_only=True)
        image = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.6, 0.4]])
        labels = torch.tensor([0, 1, 0])
        pixels = torch.rand(3, 2, 2, 2)
        masks = torch.zeros(3, 2, 2)
        for metric in (complete, fast):
            metric.update(image, labels, pixels, masks, ['widget'] * 3)
        self.assertEqual(fast.get_scores(), complete.calculate_image_scores('widget'))
        self.assertEqual(fast.stat_buffer['widget']['p_score_map'], [])

    def test_pixel_experts_keep_primary_image_scores_and_average_calibrated_maps(self):
        class Predictor(nn.Module):
            def __init__(self, image, pixels):
                super().__init__()
                self.image, self.pixels = image, pixels

            def forward(self, *args, **kwargs):
                return self.image, self.pixels, 0, 0, 0

        image = torch.tensor([[0.1, 0.9]])
        first = torch.tensor([[[[0.8]], [[0.7]]]])
        second = torch.tensor([[[[0.6]], [[0.2]]]])
        primary = Predictor(image, first)
        ensemble = PixelExpertEnsemble(primary, [Predictor(image, first), Predictor(image, second)],
                                       [types.SimpleNamespace(proxy_weight=0.2),
                                        types.SimpleNamespace(proxy_weight=0.6)])
        outputs = ensemble(types.SimpleNamespace(n_shot=2), None, None, None, None)
        torch.testing.assert_close(outputs[0], image)
        expected = ((0.2 * 0.7 + 0.8 * 0.2) + (0.6 * 0.2 + 0.4 * 0.4)) / 2
        self.assertAlmostEqual(outputs[1][0, 1, 0, 0].item(), expected)
        self.assertAlmostEqual(outputs[1][0, 0, 0, 0].item(), 1 - expected)
        extra_image = torch.tensor([[0.7, 0.3]])
        averaged = PixelExpertEnsemble(primary, [], [], [Predictor(extra_image, second)],
                                      [types.SimpleNamespace()])
        combined = averaged(types.SimpleNamespace(n_shot=2), None, None, None, None)
        torch.testing.assert_close(combined[0], torch.tensor([[0.4, 0.6]]))
        torch.testing.assert_close(combined[1], first)

if __name__ == "__main__":
    unittest.main()
