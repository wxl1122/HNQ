"""Export trainable weights plus a portable frozen-encoder identity check."""

import argparse
import hashlib
from pathlib import Path

import torch

from models.model import AML
from utils.checkpoint import frozen_encoder_signature


MODEL_KEYS = (
    'backbone_name', 'num_a_queries', 'num_n_queries', 'n_shot', 'image_size',
    'image_topk', 'image_proxy_weight', 'proxy_weight', 'proxy_orientation', 'loss_mode',
    'loss_logit_scale', 'nls_values', 'proxy_reduction', 'residual_matching',
    'support_rotations', 'support_scales',
    'matching_layers', 'matching_pool_size',
    'image_support_rotations', 'image_final_layer_matching',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dinov2-repo', default=None)
    cli = parser.parse_args()
    original = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    known = isinstance(original.get('args'), dict)
    if known:
        training = original['args']
        model_weights = original['model']
        inference = {key: training[key] for key in MODEL_KEYS if key in training}
    else:
        model_weights = original
        tokens, dim = model_weights['learnable_token.weight'].shape
        if dim != 384:
            raise ValueError('An unversioned checkpoint requires the documented ViT-S legacy profile')
        training = {}
        inference = dict(backbone_name='dinov2_vits14', num_a_queries=tokens,
                         num_n_queries=3, n_shot=1, image_size=448, image_topk=20,
                         image_proxy_weight=0.35, proxy_weight=0.2,
                         proxy_orientation='legacy_anomaly', loss_mode='legacy',
                         nls_values='spatial_mean', proxy_reduction='mean',
                         residual_matching='cosine', support_rotations=[0])
    model_config = {**inference, 'backbone_weights': training.get('backbone_weights'),
                    'dinov2_repo': cli.dinov2_repo, 'feature_cache_size': 0}
    model = AML(argparse.Namespace(**model_config))
    signature = frozen_encoder_signature(model)
    inference.update(backbone_weights=None, dinov2_repo=None, feature_cache_size=0)
    digest = hashlib.sha256()
    with cli.checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    exported = {
        'format_version': 2, 'model': model_weights, 'args': training,
        'inference_args': inference, 'frozen_encoder_signature': signature,
        'source_checkpoint': str(cli.checkpoint), 'source_sha256': digest.hexdigest(),
        'training_configuration_known': known,
    }
    if 'epoch' in original:
        exported['epoch'] = original['epoch']
    cli.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, cli.output)
    print(f'{cli.output}: {cli.output.stat().st_size / 1024 ** 2:.2f} MiB; encoder={signature}')


if __name__ == '__main__':
    main()
