"""Independent residual-normality experts for image ranking and localization."""

import torch
import math
from torch import nn


def fuse_pixel_experts(maps, reduction='mean', weights=None):
    stacked = torch.stack(maps)
    if weights is not None:
        if (reduction != 'mean' or len(weights) != len(maps)
                or any(not math.isfinite(weight) or weight < 0 for weight in weights)
                or sum(weights) <= 0):
            raise ValueError('Pixel expert weights require arithmetic fusion and one nonnegative weight per expert')
        normalized = stacked.new_tensor(weights)
        normalized = normalized / normalized.sum()
        return (stacked * normalized.reshape(-1, *([1] * (stacked.ndim - 1)))).sum(dim=0)
    if reduction == 'mean':
        return stacked.mean(dim=0)
    raise ValueError(f'Unknown pixel-expert reduction: {reduction}')


class ResidualExpertEnsemble(nn.Module):
    """Average residual-normality image experts and localization experts."""

    def __init__(self, primary, experts, expert_args, image_experts=None, image_args=None):
        super().__init__()
        self.primary = primary
        self.experts = nn.ModuleList(experts)
        self.expert_args = expert_args
        self.image_experts = nn.ModuleList(image_experts or [])
        self.image_args = image_args or []

    def forward(self, args, query_image, query_mask, query_label, support_normal,
                return_score_components=False):
        prediction = self.primary(args, query_image, query_mask, query_label, support_normal,
                                  return_score_components=return_score_components)
        outputs, components = prediction if return_score_components else (prediction, None)
        image = outputs[0]
        if self.image_experts:
            score = image[:, 1]
            for expert, config in zip(self.image_experts, self.image_args):
                config.n_shot = args.n_shot
                extra = expert(config, query_image, query_mask, query_label, support_normal)
                score = score + extra[0][:, 1]
            score = score / (len(self.image_experts) + 1)
            image = torch.stack([1.0 - score, score], dim=1)
        maps = []
        for expert, config in zip(self.experts, self.expert_args):
            config.n_shot = args.n_shot
            _, pixels, _, _, _ = expert(config, query_image, query_mask, query_label, support_normal)
            weight = getattr(config, 'proxy_weight', 0.2)
            maps.append(weight * pixels[:, 1] + (1.0 - weight) * (1.0 - pixels[:, 0]))
        pixels = outputs[1]
        if maps:
            anomaly = fuse_pixel_experts(maps, getattr(args, 'pixel_expert_reduction', 'mean'),
                                         getattr(args, 'pixel_expert_weights', None))
            pixels = torch.stack([1.0 - anomaly, anomaly], dim=1)
        result = (image, pixels, *outputs[2:])
        return (result, components) if return_score_components else result


PixelExpertEnsemble = ResidualExpertEnsemble
