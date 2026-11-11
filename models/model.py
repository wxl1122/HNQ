"""Frozen features, residual feature learning, normality learning and query scoring."""

import os
import hashlib
import copy
from collections import OrderedDict
import torch
from utils.support_views import normal_reference_views
import torch.nn as nn
import torch.nn.functional as F
import timm

from einops import rearrange
from models.attention_layer import CrossAttentionLayer, SelfAttentionLayer, PositionEmbeddingSine

from utils.loss import FocalLoss, DiceLoss


class AML(nn.Module):
    def __init__(self, args):
        super(AML, self).__init__()

        self.nheads = 4
        self.pre_norm = False
        self.backbone_name = args.backbone_name
        self.matching_layers = getattr(args, 'matching_layers', None)
        self.matching_pool_size = getattr(args, 'matching_pool_size', 1)
        if self.matching_layers is not None:
            if (not self.matching_layers or sorted(set(self.matching_layers)) != self.matching_layers
                    or self.matching_layers[0] < 0):
                raise ValueError('matching_layers requires distinct ascending blocks and final-block head features')
        if self.matching_pool_size not in (1, 3, 5):
            raise ValueError('matching_pool_size must be 1, 3, or 5')

        if "resnet" in args.backbone_name:
            local_weights = getattr(args, 'backbone_weights', None)
            self.cnn_encoder = timm.create_model(
                args.backbone_name, features_only=True, out_indices=(1, 2, 3, 4),
                pretrained=not bool(local_weights),
            )
            if local_weights:
                state = torch.load(local_weights, map_location='cpu', weights_only=True)
                state = {key: value for key, value in state.items() if not key.startswith('fc.')}
                self.cnn_encoder.load_state_dict(state, strict=True)
            self.hidden_dim = self.cnn_encoder.feature_info.channels()[-1]
            self.d_scale = 32
            self.cnn_encoder.eval()

        elif "dinov2" in args.backbone_name:
            if os.environ.get("HNQ_ENABLE_XFORMERS") != "1":
                # Avoid selecting an importable but ABI-incompatible xFormers build.
                os.environ.setdefault("XFORMERS_DISABLED", "1")
            dinov2_repo = getattr(args, "dinov2_repo", None)
            local_weights = getattr(args, 'backbone_weights', None)
            if dinov2_repo:
                self.vit_encoder = torch.hub.load(
                    dinov2_repo, args.backbone_name, source="local", pretrained=not bool(local_weights)
                )
            else:
                self.vit_encoder = torch.hub.load(
                    "facebookresearch/dinov2", args.backbone_name, pretrained=not bool(local_weights)
                )
            if local_weights:
                self.vit_encoder.load_state_dict(
                    torch.load(local_weights, map_location='cpu', weights_only=True), strict=True
                )
            self.hidden_dim = getattr(self.vit_encoder, 'embed_dim', 384)
            self.d_scale = 14
            self.vit_encoder.eval()
        else:
            raise Exception('Unavailable backbone: %s' % args.backbone_name)

        if args.n_shot > 0:

            self.learnable_token = nn.Embedding(args.num_a_queries, self.hidden_dim)
            self.first_cross_attention, self.first_self_attention = self.attention_module()

            self.later_cross_attention, self.later_self_attention = self.attention_module()

        self.pe_layer = PositionEmbeddingSine(self.hidden_dim//2, normalize=True)

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()

        self.feature_cache_size = getattr(args, 'feature_cache_size', 0)
        self._feature_cache = OrderedDict()

    def train(self, mode: bool = True):
        """Keep the frozen feature extractor in evaluation mode.

        ``nn.Module.train`` recursively switches every child to training mode.
        That made DINOv2 stochastic during training even though its parameters
        were frozen.  The trainable RFL/NLS blocks still follow ``mode``.
        """
        super().train(mode)
        if hasattr(self, "vit_encoder"):
            self.vit_encoder.eval()
        if hasattr(self, "cnn_encoder"):
            self.cnn_encoder.eval()
        return self

    def attention_module(self):
        cross_attention = CrossAttentionLayer(
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dropout=0.0,
                normalize_before=self.pre_norm,
            )
        self_attention = SelfAttentionLayer(
                d_model=self.hidden_dim,
                nhead=self.nheads,
                dropout=0.0,
                normalize_before=self.pre_norm,
            )
        return cross_attention, self_attention

    @torch.no_grad()
    def feature_forward(self, image, fixed_supports=False, head_only=False):
        """Reuse frozen descriptors through a bounded, image-content keyed CPU cache."""
        provider = self.__dict__.get('_frozen_feature_provider')
        if provider is not None:
            return provider.feature_forward(image, fixed_supports=fixed_supports, head_only=True)
        if (fixed_supports and self.feature_cache_size and image.shape[0] > 1
                and torch.equal(image, image[:1].expand_as(image))):
            return self.feature_forward(image[:1], head_only=head_only).expand(image.shape[0], -1, -1, -1)
        if not self.feature_cache_size:
            features = self._feature_forward(image)
            return features[..., :self.hidden_dim] if head_only else features
        b, num, c, h, w = image.shape
        flat = image.reshape(-1, c, h, w)
        keys = [
            hashlib.sha256(item.detach().cpu().contiguous().numpy().tobytes()).digest()
            for item in flat
        ]
        missing = list(dict.fromkeys(key for key in keys if key not in self._feature_cache))
        fresh = {}
        if missing:
            indices = [keys.index(key) for key in missing]
            encoded = self._feature_forward(flat[indices].unsqueeze(1))
            fresh = {key: encoded[i, 0].detach().cpu() for i, key in enumerate(missing)}
        values = [fresh[key] if key in fresh else self._feature_cache[key] for key in keys]
        for key, value in zip(keys, values):
            self._feature_cache[key] = value
            self._feature_cache.move_to_end(key)
        while len(self._feature_cache) > self.feature_cache_size:
            self._feature_cache.popitem(last=False)
        selected = [value[..., :self.hidden_dim] for value in values] if head_only else values
        return torch.stack(selected).to(image.device).reshape(b, num, *selected[0].shape)

    @torch.no_grad()
    def _feature_forward(self, image):
        b, num, c, h, w = image.shape
        if "resnet" in self.backbone_name:
            feat = self.cnn_encoder(image.view(-1, c, h, w))[-1]
            feat = rearrange(feat, 'b c h w -> b (h w) c')
        else:
            matching = None
            if self.matching_layers is not None:
                final = len(self.vit_encoder.blocks) - 1
                if self.matching_layers[-1] > final:
                    raise ValueError('matching_layers contains a block outside the encoder')
                indices = sorted(set([*self.matching_layers, final]))
                outputs = self.vit_encoder.get_intermediate_layers(image.view(-1, c, h, w), n=indices)
                layers = dict(zip(indices, outputs))
                feat = layers[final]
                descriptors = []
                for index in self.matching_layers:
                    descriptor = layers[index]
                    if self.matching_pool_size > 1:
                        grid = descriptor.transpose(1, 2).reshape(b * num, self.hidden_dim,
                                                                  h // self.d_scale, w // self.d_scale)
                        grid = F.avg_pool2d(grid, self.matching_pool_size, stride=1,
                                           padding=self.matching_pool_size // 2, count_include_pad=False)
                        descriptor = grid.flatten(2).transpose(1, 2)
                    descriptors.append(F.normalize(descriptor, dim=-1))
                matching = torch.cat(descriptors, dim=-1)
            else:
                feat = self.vit_encoder.get_intermediate_layers(image.view(-1, c, h, w))[0]
        if 'dinov2' in self.backbone_name and matching is not None:
            feat = torch.cat([feat, matching], dim=-1)
        return feat.reshape(b, num, -1, feat.shape[-1])

    def get_pseudo_mask(self, query_feat, support_feat, support_mask=None, mode='max'):
        '''
        query_feat: (b, num_q, M, c)
        support_feat: (b, num_s, N, c)
        support_mask: (b, num_s, h, w)
        '''
        _, num_q, M, _ = query_feat.shape
        _, num_s, N, _ = support_feat.shape
        query_feat = rearrange(F.normalize(query_feat, dim=-1), 'b num_q M c -> b (num_q M) c')
        support_feat = rearrange(F.normalize(support_feat, dim=-1), 'b num_s N c -> b (num_s N) c')

        similarity_map = torch.einsum('bmc,bnc->bmn', query_feat, support_feat)
        if mode == 'max':
            pseudo_mask = similarity_map.max(dim=-1)[0]
        elif mode == 'mean':
            pseudo_mask = similarity_map.mean(dim=-1)
        pseudo_mask = rearrange(pseudo_mask, 'b (num_q M) -> b num_q M 1', num_q=num_q)

        return pseudo_mask

    def attention_forward(self, cross_layer, self_layer, query_embed, key_feat, value_feat, feat_mask=None):
        '''
        query_embed: (num_q, c)
        key_feat: (b, num_s, M, c)
        value_feat: (b, num_s, M, c)
        feat_mask: (b, num_s, M, 1)
        '''
        B, _, _, C = value_feat.shape

        if isinstance(query_embed, nn.Embedding):
            q_supp_out = query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        elif query_embed.dim() == 2:
            q_supp_out = query_embed.unsqueeze(1).repeat(1, B, 1)
        else:
            q_supp_out = query_embed.permute(1, 0, 2)

        key = rearrange(key_feat, 'b num M c -> (num M) b c')
        pos_embedding = self.pe_layer(value_feat, None if feat_mask is None else 1 - feat_mask)
        value = rearrange(value_feat, 'b num M c -> (num M) b c')

        if feat_mask is not None:
            valid = rearrange(feat_mask.squeeze(-1), 'b num M -> b (num M)')
            attn_mask = valid[:, None, None, :].expand(
                B, self.nheads, q_supp_out.shape[0], valid.shape[-1]
            ).reshape(B * self.nheads, q_supp_out.shape[0], valid.shape[-1])
            attn_mask = -1e9*(1-attn_mask)
        else:
            attn_mask = feat_mask

        output = cross_layer(q_supp_out, key, value,
                             memory_mask=attn_mask,
                             memory_key_padding_mask=None,
                             pos=pos_embedding, query_pos=None
                             )
        output = self_layer(output,
                            tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                            )

        return output.permute(1, 0, 2)

    def get_res_feat(self, ori_feat, temp_feat, return_matched=False, metric='cosine',
                     query_matching=None, reference_matching=None, return_similarity=False):
        '''
        ori_feat: (b, num, h*w, c)
        temp_feat: (b, num, h*w, c)
        '''
        b, num, h_w, c = ori_feat.shape
        ori_feat = rearrange(ori_feat, 'b num h_w c -> b (num h_w) c')
        temp_feat = rearrange(temp_feat, 'b num h_w c -> b (num h_w) c')
        if (query_matching is None) != (reference_matching is None):
            raise ValueError('Both query and reference matching descriptors are required')
        query_descriptor = ori_feat if query_matching is None else query_matching.flatten(1, 2)
        reference_descriptor = temp_feat if reference_matching is None else reference_matching.flatten(1, 2)
        sim_map = torch.einsum(
            'bmc,bnc->bmn',
            F.normalize(query_descriptor, dim=-1) if metric == 'cosine' else query_descriptor,
            F.normalize(reference_descriptor, dim=-1) if metric == 'cosine' else reference_descriptor,
        )
        similarities, max_idx = sim_map.max(dim=-1)
        # Match with normalized descriptors; subtract original descriptors.
        most_sim_feat = torch.gather(temp_feat, 1, max_idx.unsqueeze(-1).repeat(1, 1, c))
        res_feat = ori_feat - most_sim_feat
        res_feat = rearrange(res_feat, 'b (num h_w) c -> b num h_w c', b=b, num=num)
        if return_matched:
            if return_similarity:
                return res_feat, most_sim_feat.view(b, num, h_w, c), similarities.view(b, num, h_w)
            return res_feat, most_sim_feat.view(b, num, h_w, c)
        return res_feat

    def forward(self, args, query_image, query_mask, query_label, support_normal,
                return_score_components=False, _query_features=None):

        query_all = self.feature_forward(query_image) if _query_features is None else _query_features
        query_feat = query_all[..., :self.hidden_dim]
        use_matching = self.matching_layers is not None and not getattr(args, '_final_layer_matching', False)
        query_matching = query_all[..., self.hidden_dim:] if use_matching else None
        last_query_feat = query_feat

        loss_i = 0
        loss_p = 0
        loss_a_match = 0

        if args.n_shot > 0:
            support_n_image, support_n_mask_ = support_normal
            device = query_image.device
            support_n_image = support_n_image.to(device, non_blocking=True)
            support_n_mask_ = support_n_mask_.to(device, non_blocking=True)
            support_n_image = normal_reference_views(
                support_n_image, getattr(args, 'support_rotations', [0]),
                getattr(args, 'support_scales', [1.0]))
            support_all = self.feature_forward(
                support_n_image, fixed_supports=getattr(args, 'cache_eval_supports', False),
                head_only=not use_matching)
            support_n_feat = support_all[..., :self.hidden_dim]
            reference_matching = support_all[..., self.hidden_dim:] if use_matching else None
            metric = getattr(args, 'residual_matching', 'cosine')
            query_res_feat, matched_normal_feat, normality = self.get_res_feat(
                query_feat, support_n_feat, return_matched=True,
                metric=metric, return_similarity=True,
                query_matching=query_matching, reference_matching=reference_matching,
            )
            n_out = normality if metric == 'cosine' else self.get_pseudo_mask(
                query_matching if query_matching is not None else query_feat,
                reference_matching if reference_matching is not None else support_n_feat).squeeze(-1)

            # RFL: aggregate query residuals into learned residual proxies.
            res_prototype = self.attention_forward(
                self.first_cross_attention, self.first_self_attention,
                self.learnable_token, query_res_feat, query_res_feat, None,
            )

            # NLS: translate residual proxies into normal appearance proxies.
            nls_values = getattr(args, 'nls_values', 'spatial_mean')
            nls_mask = None
            if nls_values == 'matched':
                normal_values = matched_normal_feat
            elif nls_values == 'spatial_mean':
                normal_values = support_n_feat.mean(1).unsqueeze(1)
            else:
                raise ValueError(f'Unsupported NLS values: {nls_values}')
            n_embed = self.attention_forward(
                self.later_cross_attention,
                self.later_self_attention,
                res_prototype,
                query_res_feat,
                normal_values,
                nls_mask,
            ).unsqueeze(1)
            a_out = self.get_pseudo_mask(
                last_query_feat, n_embed, mode=getattr(args, 'proxy_reduction', 'mean')
            )
            a_out = a_out.squeeze(-1)

        # HNQ: preserve legacy-head semantics; normality heads use the complement.
        proxy_anomaly = (
            1.0 - a_out
            if getattr(args, 'proxy_orientation', 'legacy_anomaly') == 'paper_normality'
            else a_out
        )
        pixel_level_logits = torch.cat([n_out, proxy_anomaly], dim=1) # (b, 2, h*w)

        image_proxy_weight = getattr(args, 'image_proxy_weight', 0.5)
        a_score = (
            image_proxy_weight * proxy_anomaly
            + (1.0 - image_proxy_weight) * (1.0 - n_out)
        )
        topk = min(getattr(args, 'image_topk', 20), a_score.shape[-1])
        a_score_topk = torch.topk(a_score, topk, dim=-1)[0].mean(dim=-1)
        image_level_logits = torch.cat([1-a_score_topk, a_score_topk], dim=-1)
        image_rotations = getattr(args, 'image_support_rotations', None)
        if image_rotations is not None:
            image_args = copy.copy(args)
            image_args.support_rotations = image_rotations
            image_args.image_support_rotations = None
            image_args._final_layer_matching = getattr(args, 'image_final_layer_matching', False)
            image_level_logits = self.forward(
                image_args, query_image, query_mask, query_label, support_normal,
                _query_features=query_all)[0]

        l = int(pixel_level_logits.shape[-1]**0.5)
        pixel_level_logits = rearrange(pixel_level_logits, 'b n (h w) -> b n h w', h=l)
        pixel_level_logits = F.interpolate(
            pixel_level_logits, size=query_mask.shape[-2:], mode='bilinear'
        )

        loss_mode = getattr(args, 'loss_mode', 'legacy')
        loss_scale = getattr(args, 'loss_logit_scale', 5.0)
        if loss_mode == 'paper' and loss_scale <= 0:
            raise ValueError('loss_logit_scale must be positive')
        if loss_mode == 'paper':
            if getattr(args, 'proxy_orientation', 'legacy_anomaly') != 'paper_normality':
                raise ValueError('paper loss requires proxy_orientation=paper_normality')
            patch_anomaly = 0.5 * (proxy_anomaly + (1.0 - n_out))
            pixel_anomaly = F.interpolate(
                patch_anomaly.view(query_image.shape[0], 1, l, l),
                size=query_mask.shape[-2:], mode='bilinear', align_corners=False,
            ).squeeze(1)
            pixel_logit = loss_scale * (pixel_anomaly - 0.3)
            image_logit = loss_scale * (patch_anomaly.amax(dim=-1).squeeze(1) - 0.3)
            loss_i = F.binary_cross_entropy_with_logits(
                image_logit, query_label.float().view(-1)
            )
            loss_p = self.focal_loss(pixel_logit, query_mask)
            loss_p = loss_p + self.dice_loss(pixel_logit.sigmoid(), query_mask)
            outputs = (image_level_logits, pixel_level_logits,
                       loss_a_match, loss_i, loss_p)
            if return_score_components:
                return outputs, (proxy_anomaly.squeeze(1), (1.0 - n_out).squeeze(1))
            return outputs
        if loss_mode != 'legacy':
            raise ValueError(f'Unsupported loss mode: {loss_mode}')
        loss_i += self.cross_entropy_loss(image_level_logits, query_label.long())
        query_mask_n = torch.stack([1-query_mask, query_mask], dim=1)
        loss_p += self.focal_loss(pixel_level_logits, query_mask_n)
        loss_p += self.dice_loss(pixel_level_logits, query_mask_n)

        outputs = (image_level_logits, pixel_level_logits, loss_a_match, loss_i, loss_p)
        if return_score_components:
            return outputs, (proxy_anomaly.squeeze(1), (1.0 - n_out).squeeze(1))
        return outputs
