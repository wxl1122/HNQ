"""Train HNQ-GAD and evaluate sampled episodes or the complete test split."""

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.model import AML
from models.ensemble import PixelExpertEnsemble
from utils.dataset import ExhaustiveEvalDataset, FSDataset
from utils.medical import MedicalEvalDataset
from utils.checkpoint import frozen_encoder_signature
from utils.metrics import FewShotMetric
from utils.utils import setup_seed





def parse_args():
    parser = argparse.ArgumentParser("HNQ-GAD")
    parser.add_argument('--config', default=None, help='JSON defaults; explicit command-line arguments take precedence.')
    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--data-mode", choices=["mvtec_visa", "realiad", "mvtec_brats"], default="mvtec_visa")
    parser.add_argument('--medical-split', default=None, help='Declared content-disjoint BraTS query/reference split.')
    parser.add_argument('--metric-set', choices=['full', 'auroc'], default='full',
                        help='Exact image/pixel AUROCs for the medical table, or the existing six-metric suite.')
    parser.add_argument("--fold", type=int, choices=[0, 1], default=0)
    parser.add_argument("--n-shot", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--train-mixed-shots", action="store_true",
                        help="Train each batch with 1, 2 or 4 normal references, sampled uniformly.")
    parser.add_argument('--train-support-split', choices=['test', 'train'], default='test',
                        help='Source normal-reference split during training; train aligns with exhaustive evaluation.')
    parser.add_argument("--support-rotations", type=int, nargs='+', choices=[0, 90, 180, 270], default=[0],
                        help="Normal-reference views generated from the same K support images.")
    parser.add_argument('--support-scales', type=float, nargs='+', choices=[1.0], default=[1.0],
                        help=argparse.SUPPRESS)
    parser.add_argument("--eval-shots", type=int, nargs='+', choices=[1, 2, 4], default=None,
                        help="Evaluate several shot counts with one model and a shared frozen-feature cache.")
    parser.add_argument("--eval-seeds", type=int, nargs='+', default=None,
                        help="Repeat evaluation over predeclared support-sampling seeds; reuse cached frozen features.")
    parser.add_argument("--a-shot", type=int, default=0)

    parser.add_argument("--backbone-name", default="dinov2_vits14")
    parser.add_argument("--backbone-weights", default=None,
                        help="Optional local pretrained state dictionary for the frozen encoder.")
    parser.add_argument('--pretrained-weights-dir', default=None,
                        help='Optional directory containing portable frozen encoder weight files for experts.')
    parser.add_argument("--feature-cache-size", type=int, default=0,
                        help="Maximum frozen image features cached in CPU memory (0 disables).")
    parser.add_argument('--image-cache-size', type=int, default=0,
                        help='Per-worker LRU cache of deterministic image/mask preprocessing.')
    parser.add_argument('--matching-layers', type=int, nargs='+', default=None,
                        help='Concatenate normalized DINOv2 blocks for normal-reference matching; retain final features for RFL/NLS.')
    parser.add_argument('--matching-pool-size', type=int, choices=[1, 3, 5], default=1,
                        help='Local averaging of matching descriptors only.')
    parser.add_argument('--image-support-rotations', type=int, nargs='+',
                        choices=[0, 90, 180, 270], default=None,
                        help='Optional separate image readout using the same head and K normal images.')
    parser.add_argument('--image-final-layer-matching', action='store_true',
                        help='Use final-block matching for the separate image readout.')
    parser.add_argument("--nls-values", choices=["spatial_mean", "matched"], default="spatial_mean",
                        help="Use legacy spatial support means or residual-aligned normal matches.")
    parser.add_argument("--proxy-reduction", choices=["mean", "max"], default="mean",
                        help="Aggregate similarity to normality proxies by mean (legacy) or best match.")
    parser.add_argument("--residual-matching", choices=["cosine", "dot"], default="cosine",
                        help="Metric selecting the normal token subtracted from each query token.")
    parser.add_argument(
        "--dinov2-repo",
        default=None,
        help="Optional local facebookresearch/dinov2 checkout (avoids torch.hub download).",
    )
    parser.add_argument(
        "--num-proxies", "--num-a-queries", dest="num_a_queries", type=int, default=25,
        help="Number of residual/normality proxies. The released checkpoint uses 25.",
    )
    parser.add_argument("--num-n-queries", dest="num_n_queries", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument('--eval-image-size', type=int, default=None,
                        help='Optional fixed evaluation mask/map resolution, independent of encoder input size.')
    parser.add_argument("--image-topk", type=int, default=20)
    parser.add_argument("--image-proxy-weight", type=float, default=0.35)
    parser.add_argument("--proxy-weight", type=float, default=0.2)
    # Identity-only fields retained so the recorded, checksummed profile stays unchanged.
    parser.add_argument('--pixel-center-weight', type=float, choices=[0.0], default=0.0,
                        help=argparse.SUPPRESS)
    parser.add_argument('--normal-tta-steps', type=int, choices=[0], default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--proxy-orientation", choices=["legacy_anomaly", "paper_normality"],
                        default="legacy_anomaly",
                        help="Interpret proxy similarity as an anomaly cue (legacy) or normality (paper Eq. 8).")
    parser.add_argument("--loss-mode", choices=["legacy", "paper"],
                        default="legacy",
                        help="Legacy checkpoint-compatible losses or normality-oriented BCE, focal and Dice losses.")
    parser.add_argument("--loss-logit-scale", type=float, default=5.0,
                        help="Train-only logit scale for the normality-oriented loss.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=2000,
                        help="Number of sampled episodes; ignored for exhaustive evaluation.")
    parser.add_argument("--export-score-components", action="store_true",
                        help="Save patch cues and image labels for auditable score diagnostics (evaluation only).")
    parser.add_argument("--image-only", action="store_true",
                        help="Fast diagnostic: compute image metrics only, saved separately from complete evaluations.")
    parser.add_argument("--eval-protocol", choices=["episodic", "exhaustive"],
                        default="episodic",
                        help="Exhaustive evaluates every test image once using fixed train supports.")
    parser.add_argument('--cache-eval-supports', action='store_true',
                        help='Cache deterministic preprocessing of fixed exhaustive normal supports.')
    parser.add_argument("--eval-label-sampling", choices=["balanced", "natural"],
                        default="balanced",
                        help="For episodic evaluation: balanced labels or natural per-category test prevalence.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument('--metric-workers', type=int, choices=[1, 2, 4], default=1,
                        help='Bounded CPU category-metric concurrency; 1 minimizes peak RAM.')
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-frequency", type=int, default=1,
                        help="Epochs between target evaluations; 0 trains without target evaluation and saves the final epoch.")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--target-gap-pp", type=float, default=0.5,
                        help="Maximum absolute difference to the paper, in percentage points.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or e.g. cuda:1")

    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate or resume.")
    parser.add_argument("--pixel-experts", nargs='+', default=None,
                        help="Versioned source checkpoints for arithmetic pixel fusion; paths may contain {shot}.")
    parser.add_argument('--pixel-support-rotations', type=int, nargs='+',
                        choices=[0, 90, 180, 270], default=None,
                        help='Optional pixel-expert reference views; unset preserves checkpoint settings.')
    parser.add_argument('--pixel-support-scales', type=float, nargs='+', choices=[1.0], default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument('--pixel-expert-reduction', choices=['mean'],
                        default='mean', help='Combine calibrated pixel-expert anomaly maps (default: arithmetic mean).')
    parser.add_argument('--pixel-expert-weights', type=float, nargs='+', default=None,
                        help='Optional global weights for arithmetic pixel-expert fusion, in checkpoint-list order.')
    parser.add_argument("--image-experts", nargs='+', default=None,
                        help="Average these versioned image experts with the primary image score; use primary support views.")
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        defaults = json.loads(Path(preliminary.config).read_text(encoding='utf-8'))
        unknown = set(defaults) - {action.dest for action in parser._actions}
        if unknown:
            raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
        parser.set_defaults(**defaults)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return device


def override_pretrained_directory(config, directory):
    if not directory:
        return
    name = config.backbone_name
    name = name[:-4] + '_reg4' if name.endswith('_reg') else name
    candidate = Path(directory) / f'{name}_pretrain.pth'
    if candidate.is_file():
        config.backbone_weights = str(candidate)


def image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_loader(args, split: str):
    episodes = args.train_episodes if split == "train" else args.eval_episodes
    dataset_class = (
        MedicalEvalDataset if args.data_mode == 'mvtec_brats' else ExhaustiveEvalDataset
        if split == "eval" and args.eval_protocol == "exhaustive"
        else FSDataset
    )
    dataset = dataset_class(
        data_root=args.data_root,
        data_mode=args.data_mode,
        fold=args.fold,
        split=split,
        shot=(4 if split == 'train' and args.train_mixed_shots else args.n_shot, args.a_shot),
        transform=image_transform(args.image_size),
        choice=episodes,
        seed=args.seed,
        eval_label_sampling=args.eval_label_sampling,
        normal_support_split=args.train_support_split if split == 'train' else 'test',
        image_cache_size=getattr(args, 'image_cache_size', 0),
        mask_size=getattr(args, 'eval_image_size', None) if split != 'train' else None,
        **({'medical_split': args.medical_split,
            'cache_supports': getattr(args, 'cache_eval_supports', False)}
           if dataset_class is MedicalEvalDataset else
           {'cache_supports': getattr(args, 'cache_eval_supports', False)}
           if dataset_class is ExhaustiveEvalDataset else {}),
    )
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == "train"),
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        generator=generator,
    )


def configure_trainable_parameters(model):
    frozen_roots = {"vit_encoder", "cnn_encoder", "mask_downsample"}
    trainable = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.split(".", 1)[0] not in frozen_roots)
        if parameter.requires_grad:
            trainable.append(parameter)
    return trainable


def run_epoch(args, model, dataloader, device, optimizer=None):
    training = optimizer is not None
    export_components = not training and getattr(args, 'export_score_components', False)
    exported = {'proxy': [], 'nearest': [], 'labels': [], 'products': [], 'image_scores': []}
    model.train(training)
    metric = None if training else FewShotMetric(
        dataloader.dataset.products, proxy_weight=args.proxy_weight,
        image_only=getattr(args, 'image_only', False),
        workers=getattr(args, 'metric_workers', 1),
        metric_set=getattr(args, 'metric_set', 'full'),
    )
    loss_sums = {"total": 0.0, "match": 0.0, "image": 0.0, "pixel": 0.0}
    evaluated_queries = 0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, data in enumerate(dataloader):
            query_image = data["query"][0].to(device, non_blocking=True)
            query_mask = data["query"][1].squeeze(1).to(device, non_blocking=True)
            image_label = data["image_level_label"].to(device, non_blocking=True)
            support_normal = data["support_normal"]
            if training and args.train_mixed_shots:
                shot = (1, 2, 4)[int(torch.randint(3, (1,)))]
                support_normal = [part[:, :shot] for part in support_normal]

            if export_components:
                outputs, (proxy, nearest) = model(
                    args, query_image, query_mask, image_label, support_normal,
                    return_score_components=True,
                )
            else:
                outputs = model(args, query_image, query_mask, image_label, support_normal)
            image_logits, pixel_logits, loss_match, loss_image, loss_pixel = outputs
            if export_components:
                exported['proxy'].append(proxy.cpu())
                exported['nearest'].append(nearest.cpu())
                exported['labels'].append(image_label.cpu())
                exported['products'].extend(data['sample_product'])
                exported['image_scores'].append(image_logits[:, 1].cpu())
            loss = loss_match + loss_image + loss_pixel

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            else:
                metric.update(
                    image_logits,
                    image_label,
                    pixel_logits,
                    query_mask,
                    data["sample_product"],
                )
                evaluated_queries += len(data['sample_product'])

            for key, value in (
                ("total", loss),
                ("match", loss_match),
                ("image", loss_image),
                ("pixel", loss_pixel),
            ):
                loss_sums[key] += float(value.detach()) if torch.is_tensor(value) else float(value)

            if training and step % args.print_freq == 0:
                count = step + 1
                print(
                    f"step={step:04d} loss={loss_sums['total']/count:.4f} "
                    f"image={loss_sums['image']/count:.4f} "
                    f"pixel={loss_sums['pixel']/count:.4f}"
                )
            elif not training and step % args.print_freq == 0:
                print(f'eval step={step:04d}/{len(dataloader)} shot={args.n_shot}', flush=True)

    losses = {key: value / len(dataloader) for key, value in loss_sums.items()}
    if training:
        return {"loss": losses}
    if evaluated_queries != len(dataloader.dataset):
        raise RuntimeError(f'Evaluated {evaluated_queries} queries; expected {len(dataloader.dataset)}')

    if export_components:
        import numpy as np
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / 'score_components.npz',
                            proxy=torch.cat(exported['proxy']).numpy(),
                            nearest=torch.cat(exported['nearest']).numpy(),
                            labels=torch.cat(exported['labels']).numpy(),
                            image_scores=torch.cat(exported['image_scores']).numpy(),
                            image_uses_pixel_components=np.asarray(
                                getattr(args, 'image_support_rotations', None) is None
                                and not getattr(args, 'image_experts', None)),
                            products=np.asarray(exported['products']))

    values = metric.get_scores(report_progress=True)
    metric.print_metrics()
    names = (("i_auroc", "p_auroc") if getattr(args, 'metric_set', 'full') == 'auroc'
             else ("i_auroc", "i_ap", "i_f1", "p_auroc", "au_pro", "p_f1")[:len(values)])
    return {
        **dict(zip(names, map(float, values))),
        "loss": losses,
        "per_category": {
            product: dict(zip(
                names,
                map(float, row),
            ))
            for product, row in zip(dataloader.dataset.products, metric.results)
        },
    }


def checkpoint_state(model, optimizer, args, epoch, metrics):
    parameters = dict(model.named_parameters())
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in parameters and parameters[name].requires_grad
    }
    return {
        "format_version": 2,
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
        "model": trainable_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }


def save_checkpoint(path: Path, state: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, model, optimizer=None):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location="cpu")

    weights = state.get("model", state)
    if state.get('frozen_encoder_signature'):
        actual = frozen_encoder_signature(model)
        if actual != state['frozen_encoder_signature']:
            raise RuntimeError('Frozen encoder weights differ from this inference checkpoint')
    incompatible = model.load_state_dict(weights, strict=False)
    parameters = dict(model.named_parameters())
    unexpected = list(incompatible.unexpected_keys)
    missing_trainable = [
        name for name in incompatible.missing_keys
        if name in parameters and parameters[name].requires_grad
    ]
    if unexpected or missing_trainable:
        raise RuntimeError(
            f"Incompatible checkpoint: unexpected={unexpected}, "
            f"missing_trainable={missing_trainable}"
        )
    if optimizer is not None and isinstance(state, dict) and state.get("optimizer"):
        optimizer.load_state_dict(state["optimizer"])
    return state if isinstance(state, dict) else {}


def validate_checkpoint_fold(state, fold, data_mode='mvtec_visa'):
    trained = state.get('args', {})
    if data_mode == 'mvtec_brats':
        if (not isinstance(trained, dict) or trained.get('fold') != 1
                or trained.get('data_mode') != 'mvtec_visa' or fold != 1):
            raise ValueError('Medical evaluation requires a versioned MVTec-trained checkpoint (fold 1)')
        return
    if (isinstance(trained, dict) and trained.get('fold') is not None
            and trained.get('data_mode', 'mvtec_visa') == data_mode
            and trained['fold'] != fold):
        raise ValueError(f"Checkpoint was trained for fold {trained['fold']}, not fold {fold}; "
                         "swapping the source/target split can expose source training categories as test data")


def paper_targets(args):
    if args.data_mode == 'mvtec_brats':
        image, pixel = {1: (0.781, 0.965), 2: (0.820, 0.965), 4: (0.854, 0.976)}[args.n_shot]
        return {'i_auroc': image, 'p_auroc': pixel}
    return PAPER_TARGETS.get((args.fold, args.n_shot))


def print_paper_gap(args, metrics):
    target = paper_targets(args)
    if not target:
        return
    print("Paper target comparison (percentage-point difference):")
    for name, expected in target.items():
        if name not in metrics:
            continue
        difference = 100.0 * (metrics[name] - expected)
        tolerance = args.target_gap_pp
        status = "exceeds" if difference > 0 else "matches" if difference == 0 else "below"
        print(
            f"  {name}: current={100*metrics[name]:.2f}, "
            f"paper={100*expected:.2f}, diff={difference:+.2f} "
            f"({status}; {'within' if abs(difference) <= tolerance else 'outside'} ±{tolerance:g} pp)"
        )


def evaluate_and_save(args, model, dataloader, device):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage = getattr(dataloader.dataset, 'evaluation_manifest', None)
    if coverage is not None:
        (output_dir / 'evaluation_manifest.json').write_text(
            json.dumps(coverage, indent=2) + '\n', encoding='utf-8')
        print(f"Exhaustive coverage: {coverage['number_of_unique_queries']} unique test images; "
              "fixed train-normal supports; zero query/support overlap")
    metrics = run_epoch(args, model, dataloader, device)
    print_paper_gap(args, metrics)
    metric_file = 'image_metrics.json' if args.image_only else 'eval_metrics.json'
    (output_dir / metric_file).write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'eval_config.json').write_text(
        json.dumps({**vars(args), 'number_of_queries': len(dataloader.dataset),
                    'effective_query_sampling': 'all_test_images_once' if coverage else args.eval_label_sampling,
                    'eval_episodes_applied': coverage is None,
                    'torch_version': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                    'resolved_pixel_experts': [path.format(shot=args.n_shot) for path in args.pixel_experts or []],
                    'resolved_image_experts': [path.format(shot=args.n_shot) for path in args.image_experts or []]},
                   indent=2) + '\n',
        encoding='utf-8',
    )
    targets = paper_targets(args)
    if targets:
        comparison = {
            'eval_protocol': args.eval_protocol,
            'comparison_note': 'Descriptive paper gap; an identical paper evaluation protocol is not established.',
            'all_metrics_available': all(k in metrics for k in targets),
            'all_metrics_exceed_paper': all(k in metrics and metrics[k] > v for k, v in targets.items()),
            'metrics': {
                k: {'measured': metrics[k], 'paper': v,
                    'difference_pp': 100.0 * (metrics[k] - v), 'exceeds': metrics[k] > v}
                for k, v in targets.items() if k in metrics
            },
        }
        (output_dir / 'paper_comparison.json').write_text(
            json.dumps(comparison, indent=2) + '\n', encoding='utf-8'
        )
    return metrics


def with_pixel_experts(args, primary, device, cache):
    if not (args.pixel_experts or args.image_experts):
        return primary
    members, configurations = [], []
    image_members, image_configurations = [], []
    for role, pattern in ([('pixel', path) for path in args.pixel_experts or []]
                          + [('image', path) for path in args.image_experts or []]):
        path = pattern.format(shot=args.n_shot)
        if path not in cache:
            state = torch.load(path, map_location='cpu', weights_only=False)
            validate_checkpoint_fold(state, args.fold, args.data_mode)
            if not isinstance(state.get('args'), dict):
                raise ValueError('Pixel experts require a versioned checkpoint with its training configuration')
            config = argparse.Namespace(**state.get('inference_args', state['args']))
            config.n_shot = args.n_shot
            config.image_size = args.image_size
            config.feature_cache_size = args.feature_cache_size
            config.dinov2_repo = args.dinov2_repo
            override_pretrained_directory(config, args.pretrained_weights_dir)
            expert = AML(config).to(device)
            configure_trainable_parameters(expert)
            load_checkpoint(path, expert)
            # Reuse the identical frozen encoder/cache across experts. The
            # trainable RFL/NLS parameters remain distinct for every checkpoint.
            for existing, existing_config in [(primary, args), *cache.values()]:
                if (config.backbone_name == existing_config.backbone_name
                        and getattr(config, 'backbone_weights', None) == getattr(existing_config, 'backbone_weights', None)
                        and getattr(config, 'cnn_local_features', False) == getattr(existing_config, 'cnn_local_features', False)):
                    for root in ('vit_encoder', 'cnn_encoder'):
                        if hasattr(expert, root):
                            setattr(expert, root, getattr(existing, root))
                    if (getattr(config, 'matching_layers', None) == getattr(existing_config, 'matching_layers', None)
                            and getattr(config, 'matching_pool_size', 1) == getattr(existing_config, 'matching_pool_size', 1)):
                        expert._feature_cache = existing._feature_cache
                    elif (getattr(config, 'matching_layers', None) is None
                          and getattr(existing_config, 'matching_layers', None) is not None):
                        # A final-block head can read a prefix of the richer cache.
                        # Keep the provider outside the module tree (no duplicate parameters).
                        object.__setattr__(expert, '_frozen_feature_provider', existing)
                    break
            expert.requires_grad_(False)
            cache[path] = expert, config
        expert, stored_config = cache[path]
        config = copy.copy(stored_config)
        if role == 'image':
            config.support_rotations = getattr(args, 'image_support_rotations', None) or args.support_rotations
            config.support_scales = args.support_scales
            image_members.append(expert)
            image_configurations.append(config)
        else:
            if args.pixel_support_rotations is not None:
                config.support_rotations = args.pixel_support_rotations
            if args.pixel_support_scales is not None:
                config.support_scales = args.pixel_support_scales
            members.append(expert)
            configurations.append(config)
    ensemble = PixelExpertEnsemble(primary, members, configurations, image_members, image_configurations).to(device)
    unique_count = sum(parameter.numel() for parameter in ensemble.parameters())
    print(f'residual ensemble: pixel_experts={len(members)} image_experts={1 + len(image_members)} '
          f'unique_parameters={unique_count / 1e6:.3f}M')
    return ensemble


def main():
    args = parse_args()
    if args.data_mode == 'mvtec_brats' and (
            args.mode != 'eval' or args.eval_protocol != 'exhaustive'
            or args.fold != 1 or not args.medical_split):
        raise ValueError('Medical transfer supports exhaustive evaluation with a declared split and MVTec source heads only')
    if args.n_shot <= 0 or args.a_shot != 0:
        raise ValueError("This released AML implementation supports normal-only few-shot input")
    if args.eval_shots and args.mode != 'eval':
        raise ValueError('--eval-shots requires --mode eval')
    if args.eval_seeds and args.mode != 'eval':
        raise ValueError('--eval-seeds requires --mode eval')
    if (args.pixel_experts or args.image_experts) and args.mode != 'eval':
        raise ValueError('Residual expert ensembles require --mode eval')
    if args.mode != 'eval' and args.image_support_rotations is not None:
        raise ValueError('Separate image readout is an evaluation setting')
    if args.export_score_components and args.mode != 'eval':
        raise ValueError('--export-score-components requires --mode eval')
    if args.image_size % 14:
        raise ValueError("--image-size must be divisible by the DINOv2 patch size (14)")
    if args.eval_image_size is not None and args.eval_image_size <= 0:
        raise ValueError('--eval-image-size must be positive')

    if (args.pixel_center_weight != 0 or args.normal_tta_steps != 0
            or args.support_scales != [1.0]
            or args.pixel_support_scales not in (None, [1.0])):
        raise ValueError('The release supports reference rotations, without scale/TTA/context experiments')
    if args.pixel_expert_reduction != 'mean':
        raise ValueError('The release supports arithmetic pixel-expert fusion only')

    setup_seed(args.seed)
    device = resolve_device(args.device)
    print(json.dumps(vars(args), indent=2))
    print(f"device={device}")

    override_pretrained_directory(args, args.pretrained_weights_dir)
    model = AML(args).to(device)
    trainable = configure_trainable_parameters(model)
    trainable_count = sum(parameter.numel() for parameter in trainable)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters: trainable={trainable_count/1e6:.3f}M total={total_count/1e6:.3f}M")

    optimizer: Optional[torch.optim.Optimizer] = None
    if args.mode == "train":
        optimizer = torch.optim.AdamW(
            trainable, lr=args.learning_rate, weight_decay=args.weight_decay
        )

    start_epoch = 0
    if args.checkpoint:
        state = load_checkpoint(args.checkpoint, model, optimizer)
        validate_checkpoint_fold(state, args.fold, args.data_mode)
        if isinstance(state.get("args"), dict):
            trained_orientation = state["args"].get(
                "proxy_orientation", "legacy_anomaly"
            )
            if trained_orientation != args.proxy_orientation:
                raise ValueError(
                    f"Checkpoint was trained with proxy_orientation={trained_orientation}; "
                    f"requested {args.proxy_orientation}. Use a matching score orientation."
                )
            if args.mode == 'eval':
                for key, fallback in (('backbone_name', 'dinov2_vits14'),
                                      ('nls_values', 'spatial_mean'), ('proxy_reduction', 'mean'),
                                      ('cnn_local_features', False), ('residual_matching', 'cosine')):
                    trained = state['args'].get(key, fallback)
                    requested = getattr(args, key, fallback)
                    if trained != requested:
                        raise ValueError(f'Checkpoint uses {key}={trained}; requested {requested}')
        start_epoch = int(state.get("epoch", -1)) + 1 if "format_version" in state else 0
        print(f"loaded checkpoint={args.checkpoint}")

    validation_loader = (
        build_loader(args, "eval")
        if (args.mode == "eval" and not (args.eval_shots or args.eval_seeds)) or (args.mode == 'train' and args.eval_frequency > 0)
        else None
    )
    output_dir = Path(args.output_dir)

    if args.mode == "eval":
        expert_cache = {}
        if not args.checkpoint:
            raise ValueError("--mode eval requires --checkpoint")
        if args.eval_shots or args.eval_seeds:
            for seed in args.eval_seeds or [args.seed]:
                for shot in args.eval_shots or [args.n_shot]:
                    evaluation_args = copy.copy(args)
                    evaluation_args.n_shot = shot
                    evaluation_args.seed = seed
                    setup_seed(seed)
                    root = output_dir / f'seed{seed}' if args.eval_seeds else output_dir
                    evaluation_args.output_dir = str(root / f'{shot}shot')
                    evaluation_model = with_pixel_experts(evaluation_args, model, device, expert_cache)
                    evaluate_and_save(evaluation_args, evaluation_model, build_loader(evaluation_args, 'eval'), device)
        else:
            evaluate_and_save(args, with_pixel_experts(args, model, device, expert_cache), validation_loader, device)
        return

    train_loader = build_loader(args, "train")
    best_score = float("-inf")
    best_path = output_dir / "checkpoints" / f"fold{args.fold}_{args.n_shot}shot_best.pth"
    for epoch in range(start_epoch, args.epochs):
        print(f"\nepoch={epoch + 1}/{args.epochs} train")
        training_metrics = run_epoch(args, model, train_loader, device, optimizer)
        if args.eval_frequency <= 0:
            final_path = output_dir / "checkpoints" / f"fold{args.fold}_{args.n_shot}shot_final.pth"
            save_checkpoint(
                final_path,
                checkpoint_state(model, optimizer, args, epoch, training_metrics),
            )
            print(f"saved epoch checkpoint without target evaluation: {final_path}")
            continue
        if (epoch + 1) % args.eval_frequency and epoch + 1 != args.epochs:
            continue
        print(f"epoch={epoch + 1}/{args.epochs} validation")
        metrics = run_epoch(args, model, validation_loader, device)
        score = metrics["i_auroc"] + metrics["p_auroc"]
        if score > best_score:
            best_score = score
            save_checkpoint(
                best_path,
                checkpoint_state(model, optimizer, args, epoch, metrics),
            )
            print(f"saved new best checkpoint: {best_path}")
        print_paper_gap(args, metrics)


if __name__ == "__main__":
    main()
