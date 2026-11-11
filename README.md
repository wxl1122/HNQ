# HNQ

Code for **Hunting Normality from Query Sample via Residual Learning for
Generalist Anomaly Detection**.

HNQ learns residual proxies from query-minus-normal features (RFL), translates
them into normal appearance proxies (NLS), and hunts normality to score query
images and localize anomalies. The visual encoder is frozen.


## Layout

```text
configs/                 Shared-Base evaluation and two source-training recipes
dataset_preparation/     MVTec AD / VisA metadata and optional layout converter
models/                  Frozen encoder, RFL, NLS and score fusion
utils/                   Data loading, reference views, losses and metrics
tools/                   Training queue, checkpoint export and coverage audits
tests/                   CPU regression tests with synthetic data/encoders
docs/                    Method, protocol and reproduction notes
run.py                   Training/evaluation entry point
```

## Environment

Use Python 3.10 and install the pinned runtime from the repository root:

```bash
python -m pip install -r requirements-eval-lock.txt
python -m pip install pandas
```

The pinned environment uses PyTorch 2.8.0 and torchvision 0.23.0. Install a
PyTorch build compatible with your GPU/driver; `requirements.txt` contains
broader constraints rather than a promise of identical behavior across versions.
xFormers is disabled by default and is not required.

DINOv2 code and pretrained **encoder** weights are loaded using torch.hub on
first use. These upstream pretrained weights are distinct from the HNQ heads
you train. For an offline setup, pass:

```text
--dinov2-repo /path/to/dinov2
--pretrained-weights-dir /path/to/pretrained
```

For the recommended profile, the latter directory must contain
`dinov2_vitb14_reg4_pretrain.pth`. Encoder signatures are checked when loading
exported heads. See the upstream [DINOv2 repository](https://github.com/facebookresearch/dinov2)
for its code, pretrained models and license.

## Prepare data

Download MVTec AD and VisA separately under their respective terms. Use the
original VisA layout with its official `split_csv/1cls.csv` file:

```text
/path/to/datasets/
├── mvtec/
│   ├── bottle/, cable/, ... (original train/test/ground_truth directories)
│   └── meta.json
└── visa/
    ├── split_csv/1cls.csv
    ├── ... (original image and mask paths)
    └── meta.json
```

Generate metadata when needed; these commands write `meta.json`:

```bash
python -m dataset_preparation.mvtec --root /path/to/datasets/mvtec
python -m dataset_preparation.visa --root /path/to/datasets/visa
```

`dataset_preparation/convert_visa_to_mvtec.py` is an optional alternative
converter, not a required step for the original-layout commands above. Its
Amazon copyright and Apache-2.0 notice are retained.

## Train the two source heads

The included recipes use **VisA as the source domain** and reserve MVTec AD as
the target (`fold=0`). They train from scratch on top of the frozen pretrained
encoder, disable target evaluation, and save the final epoch. Source training
uses labeled source images/masks, including source metadata's test partition;
it must not be described as normal-only or source-train-partition-only learning.

```bash
python -m tools.train_source_heads \
  --recipes configs/train/base_register.json configs/train/hnq_base_normality.json \
  --gpus 0 2 \
  --data-root /path/to/datasets \
  --dinov2-repo /path/to/dinov2 \
  --pretrained-weights-dir /path/to/pretrained \
  --output-dir output/source_training
```

The queue accepts one or two distinct physical GPUs and resolves indices to
UUIDs. Pass `--gpus 0` to train sequentially on one card. The output directory
must be new. Alternatively, run each recipe directly with `python run.py
--config ... --data-root ... --device cuda:0`.

Export the trained heads to the locations expected by the evaluation profile:

```bash
python -m tools.export_inference_checkpoint \
  --checkpoint output/source_training/base_register/checkpoints/fold0_2shot_final.pth \
  --output checkpoints/base_register.pth \
  --dinov2-repo /path/to/dinov2

python -m tools.export_inference_checkpoint \
  --checkpoint output/source_training/hnq_base_normality/checkpoints/fold0_2shot_final.pth \
  --output checkpoints/normality.pth \
  --dinov2-repo /path/to/dinov2
```

Export preserves trainable tensors and records the frozen encoder identity; it
does not train or bundle a second encoder. Only load trusted checkpoints.

## Evaluate every target test image

After training and exporting both heads:

```bash
python run.py --config configs/hnq_base.json \
  --data-root /path/to/datasets --device cuda:0 \
  --dinov2-repo /path/to/dinov2 \
  --pretrained-weights-dir /path/to/pretrained \
  --output-dir output/evaluation
```

The profile declares 1/2/4-shot and support seeds 111/222/333/444/555. Each
run evaluates every target test image exactly once, with fixed train-normal
references, nested support sets and no query/reference overlap. In this mode
`--eval-episodes` does not limit the query set. Metrics are category macro
averages at 448 x 448, Gaussian sigma 4, and exact AUPRO up to FPR 0.3.
F1-max uses evaluation-set thresholds, not calibrated deployment thresholds.

The FP32 multi-block feature cache needs substantial **host RAM**, in addition
to GPU memory. On lower-memory hosts, override `--feature-cache-size 0
--workers 0 --metric-workers 1`; this changes runtime settings, not the scoring
equations. Batch size and software changes can still affect numerical details.

Summarize all declared seeds without silently dropping missing runs:

```bash
python -m tools.summarize_seeds output/evaluation \
  --seeds 111 222 333 444 555 --output output/seed_summary.json
```

For an audited two-GPU replay, first create manifests **from your own trained
heads and local data**:

```bash
python -m tools.check_release --config configs/hnq_base.json \
  --dinov2-repo /path/to/dinov2 --output output/model_manifest.json

python -m tools.audit_exhaustive_dataset --data-root /path/to/datasets \
  --seeds 111 222 333 444 555 --output output/dataset_audit.json

python -m tools.evaluate_seed_queue --config configs/hnq_base.json \
  --manifest output/model_manifest.json --gpus 0 2 \
  --data-root /path/to/datasets \
  --dinov2-repo /path/to/dinov2 \
  --pretrained-weights-dir /path/to/pretrained \
  --output-dir output/evaluation_queue

python -m tools.summarize_benchmark --root output/evaluation_queue \
  --config configs/hnq_base.json --audit output/dataset_audit.json \
  --manifest output/model_manifest.json --output output/checked_benchmark.json
```

All output paths are local, untracked artifacts. Use new output directories;
do not overwrite runs or change a profile after creating its manifest.

## Correctness checks

```bash
python -m unittest discover -s tests -v
```

The tests use synthetic data/encoders on CPU; no dataset or checkpoint download
is required. They cover residual matching, shared features, score fusion,
label-independent predictions, full coverage, nested supports and split safety.

## Reproduction scope

The shared-Base settings were developed using target-domain metrics. Repeating
the fixed protocol is not independent held-out validation of method selection.
Support seeds are not independent model-training seeds, and a natural full-test
protocol is not identical to sampled balanced episodes. Retraining is not
guaranteed to reproduce historical numbers bit-for-bit. This release deliberately
contains no performance tables or measured result files.

See [method and configuration notes](docs/METHOD.md). Reverse transfer requires
independently trained MVTec-source heads (`--fold 1`); never reuse VisA-trained
heads to claim generalization to VisA. Generic medical split safeguards remain
in the code, but no medical cohort manifest or dataset is distributed.

## Acknowledgement and license

We retain the original repository's acknowledgement of
[InCTRL](https://github.com/mala-lab/InCTRL),
[ResAD](https://github.com/xcyao00/ResAD),
[AnomalyDINO](https://github.com/dammsi/AnomalyDINO),
[AnomalyCLIP](https://github.com/zqhang/AnomalyCLIP), and
[WinCLIP](https://github.com/zqhang/Accurate-WinCLIP-pytorch).

Project code is covered by the existing [MIT license](LICENSE), except where
third-party notices specify otherwise. Datasets, DINOv2 code and pretrained
weights retain their own licenses and are not redistributed here. The project
license does not relicense any dataset.
