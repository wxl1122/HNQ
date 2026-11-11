# Method and configuration notes

## Shared frozen features

The recommended profile uses DINOv2 ViT-B/14 with register tokens. The encoder
is frozen and remains in evaluation mode even when its RFL/NLS head is trained.
Two independently trained heads share the same encoder at inference. Each head
retains residual feature learning and normality learning rather than replacing
HNQ with a task-specific target-trained classifier.

## Normal reference matching and localization

For localization, concatenate individually normalized features from blocks
6, 9 and 12 (zero-based indices 5, 8 and 11). Use this descriptor to find the
nearest normal reference patch. Form the residual by subtracting the matched
patch's original final-block feature from the query's final-block feature.
Thus multi-block descriptors determine correspondence; the learned RFL/NLS
modules still receive final-block features in their original feature space.

RFL aggregates residuals into learned proxies. NLS uses matched normal values
to translate these into appearance proxies. The compatibility head's pixel
score combines its proxy cue with the multi-block nearest-normal distance,
using proxy weight 0.025. Pixel matching uses the unrotated K references.

## Image score

Both image heads use final-block matching and 0/90/180/270-degree views of the
same K reference images. The views do not introduce extra reference identities.

- Compatibility head: legacy-anomaly proxy semantics, proxy weight 0.35,
  nearest-normal weight 0.65, and top-20 patch-score averaging.
- Normality head: complement the learned normality similarity to obtain an
  anomaly cue; use proxy weight 0.2, nearest-normal weight 0.8, and top-10
  patch-score averaging.

The final image score is the equal arithmetic mean of the two image scores.
There are no per-category weights or label-dependent prediction corrections.
The same weights and settings serve all three support budgets.

## Source training

The two recipes use VisA labels/masks, not target-domain gradients:

- `configs/train/base_register.json`: compatibility orientation/loss, 2-shot
  episodes, 15 epochs of 1,000 episodes, AdamW learning rate 1e-4.
- `configs/train/hnq_base_normality.json`: normality-oriented loss, mixed
  1/2/4-shot episodes, 20 epochs of 1,000 episodes, AdamW learning rate 5e-5.

Both use weight decay 0.01, batch size 8, training seed 111, matched NLS values
and 25 learned proxies. The normality-oriented objective combines image BCE
with pixel focal/Dice losses, using logit scale 5 and center 0.3. Both recipes
disable target evaluation and save the final epoch, not a target-selected epoch.
Source metadata test-partition labels are used during training; this is not
normal-only or source-train-partition-only learning.

## Runtime and evaluation safeguards

Frozen descriptors are cached by image content in bounded CPU memory. The
normality head reuses the final-block prefix of the richer primary feature
cache. Cache reuse must not overwrite multi-block entries with final-only
features. Repeated fixed support batches are encoded once when possible.

Full evaluation visits each target query once, uses fixed normal-training
references, enforces nested 1/2/4-shot draws, and rejects query/support overlap.
Coverage audits also check VisA against its official split CSV. Medical helpers
exclude reference/query duplicate content when a new cohort manifest is built.

The tests verify that prediction values do not depend on query labels or mask
contents. Labels and masks are used for source supervision or metric/loss
calculation, not to adjust inference predictions.

## Limitations

The larger encoder, additional head and reference views change the original
conference compute/protocol. Inference settings were developed using target
metrics; neither a replay nor the absence of target gradients establishes
independent held-out method selection. Full-test prevalence affects AP/F1,
and support-seed variation is not training-seed variation. No measured results
or claims of matching every independently retrained run are bundled here.
