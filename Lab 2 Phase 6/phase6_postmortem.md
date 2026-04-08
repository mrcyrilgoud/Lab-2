# Phase 6 Post-Mortem: Paired HR-LR PSNR Plateau

Date: 2026-04-06

## Executive Summary

Phase 6 has not yet shown a meaningful improvement on the paired HR-LR validation set. The best completed Phase 6 screening result is `wide_se / coco_plus_imagenet` at `21.826 dB` paired-val PSNR. That is only about `+0.13 dB` over the Phase 3 paired-val diagnostic of `21.696 dB`, while the combined validation score is lower than the earlier 25.5-25.7 dB headline results.

The dominant issue appears to be a domain mismatch, not a lack of generic natural-image data. COCO/ImageNet synthetic pretraining improves synthetic validation performance but does not transfer strongly to the paired validation split. The paired training split is also much easier than the paired validation split by raw LR-vs-HR PSNR, so the paired fine-tune stage can fit the training distribution without solving the harder validation distribution.

There are also plausible architectural contributors: pervasive BatchNorm with small screening batches, EMA weights paired with non-EMA BatchNorm buffers, regularization designed for classification-style generalization, and attention/gating blocks that may suppress high-frequency residuals under synthetic pretraining.

## Current Evidence

Completed Phase 6 screening configs:

| Rank | Model | Mix | Params | Paired Val PSNR | Combined Val PSNR | COCO Val PSNR | ImageNet Val PSNR |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | `wide_se` | `coco_plus_imagenet` | 1,222,723 | 21.826 | 24.849 | 25.082 | 25.468 |
| 2 | `wide_se` | `coco_only` | 1,222,723 | 21.809 | 24.541 | 25.087 | 25.481 |
| 3 | `dsdan` | `coco_plus_imagenet` | 622,467 | 21.693 | 24.830 | 25.080 | 25.457 |

Earlier reference points:

| Phase | Model / Note | Metric | Value |
|---|---|---|---:|
| Phase 3 | ResNet-SE diagnostic | paired-val PSNR | 21.696 |
| Phase 3 | ResNet-SE diagnostic | ImageNet-val PSNR | 26.825 |
| Phase 3 | ResNet-SE diagnostic | combined-val PSNR | 25.543 |
| Phase 4A | Wide-SE Colab run | best combined `val_psnr` | 25.713 |
| Phase 5D | Large-kernel DW, user-reported | mean PSNR after 40 epochs | ~25.6 |

Important comparison caveat: Phase 6 combined validation includes paired, COCO, and sometimes ImageNet validation, while earlier combined validation generally meant paired + ImageNet. The paired-val comparison is cleaner than the combined-val comparison.

## Data Findings

### 1. COCO Helps Synthetic Validation, Not Paired Validation

For completed Stage 1 synthetic pretraining runs:

| Config | Best Paired Epoch | Best Paired | Last Paired | Best / Last Combined |
|---|---:|---:|---:|---:|
| `wide_se / coco_only` | 4 | 21.145 | 20.811 | 25.198 / 25.621 |
| `wide_se / coco_plus_imagenet` | 3 | 21.138 | 20.566 | 25.635 / 26.109 |
| `dsdan / coco_plus_imagenet` | 2 | 21.301 | 20.734 | 25.377 / 25.950 |

This is the main failure signature. As synthetic training progresses, synthetic/combined validation improves, but paired validation gets worse. That means the pretraining distribution is moving the model away from the paired validation objective.

Phase 6 Stage 1 is synthetic-only: COCO is always in the Stage 1 train set, ImageNet is added only for `coco_plus_imagenet`, and paired data is not in Stage 1. Stage 2 then switches to paired-only fine-tuning. This is implemented in `build_phase6_data_bundle`, where Stage 1 sets `train_parts` to COCO plus optional ImageNet and Stage 2 sets `train_parts` to only `paired_train`.

### 2. Paired Train and Paired Validation Are Not the Same Difficulty

Raw LR-vs-HR baseline PSNR measured locally:

| Split | n | Mean Baseline PSNR | Median | p10 | p90 |
|---|---:|---:|---:|---:|---:|
| `HR_train1` | 663 | 26.112 | 25.400 | 20.502 | 32.365 |
| `HR_train2` | 782 | 26.734 | 26.117 | 19.752 | 34.089 |
| `HR_train3` | 810 | 26.998 | 26.408 | 19.994 | 34.357 |
| `HR_train4` | 781 | 27.078 | 26.418 | 19.915 | 34.650 |
| `HR_val` | 100 | 21.336 | 21.548 | 17.914 | 24.072 |

The validation LR images are much farther from HR than the paired training LR images. This explains why Phase 6 Stage 2 can reach train-eval PSNR around `26.86-27.01 dB` but still only reach paired-val PSNR around `21.69-21.83 dB`.

This also means paired-only fine-tuning is not necessarily training on examples that resemble the hard validation distribution.

### 3. The Synthetic Degradation Recipe Is Still Hand-Written

Phase 6 synthetic LR generation uses random blur, downsample/upsample, JPEG compression, and optional tensor noise/cutout. That is broader than Phase 5, but it is still not learned from the actual paired LR images. The current Phase 6 synthetic degradation includes:

- Gaussian blur with radius `0.2-1.6`
- downsample scales `(2, 3, 4)`
- resize modes `bicubic`, `bilinear`, `lanczos`
- JPEG quality `25-90`
- train-time LR noise probability `0.30`
- train-time cutout probability `0.35`

This can create a useful restoration prior, but it does not guarantee the same artifact distribution as `LR_val`. The Stage 1 metric trajectory suggests it does not match well enough.

## Architecture Findings

### 1. BatchNorm Is Everywhere, Including Small-Batch Screening

All current candidates use BatchNorm heavily:

| Model | Params | BN Count | Notes |
|---|---:|---:|---|
| `wide_se` | 1,222,723 | 33 | SE residual stack |
| `dsdan` | 622,467 | 61 | depthwise separable dual attention stack |
| `repconv` | 2,117,091 | 49 | RepConv branches with BN fusion |
| `large_kernel_dw` | 650,595 | 43 | large-kernel depthwise blocks |
| `large_kernel_se` | 717,123 | 43 | large-kernel DW plus SE |
| `hybrid_rep_large_kernel` | 1,337,763 | 43 | alternating RepConv and large-kernel DW |

BatchNorm can be problematic for image restoration when:

- batch size is small; Phase 6 screening currently uses batch size `4`
- train and validation distributions differ
- pretraining and fine-tuning distributions differ
- evaluation uses EMA parameters while BN running statistics come from the non-EMA training trajectory

Phase 6’s EMA implementation tracks only `model.named_parameters()`, not buffers such as BatchNorm `running_mean` and `running_var`. Checkpoints save the model state and an `ema_shadow` parameter dict. When loading weights only, the code loads the full checkpoint state first, then overwrites parameters from EMA, leaving BN buffers from the non-EMA model. That can produce a parameter/statistics mismatch.

This is not proven to be the primary bottleneck, but it is one of the most actionable architecture-level suspects.

### 2. Regularization May Be Too Strong for Low-Data Paired Fine-Tuning

The models use dropout and stochastic depth inside restoration residual blocks:

| Model | Dropout | Max Drop Path |
|---|---:|---:|
| `wide_se` | 0.08 | 0.10 |
| `dsdan` | 0.08 | 0.08 |
| `repconv` | 0.06 | 0.08 |
| `large_kernel_dw` | 0.04 | 0.06 |
| `large_kernel_se` | 0.04 | 0.06 |
| `hybrid_rep_large_kernel` | 0.05 | 0.08 |

This may be reasonable for broad synthetic pretraining, but it can reduce the model’s ability to learn subtle, deterministic residual corrections from only 3,036 paired training samples. Since the validation improvement target is only a small residual over the LR input, this kind of regularization can become a ceiling rather than a safety net.

Recommended test: for paired fine-tuning, freeze or disable dropout and stochastic depth, then compare paired-val PSNR against the current Stage 2 recipe.

### 3. Attention and Gating May Learn the Wrong Domain Prior

`wide_se` uses SE gating in every residual block. `dsdan` uses both channel SE and a spatial depthwise gate. These mechanisms can learn to suppress or amplify residual features based on statistics learned during synthetic pretraining. If the synthetic artifacts are blur/JPEG/resize artifacts and the paired validation artifacts are different, the gates can reinforce the wrong residual behavior.

This is consistent with the `dsdan` result: it is more heavily gated and depthwise than `wide_se`, has roughly half the parameter count, and underperforms `wide_se` on paired validation in completed screening.

This is a hypothesis, not proven. The necessary ablation is a no-attention residual model or attention-disabled fine-tune of the same backbone.

### 4. Depthwise-Heavy Blocks May Be Capacity-Limited for Cross-Channel Corrections

The `dsdan`, `large_kernel_dw`, `large_kernel_se`, and `hybrid_rep_large_kernel` families rely on depthwise operations. Depthwise blocks are efficient and NPU-friendly, but they separate spatial filtering from cross-channel mixing. That can be a limitation if the paired LR artifacts require color-channel correction, demosaicing-like corrections, or chroma/luma coupling.

Only `dsdan` has completed so far, so this is not settled for the large-kernel models. However, it is a plausible architectural risk for the untested candidates.

### 5. Same-Resolution Residual Prediction May Be Too Conservative

Every Phase 6 architecture predicts:

```python
return x + self.tail(self.body(...))
```

This is a sensible default for restoration. But with paired validation baseline already around `21.336 dB`, the model only needs to learn a small, accurate residual. If the LR/HR pair has misalignment, non-invertible degradation, or content changes, the L1/Charbonnier objective can learn to output a conservative residual that avoids making PSNR worse. That would produce exactly the observed small paired-val improvements.

This is not an argument to remove the skip connection. It is an argument to inspect residual magnitude and per-sample residual direction on hard validation samples.

### 6. Loss Function Optimizes Smooth Fidelity, Not Necessarily Hard-Artifact Recovery

Phase 6 uses a combined Charbonnier + L1 loss. That is stable and PSNR-aligned enough for many restoration tasks, but it can under-emphasize the rare hard validation examples where the LR image differs significantly from HR. Since paired validation includes hard examples down to baseline PSNR ~16.3 dB, the mean loss may not push strongly enough on those cases.

Recommended test: paired fine-tune with hard-example weighting or stratified sampling by baseline LR-vs-HR PSNR.

## Training Protocol Findings

### 1. Stage 1 Best Paired Epoch Happens Very Early

For synthetic pretraining, paired validation peaks at epochs `2-4`, while synthetic validation continues improving. Because Stage 2 initializes from Stage 1 `best.pt`, this means only early synthetic pretraining is useful for the paired objective. Long synthetic pretraining is not currently buying much for paired validation.

### 2. Stage 2 May Be Too Short, But It Is Not the Whole Explanation

Completed Stage 2 runs have best paired-val PSNR at the final epoch:

| Config | Stage 2 Rows | Best Epoch | Best Paired |
|---|---:|---:|---:|
| `wide_se / coco_only` | 8 | 8 | 21.809 |
| `wide_se / coco_plus_imagenet` | 8 | 8 | 21.826 |
| `dsdan / coco_plus_imagenet` | 8 | 8 | 21.693 |

This implies paired fine-tuning may still be improving at epoch 8. A longer Stage 2 could help, but the train-eval vs paired-val gap is large enough that simply extending Stage 2 is unlikely to solve the full plateau by itself.

### 3. Current Screening Batch Size Is Much Smaller Than Earlier Colab Runs

Phase 5/Colab-style notebooks commonly used batch size `32`; Modal screening uses batch size `4`. This matters because the architectures are BatchNorm-heavy. Small batch statistics can alter training dynamics and make cross-domain fine-tuning less stable.

If cost allows, test at least one finalist with batch size `16` or gradient accumulation plus a normalization change. Gradient accumulation alone does not fix BatchNorm statistics, so the better ablation is “no BatchNorm / frozen BatchNorm / GroupNorm-free alternative” rather than just accumulation.

## Most Likely Root Causes

Ranked by confidence:

1. High confidence: COCO/ImageNet synthetic degradation does not match paired validation LR artifacts. Evidence: synthetic validation improves while paired validation degrades during Stage 1.
2. High confidence: paired validation is much harder than paired training. Evidence: raw baseline `HR_val` PSNR is ~21.3 dB vs ~26-27 dB for paired train folders.
3. Medium-high confidence: Stage 2 is adapting to paired train but not the paired validation distribution. Evidence: Stage 2 train-eval ~27 dB vs paired-val ~21.7-21.8 dB.
4. Medium confidence: BatchNorm is amplifying domain shift. Evidence: all candidates use many BN layers, screening batch size is 4, and EMA does not track BN buffers.
5. Medium confidence: dropout/stochastic depth are hurting precise restoration in paired fine-tuning. Evidence: all model configs use dropout/drop-path inside residual restoration blocks, and target gains are small residual corrections.
6. Medium-low confidence: attention/gating is learning synthetic-domain residual priors. Evidence: gated models do not outperform a simpler Wide-SE model so far, but only two architectures have completed.
7. Medium-low confidence: depthwise-heavy blocks are underpowered for paired-val artifacts. Evidence is currently insufficient because large-kernel depthwise models have not completed Phase 6 screening.

## Recommended Next Experiments

### Immediate, Low-Cost Diagnostics

1. Add a paired train/val degradation report to Phase 6 artifacts:
   - LR-vs-HR baseline PSNR by folder
   - color mean/std deltas
   - edge residual statistics
   - hard/easy sample IDs

2. Add residual diagnostics for the current best checkpoints:
   - predicted residual magnitude vs target residual magnitude
   - per-sample PSNR delta over LR baseline
   - hard-sample deltas for the 10 worst paired-val examples

3. Recompute paired-val PSNR with BN behavior variants:
   - standard eval mode
   - recalibrated BN statistics on paired train
   - train-mode BN for evaluation as a diagnostic only
   - EMA parameters disabled

### Next Screening Ablations

1. `wide_se` Stage 2 ablation:
   - initialize from the current `wide_se / coco_plus_imagenet` Stage 1 best
   - run paired fine-tune for 20 epochs
   - set dropout and drop-path to zero during Stage 2
   - keep selection on paired-val PSNR

2. BatchNorm ablation:
   - create a `wide_se_no_bn` or `wide_se_frozen_bn` screening candidate
   - keep the rest of the architecture identical
   - run only `coco_plus_imagenet`

3. Hard paired sampling:
   - compute train-pair baseline LR-vs-HR PSNR
   - oversample harder paired-train examples during Stage 2
   - compare against current uniform paired Stage 2

4. Keep a small hard-synthetic stream in Stage 2:
   - instead of paired-only Stage 2, use mostly paired data plus a small synthetic stream tuned to match paired-val baseline difficulty
   - do not use generic COCO degradation unchanged

5. Synthetic degradation calibration:
   - tune synthetic degradation so synthetic baseline LR-vs-HR PSNR distribution overlaps paired-val, not paired-train
   - explicitly include stronger non-blur artifacts if residual analysis shows they exist

### Model Search Priority

Continue screening the remaining models, but interpret results carefully:

1. `repconv`: useful because it has the highest parameter count and fewer attention gates, but still has heavy BN.
2. `large_kernel_dw`: useful to test whether larger spatial context helps hard validation examples.
3. `large_kernel_se`: test only after `large_kernel_dw`, because it isolates whether SE helps or hurts the large-kernel backbone.
4. `hybrid_rep_large_kernel`: useful only if either RepConv or large-kernel DW shows a positive signal.

If none of these moves paired-val PSNR beyond ~22.0 in screening, prioritize data/degradation and normalization changes over larger architectures.

## Decision Recommendation

Do not move current Phase 6 winners directly into full 40+20 training as-is. The current signal is too weak:

- best paired-val gain over Phase 3 is only about `+0.13 dB`
- combined validation is lower than earlier headline results
- Stage 1 clearly optimizes synthetic validation at the expense of paired validation
- Stage 2 still has a large paired-train vs paired-val gap

Proceed with the remaining one-by-one screening if the goal is to finish the architecture comparison, but treat it as diagnostic screening, not finalist selection. Before full training, add at least one normalization ablation and one paired-hardness/degradation ablation.

