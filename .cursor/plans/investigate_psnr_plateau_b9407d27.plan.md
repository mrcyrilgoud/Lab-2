---
name: Investigate PSNR Plateau
overview: Diagnose why training PSNR keeps rising while validation PSNR stalls in the super-resolution notebooks, extend the plan to augment the current SR data pipeline with the new ImageNet-based train/val manifests, and do the implementation in a new notebook under `Lab 2 Phase 3`.
todos:
  - id: eval-train-psnr
    content: Measure training PSNR in eval mode on a fixed train subset to remove BatchNorm logging bias
    status: completed
  - id: integrate-imagenet-manifests
    content: Wire `Data/ImageNet` manifests and `imagenet_train20a` / `imagenet_val20a` folders into the notebook data-loading pipeline as an additional source
    status: completed
  - id: evaluate-regularization
    content: Evaluate architecture-level regularization options such as dropout, stochastic depth, or reduced capacity to lower the train/val PSNR gap
    status: completed
  - id: expand-augmentations
    content: Evaluate data-side regularization such as random crops, cutout/coarse dropout, mix-style corruption, and train-only degradation augmentation
    status: completed
  - id: compare-distributions
    content: Compare per-image PSNR and sample difficulty across the original split and the augmented ImageNet-backed split
    status: completed
  - id: assess-split-shift
    content: Check whether the current val folder and the new ImageNet val manifest represent different difficulty or domain characteristics from training
    status: completed
  - id: choose-fix-path
    content: Decide whether to prioritize logging correction, ImageNet data augmentation, early stopping/regularization, or validation split changes
    status: completed
isProject: false
---

# Investigate Validation PSNR Plateau

## Implementation Target
This work should be implemented in a new Phase 3 notebook instead of editing the current Phase 2 notebook in place.

- Create a new folder: [Lab 2 Phase 3](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 3)
- Create the new working notebook there, based on the current SE-ResNet notebook:
  [lab2_phase2_resnet_se.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/lab2_phase2_resnet_se.ipynb)
- The intended destination notebook should be a new Phase 3 copy, for example:
  [lab2_phase3_resnet_se.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 3/lab2_phase3_resnet_se.ipynb)

This keeps Phase 2 as the baseline and makes Phase 3 the place for experiments involving ImageNet augmentation, architecture regularization, and stronger data manipulation.

## Current Diagnosis
The strongest evidence points to two effects happening at once:

1. Metric mismatch during logging
   In both notebooks, training PSNR is computed while the model is still in `train()` mode, but validation PSNR is computed in `eval()` mode. Because these models use `BatchNorm2d`, the train and val PSNR numbers are not directly comparable.

```454:480:/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 1/lab2_phase1_resnet.ipynb
"def train_one_epoch(model, loader, optimizer, cfg):\n",
"    model.train()\n",
"    ...\n",
"        with torch.no_grad():\n",
"            total_psnr += compute_psnr(pred.detach(), hr_img).sum().item()\n",
"...\n",
"@torch.no_grad()\n",
"def validate(model, loader, cfg):\n",
"    model.eval()\n",
```

The same pattern exists in [lab2_phase2_resnet_se.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/lab2_phase2_resnet_se.ipynb).

2. Real overfitting / generalization ceiling
   The saved metrics show training PSNR continuing to improve while validation PSNR plateaus early.

- Phase 1: train PSNR rises from `15.49 -> 26.01`, while val PSNR rises to about `21.32` and then stalls. Source: [metrics.jsonl](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 1/runs/resnet_sr/metrics.jsonl)
- Phase 2: train PSNR rises from `18.87 -> 27.14` by epoch 41, while val PSNR only reaches about `21.74`. Source: [metrics.jsonl](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/runs/resnet_se_sr/metrics.jsonl)

3. Validation may be a harder or shifted distribution
   Training samples come from nested `HR_train*` / `LR_train*` folders, while validation comes from a separate flat `val/HR_val` / `val/LR_val` split. There is no obvious pairing bug, but the split construction suggests a real train-vs-val distribution gap.

```121:167:/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/lab2_phase2_resnet_se.ipynb
"HR_TRAIN_ROOT = DATA_ROOT / \"HR_train\"\n",
"LR_TRAIN_ROOT = DATA_ROOT / \"LR_train\"\n",
"HR_VAL_DIR = DATA_ROOT / \"val\" / \"HR_val\"\n",
"LR_VAL_DIR = DATA_ROOT / \"val\" / \"LR_val\"\n",
"...\n",
"train_pairs = collect_paired_by_subfolder(LR_TRAIN_ROOT, HR_TRAIN_ROOT)\n",
"...\n",
"val_pairs = collect_paired_flat(LR_VAL_DIR, HR_VAL_DIR)\n",
```

4. Validation is small
   Both notebooks report `3036` train pairs and only `100` val pairs, so validation PSNR will be noisier and easier to plateau visually.

5. The new ImageNet data can help, but it needs ingestion work first
   `Data/ImageNet` is not laid out like the current SR data roots. It currently exposes two manifest files:

- [imagenet_train20.txt](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Data/ImageNet/imagenet_train20.txt) with `6000` entries
- [imagenet_val20.txt](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Data/ImageNet/imagenet_val20.txt) with `1000` entries

   Each line is a JPEG filename plus a class ID, so the plan needs to parse these manifests and resolve them against the actual image folders you identified: `imagenet_train20a` and `imagenet_val20a`. That makes this an augmentation task, not just a root-path swap.

6. The current regularization story is minimal
   In the current notebook, the model uses BatchNorm and weight decay, but there is no explicit architecture-level regularization such as dropout, stochastic depth, or feature-level noise injection. On the data side, training only uses flips and 90-degree rotations.

```315:340:/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/lab2_phase2_resnet_se.ipynb
"        self.bn1 = nn.BatchNorm2d(channels)\n",
"        self.act = nn.PReLU(num_parameters=channels)\n",
"        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)\n",
"        self.bn2 = nn.BatchNorm2d(channels)\n",
"        self.se = SEBlock(channels, reduction)\n",
"...\n",
"        self.stem = nn.Sequential(\n",
"            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),\n",
"            nn.BatchNorm2d(channels),\n",
"            nn.PReLU(num_parameters=channels),\n",
```

```188:217:/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/lab2_phase2_resnet_se.ipynb
"        if self.train:\n",
"            if random.random() > 0.5:\n",
"                lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)\n",
"                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)\n",
"            if random.random() > 0.5:\n",
"                lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)\n",
"                hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)\n",
"            k = random.randint(0, 3)\n",
```

## What This Most Likely Means
The high training PSNR is partly an artifact of how it is measured, but not entirely. Even after correcting the logging, the saved curves already look like a classic generalization gap:

- the model keeps fitting the training set
- the validation set improves early, then saturates
- the gap grows to about `4.5-5.5 dB`

That combination is much more consistent with overfitting and/or dataset mismatch than with a broken PSNR formula.

## Recommended Next Checks
If you want to verify this before changing architecture or hyperparameters, do these in order:

1. Recompute training PSNR in `eval()` mode on a fixed subset of the training set
   This isolates how much of the train/val gap is caused by BatchNorm logging behavior.

2. Add an ImageNet-backed data source alongside the current SR split
   Parse the two manifest files, resolve filenames into `imagenet_train20a` / `imagenet_val20a`, and decide how those samples are converted into the LR/HR pairing format expected by the notebooks.

3. Evaluate architecture-level anti-overfitting changes before making the network larger
   Compare simple interventions first: dropout inside residual blocks, spatial dropout / dropout2d on features, stochastic depth on residual branches, and reducing width or block count. The goal is to test whether the current SE-ResNet is memorizing rather than failing for purely data reasons.

4. Evaluate stronger train-only data manipulations
   Extend beyond flips and rotations with patch-based training, random crops, cutout/coarse dropout, blur/noise/compression perturbations on LR inputs, and possibly mixup-style corruption only if it preserves the paired SR objective.

5. Compare per-image PSNR distributions, not just epoch averages
   Measure train-subset PSNR and val PSNR image-by-image to see whether the original val split or the ImageNet-backed split is uniformly harder, or whether a few difficult samples dominate the mean.

6. Inspect whether the validation sets differ visually from training
   Compare the current `val/*` images against the ImageNet-backed val samples to check for differences in content, blur level, noise, crop style, or compression that could cap PSNR.

7. Use best-val checkpoint and stop treating later train improvements as meaningful
   In Phase 1, best val is already around epoch 9. In Phase 2, gains after the mid-20s are very small. That suggests the model is already near its val ceiling under the current setup.

## Most Likely Fix Directions
If the user wants implementation next, prioritize:

- make train-side PSNR logging fair by evaluating under `model.eval()` on a held-out train subset
- add a manifest-driven ImageNet ingestion path that augments the existing train and val datasets instead of replacing them
- try the smallest architecture regularizers first: dropout or dropout2d inside residual paths, stochastic depth, or a modest reduction in width/depth
- strengthen train-only augmentation with crop-based sampling, cutout/coarse dropout, and LR-only degradations that mimic harder validation examples
- reduce overfitting pressure with either a smaller model, stronger regularization, or earlier stopping
- strengthen validation reliability with a larger or more representative val split, potentially using the new ImageNet val manifest
- only after that, tune loss or architecture further

## Relevant Files
- [lab2_phase1_resnet.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 1/lab2_phase1_resnet.ipynb)
- [lab2_phase2_resnet_se.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/lab2_phase2_resnet_se.ipynb)
- [Lab 2 Phase 3](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 3)
- [lab2_phase3_resnet_se.ipynb](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 3/lab2_phase3_resnet_se.ipynb)
- [imagenet_train20.txt](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Data/ImageNet/imagenet_train20.txt)
- [imagenet_val20.txt](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Data/ImageNet/imagenet_val20.txt)
- [Phase 1 metrics.jsonl](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 1/runs/resnet_sr/metrics.jsonl)
- [Phase 2 metrics.jsonl](/Users/cyrilgoud/Desktop/repos/personal/Lab 2/Lab 2 Phase 2/runs/resnet_se_sr/metrics.jsonl)