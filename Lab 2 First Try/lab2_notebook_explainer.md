# Lab 2 Notebook Explanation

This document explains how the Lab 2 notebook works, why it was structured this way, and the main tradeoffs in the design.

Note: the notebook currently in the workspace is [lab2_same_resolution_residual_attention_pipeline.ipynb](/Users/mrcyrilgoud/Desktop/Data 255/Lab2/lab2_same_resolution_residual_attention_pipeline.ipynb). This is the notebook that replaced the earlier `lab2_residual_attention_starter.ipynb` name.

## Overview

The notebook is a single-file prototype for a same-resolution image restoration workflow. Its purpose is to let you validate the full Lab 2 pipeline in one place before or while moving pieces into a package-style project layout such as `src/`, `train.py`, and `export_onnx.py`.

At a high level, it does five things:

1. Validates that the dataset uses the expected canonical train/validation split folders.
2. Loads paired low-resolution and high-resolution RGB images with synchronized augmentation.
3. Defines a same-resolution residual attention model in PyTorch.
4. Trains and validates the model using L1 loss and PSNR-based checkpoint selection.
5. Exports the trained model to ONNX and optionally checks PyTorch vs ONNX Runtime parity.

## How the Notebook Works

### 1. Imports and environment setup

The first code cell imports PyTorch, ONNX-related packages, and the dataset helpers from [src/data.py](/Users/mrcyrilgoud/Desktop/Data 255/Lab2/src/data.py). It also selects the device:

- `cuda` if a GPU is available
- otherwise `cpu`

This keeps the rest of the notebook device-agnostic.

### 2. Dataset contract and validation

The notebook assumes the dataset is already organized as:

- `training_data/LR_train`
- `training_data/HR_train`
- `training_data/LR_val`
- `training_data/HR_val`

It calls `require_canonical_split_dirs(...)` to fail early if that layout is missing. This is intentional: the training code should not guess dataset structure.

Then it runs `verify_split(...)`, which:

- collects LR/HR image pairs by basename
- checks that every basename exists in both folders
- loads every image as RGB
- verifies each image is `256x256`

This catches dataset problems before any model code runs.

### 3. Paired dataset and loaders

The notebook builds `PairedImageDataset` instances for train and validation.

That dataset class:

- pairs LR and HR files strictly by basename within a split
- converts images to float tensors in `[0,1]`
- applies train-only augmentation with synchronized horizontal flip, vertical flip, and `k * 90` rotation

The notebook then wraps those datasets in `DataLoader`s and inspects one batch to confirm:

- batch shape
- dtype
- value range
- a few example basenames

This is a fast sanity check that the data pipeline is behaving as expected.

### 4. Model definition

The notebook defines `MultiScaleResidualAttentionNet`, which follows a same-resolution restoration design:

- a `3 -> 48` convolution stem
- three residual groups
- three residual blocks per group
- stage attention after each group
- a `48 -> 24 -> 3` tail
- a final residual add with the original input image

The model keeps the input and output spatial size the same, so it is restoration rather than explicit upscaling.

It also checks for forbidden modules such as batch normalization or ReLU-based layers that would violate the export-safe design constraints.

Finally, it runs a forward smoke test on a random `[1, 3, 256, 256]` tensor and prints:

- parameter count
- output tensor shape

That makes architecture regressions easy to spot.

### 5. Metrics, scheduling, and training loop

The training section defines:

- `compute_psnr(...)`
- `lr_for_epoch(...)`
- `train_one_epoch(...)`
- `validate(...)`
- `save_checkpoint(...)`
- `fit(...)`

The training behavior is:

- optimize with `AdamW`
- use L1 loss for training
- use linear warmup followed by cosine decay for the learning rate
- compute validation PSNR on clamped `[0,1]` predictions
- save `last.pt` every epoch
- save `best.pt` when validation PSNR improves
- append epoch metrics to `metrics.jsonl`

This mirrors what a script-based training pipeline would do, but keeps it notebook-friendly for debugging and inspection.

### 6. Tiny smoke test

The notebook includes a tiny smoke test that trains on a very small subset of samples.

This is useful because it answers a practical question early:

“Does the pipeline run end-to-end without crashing?”

That is often more important at first than absolute PSNR performance.

### 7. ONNX export and parity check

The export section loads a checkpoint into the same model definition used for training and validation, then exports it with:

- fixed input shape `[1, 3, 256, 256]`
- opset `13`
- input name `input`
- output name `output`
- no dynamic axes

If `onnx` is installed, it runs the ONNX checker.

If `verify=True` and `onnxruntime` is installed, it also:

- loads one fixed sample image
- runs both PyTorch and ONNX Runtime
- measures `max_abs_diff`
- measures `mean_abs_diff`

That helps confirm the exported graph matches the PyTorch model closely enough.

## Thought Process Behind the Notebook

The notebook structure comes from a few deliberate decisions.

### Start with a full pipeline, not isolated fragments

Instead of only defining the model, the notebook includes dataset checks, training utilities, and export logic. The reasoning is that Lab 2 is not just an architecture exercise. It is also a workflow exercise:

- can the dataset be loaded correctly?
- can the model train?
- can checkpoints be selected correctly?
- can the model be exported safely?

Putting the entire path in one notebook makes early debugging much faster.

### Fail early on dataset mistakes

A lot of training failures are really data failures. That is why the notebook validates split folders, basename pairing, image mode, and spatial size before doing anything expensive. The goal was to move errors earlier and make them easier to interpret.

### Keep the model export-safe from the beginning

The architecture avoids layers and operations that could complicate ONNX export. That design choice reduces the chance of building a model that trains fine in PyTorch but becomes painful to export later.

### Separate training behavior from model behavior

The notebook clamps predictions only for PSNR computation, not inside the model. The thought process here is that evaluation constraints and model graph constraints should stay separate. That makes the exported model simpler and more conservative.

### Include a smoke-test path

A small overfit or smoke-test mode is valuable because it shortens iteration time. Before committing to a long run, you want evidence that:

- tensors line up
- loss decreases
- validation executes
- checkpoint writing works

The tiny test gives that quickly.

### Keep the notebook close to a future script layout

Even though this is a notebook, its sections map naturally to a project structure:

- data utilities
- model definition
- metrics
- training loop
- export logic

That makes it easier to convert into `src/model.py`, `src/metrics.py`, `train.py`, and `export_onnx.py` later.

## Advantages

### 1. End-to-end visibility

Everything important is visible in one place:

- dataset assumptions
- model design
- training behavior
- checkpoint logic
- export behavior

That makes it easier to understand and debug the full system.

### 2. Strong early validation

The notebook checks many failure points up front:

- missing split folders
- mismatched LR/HR basenames
- wrong image sizes
- forbidden modules
- output shape mismatches
- non-finite PSNR values

That reduces silent failure.

### 3. Good debugging ergonomics

The batch inspection cell, smoke test, parameter count print, and export verification all help narrow down issues quickly.

### 4. Clear path to production-style scripts

Although the notebook is interactive, the logic is organized in a way that is easy to migrate into standalone scripts.

### 5. Export-aware architecture

The model was designed with ONNX export in mind from the start, which lowers downstream integration risk.

## Drawbacks

### 1. Notebook coupling

Even though the structure is clean, notebooks are still less modular than scripts. State can leak across cells, and rerunning cells out of order can create confusion.

### 2. Some duplication with the `src/` package

The notebook uses `src.data`, but it still defines the model, metrics, training loop, and export logic inline. That is convenient for experimentation, but it means there is still code that eventually should live in reusable modules.

### 3. Limited experiment management

The notebook is good for a baseline workflow, but it is not yet ideal for repeated controlled experiments with many configurations, multiple runs, or more formal logging.

### 4. CPU-first notebook behavior can be slow

If the environment falls back to CPU, the smoke test and any real training can be slow. The notebook handles that correctly, but not efficiently.

### 5. Export verification depends on optional packages

The parity check is useful, but it only runs if `onnx` and `onnxruntime` are installed. Without them, export validation is incomplete.

## When This Notebook Works Best

This notebook is strongest as:

- a reference implementation of the full Lab 2 flow
- a debugging workspace for architecture and data issues
- a staging area before code is split into scripts

It is weaker as:

- a final reproducible training system for many experiments
- a long-term maintainable codebase without further refactoring

## Recommended Next Step

The clean next step is to treat this notebook as the verified prototype, then move the remaining inline logic into reusable project files:

- `src/model.py`
- `src/metrics.py`
- `train.py`
- `export_onnx.py`

That gives you the same behavior with better maintainability and easier reruns.
