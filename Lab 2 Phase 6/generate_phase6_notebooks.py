from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase6_screening_common import (
    MODEL_ORDER,
    MIX_ORDER,
    PORTABLE_MODEL_SOURCES,
    REPO_ROOT,
    create_portable_notebook,
)


SCREENING_NOTEBOOK_PATH = Path(__file__).with_name("lab2_phase6a_screening_matrix.ipynb")


def screening_notebook() -> dict:
    code = """from __future__ import annotations

from pathlib import Path
import json
import os

from phase6_screening_common import (
    assert_npu_compatible,
    build_phase6_data_bundle,
    configure_runtime,
    choose_amp_policy,
    env_flag,
    env_int,
    export_to_onnx,
    fit_stage,
    get_model_spec,
    print_data_summary,
    resolve_phase6_workspace,
    run_diagnostics,
)

MODEL_ID = os.environ.get("LAB2_MODEL_ID", "wide_se")
PRETRAIN_MIX = os.environ.get("LAB2_PRETRAIN_MIX", "coco_only")
STAGE_NAME = os.environ.get("LAB2_STAGE", "stage1_pretrain")
EPOCHS = env_int("LAB2_STAGE_EPOCHS", 12 if STAGE_NAME == "stage1_pretrain" else 8)
BATCH_SIZE = env_int("LAB2_BATCH_SIZE", 4)
NUM_WORKERS = env_int("LAB2_NUM_WORKERS", 2)
SEED = env_int("LAB2_SEED", 255)
RESUME_TRAINING = env_flag("LAB2_RESUME_TRAINING", True)
RUN_DIAGNOSTICS = env_flag("LAB2_RUN_DIAGNOSTICS", True)
RUN_ONNX_EXPORT = env_flag("LAB2_RUN_ONNX_EXPORT", False)
USE_AMP = env_flag("LAB2_USE_AMP", True)
CHANNELS_LAST = env_flag("LAB2_CHANNELS_LAST", True)
INIT_CHECKPOINT = os.environ.get("LAB2_INIT_CHECKPOINT")

workspace = resolve_phase6_workspace(output_subdir="phase6_screening")
device = configure_runtime()
amp_policy = choose_amp_policy(device)
if not USE_AMP:
    amp_policy = {"enabled": False, "dtype": None, "use_scaler": False, "label": "fp32"}
channels_last = bool(CHANNELS_LAST and device.type == "cuda")

spec = get_model_spec(MODEL_ID)
stage_cfg = {
    "stage_name": STAGE_NAME,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": 3e-4 if STAGE_NAME == "stage1_pretrain" else 1.5e-4,
    "weight_decay": 2e-4,
    "warmup_epochs": 2 if STAGE_NAME == "stage1_pretrain" else 1,
    "min_lr_ratio": 0.05,
    "checkpoint_interval": 4 if STAGE_NAME == "stage1_pretrain" else 2,
    "train_eval_interval": 2 if STAGE_NAME == "stage1_pretrain" else 1,
    "seed": SEED,
    "early_stop_patience": 6 if STAGE_NAME == "stage1_pretrain" else 5,
    "grad_clip_norm": 1.0,
    "ema_decay": 0.999,
    "charb_eps": 1e-6,
    "selection_metric": "paired_val_psnr",
}
data_cfg = {
    "train_patch_size": 224,
    "eval_size": 256,
    "random_scale_pad": 48,
    "cutout_prob": 0.35,
    "cutout_ratio": 0.18,
    "lr_noise_prob": 0.30,
    "lr_noise_std": 0.02,
    "lr_blur_prob": 0.80,
    "blur_radius_min": 0.2,
    "blur_radius_max": 1.6,
    "jpeg_prob": 0.60,
    "jpeg_quality_min": 25,
    "jpeg_quality_max": 90,
    "downsample_scales": (2, 3, 4),
    "resize_modes": ("bicubic", "bilinear", "lanczos"),
    "imagenet_train_limit": 6000,
    "imagenet_val_limit": 300,
    "coco_train_limit": 12000,
    "coco_val_limit": 500,
    "train_eval_subset_size": 128,
}

print(json.dumps({
    "model_id": MODEL_ID,
    "mix_id": PRETRAIN_MIX,
    "stage_name": STAGE_NAME,
    "stage_cfg": stage_cfg,
    "data_cfg": data_cfg,
    "workspace": {k: str(v) for k, v in workspace.items()},
    "params": spec["params"],
    "ops": spec["ops"],
}, indent=2))

model_for_check = spec["build_model"](**spec["model_cfg"])
assert_npu_compatible(model_for_check)

data_bundle = build_phase6_data_bundle(
    workspace=workspace,
    data_cfg=data_cfg,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    device=device,
    seed=SEED,
    pretrain_mix=PRETRAIN_MIX,
    stage_name=STAGE_NAME,
)
print_data_summary(data_bundle)

history = fit_stage(
    model=spec["build_model"](**spec["model_cfg"]),
    train_loader=data_bundle["train_loader"],
    train_eval_loader=data_bundle["train_eval_loader"],
    eval_loaders={
        "paired_val": data_bundle["paired_val_loader"],
        "combined_val": data_bundle["combined_val_loader"],
        "coco_val": data_bundle["coco_val_loader"],
        "imagenet_val": data_bundle["imagenet_val_loader"],
    },
    output_dir=Path(workspace["output_dir"]),
    model_cfg=spec["model_cfg"],
    train_cfg=stage_cfg,
    data_cfg=data_cfg,
    device=device,
    amp_policy=amp_policy,
    model_id=MODEL_ID,
    mix_id=PRETRAIN_MIX,
    stage_name=STAGE_NAME,
    channels_last=channels_last,
    resume=RESUME_TRAINING,
    init_checkpoint_path=Path(INIT_CHECKPOINT) if INIT_CHECKPOINT else None,
)
print(f"Recorded epochs: {len(history)}")

if RUN_DIAGNOSTICS:
    run_diagnostics(
        build_model=spec["build_model"],
        model_cfg=spec["model_cfg"],
        output_dir=Path(workspace["output_dir"]),
        data_bundle=data_bundle,
        device=device,
        prepare_export_model=spec["prepare_export_model"],
    )

if RUN_ONNX_EXPORT:
    export_to_onnx(
        build_model=spec["build_model"],
        model_cfg=spec["model_cfg"],
        checkpoint_path=Path(workspace["output_dir"]) / "best.pt",
        onnx_path=Path(workspace["output_dir"]) / "best.onnx",
        data_cfg=data_cfg,
        device=device,
        prepare_export_model=spec["prepare_export_model"],
        verify=True,
        sample_loader=data_bundle["paired_val_loader"],
    )
"""
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Phase 6A Screening Matrix\n",
                    "\n",
                    "Parameterized screening notebook for Modal execution. Inputs come from environment variables.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [code],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_screening_notebook(path: Path = SCREENING_NOTEBOOK_PATH) -> Path:
    path.write_text(json.dumps(screening_notebook(), indent=2))
    return path


def generate_finalists_from_leaderboard(leaderboard_path: Path, output_dir: Path) -> list[Path]:
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"Leaderboard not found: {leaderboard_path}")
    payload = json.loads(leaderboard_path.read_text())
    promoted = [(row["model_id"], row["mix_id"]) for row in payload["rows"] if str(row.get("promotion_status", "")).startswith("promoted")]
    created = []
    for model_id, mix_id in promoted:
        output_path = output_dir / f"lab2_phase6_{model_id}_full.ipynb"
        created.append(create_portable_notebook(model_id, mix_id, output_path))
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 screening/finalist notebooks.")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    screening_parser = subparsers.add_parser("screening", help="Write the parameterized screening notebook.")
    screening_parser.add_argument("--output", type=Path, default=SCREENING_NOTEBOOK_PATH)

    finalists_parser = subparsers.add_parser("finalists", help="Generate standalone finalist notebooks from leaderboard.json.")
    finalists_parser.add_argument("--leaderboard", type=Path, default=REPO_ROOT / "runs" / "phase6_screening" / "leaderboard.json")
    finalists_parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)

    one_parser = subparsers.add_parser("one", help="Generate one standalone notebook directly.")
    one_parser.add_argument("--model-id", choices=sorted(PORTABLE_MODEL_SOURCES), required=True)
    one_parser.add_argument("--mix-id", choices=MIX_ORDER, required=True)
    one_parser.add_argument("--output", type=Path, required=True)
    one_parser.add_argument("--stage1-epochs", type=int, default=40)
    one_parser.add_argument("--stage2-epochs", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "screening":
        path = write_screening_notebook(args.output)
        print(path)
    elif args.cmd == "finalists":
        created = generate_finalists_from_leaderboard(args.leaderboard, args.output_dir)
        for path in created:
            print(path)
    else:
        path = create_portable_notebook(args.model_id, args.mix_id, args.output, stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs)
        print(path)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
