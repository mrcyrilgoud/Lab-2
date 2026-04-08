#!/usr/bin/env python3
"""
Compile Phase 7B ONNX exports to MXQ using Qubee calibration.

Expected notebook export layout:
  <run_root>/exports/best.onnx
  <run_root>/exports/calibration/
  <run_root>/exports/calibration/manifest.json

The Phase 7B super-resolution model expects LR image tensors equivalent to
torchvision.transforms.ToTensor(), so calibration images are converted to RGB
float32 arrays in [0, 1] without ImageNet normalization.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Sequence, Tuple

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ONNX_NAME = "best.onnx"
DEFAULT_CALIBRATION_DIRNAME = "calibration"
DEFAULT_CALIB_IMAGE_STAGING_DIRNAME = ".phase7b_calibration_images_auto"
CALIB_DATA_DIR_PREFIX = ".phase7b_calibration_data_"
DEFAULT_QUANTIZATION_OUTPUT_INDEX = 1

BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile Lab 2 Phase 7B ONNX exports into MXQ with SR calibration images."
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        help="Export directory containing best.onnx and calibration/, or a run root containing exports/.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="Run directory containing exports/, or an exports directory directly.",
    )
    parser.add_argument("--onnx-model", type=Path, help="Explicit ONNX model path.")
    parser.add_argument("--calibration-dir", type=Path, help="Explicit calibration image directory.")
    parser.add_argument(
        "--output-mxq",
        type=Path,
        help="Output MXQ path. Default: ONNX path with .mxq suffix.",
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=256,
        help="Fallback model input height if ONNX shape is dynamic/unavailable (default: 256).",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=256,
        help="Fallback model input width if ONNX shape is dynamic/unavailable (default: 256).",
    )
    parser.add_argument(
        "--quantize-method",
        default="maxpercentile",
        help="Qubee quantize method (default: maxpercentile).",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.999,
        help="Quantization percentile (default: 0.999).",
    )
    parser.add_argument(
        "--topk-ratio",
        type=float,
        default=0.01,
        help="Top-k ratio for maxpercentile (default: 0.01).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated calibration image/data directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and preprocessing settings without invoking Qubee.",
    )
    return parser.parse_args()


def normalize_export_dir(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved / DEFAULT_ONNX_NAME).is_file() or (
        resolved / DEFAULT_CALIBRATION_DIRNAME
    ).is_dir():
        return resolved
    if (resolved / "exports").is_dir():
        return (resolved / "exports").resolve()
    return resolved


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    export_dir = None
    if args.export_dir is not None:
        export_dir = normalize_export_dir(args.export_dir)
    elif args.run_dir is not None:
        export_dir = normalize_export_dir(args.run_dir)

    onnx_model = args.onnx_model.expanduser().resolve() if args.onnx_model else None
    calibration_dir = (
        args.calibration_dir.expanduser().resolve() if args.calibration_dir else None
    )

    if export_dir is not None:
        if onnx_model is None:
            onnx_model = (export_dir / DEFAULT_ONNX_NAME).resolve()
        if calibration_dir is None:
            calibration_dir = (export_dir / DEFAULT_CALIBRATION_DIRNAME).resolve()

    if onnx_model is None or calibration_dir is None:
        raise ValueError(
            "Provide --export-dir/--run-dir, or provide both --onnx-model and --calibration-dir."
        )

    output_mxq = (
        args.output_mxq.expanduser().resolve()
        if args.output_mxq
        else onnx_model.with_suffix(".mxq")
    )

    return {
        "export_dir": export_dir if export_dir is not None else onnx_model.parent,
        "onnx_model": onnx_model,
        "calibration_dir": calibration_dir,
        "output_mxq": output_mxq,
    }


def load_manifest(calibration_dir: Path) -> dict[str, Any] | None:
    manifest_path = calibration_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Calibration manifest must be a JSON object: {manifest_path}")
    return manifest


def collect_calibration_images(calibration_dir: Path) -> list[Path]:
    return sorted(
        path.resolve()
        for path in calibration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def validate_inputs(paths: dict[str, Path]) -> dict[str, Any]:
    onnx_model = paths["onnx_model"]
    calibration_dir = paths["calibration_dir"]

    if not onnx_model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model}")
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"Calibration directory not found: {calibration_dir}")

    manifest = load_manifest(calibration_dir)
    calibration_images = collect_calibration_images(calibration_dir)
    if not calibration_images:
        raise RuntimeError(f"No calibration images found under: {calibration_dir}")

    manifest_samples = None
    manifest_summary = None
    if manifest is not None:
        samples = manifest.get("samples")
        manifest_samples = len(samples) if isinstance(samples, list) else None
        manifest_summary = manifest.get("summary")

    return {
        "manifest": manifest,
        "manifest_samples": manifest_samples,
        "manifest_summary": manifest_summary,
        "calibration_images": calibration_images,
    }


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def calibration_data_path_for_model(onnx_model: Path, temp_root: Path) -> Path:
    try:
        model_key = str(onnx_model.relative_to(temp_root.parent))
    except ValueError:
        model_key = onnx_model.name
    return temp_root / f"{CALIB_DATA_DIR_PREFIX}{sanitize_id(model_key)}"


def infer_onnx_hw(onnx_path: Path, fallback: Tuple[int, int]) -> Tuple[int, int]:
    try:
        import onnx  # type: ignore
    except Exception:
        return fallback

    try:
        model = onnx.load(str(onnx_path))
        first_input = model.graph.input[0]
        dims = first_input.type.tensor_type.shape.dim
        if len(dims) >= 4:
            h = int(dims[2].dim_value or 0)
            w = int(dims[3].dim_value or 0)
            if h > 0 and w > 0:
                return h, w
    except Exception as exc:
        print(f"Warning: failed to infer input shape from {onnx_path.name}: {exc}")
    return fallback


def build_preprocess(height: int, width: int):
    def _preprocess(image_path: str):
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), BILINEAR)
            return (np.asarray(rgb, dtype=np.float32) / 255.0).astype(np.float32)

    return _preprocess


def link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    try:
        os.link(src, dst)
        return
    except OSError:
        shutil.copy2(src, dst)


def stage_calibration_images(sample_paths: Sequence[Path], stage_dir: Path) -> None:
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(sample_paths):
        suffix = src.suffix.lower() if src.suffix else ".png"
        link_or_copy(src, stage_dir / f"{idx:06d}{suffix}")


def quantization_mode_index(quantize_method: str) -> int:
    normalized = re.sub(r"[^a-z0-9]+", "", quantize_method.lower())
    mode_map = {
        "percentile": 0,
        "max": 1,
        "maxpercentile": 2,
        "fastpercentile": 3,
        "histogramkl": 4,
        "histogrammse": 5,
    }
    if normalized not in mode_map:
        valid = ", ".join(sorted(mode_map))
        raise ValueError(
            f"Unsupported --quantize-method '{quantize_method}'. Valid values map to: {valid}"
        )
    return mode_map[normalized]


def call_mxq_compile_compatible(
    mxq_compile,
    onnx_model: Path,
    calib_data_path: Path,
    output_mxq: Path,
    quantize_method: str,
    percentile: float,
    topk_ratio: float,
) -> None:
    base_kwargs = {
        "model": str(onnx_model),
        "calib_data_path": str(calib_data_path),
        "topk_ratio": topk_ratio,
        "save_path": str(output_mxq),
        "backend": "onnx",
    }

    new_style_kwargs = {
        **base_kwargs,
        "quantization_mode": quantization_mode_index(quantize_method),
        "quantization_output": DEFAULT_QUANTIZATION_OUTPUT_INDEX,
        "percentile": percentile,
    }
    old_style_kwargs = {
        **base_kwargs,
        "quantize_method": quantize_method,
        "is_quant_ch": True,
        "quantize_percentile": percentile,
        "quant_output": "layer",
    }

    attempts = [("new", new_style_kwargs), ("old", old_style_kwargs)]
    try:
        params = inspect.signature(mxq_compile).parameters
        if "quantization_mode" in params:
            attempts = [("new", new_style_kwargs), ("old", old_style_kwargs)]
        elif "quantize_method" in params:
            attempts = [("old", old_style_kwargs), ("new", new_style_kwargs)]
    except (TypeError, ValueError):
        pass

    last_error = None
    for _, kwargs in attempts:
        try:
            mxq_compile(**kwargs)
            return
        except (TypeError, ValueError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise RuntimeError("mxq_compile failed without an explicit error.")


def compile_model(
    onnx_model: Path,
    output_mxq: Path,
    calib_image_dir: Path,
    temp_root: Path,
    fallback_hw: Tuple[int, int],
    quantize_method: str,
    percentile: float,
    topk_ratio: float,
) -> Path:
    try:
        from qubee import mxq_compile
        from qubee.calibration import make_calib_man
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: 'qubee'. Install it in this environment before running MXQ compile."
        ) from exc

    calib_data_path = calibration_data_path_for_model(onnx_model, temp_root)
    if calib_data_path.exists():
        shutil.rmtree(calib_data_path)

    h, w = infer_onnx_hw(onnx_model, fallback=fallback_hw)
    preprocess_fn = build_preprocess(h, w)
    calib_data_name = calib_data_path.name

    make_calib_man(
        pre_ftn=preprocess_fn,
        data_dir=str(calib_image_dir),
        save_dir=str(temp_root),
        save_name=calib_data_name,
        max_size=len(os.listdir(calib_image_dir)),
    )

    output_mxq.parent.mkdir(parents=True, exist_ok=True)
    call_mxq_compile_compatible(
        mxq_compile=mxq_compile,
        onnx_model=onnx_model,
        calib_data_path=calib_data_path,
        output_mxq=output_mxq,
        quantize_method=quantize_method,
        percentile=percentile,
        topk_ratio=topk_ratio,
    )
    return calib_data_path


def print_validation_summary(
    paths: dict[str, Path],
    validation: dict[str, Any],
    input_hw: Tuple[int, int],
) -> None:
    print(f"Export dir: {paths['export_dir']}")
    print(f"ONNX model: {paths['onnx_model']}")
    print(f"Calibration dir: {paths['calibration_dir']}")
    print(f"Calibration images: {len(validation['calibration_images'])}")
    if validation["manifest_samples"] is not None:
        print(f"Manifest samples: {validation['manifest_samples']}")
    if validation["manifest_summary"] is not None:
        print(f"Manifest summary: {json.dumps(validation['manifest_summary'], default=str)}")
    print(f"Preprocess: RGB float32 [0, 1], HWC, resize to {input_hw[0]}x{input_hw[1]}")
    print(f"Output MXQ: {paths['output_mxq']}")


def main() -> int:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    temp_root = script_dir

    paths = resolve_paths(args)
    validation = validate_inputs(paths)
    fallback_hw = (args.input_height, args.input_width)
    input_hw = infer_onnx_hw(paths["onnx_model"], fallback=fallback_hw)
    print_validation_summary(paths, validation, input_hw)

    if args.dry_run:
        print("Dry run validated inputs. Qubee was not invoked.")
        return 0

    staged_calib_dir = temp_root / DEFAULT_CALIB_IMAGE_STAGING_DIRNAME
    expected_calib_data_path = calibration_data_path_for_model(paths["onnx_model"], temp_root)
    generated_calib_paths = [expected_calib_data_path]

    try:
        stage_calibration_images(validation["calibration_images"], staged_calib_dir)
        calib_data_path = compile_model(
            onnx_model=paths["onnx_model"],
            output_mxq=paths["output_mxq"],
            calib_image_dir=staged_calib_dir,
            temp_root=temp_root,
            fallback_hw=fallback_hw,
            quantize_method=args.quantize_method,
            percentile=args.percentile,
            topk_ratio=args.topk_ratio,
        )
        if calib_data_path not in generated_calib_paths:
            generated_calib_paths.append(calib_data_path)
        size_mb = paths["output_mxq"].stat().st_size / (1024 * 1024)
        print(f"Compiled: {paths['output_mxq']} ({size_mb:.1f} MB)")
        return 0
    finally:
        if not args.keep_temp:
            if staged_calib_dir.exists():
                shutil.rmtree(staged_calib_dir)
            for calib_data_path in generated_calib_paths:
                if calib_data_path.exists():
                    shutil.rmtree(calib_data_path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError, ModuleNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
