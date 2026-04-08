#!/usr/bin/env python3
"""
Convert lab ONNX models to MXQ with automatic calibration sampling.

Supports the exported models produced by:
- lab1/lab1.ipynb
- lab 1 trial/lab1_v1.1.ipynb
- lab1 v1.5/lab1_v1_5.ipynb
- lab1 v2/lab1_v2.ipynb
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image

# Path("lab1_v1_5_model.onnx"), Path("lab1_v2_model.onnx"),


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Path configuration: edit these defaults for easy path changes.
DEFAULT_ONNX_RELATIVE_PATHS = (
    Path("lab1v11_model.onnx"),
)
DEFAULT_TRAIN_DIR = "imagenet_train20"
DEFAULT_TRAIN_MANIFEST = "imagenet_train20.txt"
DEFAULT_CALIB_IMAGE_STAGING_DIRNAME = ".calibration_images_auto"
CALIB_DATA_DIR_PREFIX = ".calibration_data_"
DEFAULT_QUANTIZATION_OUTPUT_INDEX = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile lab ONNX models to MXQ with automatic calibration subset sampling."
    )
    parser.add_argument(
        "--onnx-model",
        action="append",
        default=[],
        help="Path to an ONNX model. Can be passed multiple times. Default: auto-discover the 4 lab models.",
    )
    parser.add_argument(
        "--output-mxq",
        default=None,
        help="Output MXQ path (valid only when compiling exactly one ONNX model).",
    )
    parser.add_argument(
        "--train-dir",
        default=DEFAULT_TRAIN_DIR,
        help=f"Training image root directory (default: {DEFAULT_TRAIN_DIR}).",
    )
    parser.add_argument(
        "--train-manifest",
        default=DEFAULT_TRAIN_MANIFEST,
        help=f"Training manifest path (default: {DEFAULT_TRAIN_MANIFEST}).",
    )
    parser.add_argument(
        "--sampling",
        choices=("stratified", "random"),
        default="stratified",
        help="Calibration sampling strategy (default: stratified).",
    )
    parser.add_argument(
        "--calib-ratio",
        type=float,
        default=0.10,
        help="Calibration subset ratio of available training images (default: 0.10).",
    )
    parser.add_argument(
        "--min-calib-images",
        type=int,
        default=128,
        help="Minimum calibration images when available (default: 128).",
    )
    parser.add_argument(
        "--max-calib-images",
        type=int,
        default=512,
        help="Maximum calibration images (default: 512).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42).")
    parser.add_argument(
        "--input-height",
        type=int,
        default=240,
        help="Fallback model input height if ONNX shape is dynamic/unavailable (default: 240).",
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=240,
        help="Fallback model input width if ONNX shape is dynamic/unavailable (default: 240).",
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
    return parser.parse_args()


def build_search_roots(script_dir: Path) -> List[Path]:
    roots: List[Path] = []
    seen = set()
    for base in (Path.cwd(), script_dir):
        for root in (base, *base.parents):
            resolved = root.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            roots.append(resolved)
    return roots


def resolve_required_path(raw_path: str, expect_dir: bool, search_roots: Sequence[Path]) -> Path:
    candidate = Path(raw_path)
    attempted: List[str] = []

    if candidate.is_absolute():
        attempted.append(str(candidate))
        if (expect_dir and candidate.is_dir()) or ((not expect_dir) and candidate.is_file()):
            return candidate.resolve()
    else:
        for root in search_roots:
            resolved_candidate = (root / candidate).resolve()
            attempted.append(str(resolved_candidate))
            if expect_dir and resolved_candidate.is_dir():
                return resolved_candidate
            if (not expect_dir) and resolved_candidate.is_file():
                return resolved_candidate

    expected = "directory" if expect_dir else "file"
    checked = "\n- ".join(attempted)
    raise FileNotFoundError(
        f"Could not find required {expected} for '{raw_path}'. Checked:\n- {checked}"
    )


def resolve_onnx_models(cli_models: Sequence[str], script_dir: Path) -> List[Path]:
    if cli_models:
        models = []
        for raw in cli_models:
            model_path = Path(raw)
            if not model_path.exists():
                raise FileNotFoundError(f"ONNX model not found: {raw}")
            models.append(model_path.resolve())
        return models

    discovered: List[Path] = []
    for rel_path in DEFAULT_ONNX_RELATIVE_PATHS:
        candidate = (script_dir / rel_path).resolve()
        if candidate.is_file():
            discovered.append(candidate)

    if not discovered:
        names = ", ".join(str(p) for p in DEFAULT_ONNX_RELATIVE_PATHS)
        raise FileNotFoundError(
            "Could not auto-discover lab ONNX models. Expected one or more of: "
            f"{names}"
        )
    return discovered


def load_training_image_index(
    train_manifest: Path, train_dir: Path
) -> Tuple[Dict[int, List[Path]], int, int]:
    by_class: Dict[int, List[Path]] = defaultdict(list)
    valid_count = 0
    missing_count = 0

    with train_manifest.open("r", encoding="utf-8") as manifest_file:
        for line_num, line in enumerate(manifest_file, start=1):
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            filename = parts[0]
            try:
                label = int(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid label at {train_manifest}:{line_num}: '{parts[1]}'"
                ) from exc

            wnid = filename.split("_", 1)[0]
            img_path = (train_dir / wnid / filename).resolve()
            if img_path.is_file():
                by_class[label].append(img_path)
                valid_count += 1
            else:
                missing_count += 1

    if valid_count == 0:
        raise RuntimeError(
            "No training samples found from the manifest.\n"
            f"Manifest: {train_manifest}\n"
            f"Train dir: {train_dir}"
        )

    return by_class, valid_count, missing_count


def select_calibration_subset(
    by_class: Dict[int, List[Path]],
    total_available: int,
    sampling: str,
    ratio: float,
    min_images: int,
    max_images: int,
    seed: int,
) -> List[Path]:
    if ratio <= 0:
        raise ValueError(f"--calib-ratio must be > 0. Got {ratio}.")
    if min_images <= 0 or max_images <= 0:
        raise ValueError("--min-calib-images and --max-calib-images must be > 0.")
    if min_images > max_images:
        raise ValueError("--min-calib-images cannot be greater than --max-calib-images.")

    ratio_target = int(total_available * ratio)
    target = max(min_images, ratio_target)
    target = min(target, max_images, total_available)
    target = max(1, target)

    rng = random.Random(seed)
    flattened = [img for _, imgs in sorted(by_class.items()) for img in imgs]
    if sampling == "random":
        rng.shuffle(flattened)
        return flattened[:target]

    class_order = sorted(by_class.keys())
    rng.shuffle(class_order)
    pools: Dict[int, List[Path]] = {}
    for label in class_order:
        paths = list(by_class[label])
        rng.shuffle(paths)
        pools[label] = paths

    selected: List[Path] = []
    if target <= len(class_order):
        for label in class_order[:target]:
            selected.append(pools[label].pop())
        return selected

    for label in class_order:
        if pools[label]:
            selected.append(pools[label].pop())

    while len(selected) < target:
        any_added = False
        for label in class_order:
            if len(selected) >= target:
                break
            if pools[label]:
                selected.append(pools[label].pop())
                any_added = True
        if not any_added:
            break

    return selected


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
        suffix = src.suffix if src.suffix else ".jpg"
        dst = stage_dir / f"{idx:06d}{suffix.lower()}"
        link_or_copy(src, dst)


def sanitize_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value).strip("_").lower()


def calibration_data_path_for_model(onnx_model: Path, temp_root: Path) -> Path:
    try:
        model_key = str(onnx_model.relative_to(temp_root.parent))
    except ValueError:
        model_key = onnx_model.name
    model_id = sanitize_id(model_key)
    return temp_root / f"{CALIB_DATA_DIR_PREFIX}{model_id}"


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
            resized = img.convert("RGB").resize((width, height), Image.BILINEAR)
        img_array = np.array(resized, dtype=np.float32) / 255.0
        return ((img_array - MEAN) / STD).astype(np.float32)

    return _preprocess


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
            f"Unsupported --quantize-method '{quantize_method}'. "
            f"Valid values map to: {valid}"
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

    attempts = [
        ("new", new_style_kwargs),
        ("old", old_style_kwargs),
    ]

    try:
        params = inspect.signature(mxq_compile).parameters
        if "quantization_mode" in params:
            attempts = [("new", new_style_kwargs), ("old", old_style_kwargs)]
        elif "quantize_method" in params:
            attempts = [("old", old_style_kwargs), ("new", new_style_kwargs)]
    except (TypeError, ValueError):
        # Some compiled callables may not expose a usable signature.
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
    calib_data_name = calib_data_path.name

    if calib_data_path.exists():
        shutil.rmtree(calib_data_path)

    h, w = infer_onnx_hw(onnx_model, fallback=fallback_hw)
    preprocess_fn = build_preprocess(h, w)

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


def compute_histogram(sample_paths: Sequence[Path], by_class: Dict[int, List[Path]]) -> Dict[int, int]:
    path_to_class = {}
    for label, paths in by_class.items():
        for path in paths:
            path_to_class[path] = label

    hist: Dict[int, int] = defaultdict(int)
    for path in sample_paths:
        hist[path_to_class[path]] += 1
    return dict(sorted(hist.items()))


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    search_roots = build_search_roots(script_dir)

    onnx_models = resolve_onnx_models(args.onnx_model, script_dir)
    if args.output_mxq and len(onnx_models) != 1:
        raise ValueError("--output-mxq can only be used when compiling exactly one ONNX model.")

    train_dir = resolve_required_path(args.train_dir, expect_dir=True, search_roots=search_roots)
    train_manifest = resolve_required_path(
        args.train_manifest, expect_dir=False, search_roots=search_roots
    )

    by_class, valid_count, missing_count = load_training_image_index(train_manifest, train_dir)
    sample_paths = select_calibration_subset(
        by_class=by_class,
        total_available=valid_count,
        sampling=args.sampling,
        ratio=args.calib_ratio,
        min_images=args.min_calib_images,
        max_images=args.max_calib_images,
        seed=args.seed,
    )

    temp_root = script_dir
    calib_image_dir = temp_root / DEFAULT_CALIB_IMAGE_STAGING_DIRNAME
    stage_calibration_images(sample_paths, calib_image_dir)
    hist = compute_histogram(sample_paths, by_class)

    print(f"Training dir: {train_dir}")
    print(f"Training manifest: {train_manifest}")
    print(f"Training images available: {valid_count} (missing entries skipped: {missing_count})")
    print(
        f"Calibration subset: {len(sample_paths)} images "
        f"(sampling={args.sampling}, ratio={args.calib_ratio}, min={args.min_calib_images}, max={args.max_calib_images})"
    )
    print(f"Calibration class histogram: {hist}")
    print("ONNX models to compile:")
    for model_path in onnx_models:
        print(f"  - {model_path}")

    generated_calib_paths: List[Path] = []
    fallback_hw = (args.input_height, args.input_width)

    try:
        for onnx_model in onnx_models:
            if args.output_mxq:
                output_mxq = Path(args.output_mxq).resolve()
            else:
                output_mxq = onnx_model.with_suffix(".mxq")

            expected_calib_data_path = calibration_data_path_for_model(onnx_model, temp_root)
            if expected_calib_data_path not in generated_calib_paths:
                generated_calib_paths.append(expected_calib_data_path)

            calib_data_path = compile_model(
                onnx_model=onnx_model,
                output_mxq=output_mxq,
                calib_image_dir=calib_image_dir,
                temp_root=temp_root,
                fallback_hw=fallback_hw,
                quantize_method=args.quantize_method,
                percentile=args.percentile,
                topk_ratio=args.topk_ratio,
            )
            if calib_data_path not in generated_calib_paths:
                generated_calib_paths.append(calib_data_path)

            size_mb = output_mxq.stat().st_size / (1024 * 1024)
            print(f"Compiled: {output_mxq} ({size_mb:.1f} MB)")
    except ModuleNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    finally:
        if not args.keep_temp:
            if calib_image_dir.exists():
                shutil.rmtree(calib_image_dir)
            for calib_data_path in generated_calib_paths:
                if calib_data_path.exists():
                    shutil.rmtree(calib_data_path)


if __name__ == "__main__":
    main()
