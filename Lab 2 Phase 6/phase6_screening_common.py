from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable
import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import tarfile
import time
import urllib.request
import warnings
import zipfile

warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parent
DATA_ROOT = REPO_ROOT / "Data"
RUNS_ROOT = REPO_ROOT / "runs"

TO_TENSOR = transforms.ToTensor()
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
FORBIDDEN_TYPES = (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.LayerNorm, nn.GroupNorm)
INTERPOLATION_BANK = {
    "bicubic": BICUBIC,
    "bilinear": Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR,
    "lanczos": LANCZOS,
}

COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}
MODEL_ORDER = [
    "wide_se",
    "dsdan",
    "repconv",
    "large_kernel_dw",
    "large_kernel_se",
    "hybrid_rep_large_kernel",
]
MIX_ORDER = ["coco_only", "coco_plus_imagenet"]
SCREENING_STAGE_ORDER = ["stage1_pretrain", "stage2_finetune"]
PROMOTION_COUNT = 3
DEFAULT_TIE_MARGIN = 0.05
STATE_VERSION = 1


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw is not None else default


def slugify_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part)[:80] or "sample"


def default_screening_data_cfg() -> dict[str, Any]:
    return {
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


def default_fullrun_data_cfg() -> dict[str, Any]:
    cfg = default_screening_data_cfg()
    cfg.update(
        {
            "coco_train_limit": 30000,
            "coco_val_limit": 1000,
            "train_eval_subset_size": 256,
        }
    )
    return cfg


def default_stage_cfg(stage_name: str, batch_size: int, epochs: int, seed: int) -> dict[str, Any]:
    base = {
        "stage_name": stage_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 3e-4 if stage_name == "stage1_pretrain" else 1.5e-4,
        "weight_decay": 2e-4,
        "warmup_epochs": 2 if stage_name == "stage1_pretrain" else 1,
        "min_lr_ratio": 0.05,
        "checkpoint_interval": 4,
        "train_eval_interval": 2,
        "seed": seed,
        "early_stop_patience": 6 if stage_name == "stage1_pretrain" else 5,
        "grad_clip_norm": 1.0,
        "ema_decay": 0.999,
        "charb_eps": 1e-6,
        "selection_metric": "paired_val_psnr",
    }
    if stage_name == "stage2_finetune":
        base["checkpoint_interval"] = 2
        base["train_eval_interval"] = 1
    return base


def screening_recipe(batch_size: int = 4, seed: int = 255) -> dict[str, Any]:
    return {
        "data_cfg": default_screening_data_cfg(),
        "stage_cfgs": {
            "stage1_pretrain": default_stage_cfg("stage1_pretrain", batch_size, epochs=12, seed=seed),
            "stage2_finetune": default_stage_cfg("stage2_finetune", batch_size, epochs=8, seed=seed),
        },
    }


def fullrun_recipe(batch_size: int = 4, seed: int = 255) -> dict[str, Any]:
    return {
        "data_cfg": default_fullrun_data_cfg(),
        "stage_cfgs": {
            "stage1_pretrain": default_stage_cfg("stage1_pretrain", batch_size, epochs=40, seed=seed),
            "stage2_finetune": default_stage_cfg("stage2_finetune", batch_size, epochs=20, seed=seed),
        },
    }


def configure_runtime() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    return device


def choose_amp_policy(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"enabled": False, "dtype": None, "use_scaler": False, "label": "fp32"}
    bf16_ok = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
    if bf16_ok:
        return {"enabled": True, "dtype": torch.bfloat16, "use_scaler": False, "label": "bf16"}
    return {"enabled": True, "dtype": torch.float16, "use_scaler": True, "label": "fp16"}


def make_grad_scaler(amp_policy: dict[str, Any]):
    if not amp_policy["enabled"] or not amp_policy["use_scaler"]:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def autocast_context(device: torch.device, amp_policy: dict[str, Any]):
    if device.type != "cuda" or not amp_policy["enabled"]:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=amp_policy["dtype"])
    return torch.cuda.amp.autocast(dtype=amp_policy["dtype"])


def optimizer_with_fallback(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    kwargs = {"lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]}
    if torch.cuda.is_available():
        try:
            return AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return AdamW(model.parameters(), **kwargs)


def resolve_phase6_workspace(output_subdir: str, data_root: str | Path | None = None) -> dict[str, Path]:
    data_override = os.environ.get("LAB2_DATA_ROOT")
    output_override = os.environ.get("LAB2_OUTPUT_DIR")
    resolved_data = Path(data_override) if data_override else (Path(data_root) if data_root else DATA_ROOT)
    resolved_output = Path(output_override) if output_override else RUNS_ROOT / output_subdir
    resolved_output.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": REPO_ROOT,
        "data_root": resolved_data,
        "output_dir": resolved_output,
        "workspace_root": REPO_ROOT,
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_npu_compatible(model: nn.Module) -> None:
    for name, mod in model.named_modules():
        if isinstance(mod, FORBIDDEN_TYPES):
            raise TypeError(f"Forbidden NPU op '{name}': {mod.__class__.__name__}")


def summarize_npu_ops(model: nn.Module) -> dict[str, int]:
    ops: dict[str, int] = defaultdict(int)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            key = "DWConv" if module.groups == module.in_channels and module.in_channels > 1 else "Conv"
            ops[key] += 1
        elif isinstance(module, nn.BatchNorm2d):
            ops["BN"] += 1
        elif isinstance(module, nn.PReLU):
            ops["PReLU"] += 1
        elif isinstance(module, nn.Hardsigmoid):
            ops["HardSigmoid"] += 1
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            ops["GlobalAvgPool"] += 1
    return dict(ops)


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def ensure_unzipped(zip_path: Path, extracted_dir: Path) -> Path:
    if extracted_dir.exists():
        return extracted_dir
    if not zip_path.exists():
        return extracted_dir
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    return extracted_dir


def ensure_tar_extracted(archive_path: Path, dest_root: Path) -> None:
    if not archive_path.exists():
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(dest_root)


def download_url(url: str, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return dest_path
    with urllib.request.urlopen(url) as response, dest_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    return dest_path


def coco_root(data_root: Path) -> Path:
    return data_root / "course_files_export" / "coco2017"


def build_image_manifest(image_dir: Path, manifest_path: Path) -> Path:
    images = sorted(
        [
            str(path.relative_to(image_dir.parent))
            for path in image_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(images) + ("\n" if images else ""))
    return manifest_path


def stage_coco2017(data_root: Path, download_missing: bool = True) -> dict[str, str]:
    root = coco_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    info: dict[str, str] = {}
    for split in ("train2017", "val2017"):
        zip_path = root / f"{split}.zip"
        image_dir = root / split
        manifest = root / f"coco_{split}.txt"
        if download_missing and not zip_path.exists():
            download_url(COCO_URLS[split], zip_path)
            info[f"{split}_downloaded"] = str(zip_path)
        ensure_unzipped(zip_path, image_dir)
        build_image_manifest(image_dir, manifest)
        info[f"{split}_dir"] = str(image_dir)
        info[f"{split}_manifest"] = str(manifest)
    return info


def read_manifest_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_imagenet_manifest(path: Path) -> list[tuple[str, int]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append((parts[0], int(parts[1])))
    return rows


def collect_imagenet_records(rows: list[tuple[str, int]], root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for filename, class_id in rows:
        synset = filename.split("_")[0]
        path = (root / synset / filename) if split == "train" else (root / filename)
        if not path.exists():
            continue
        records.append(
            {
                "path": path,
                "stem": path.stem,
                "class_id": class_id,
                "split": split,
                "source_name": f"imagenet_{split}",
            }
        )
    return records


def collect_coco_records(manifest_lines: list[str], root: Path, split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in manifest_lines:
        path = root / rel_path
        if not path.exists():
            continue
        records.append(
            {
                "path": path,
                "stem": path.stem,
                "class_id": -1,
                "split": split,
                "source_name": f"coco_{split}",
            }
        )
    return records


def take_manifest_subset(records: list[Any], limit: int | None, seed: int) -> list[Any]:
    if limit is None or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    return items[:limit]


def seeded_rng(key: str) -> random.Random:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def collect_paired_by_subfolder(lr_root: Path, hr_root: Path) -> list[tuple[Path, Path, str]]:
    pairs: list[tuple[Path, Path, str]] = []
    for hr_dir in sorted(p for p in hr_root.iterdir() if p.is_dir()):
        suffix = hr_dir.name.replace("HR_train", "")
        lr_dir = lr_root / f"LR_train{suffix}"
        if not lr_dir.exists():
            continue
        hr_imgs = {p.stem: p for p in sorted(hr_dir.glob("*.png"))}
        lr_imgs = {p.stem: p for p in sorted(lr_dir.glob("*.png"))}
        common = sorted(set(hr_imgs) & set(lr_imgs))
        pairs.extend((lr_imgs[s], hr_imgs[s], f"{hr_dir.name}/{s}") for s in common)
    return pairs


def collect_paired_flat(lr_dir: Path, hr_dir: Path) -> list[tuple[Path, Path, str]]:
    hr_imgs = {p.stem: p for p in sorted(hr_dir.glob("*.png"))}
    lr_imgs = {p.stem: p for p in sorted(lr_dir.glob("*.png"))}
    common = sorted(set(hr_imgs) & set(lr_imgs))
    return [(lr_imgs[s], hr_imgs[s], s) for s in common]


def collect_train_pairs(data_root: Path) -> list[tuple[Path, Path, str]]:
    structured_lr_root = data_root / "LR_train"
    structured_hr_root = data_root / "HR_train"
    if structured_lr_root.exists() and structured_hr_root.exists():
        pairs = collect_paired_by_subfolder(structured_lr_root, structured_hr_root)
        if pairs:
            return pairs
    flat_lr_root = data_root / "train" / "LR"
    flat_hr_root = data_root / "train" / "HR"
    if flat_lr_root.exists() and flat_hr_root.exists():
        return collect_paired_flat(flat_lr_root, flat_hr_root)
    return []


def collect_val_pairs(data_root: Path) -> list[tuple[Path, Path, str]]:
    candidates = [
        (data_root / "LR_val", data_root / "HR_val"),
        (data_root / "val" / "LR_val", data_root / "val" / "HR_val"),
        (data_root / "val" / "LR", data_root / "val" / "HR"),
    ]
    for lr_dir, hr_dir in candidates:
        if lr_dir.exists() and hr_dir.exists():
            pairs = collect_paired_flat(lr_dir, hr_dir)
            if pairs:
                return pairs
    return []


def random_crop_pair(lr_img: Image.Image, hr_img: Image.Image, size: int, rng: random.Random):
    if lr_img.width == size and lr_img.height == size:
        return lr_img, hr_img
    x0 = rng.randint(0, lr_img.width - size)
    y0 = rng.randint(0, lr_img.height - size)
    box = (x0, y0, x0 + size, y0 + size)
    return lr_img.crop(box), hr_img.crop(box)


def random_crop_single(img: Image.Image, size: int, rng: random.Random):
    if img.width == size and img.height == size:
        return img
    x0 = rng.randint(0, img.width - size)
    y0 = rng.randint(0, img.height - size)
    return img.crop((x0, y0, x0 + size, y0 + size))


def augment_pair(lr_img: Image.Image, hr_img: Image.Image, rng: random.Random):
    if rng.random() > 0.5:
        lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
        hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randint(0, 3)
    if k > 0:
        rot = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}[k]
        lr_img = lr_img.transpose(rot)
        hr_img = hr_img.transpose(rot)
    return lr_img, hr_img


def augment_single(img: Image.Image, rng: random.Random):
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randint(0, 3)
    if k > 0:
        img = img.transpose({1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}[k])
    return img


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_from_hr(hr_img: Image.Image, rng: random.Random, cfg: dict[str, Any]) -> Image.Image:
    lr_img = hr_img.copy()
    if rng.random() < cfg["lr_blur_prob"]:
        radius = rng.uniform(cfg["blur_radius_min"], cfg["blur_radius_max"])
        lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=radius))
    scale = rng.choice(cfg["downsample_scales"])
    resize_name = rng.choice(cfg["resize_modes"])
    resize_mode = INTERPOLATION_BANK[resize_name]
    small = (max(1, hr_img.width // scale), max(1, hr_img.height // scale))
    lr_img = lr_img.resize(small, resample=resize_mode).resize(hr_img.size, resample=resize_mode)
    if rng.random() < cfg["jpeg_prob"]:
        lr_img = jpeg_roundtrip(lr_img, rng.randint(cfg["jpeg_quality_min"], cfg["jpeg_quality_max"]))
    return lr_img


def apply_tensor_regularization(lr_t: torch.Tensor, rng: random.Random, cfg: dict[str, Any], train: bool) -> torch.Tensor:
    if not train:
        return lr_t
    if cfg["lr_noise_prob"] > 0 and rng.random() < cfg["lr_noise_prob"]:
        lr_t = (lr_t + torch.randn_like(lr_t) * cfg["lr_noise_std"]).clamp(0.0, 1.0)
    if cfg["cutout_prob"] > 0 and rng.random() < cfg["cutout_prob"]:
        _, h, w = lr_t.shape
        cut = max(8, int(min(h, w) * cfg["cutout_ratio"]))
        x0 = rng.randint(0, w - cut)
        y0 = rng.randint(0, h - cut)
        fill = lr_t.mean().item()
        lr_t[:, y0 : y0 + cut, x0 : x0 + cut] = fill
    return lr_t


class PairedSRDataset(Dataset):
    def __init__(self, pairs, train: bool, data_cfg: dict[str, Any], source_name: str, seed: int):
        self.pairs = pairs
        self.train = train
        self.data_cfg = data_cfg
        self.source_name = source_name
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, stem = self.pairs[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        rng = random.Random(self.seed + idx) if self.train else seeded_rng(f"{self.source_name}:{stem}")
        if self.train:
            lr_img, hr_img = random_crop_pair(lr_img, hr_img, self.data_cfg["train_patch_size"], rng)
            lr_img, hr_img = augment_pair(lr_img, hr_img, rng)
        else:
            lr_img = ImageOps.fit(lr_img, (self.data_cfg["eval_size"], self.data_cfg["eval_size"]), method=BICUBIC)
            hr_img = ImageOps.fit(hr_img, (self.data_cfg["eval_size"], self.data_cfg["eval_size"]), method=BICUBIC)
        lr_t = TO_TENSOR(lr_img)
        hr_t = TO_TENSOR(hr_img)
        lr_t = apply_tensor_regularization(lr_t, rng, self.data_cfg, train=self.train)
        return lr_t, hr_t, stem, self.source_name


class NaturalImageSyntheticSRDataset(Dataset):
    def __init__(self, records, train: bool, data_cfg: dict[str, Any], seed: int):
        self.records = records
        self.train = train
        self.data_cfg = data_cfg
        self.seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        hr_img = Image.open(record["path"]).convert("RGB")
        source_name = record["source_name"]
        rng = random.Random(self.seed + idx) if self.train else seeded_rng(f"{source_name}:{record['stem']}")
        if self.train:
            base_size = max(self.data_cfg["eval_size"], self.data_cfg["train_patch_size"] + self.data_cfg["random_scale_pad"])
            hr_img = ImageOps.fit(hr_img, (base_size, base_size), method=BICUBIC)
            hr_img = random_crop_single(hr_img, self.data_cfg["train_patch_size"], rng)
            hr_img = augment_single(hr_img, rng)
        else:
            hr_img = ImageOps.fit(hr_img, (self.data_cfg["eval_size"], self.data_cfg["eval_size"]), method=BICUBIC)
        lr_img = degrade_from_hr(hr_img, rng, self.data_cfg)
        lr_t = TO_TENSOR(lr_img)
        hr_t = TO_TENSOR(hr_img)
        lr_t = apply_tensor_regularization(lr_t, rng, self.data_cfg, train=self.train)
        return lr_t, hr_t, record["stem"], source_name


def loader_kwargs(num_workers: int, pin_memory: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def make_fixed_subset_loader(dataset, subset_size: int, batch_size: int, seed: int, num_workers: int, pin_memory: bool):
    subset_size = min(subset_size, len(dataset))
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:subset_size])
    return DataLoader(subset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))


def paired_finetune_data_cfg(base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["cutout_prob"] = 0.0
    cfg["lr_noise_prob"] = 0.0
    return cfg


def build_phase6_data_bundle(
    workspace: dict[str, Path],
    data_cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    pretrain_mix: str,
    stage_name: str,
) -> dict[str, Any]:
    if pretrain_mix not in MIX_ORDER:
        raise ValueError(f"Unsupported pretrain_mix: {pretrain_mix}")

    data_root = Path(workspace["data_root"])
    course_export_root = data_root / "course_files_export"
    legacy_image_root = data_root / "ImageNet"
    coco_stage_info = stage_coco2017(data_root, download_missing=False)

    hr_train_root = data_root / "HR_train"
    lr_train_root = data_root / "LR_train"
    hr_val_dir = first_existing(data_root / "HR_val", data_root / "val" / "HR_val")
    lr_val_dir = first_existing(data_root / "LR_val", data_root / "val" / "LR_val")

    imagenet_train_list = first_existing(course_export_root / "imagenet_train20.txt", legacy_image_root / "imagenet_train20.txt")
    imagenet_val_list = first_existing(course_export_root / "imagenet_val20.txt", legacy_image_root / "imagenet_val20.txt")
    imagenet_train_root = ensure_unzipped(
        course_export_root / "imagenet_train20.zip",
        first_existing(course_export_root / "imagenet_train20a", legacy_image_root / "imagenet_train20a"),
    )
    imagenet_val_root = ensure_unzipped(
        course_export_root / "imagenet_val20.zip",
        first_existing(course_export_root / "imagenet_val20", legacy_image_root / "imagenet_val20"),
    )
    coco_train_root = Path(coco_stage_info.get("train2017_dir", str(coco_root(data_root) / "train2017")))
    coco_val_root = Path(coco_stage_info.get("val2017_dir", str(coco_root(data_root) / "val2017")))
    coco_train_manifest = Path(coco_stage_info.get("train2017_manifest", str(coco_root(data_root) / "coco_train2017.txt")))
    coco_val_manifest = Path(coco_stage_info.get("val2017_manifest", str(coco_root(data_root) / "coco_val2017.txt")))

    required_paths = [
        hr_train_root,
        lr_train_root,
        hr_val_dir,
        lr_val_dir,
        imagenet_train_root,
        imagenet_val_root,
        imagenet_train_list,
        imagenet_val_list,
        coco_train_root,
        coco_val_root,
        coco_train_manifest,
        coco_val_manifest,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required Phase 6 data paths under {data_root}: {joined}")

    train_pairs = collect_paired_by_subfolder(lr_train_root, hr_train_root)
    val_pairs = collect_paired_flat(lr_val_dir, hr_val_dir)
    if not train_pairs:
        raise FileNotFoundError(f"No paired training PNGs found under {hr_train_root} and {lr_train_root}.")
    if not val_pairs:
        raise FileNotFoundError(f"No paired validation PNGs found under {hr_val_dir} and {lr_val_dir}.")

    imagenet_train_records = collect_imagenet_records(read_imagenet_manifest(imagenet_train_list), imagenet_train_root, split="train")
    imagenet_val_records = collect_imagenet_records(read_imagenet_manifest(imagenet_val_list), imagenet_val_root, split="val")
    coco_train_records = collect_coco_records(read_manifest_lines(coco_train_manifest), coco_root(data_root), split="train")
    coco_val_records = collect_coco_records(read_manifest_lines(coco_val_manifest), coco_root(data_root), split="val")

    imagenet_train_used = take_manifest_subset(imagenet_train_records, data_cfg["imagenet_train_limit"], seed)
    imagenet_val_used = take_manifest_subset(imagenet_val_records, data_cfg["imagenet_val_limit"], seed)
    coco_train_used = take_manifest_subset(coco_train_records, data_cfg["coco_train_limit"], seed)
    coco_val_used = take_manifest_subset(coco_val_records, data_cfg["coco_val_limit"], seed)

    synthetic_cfg = dict(data_cfg)
    finetune_cfg = paired_finetune_data_cfg(data_cfg)

    paired_train_dataset = PairedSRDataset(train_pairs, train=True, data_cfg=finetune_cfg if stage_name == "stage2_finetune" else data_cfg, source_name="paired_train", seed=seed)
    paired_train_eval_dataset = PairedSRDataset(train_pairs, train=False, data_cfg=data_cfg, source_name="paired_train", seed=seed)
    paired_val_dataset = PairedSRDataset(val_pairs, train=False, data_cfg=data_cfg, source_name="paired_val", seed=seed)
    imagenet_train_dataset = NaturalImageSyntheticSRDataset(imagenet_train_used, train=True, data_cfg=synthetic_cfg, seed=seed)
    imagenet_train_eval_dataset = NaturalImageSyntheticSRDataset(imagenet_train_used, train=False, data_cfg=data_cfg, seed=seed)
    imagenet_val_dataset = NaturalImageSyntheticSRDataset(imagenet_val_used, train=False, data_cfg=data_cfg, seed=seed)
    coco_train_dataset = NaturalImageSyntheticSRDataset(coco_train_used, train=True, data_cfg=synthetic_cfg, seed=seed)
    coco_train_eval_dataset = NaturalImageSyntheticSRDataset(coco_train_used, train=False, data_cfg=data_cfg, seed=seed)
    coco_val_dataset = NaturalImageSyntheticSRDataset(coco_val_used, train=False, data_cfg=data_cfg, seed=seed)

    if stage_name == "stage1_pretrain":
        train_parts: list[Dataset] = [coco_train_dataset]
        train_eval_parts: list[Dataset] = [coco_train_eval_dataset]
    elif stage_name == "stage2_finetune":
        train_parts = [paired_train_dataset]
        train_eval_parts = [paired_train_eval_dataset]
    else:
        raise ValueError(f"Unsupported stage_name: {stage_name}")

    combined_val_parts: list[Dataset] = [paired_val_dataset, coco_val_dataset]
    calibration_datasets = {
        "paired_train": paired_train_eval_dataset,
        "coco_train": coco_train_eval_dataset,
    }
    if pretrain_mix == "coco_plus_imagenet":
        if stage_name == "stage1_pretrain":
            train_parts.append(imagenet_train_dataset)
            train_eval_parts.append(imagenet_train_eval_dataset)
        combined_val_parts.append(imagenet_val_dataset)
        calibration_datasets["imagenet_train"] = imagenet_train_eval_dataset

    train_dataset = ConcatDataset(train_parts)
    train_eval_dataset = ConcatDataset(train_eval_parts)
    combined_val_dataset = ConcatDataset(combined_val_parts)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs(num_workers, pin_memory))
    train_eval_loader = make_fixed_subset_loader(train_eval_dataset, data_cfg["train_eval_subset_size"], batch_size, seed, num_workers, pin_memory)
    paired_val_loader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    coco_val_loader = DataLoader(coco_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    imagenet_val_loader = DataLoader(imagenet_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))

    return {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "imagenet_train_records": imagenet_train_used,
        "imagenet_val_records": imagenet_val_used,
        "coco_train_records": coco_train_used,
        "coco_val_records": coco_val_used,
        "train_dataset": train_dataset,
        "train_eval_dataset": train_eval_dataset,
        "combined_val_dataset": combined_val_dataset,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "paired_val_loader": paired_val_loader,
        "coco_val_loader": coco_val_loader,
        "imagenet_val_loader": imagenet_val_loader,
        "combined_val_loader": combined_val_loader,
        "calibration_datasets": calibration_datasets,
    }


def print_data_summary(bundle: dict[str, Any]) -> None:
    lr_batch, hr_batch, _, _ = next(iter(bundle["train_loader"]))
    print(f"Train dataset: {len(bundle['train_dataset'])}")
    print(f"Combined val:  {len(bundle['combined_val_dataset'])}")
    print(f"Train-eval subset: {len(bundle['train_eval_loader'].dataset)}")
    print(f"Batch shapes: LR={tuple(lr_batch.shape)}, HR={tuple(hr_batch.shape)}")


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def combined_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 0.5 * charbonnier_loss(pred, target, eps) + 0.5 * F.l1_loss(pred, target)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


def lr_for_epoch(epoch: int, total: int, base_lr: float, warmup: int, min_ratio: float) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_ratio
    return min_lr + (base_lr - min_lr) * cosine


def _move_batch(lr_img: torch.Tensor, hr_img: torch.Tensor, device: torch.device, channels_last: bool):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        lr_img = lr_img.contiguous(memory_format=torch.channels_last)
        hr_img = hr_img.contiguous(memory_format=torch.channels_last)
    return lr_img, hr_img


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    train_cfg: dict[str, Any],
    device: torch.device,
    amp_policy: dict[str, Any],
    ema: EMA | None = None,
    scaler=None,
    channels_last: bool = True,
) -> dict[str, float]:
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for lr_img, hr_img, _, _ in loader:
        lr_img, hr_img = _move_batch(lr_img, hr_img, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast_context(device, amp_policy):
                pred = model(lr_img)
                loss = combined_loss(pred, hr_img, eps=train_cfg["charb_eps"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_context(device, amp_policy):
                pred = model(lr_img)
                loss = combined_loss(pred, hr_img, eps=train_cfg["charb_eps"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip_norm"])
            optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        with torch.no_grad():
            total_psnr += compute_psnr(pred.detach(), hr_img).sum().item()
        n += bs
    return {"train_loss": total_loss / max(1, n), "train_psnr_online": total_psnr / max(1, n)}


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    train_cfg: dict[str, Any],
    device: torch.device,
    amp_policy: dict[str, Any],
    split_name: str,
    channels_last: bool = True,
) -> dict[str, float]:
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for lr_img, hr_img, _, _ in loader:
        lr_img, hr_img = _move_batch(lr_img, hr_img, device, channels_last)
        with autocast_context(device, amp_policy):
            pred = model(lr_img)
            loss = combined_loss(pred, hr_img, eps=train_cfg["charb_eps"])
        psnr = compute_psnr(pred, hr_img)
        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        total_psnr += psnr.sum().item()
        n += bs
    return {
        f"{split_name}_loss": total_loss / max(1, n),
        f"{split_name}_psnr": total_psnr / max(1, n),
    }


def load_history(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    best_metric: float,
    best_epoch: int,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    amp_policy: dict[str, Any],
    stage_name: str,
    ema: EMA | None = None,
    scaler=None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "data_cfg": data_cfg,
        "amp_policy": amp_policy,
        "stage_name": stage_name,
    }
    if ema is not None:
        state["ema_shadow"] = ema.shadow
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return ckpt


def load_weights_only(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = load_checkpoint(model, checkpoint_path, map_location=map_location)
    if "ema_shadow" in ckpt:
        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name])
    return ckpt


def should_run_train_eval(epoch_num: int, train_cfg: dict[str, Any]) -> bool:
    interval = train_cfg.get("train_eval_interval", 5)
    if epoch_num == train_cfg["epochs"]:
        return True
    if interval > 0 and epoch_num % interval == 0:
        return True
    checkpoint_interval = train_cfg.get("checkpoint_interval", 10)
    return checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0


def fit_stage(
    model: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    eval_loaders: dict[str, DataLoader],
    output_dir: Path,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    device: torch.device,
    amp_policy: dict[str, Any],
    model_id: str,
    mix_id: str,
    stage_name: str,
    channels_last: bool = True,
    resume: bool = True,
    init_checkpoint_path: Path | None = None,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    last_ckpt_path = output_dir / "last.pt"
    best_ckpt_path = output_dir / "best.pt"
    selection_metric = train_cfg["selection_metric"]

    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = optimizer_with_fallback(model, train_cfg)
    ema = EMA(model, decay=train_cfg["ema_decay"])
    scaler = make_grad_scaler(amp_policy)
    start_epoch, best_metric, best_epoch = 0, float("-inf"), -1
    history: list[dict[str, Any]] = []

    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_metric = ckpt.get("best_metric", float("-inf"))
        best_epoch = ckpt.get("best_epoch", -1)
        history = load_history(metrics_path)
        if "ema_shadow" in ckpt:
            ema.shadow = ckpt["ema_shadow"]
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"RESUMING {stage_name} from epoch {start_epoch}/{train_cfg['epochs']}")
    else:
        set_seed(train_cfg["seed"])
        if init_checkpoint_path is not None and init_checkpoint_path.exists():
            load_weights_only(model, init_checkpoint_path, map_location=device)
            ema = EMA(model, decay=train_cfg["ema_decay"])
            print(f"INITIALIZED {stage_name} from {init_checkpoint_path}")
        metrics_path.write_text("")
        print(f"FRESH START: {stage_name}")

    if start_epoch >= train_cfg["epochs"]:
        print(f"{stage_name} already trained {start_epoch} epochs. Increase epochs to continue.")
        return history

    print(
        f"\n{'epoch':>5} | {'lr':>8} | {'train_loss':>10} {'train_online':>12} {'train_eval':>11} | "
        f"{'paired':>8} {'combined':>8} {'coco':>8} {'imagenet':>8} | {'best':>8} | {'time':>7}"
    )
    print("-" * 130)

    epochs_without_improve = 0
    for epoch in range(start_epoch, train_cfg["epochs"]):
        epoch_num = epoch + 1
        epoch_lr = lr_for_epoch(epoch, train_cfg["epochs"], train_cfg["lr"], train_cfg["warmup_epochs"], train_cfg["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr

        start_time = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            train_cfg,
            device,
            amp_policy,
            ema=ema,
            scaler=scaler,
            channels_last=channels_last,
        )
        ema.apply_shadow(model)
        train_eval_metrics: dict[str, Any] = {}
        if should_run_train_eval(epoch_num, train_cfg):
            train_eval_metrics = evaluate_loader(
                model,
                train_eval_loader,
                train_cfg,
                device,
                amp_policy,
                "train_eval",
                channels_last=channels_last,
            )

        split_metrics: dict[str, Any] = {}
        for split_name, loader in eval_loaders.items():
            split_metrics.update(
                evaluate_loader(
                    model,
                    loader,
                    train_cfg,
                    device,
                    amp_policy,
                    split_name,
                    channels_last=channels_last,
                )
            )
        ema.restore(model)

        seconds = round(time.time() - start_time, 1)
        row: dict[str, Any] = {
            "epoch": epoch_num,
            "lr": epoch_lr,
            "stage": stage_name,
            "model_id": model_id,
            "mix_id": mix_id,
            **train_metrics,
            **train_eval_metrics,
            **split_metrics,
            "seconds": seconds,
        }
        row["selection_metric"] = selection_metric
        row["selection_metric_value"] = row[selection_metric]

        history.append(row)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        metric_value = row[selection_metric]
        improved = metric_value > best_metric
        if improved:
            best_metric = metric_value
            best_epoch = epoch_num
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        save_checkpoint(
            last_ckpt_path,
            model,
            optimizer,
            epoch_num,
            row,
            best_metric,
            best_epoch,
            model_cfg,
            train_cfg,
            data_cfg,
            amp_policy,
            stage_name,
            ema=ema,
            scaler=scaler,
        )
        if improved:
            save_checkpoint(
                best_ckpt_path,
                model,
                optimizer,
                epoch_num,
                row,
                best_metric,
                best_epoch,
                model_cfg,
                train_cfg,
                data_cfg,
                amp_policy,
                stage_name,
                ema=ema,
                scaler=scaler,
            )
        should_archive = epoch_num % train_cfg["checkpoint_interval"] == 0
        should_stop = epochs_without_improve >= train_cfg["early_stop_patience"]
        if should_archive or should_stop:
            save_checkpoint(
                output_dir / f"epoch_{epoch_num:03d}.pt",
                model,
                optimizer,
                epoch_num,
                row,
                best_metric,
                best_epoch,
                model_cfg,
                train_cfg,
                data_cfg,
                amp_policy,
                stage_name,
                ema=ema,
                scaler=scaler,
            )

        train_eval_str = f"{row['train_eval_psnr']:.3f}" if "train_eval_psnr" in row else "     -"
        mark = "*" if improved else " "
        print(
            f"{epoch_num:5d} | {epoch_lr:8.2e} | {row['train_loss']:10.6f} {row['train_psnr_online']:12.3f} {train_eval_str:>11} | "
            f"{row.get('paired_val_psnr', float('nan')):8.3f} {row.get('combined_val_psnr', float('nan')):8.3f} "
            f"{row.get('coco_val_psnr', float('nan')):8.3f} {row.get('imagenet_val_psnr', float('nan')):8.3f} | "
            f"{best_metric:8.3f} | {seconds:6.1f}s {mark}"
        )

        if should_stop:
            print(f"Early stopping after {train_cfg['early_stop_patience']} epochs without improvement.")
            break

    summary = summarize_stage_run(output_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nStage complete. Best epoch {summary.get('best_epoch')} with {selection_metric}={summary.get(selection_metric):.3f} dB")
    print(f"Artifacts: {output_dir}")
    return history


def summarize_stage_run(output_dir: Path) -> dict[str, Any]:
    output_dir = Path(output_dir)
    metrics = load_history(output_dir / "metrics.jsonl")
    if not metrics:
        return {
            "output_dir": str(output_dir),
            "metrics_rows": 0,
            "best_epoch": None,
            "paired_val_psnr": None,
            "combined_val_psnr": None,
            "coco_val_psnr": None,
            "imagenet_val_psnr": None,
            "best_ckpt_exists": (output_dir / "best.pt").exists(),
        }
    selection_metric = metrics[-1].get("selection_metric", "paired_val_psnr")
    best_row = max(metrics, key=lambda row: row.get(selection_metric, float("-inf")))
    return {
        "output_dir": str(output_dir),
        "metrics_rows": len(metrics),
        "elapsed_seconds_total": round(sum(float(row.get("seconds", 0.0)) for row in metrics), 1),
        "selection_metric": selection_metric,
        "best_epoch": best_row["epoch"],
        "paired_val_psnr": best_row.get("paired_val_psnr"),
        "combined_val_psnr": best_row.get("combined_val_psnr"),
        "coco_val_psnr": best_row.get("coco_val_psnr"),
        "imagenet_val_psnr": best_row.get("imagenet_val_psnr"),
        "train_eval_psnr": best_row.get("train_eval_psnr"),
        "best_ckpt_exists": (output_dir / "best.pt").exists(),
        "last_metric": metrics[-1],
    }


@torch.no_grad()
def collect_psnr_records(model: nn.Module, loader: DataLoader, device: torch.device, channels_last: bool = True, max_items: int | None = None):
    model.eval()
    records = []
    for lr_img, hr_img, names, sources in loader:
        lr_img, hr_img = _move_batch(lr_img, hr_img, device, channels_last)
        pred = model(lr_img)
        pred_psnr = compute_psnr(pred, hr_img).cpu().tolist()
        input_psnr = compute_psnr(lr_img, hr_img).cpu().tolist()
        for name, source, pred_p, input_p in zip(names, sources, pred_psnr, input_psnr):
            records.append({"name": name, "source": source, "pred_psnr": float(pred_p), "input_psnr": float(input_p)})
            if max_items is not None and len(records) >= max_items:
                return records
    return records


def summarize_records(title: str, records: list[dict[str, Any]]) -> None:
    if not records:
        print(f"{title}: no records")
        return
    psnrs = torch.tensor([row["pred_psnr"] for row in records], dtype=torch.float32)
    baselines = torch.tensor([row["input_psnr"] for row in records], dtype=torch.float32)
    print(f"{title}: n={len(records)} | model={psnrs.mean():.3f} dB | baseline={baselines.mean():.3f} dB")


def run_diagnostics(
    build_model: Callable[..., nn.Module],
    model_cfg: dict[str, Any],
    output_dir: Path,
    data_bundle: dict[str, Any],
    device: torch.device,
    prepare_export_model: Callable[[nn.Module], nn.Module] | None = None,
) -> None:
    best_ckpt = Path(output_dir) / "best.pt"
    if not best_ckpt.exists():
        print(f"Run training first so {best_ckpt} exists.")
        return
    diag_model = build_model(**model_cfg).to(device)
    load_weights_only(diag_model, best_ckpt, map_location=device)
    if prepare_export_model is not None:
        diag_model = prepare_export_model(diag_model)
    summarize_records("train_eval", collect_psnr_records(diag_model, data_bundle["train_eval_loader"], device, max_items=128))
    summarize_records("paired_val", collect_psnr_records(diag_model, data_bundle["paired_val_loader"], device))
    summarize_records("coco_val", collect_psnr_records(diag_model, data_bundle["coco_val_loader"], device))
    summarize_records("imagenet_val", collect_psnr_records(diag_model, data_bundle["imagenet_val_loader"], device))
    summarize_records("combined_val", collect_psnr_records(diag_model, data_bundle["combined_val_loader"], device))


def export_to_onnx(
    build_model: Callable[..., nn.Module],
    model_cfg: dict[str, Any],
    checkpoint_path: Path,
    onnx_path: Path,
    data_cfg: dict[str, Any],
    device: torch.device,
    prepare_export_model: Callable[[nn.Module], nn.Module] | None = None,
    verify: bool = False,
    sample_loader: DataLoader | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_model = build_model(**model_cfg).to(device)
    load_weights_only(export_model, checkpoint_path, map_location=device)
    if prepare_export_model is not None:
        export_model = prepare_export_model(export_model)
    export_model.eval()

    dummy = torch.randn(1, 3, data_cfg["eval_size"], data_cfg["eval_size"], device=device)
    export_kwargs = dict(
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )
    try:
        torch.onnx.export(export_model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(export_model, dummy, str(onnx_path), **export_kwargs)

    if onnx is not None:
        onnx.checker.check_model(onnx.load(str(onnx_path)))
    if verify and ort is not None and sample_loader is not None:
        sample_lr, _, _, _ = next(iter(sample_loader))
        sample_lr = sample_lr[:1].to(device)
        with torch.no_grad():
            torch_out = export_model(sample_lr).cpu().numpy()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["output"], {"input": sample_lr.cpu().numpy()})[0]
        diff = abs(torch_out - ort_out)
        print(f"Parity: max_diff={diff.max():.8f}, mean_diff={diff.mean():.8f}")
    return onnx_path


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.act = nn.PReLU(num_parameters=mid)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        weights = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * weights


class SEResBlock(nn.Module):
    def __init__(self, channels, reduction=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(self.dropout(out))
        return x + out


class WideSEResNetSR(nn.Module):
    def __init__(self, num_blocks=16, channels=64, reduction=4, dropout=0.08, max_drop_path=0.10):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(*[SEResBlock(channels, reduction, dropout, drops[i]) for i in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


class SpatialGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.gate(self.bn(self.dw(x)))


class DSDABlock(nn.Module):
    def __init__(self, channels, reduction=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU(num_parameters=channels)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.PReLU(num_parameters=channels)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.sa = SpatialGate(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act1(self.bn1(self.dw1(x)))
        out = self.act2(self.bn2(self.pw1(out)))
        out = self.bn3(self.dw2(out))
        out = self.bn4(self.pw2(out))
        out = self.se(out)
        out = self.sa(out)
        out = self.drop_path(self.dropout(out))
        return x + out


class DSDANSR(nn.Module):
    def __init__(self, num_blocks=12, channels=128, reduction=4, dropout=0.08, max_drop_path=0.08):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(*[DSDABlock(channels, reduction, dropout, drops[i]) for i in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel = conv.weight
    bias = conv.bias
    if bias is None:
        bias = torch.zeros(conv.out_channels, device=kernel.device, dtype=kernel.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = bn.bias + (bias - bn.running_mean) * (bn.weight / std)
    return fused_kernel, fused_bias


def _fuse_identity_bn(bn: nn.BatchNorm2d, channels: int, device: torch.device, dtype: torch.dtype):
    kernel = torch.zeros((channels, channels, 3, 3), device=device, dtype=dtype)
    diag = torch.arange(channels, device=device)
    kernel[diag, diag, 1, 1] = 1.0
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = bn.bias - bn.running_mean * (bn.weight / std)
    return fused_kernel, fused_bias


def _pad_1x1_to_3x3(kernel: torch.Tensor):
    return torch.nn.functional.pad(kernel, [1, 1, 1, 1])


def _make_conv_from_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel, bias = _fuse_conv_bn_pair(conv, bn)
    fused = nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    ).to(device=kernel.device, dtype=kernel.dtype)
    fused.weight.data.copy_(kernel)
    fused.bias.data.copy_(bias)
    return fused


class ConvBN(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class RepConvBN(nn.Module):
    def __init__(self, channels, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        else:
            self.rbr_dense = ConvBN(channels, 3)
            self.rbr_1x1 = ConvBN(channels, 1)
            self.rbr_identity = nn.BatchNorm2d(channels)
            self.reparam = None

    def forward(self, x):
        if self.reparam is not None:
            return self.reparam(x)
        return self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)

    def get_equivalent_kernel_bias(self):
        if self.reparam is not None:
            return self.reparam.weight, self.reparam.bias
        dense_kernel, dense_bias = _fuse_conv_bn_pair(self.rbr_dense.conv, self.rbr_dense.bn)
        one_kernel, one_bias = _fuse_conv_bn_pair(self.rbr_1x1.conv, self.rbr_1x1.bn)
        id_kernel, id_bias = _fuse_identity_bn(self.rbr_identity, self.channels, dense_kernel.device, dense_kernel.dtype)
        return dense_kernel + _pad_1x1_to_3x3(one_kernel) + id_kernel, dense_bias + one_bias + id_bias

    def switch_to_deploy(self):
        if self.reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=True).to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)
        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.deploy = True


class RepResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.rep = RepConvBN(channels)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act(self.rep(x))
        out = self.bn(self.conv(out))
        out = self.drop_path(self.dropout(out))
        return x + out

    def switch_to_deploy(self):
        self.rep.switch_to_deploy()
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = _make_conv_from_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class RepConvSR(nn.Module):
    def __init__(self, num_blocks=12, channels=96, dropout=0.06, max_drop_path=0.08):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_act = nn.PReLU(num_parameters=channels)
        self.body = nn.Sequential(*[RepResidualBlock(channels, dropout=dropout, drop_path=drops[i]) for i in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        feat = self.stem_act(self.stem_bn(self.stem_conv(x)))
        return x + self.tail(self.body(feat))

    def switch_to_deploy(self):
        if isinstance(self.stem_bn, nn.BatchNorm2d):
            self.stem_conv = _make_conv_from_bn(self.stem_conv, self.stem_bn)
            self.stem_bn = nn.Identity()
        for block in self.body:
            if hasattr(block, "switch_to_deploy"):
                block.switch_to_deploy()
        return self


class LargeKernelDWBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, expansion=2, dropout=0.0, drop_path=0.0):
        super().__init__()
        hidden = channels * expansion
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU(num_parameters=channels)
        self.pw_expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.PReLU(num_parameters=hidden)
        self.pw_project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def body_forward(self, x):
        out = self.act1(self.bn1(self.dw(x)))
        out = self.act2(self.bn2(self.pw_expand(out)))
        return self.bn3(self.pw_project(out))

    def forward(self, x):
        out = self.body_forward(x)
        out = self.drop_path(self.dropout(out))
        return x + out


class LargeKernelSEBlock(LargeKernelDWBlock):
    def __init__(self, channels, kernel_size=7, expansion=2, reduction=4, dropout=0.0, drop_path=0.0):
        super().__init__(channels, kernel_size=kernel_size, expansion=expansion, dropout=dropout, drop_path=drop_path)
        self.se = SEBlock(channels, reduction=reduction)

    def forward(self, x):
        out = self.se(self.body_forward(x))
        out = self.drop_path(self.dropout(out))
        return x + out


class LargeKernelDWSR(nn.Module):
    def __init__(self, num_blocks=14, channels=96, expansion=2, kernels=(7, 11), dropout=0.04, max_drop_path=0.06):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        kernel_schedule = [kernels[i % len(kernels)] for i in range(num_blocks)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(
            *[
                LargeKernelDWBlock(channels, kernel_size=kernel_schedule[i], expansion=expansion, dropout=dropout, drop_path=drops[i])
                for i in range(num_blocks)
            ]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


class LargeKernelSESR(nn.Module):
    def __init__(self, num_blocks=14, channels=96, expansion=2, kernels=(7, 11), reduction=4, dropout=0.04, max_drop_path=0.06):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        kernel_schedule = [kernels[i % len(kernels)] for i in range(num_blocks)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(
            *[
                LargeKernelSEBlock(
                    channels,
                    kernel_size=kernel_schedule[i],
                    expansion=expansion,
                    reduction=reduction,
                    dropout=dropout,
                    drop_path=drops[i],
                )
                for i in range(num_blocks)
            ]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


class HybridRepLargeKernelBlock(nn.Module):
    def __init__(self, channels, block_index: int, kernels=(7, 11), expansion=2, dropout=0.05, drop_path=0.0):
        super().__init__()
        if block_index % 2 == 0:
            self.block = RepResidualBlock(channels, dropout=dropout, drop_path=drop_path)
        else:
            kernel = kernels[(block_index // 2) % len(kernels)]
            self.block = LargeKernelDWBlock(channels, kernel_size=kernel, expansion=expansion, dropout=dropout, drop_path=drop_path)

    def forward(self, x):
        return self.block(x)

    def switch_to_deploy(self):
        if hasattr(self.block, "switch_to_deploy"):
            self.block.switch_to_deploy()


class HybridRepLargeKernelSR(nn.Module):
    def __init__(self, num_blocks=12, channels=96, kernels=(7, 11), expansion=2, dropout=0.05, max_drop_path=0.08):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_act = nn.PReLU(num_parameters=channels)
        self.body = nn.Sequential(
            *[
                HybridRepLargeKernelBlock(
                    channels=channels,
                    block_index=i,
                    kernels=kernels,
                    expansion=expansion,
                    dropout=dropout,
                    drop_path=drops[i],
                )
                for i in range(num_blocks)
            ]
        )
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        feat = self.stem_act(self.stem_bn(self.stem_conv(x)))
        return x + self.tail(self.body(feat))

    def switch_to_deploy(self):
        if isinstance(self.stem_bn, nn.BatchNorm2d):
            self.stem_conv = _make_conv_from_bn(self.stem_conv, self.stem_bn)
            self.stem_bn = nn.Identity()
        for block in self.body:
            if hasattr(block, "switch_to_deploy"):
                block.switch_to_deploy()
        return self


def prepare_repconv_export(model):
    model.switch_to_deploy()
    return model


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "wide_se": {
        "display_name": "WideSEResNetSR",
        "build_model": lambda **cfg: WideSEResNetSR(**cfg),
        "model_cfg": {"num_blocks": 16, "channels": 64, "reduction": 4, "dropout": 0.08, "max_drop_path": 0.10},
        "prepare_export_model": None,
    },
    "dsdan": {
        "display_name": "DSDANSR",
        "build_model": lambda **cfg: DSDANSR(**cfg),
        "model_cfg": {"num_blocks": 12, "channels": 128, "reduction": 4, "dropout": 0.08, "max_drop_path": 0.08},
        "prepare_export_model": None,
    },
    "repconv": {
        "display_name": "RepConvSR",
        "build_model": lambda **cfg: RepConvSR(**cfg),
        "model_cfg": {"num_blocks": 12, "channels": 96, "dropout": 0.06, "max_drop_path": 0.08},
        "prepare_export_model": prepare_repconv_export,
    },
    "large_kernel_dw": {
        "display_name": "LargeKernelDWSR",
        "build_model": lambda **cfg: LargeKernelDWSR(**cfg),
        "model_cfg": {"num_blocks": 14, "channels": 96, "expansion": 2, "kernels": (7, 11), "dropout": 0.04, "max_drop_path": 0.06},
        "prepare_export_model": None,
    },
    "large_kernel_se": {
        "display_name": "LargeKernelSESR",
        "build_model": lambda **cfg: LargeKernelSESR(**cfg),
        "model_cfg": {"num_blocks": 14, "channels": 96, "expansion": 2, "kernels": (7, 11), "reduction": 4, "dropout": 0.04, "max_drop_path": 0.06},
        "prepare_export_model": None,
    },
    "hybrid_rep_large_kernel": {
        "display_name": "HybridRepLargeKernelSR",
        "build_model": lambda **cfg: HybridRepLargeKernelSR(**cfg),
        "model_cfg": {"num_blocks": 12, "channels": 96, "kernels": (7, 11), "expansion": 2, "dropout": 0.05, "max_drop_path": 0.08},
        "prepare_export_model": prepare_repconv_export,
    },
}


def get_model_spec(model_id: str) -> dict[str, Any]:
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id: {model_id}")
    spec = copy.deepcopy(MODEL_REGISTRY[model_id])
    probe = spec["build_model"](**spec["model_cfg"])
    spec["params"] = count_parameters(probe)
    spec["ops"] = summarize_npu_ops(probe)
    return spec


def stage_output_dir(output_root: Path, model_id: str, mix_id: str, stage_name: str, seed: int = 255) -> Path:
    root = Path(output_root)
    if seed == 255:
        return root / model_id / mix_id / stage_name
    return root / model_id / mix_id / f"{stage_name}_seed{seed}"


def config_output_dir(output_root: Path, model_id: str, mix_id: str) -> Path:
    return Path(output_root) / model_id / mix_id


def load_stage_summary(output_root: Path, model_id: str, mix_id: str, stage_name: str, seed: int = 255) -> dict[str, Any]:
    summary_path = stage_output_dir(output_root, model_id, mix_id, stage_name, seed=seed) / "summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text())


def build_config_summary(output_root: Path, model_id: str, mix_id: str, spec: dict[str, Any], seed: int = 255) -> dict[str, Any]:
    stage1 = load_stage_summary(output_root, model_id, mix_id, "stage1_pretrain", seed=seed)
    stage2 = load_stage_summary(output_root, model_id, mix_id, "stage2_finetune", seed=seed)
    summary = {
        "model_id": model_id,
        "mix_id": mix_id,
        "params": spec["params"],
        "display_name": spec["display_name"],
        "stage1": stage1,
        "stage2": stage2,
        "paired_val_psnr": stage2.get("paired_val_psnr"),
        "combined_val_psnr": stage2.get("combined_val_psnr"),
        "coco_val_psnr": stage2.get("coco_val_psnr"),
        "imagenet_val_psnr": stage2.get("imagenet_val_psnr"),
        "gpu_type": stage2.get("gpu_type") or stage1.get("gpu_type"),
        "modal_identifiers": {
            "stage1": stage1.get("modal_identifiers", {}),
            "stage2": stage2.get("modal_identifiers", {}),
        },
        "elapsed_seconds": round(float(stage1.get("elapsed_seconds_total", 0.0)) + float(stage2.get("elapsed_seconds_total", 0.0)), 1),
        "output_dir": str(config_output_dir(output_root, model_id, mix_id)),
        "seed": seed,
    }
    return summary


def sort_config_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]):
        return (
            -(row.get("paired_val_psnr") or float("-inf")),
            -(row.get("combined_val_psnr") or float("-inf")),
            row.get("params") or float("inf"),
            row.get("model_id") or "",
            row.get("mix_id") or "",
        )

    return sorted(rows, key=key)


def select_promotions(sorted_rows: list[dict[str, Any]], promotion_count: int = PROMOTION_COUNT) -> list[tuple[str, str]]:
    if not sorted_rows:
        return []
    selected: list[tuple[str, str]] = []
    selected_models: set[str] = set()

    for row in sorted_rows:
        if len(selected) >= min(2, promotion_count):
            break
        selected.append((row["model_id"], row["mix_id"]))
        selected_models.add(row["model_id"])

    while len(selected) < promotion_count:
        candidate = next((row for row in sorted_rows if (row["model_id"], row["mix_id"]) not in selected and row["model_id"] not in selected_models), None)
        if candidate is None:
            candidate = next((row for row in sorted_rows if (row["model_id"], row["mix_id"]) not in selected), None)
        if candidate is None:
            break
        selected.append((candidate["model_id"], candidate["mix_id"]))
        selected_models.add(candidate["model_id"])
    return selected


def detect_near_tie_candidates(sorted_rows: list[dict[str, Any]], selected: list[tuple[str, str]], threshold: float = DEFAULT_TIE_MARGIN) -> list[tuple[str, str]]:
    if not sorted_rows or not selected:
        return []
    promoted_scores = {
        (row["model_id"], row["mix_id"]): row.get("paired_val_psnr") for row in sorted_rows if (row["model_id"], row["mix_id"]) in selected
    }
    boundary = min(score for score in promoted_scores.values() if score is not None)
    candidates = []
    for row in sorted_rows:
        score = row.get("paired_val_psnr")
        if score is None:
            continue
        if abs(score - boundary) <= threshold:
            candidates.append((row["model_id"], row["mix_id"]))
    return candidates


def build_leaderboard_rows(config_summaries: list[dict[str, Any]], tie_threshold: float = DEFAULT_TIE_MARGIN) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked = sort_config_summaries([row for row in config_summaries if row.get("paired_val_psnr") is not None])
    selected = select_promotions(ranked, PROMOTION_COUNT)
    near_tie = detect_near_tie_candidates(ranked, selected, threshold=tie_threshold)
    rows = []
    for index, row in enumerate(ranked, start=1):
        model_mix = (row["model_id"], row["mix_id"])
        promotion_status = "promoted" if model_mix in selected else "screened"
        if model_mix in near_tie:
            promotion_status = f"{promotion_status}_near_tie"
        rows.append(
            {
                **row,
                "screening_rank": index,
                "promotion_status": promotion_status,
            }
        )
    meta = {
        "promotion_targets": [{"model_id": model_id, "mix_id": mix_id} for model_id, mix_id in selected],
        "near_tie_candidates": [{"model_id": model_id, "mix_id": mix_id} for model_id, mix_id in near_tie],
        "tie_threshold": tie_threshold,
    }
    return rows, meta


def default_screening_state(output_root: Path) -> dict[str, Any]:
    return {
        "state_version": STATE_VERSION,
        "output_root": str(output_root),
        "coco_prep_completed": False,
        "completed_models": [],
        "model_summaries": {},
        "leaderboard_meta": {},
        "tiebreak_runs": {},
    }


def load_screening_state(state_path: Path, output_root: Path) -> dict[str, Any]:
    if not state_path.exists():
        return default_screening_state(output_root)
    return json.loads(state_path.read_text())


def write_screening_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def write_leaderboard(leaderboard_path: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(json.dumps({"rows": rows, "meta": meta}, indent=2))


def current_modal_identifiers() -> dict[str, str]:
    ids = {name: os.environ.get(name) for name in ["MODAL_TASK_ID", "MODAL_CONTAINER_ID", "MODAL_APP_ID", "MODAL_ENVIRONMENT_NAME"] if os.environ.get(name)}
    try:
        import modal  # type: ignore

        if hasattr(modal, "current_function_call_id"):
            call_id = modal.current_function_call_id()
            if call_id:
                ids["modal_function_call_id"] = str(call_id)
    except Exception:
        pass
    return ids


PORTABLE_NOTEBOOK_SETUP = """# Optional one-time dependency bootstrap. Run if this environment is missing the required packages.\n# %pip install torch torchvision pillow onnx onnxruntime nbformat\n"""


def portable_runtime_source(model_source: str, default_model_id: str, default_mix: str) -> str:
    model_source = model_source.rstrip()
    return f"""from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import tarfile
import time
import urllib.request
import warnings
import zipfile

warnings.filterwarnings("ignore", message=".*legacy TorchScript-based ONNX.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps
from torch.optim import AdamW
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

TO_TENSOR = transforms.ToTensor()
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
INTERPOLATION_BANK = {{
    "bicubic": BICUBIC,
    "bilinear": Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR,
    "lanczos": LANCZOS,
}}
FORBIDDEN_TYPES = (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.LayerNorm, nn.GroupNorm)
COCO_URLS = {{
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}}


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {{"0", "false", "no", "off"}}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw is not None else default


def default_data_cfg() -> dict[str, object]:
    return {{
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
        "coco_train_limit": 30000,
        "coco_val_limit": 1000,
        "train_eval_subset_size": 256,
    }}


def default_stage_cfg(stage_name: str, batch_size: int, epochs: int, seed: int) -> dict[str, object]:
    base = {{
        "stage_name": stage_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": 3e-4 if stage_name == "stage1_pretrain" else 1.5e-4,
        "weight_decay": 2e-4,
        "warmup_epochs": 2 if stage_name == "stage1_pretrain" else 1,
        "min_lr_ratio": 0.05,
        "checkpoint_interval": 4 if stage_name == "stage1_pretrain" else 2,
        "train_eval_interval": 2 if stage_name == "stage1_pretrain" else 1,
        "seed": seed,
        "early_stop_patience": 10,
        "grad_clip_norm": 1.0,
        "ema_decay": 0.999,
        "charb_eps": 1e-6,
        "selection_metric": "paired_val_psnr",
    }}
    return base


def configure_runtime() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except AttributeError:
            pass
    return device


def choose_amp_policy(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {{"enabled": False, "dtype": None, "use_scaler": False, "label": "fp32"}}
    bf16_ok = False
    if hasattr(torch.cuda, "is_bf16_supported"):
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
    if bf16_ok:
        return {{"enabled": True, "dtype": torch.bfloat16, "use_scaler": False, "label": "bf16"}}
    return {{"enabled": True, "dtype": torch.float16, "use_scaler": True, "label": "fp16"}}


def make_grad_scaler(amp_policy: dict[str, object]):
    if not amp_policy["enabled"] or not amp_policy["use_scaler"]:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def autocast_context(device: torch.device, amp_policy: dict[str, object]):
    if device.type != "cuda" or not amp_policy["enabled"]:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=amp_policy["dtype"])
    return torch.cuda.amp.autocast(dtype=amp_policy["dtype"])


def optimizer_with_fallback(model: nn.Module, train_cfg: dict[str, object]) -> torch.optim.Optimizer:
    kwargs = {{"lr": train_cfg["lr"], "weight_decay": train_cfg["weight_decay"]}}
    if torch.cuda.is_available():
        try:
            return AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return AdamW(model.parameters(), **kwargs)


def seeded_rng(key: str) -> random.Random:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def ensure_unzipped(zip_path: Path, extracted_dir: Path) -> Path:
    if extracted_dir.exists():
        return extracted_dir
    if not zip_path.exists():
        return extracted_dir
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    return extracted_dir


def download_url(url: str, dest_path: Path) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        return dest_path
    with urllib.request.urlopen(url) as response, dest_path.open("wb") as f:
        shutil.copyfileobj(response, f)
    return dest_path


def stage_coco2017(data_root: Path) -> None:
    root = data_root / "course_files_export" / "coco2017"
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train2017", "val2017"):
        zip_path = root / f"{{split}}.zip"
        image_dir = root / split
        manifest = root / f"coco_{{split}}.txt"
        if not zip_path.exists():
            download_url(COCO_URLS[split], zip_path)
        ensure_unzipped(zip_path, image_dir)
        images = sorted(
            [
                str(path.relative_to(image_dir.parent))
                for path in image_dir.rglob("*")
                if path.suffix.lower() in {{".jpg", ".jpeg", ".png"}}
            ]
        )
        manifest.write_text("\\n".join(images) + ("\\n" if images else ""))


def read_manifest_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_imagenet_manifest(path: Path) -> list[tuple[str, int]]:
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        rows.append((parts[0], int(parts[1])))
    return rows


def collect_imagenet_records(rows: list[tuple[str, int]], root: Path, split: str) -> list[dict[str, object]]:
    records = []
    for filename, class_id in rows:
        synset = filename.split("_")[0]
        path = (root / synset / filename) if split == "train" else (root / filename)
        if path.exists():
            records.append({{"path": path, "stem": path.stem, "class_id": class_id, "source_name": f"imagenet_{{split}}"}})
    return records


def collect_coco_records(lines: list[str], root: Path, split: str) -> list[dict[str, object]]:
    records = []
    for rel in lines:
        path = root / rel
        if path.exists():
            records.append({{"path": path, "stem": path.stem, "class_id": -1, "source_name": f"coco_{{split}}"}})
    return records


def take_manifest_subset(records: list[object], limit: int | None, seed: int) -> list[object]:
    if limit is None or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    return items[:limit]


def collect_paired_by_subfolder(lr_root: Path, hr_root: Path) -> list[tuple[Path, Path, str]]:
    pairs = []
    for hr_dir in sorted(p for p in hr_root.iterdir() if p.is_dir()):
        suffix = hr_dir.name.replace("HR_train", "")
        lr_dir = lr_root / f"LR_train{{suffix}}"
        if not lr_dir.exists():
            continue
        hr_imgs = {{p.stem: p for p in sorted(hr_dir.glob("*.png"))}}
        lr_imgs = {{p.stem: p for p in sorted(lr_dir.glob("*.png"))}}
        common = sorted(set(hr_imgs) & set(lr_imgs))
        pairs.extend((lr_imgs[s], hr_imgs[s], f"{{hr_dir.name}}/{{s}}") for s in common)
    return pairs


def collect_paired_flat(lr_dir: Path, hr_dir: Path) -> list[tuple[Path, Path, str]]:
    hr_imgs = {{p.stem: p for p in sorted(hr_dir.glob("*.png"))}}
    lr_imgs = {{p.stem: p for p in sorted(lr_dir.glob("*.png"))}}
    common = sorted(set(hr_imgs) & set(lr_imgs))
    return [(lr_imgs[s], hr_imgs[s], s) for s in common]


def random_crop_pair(lr_img: Image.Image, hr_img: Image.Image, size: int, rng: random.Random):
    if lr_img.width == size and lr_img.height == size:
        return lr_img, hr_img
    x0 = rng.randint(0, lr_img.width - size)
    y0 = rng.randint(0, lr_img.height - size)
    box = (x0, y0, x0 + size, y0 + size)
    return lr_img.crop(box), hr_img.crop(box)


def random_crop_single(img: Image.Image, size: int, rng: random.Random):
    if img.width == size and img.height == size:
        return img
    x0 = rng.randint(0, img.width - size)
    y0 = rng.randint(0, img.height - size)
    return img.crop((x0, y0, x0 + size, y0 + size))


def augment_pair(lr_img: Image.Image, hr_img: Image.Image, rng: random.Random):
    if rng.random() > 0.5:
        lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
        hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        lr_img = lr_img.transpose(Image.FLIP_TOP_BOTTOM)
        hr_img = hr_img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randint(0, 3)
    if k > 0:
        rot = {{1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}}[k]
        lr_img = lr_img.transpose(rot)
        hr_img = hr_img.transpose(rot)
    return lr_img, hr_img


def augment_single(img: Image.Image, rng: random.Random):
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() > 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randint(0, 3)
    if k > 0:
        img = img.transpose({{1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}}[k])
    return img


def jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def degrade_from_hr(hr_img: Image.Image, rng: random.Random, cfg: dict[str, object]) -> Image.Image:
    lr_img = hr_img.copy()
    if rng.random() < cfg["lr_blur_prob"]:
        radius = rng.uniform(cfg["blur_radius_min"], cfg["blur_radius_max"])
        lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=radius))
    scale = rng.choice(cfg["downsample_scales"])
    resize_mode = INTERPOLATION_BANK[rng.choice(cfg["resize_modes"])]
    small = (max(1, hr_img.width // scale), max(1, hr_img.height // scale))
    lr_img = lr_img.resize(small, resample=resize_mode).resize(hr_img.size, resample=resize_mode)
    if rng.random() < cfg["jpeg_prob"]:
        lr_img = jpeg_roundtrip(lr_img, rng.randint(cfg["jpeg_quality_min"], cfg["jpeg_quality_max"]))
    return lr_img


def apply_tensor_regularization(lr_t: torch.Tensor, rng: random.Random, cfg: dict[str, object], train: bool) -> torch.Tensor:
    if not train:
        return lr_t
    if cfg["lr_noise_prob"] > 0 and rng.random() < cfg["lr_noise_prob"]:
        lr_t = (lr_t + torch.randn_like(lr_t) * cfg["lr_noise_std"]).clamp(0.0, 1.0)
    if cfg["cutout_prob"] > 0 and rng.random() < cfg["cutout_prob"]:
        _, h, w = lr_t.shape
        cut = max(8, int(min(h, w) * cfg["cutout_ratio"]))
        x0 = rng.randint(0, w - cut)
        y0 = rng.randint(0, h - cut)
        fill = lr_t.mean().item()
        lr_t[:, y0 : y0 + cut, x0 : x0 + cut] = fill
    return lr_t


class PairedSRDataset(Dataset):
    def __init__(self, pairs, train: bool, data_cfg: dict[str, object], source_name: str, seed: int):
        self.pairs = pairs
        self.train = train
        self.data_cfg = data_cfg
        self.source_name = source_name
        self.seed = seed

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        lr_path, hr_path, stem = self.pairs[idx]
        lr_img = Image.open(lr_path).convert("RGB")
        hr_img = Image.open(hr_path).convert("RGB")
        rng = random.Random(self.seed + idx) if self.train else seeded_rng(f"{{self.source_name}}:{{stem}}")
        if self.train:
            lr_img, hr_img = random_crop_pair(lr_img, hr_img, self.data_cfg["train_patch_size"], rng)
            lr_img, hr_img = augment_pair(lr_img, hr_img, rng)
        else:
            lr_img = ImageOps.fit(lr_img, (self.data_cfg["eval_size"], self.data_cfg["eval_size"]), method=BICUBIC)
            hr_img = ImageOps.fit(hr_img, (self.data_cfg["eval_size"], self.data_cfg["eval_size"]), method=BICUBIC)
        lr_t = TO_TENSOR(lr_img)
        hr_t = TO_TENSOR(hr_img)
        lr_t = apply_tensor_regularization(lr_t, rng, self.data_cfg, train=self.train)
        return lr_t, hr_t, stem, self.source_name


class NaturalImageSyntheticSRDataset(Dataset):
    def __init__(self, records, train: bool, data_cfg: dict[str, object], seed: int):
        self.records = records
        self.train = train
        self.data_cfg = data_cfg
        self.seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        hr_img = Image.open(record["path"]).convert("RGB")
        source_name = record["source_name"]
        rng = random.Random(self.seed + idx) if self.train else seeded_rng(f"{{source_name}}:{{record['stem']}}")
        if self.train:
            base_size = max(self.data_cfg["eval_size"], self.data_cfg["train_patch_size"] + self.data_cfg["random_scale_pad"])
            hr_img = ImageOps.fit(hr_img, (base_size, base_size), method=BICUBIC)
            hr_img = random_crop_single(hr_img, self.data_cfg["train_patch_size"], rng)
            hr_img = augment_single(hr_img, rng)
        else:
            hr_img = ImageOps.fit(hr_img, (self.data_cfg["eval_size"], self.data_cfg["eval_size"]), method=BICUBIC)
        lr_img = degrade_from_hr(hr_img, rng, self.data_cfg)
        lr_t = TO_TENSOR(lr_img)
        hr_t = TO_TENSOR(hr_img)
        lr_t = apply_tensor_regularization(lr_t, rng, self.data_cfg, train=self.train)
        return lr_t, hr_t, record["stem"], source_name


def loader_kwargs(num_workers: int, pin_memory: bool) -> dict[str, object]:
    kwargs: dict[str, object] = {{"num_workers": num_workers, "pin_memory": pin_memory}}
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return kwargs


def make_fixed_subset_loader(dataset, subset_size: int, batch_size: int, seed: int, num_workers: int, pin_memory: bool):
    subset_size = min(subset_size, len(dataset))
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    subset = Subset(dataset, indices[:subset_size])
    return DataLoader(subset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))


def paired_finetune_data_cfg(base_cfg: dict[str, object]) -> dict[str, object]:
    cfg = dict(base_cfg)
    cfg["cutout_prob"] = 0.0
    cfg["lr_noise_prob"] = 0.0
    return cfg


def build_data_bundle(data_root: Path, data_cfg: dict[str, object], batch_size: int, num_workers: int, device: torch.device, seed: int, pretrain_mix: str, stage_name: str):
    stage_coco2017(data_root)
    course_export_root = data_root / "course_files_export"
    legacy_image_root = data_root / "ImageNet"
    coco_stage_root = course_export_root / "coco2017"

    hr_train_root = data_root / "HR_train"
    lr_train_root = data_root / "LR_train"
    hr_val_dir = first_existing(data_root / "HR_val", data_root / "val" / "HR_val")
    lr_val_dir = first_existing(data_root / "LR_val", data_root / "val" / "LR_val")

    imagenet_train_list = first_existing(course_export_root / "imagenet_train20.txt", legacy_image_root / "imagenet_train20.txt")
    imagenet_val_list = first_existing(course_export_root / "imagenet_val20.txt", legacy_image_root / "imagenet_val20.txt")
    imagenet_train_root = ensure_unzipped(course_export_root / "imagenet_train20.zip", first_existing(course_export_root / "imagenet_train20a", legacy_image_root / "imagenet_train20a"))
    imagenet_val_root = ensure_unzipped(course_export_root / "imagenet_val20.zip", first_existing(course_export_root / "imagenet_val20", legacy_image_root / "imagenet_val20"))

    train_pairs = collect_paired_by_subfolder(lr_train_root, hr_train_root)
    val_pairs = collect_paired_flat(lr_val_dir, hr_val_dir)
    imagenet_train_records = collect_imagenet_records(read_imagenet_manifest(imagenet_train_list), imagenet_train_root, split="train")
    imagenet_val_records = collect_imagenet_records(read_imagenet_manifest(imagenet_val_list), imagenet_val_root, split="val")
    coco_train_records = collect_coco_records(read_manifest_lines(coco_stage_root / "coco_train2017.txt"), coco_stage_root, split="train")
    coco_val_records = collect_coco_records(read_manifest_lines(coco_stage_root / "coco_val2017.txt"), coco_stage_root, split="val")

    imagenet_train_used = take_manifest_subset(imagenet_train_records, data_cfg["imagenet_train_limit"], seed)
    imagenet_val_used = take_manifest_subset(imagenet_val_records, data_cfg["imagenet_val_limit"], seed)
    coco_train_used = take_manifest_subset(coco_train_records, data_cfg["coco_train_limit"], seed)
    coco_val_used = take_manifest_subset(coco_val_records, data_cfg["coco_val_limit"], seed)

    synthetic_cfg = dict(data_cfg)
    finetune_cfg = paired_finetune_data_cfg(data_cfg)

    paired_train_dataset = PairedSRDataset(train_pairs, train=True, data_cfg=finetune_cfg if stage_name == "stage2_finetune" else data_cfg, source_name="paired_train", seed=seed)
    paired_train_eval_dataset = PairedSRDataset(train_pairs, train=False, data_cfg=data_cfg, source_name="paired_train", seed=seed)
    paired_val_dataset = PairedSRDataset(val_pairs, train=False, data_cfg=data_cfg, source_name="paired_val", seed=seed)
    imagenet_train_dataset = NaturalImageSyntheticSRDataset(imagenet_train_used, train=True, data_cfg=synthetic_cfg, seed=seed)
    imagenet_train_eval_dataset = NaturalImageSyntheticSRDataset(imagenet_train_used, train=False, data_cfg=data_cfg, seed=seed)
    imagenet_val_dataset = NaturalImageSyntheticSRDataset(imagenet_val_used, train=False, data_cfg=data_cfg, seed=seed)
    coco_train_dataset = NaturalImageSyntheticSRDataset(coco_train_used, train=True, data_cfg=synthetic_cfg, seed=seed)
    coco_train_eval_dataset = NaturalImageSyntheticSRDataset(coco_train_used, train=False, data_cfg=data_cfg, seed=seed)
    coco_val_dataset = NaturalImageSyntheticSRDataset(coco_val_used, train=False, data_cfg=data_cfg, seed=seed)

    if stage_name == "stage1_pretrain":
        train_parts = [coco_train_dataset]
        train_eval_parts = [coco_train_eval_dataset]
    else:
        train_parts = [paired_train_dataset]
        train_eval_parts = [paired_train_eval_dataset]

    combined_val_parts = [paired_val_dataset, coco_val_dataset]
    if pretrain_mix == "coco_plus_imagenet":
        if stage_name == "stage1_pretrain":
            train_parts.append(imagenet_train_dataset)
            train_eval_parts.append(imagenet_train_eval_dataset)
        combined_val_parts.append(imagenet_val_dataset)

    train_dataset = ConcatDataset(train_parts)
    train_eval_dataset = ConcatDataset(train_eval_parts)
    combined_val_dataset = ConcatDataset(combined_val_parts)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs(num_workers, pin_memory))
    train_eval_loader = make_fixed_subset_loader(train_eval_dataset, data_cfg["train_eval_subset_size"], batch_size, seed, num_workers, pin_memory)
    paired_val_loader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    coco_val_loader = DataLoader(coco_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    imagenet_val_loader = DataLoader(imagenet_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    combined_val_loader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    return {{
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "paired_val_loader": paired_val_loader,
        "coco_val_loader": coco_val_loader,
        "imagenet_val_loader": imagenet_val_loader,
        "combined_val_loader": combined_val_loader,
    }}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def assert_npu_compatible(model: nn.Module) -> None:
    for name, mod in model.named_modules():
        if isinstance(mod, FORBIDDEN_TYPES):
            raise TypeError(f"Forbidden NPU op '{{name}}': {{mod.__class__.__name__}}")


def summarize_npu_ops(model: nn.Module) -> dict[str, int]:
    ops: dict[str, int] = defaultdict(int)
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            key = "DWConv" if module.groups == module.in_channels and module.in_channels > 1 else "Conv"
            ops[key] += 1
        elif isinstance(module, nn.BatchNorm2d):
            ops["BN"] += 1
        elif isinstance(module, nn.PReLU):
            ops["PReLU"] += 1
        elif isinstance(module, nn.Hardsigmoid):
            ops["HardSigmoid"] += 1
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            ops["GlobalAvgPool"] += 1
    return dict(ops)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def charbonnier_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def combined_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return 0.5 * charbonnier_loss(pred, target, eps) + 0.5 * F.l1_loss(pred, target)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return -10.0 * torch.log10(mse)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {{}}
        self.backup = {{}}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {{}}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {{}}


def lr_for_epoch(epoch: int, total: int, base_lr: float, warmup: int, min_ratio: float) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_ratio
    return min_lr + (base_lr - min_lr) * cosine


def _move_batch(lr_img: torch.Tensor, hr_img: torch.Tensor, device: torch.device, channels_last: bool):
    lr_img = lr_img.to(device, non_blocking=True)
    hr_img = hr_img.to(device, non_blocking=True)
    if channels_last and device.type == "cuda":
        lr_img = lr_img.contiguous(memory_format=torch.channels_last)
        hr_img = hr_img.contiguous(memory_format=torch.channels_last)
    return lr_img, hr_img


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, train_cfg: dict[str, object], device: torch.device, amp_policy: dict[str, object], ema: EMA | None = None, scaler=None, channels_last: bool = True):
    model.train()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for lr_img, hr_img, _, _ in loader:
        lr_img, hr_img = _move_batch(lr_img, hr_img, device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast_context(device, amp_policy):
                pred = model(lr_img)
                loss = combined_loss(pred, hr_img, eps=train_cfg["charb_eps"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip_norm"])
            scaler.step(optimizer)
            scaler.update()
        else:
            with autocast_context(device, amp_policy):
                pred = model(lr_img)
                loss = combined_loss(pred, hr_img, eps=train_cfg["charb_eps"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_cfg["grad_clip_norm"])
            optimizer.step()
        if ema is not None:
            ema.update(model)
        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        with torch.no_grad():
            total_psnr += compute_psnr(pred.detach(), hr_img).sum().item()
        n += bs
    return {{"train_loss": total_loss / max(1, n), "train_psnr_online": total_psnr / max(1, n)}}


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, train_cfg: dict[str, object], device: torch.device, amp_policy: dict[str, object], split_name: str, channels_last: bool = True):
    model.eval()
    total_loss, total_psnr, n = 0.0, 0.0, 0
    for lr_img, hr_img, _, _ in loader:
        lr_img, hr_img = _move_batch(lr_img, hr_img, device, channels_last)
        with autocast_context(device, amp_policy):
            pred = model(lr_img)
            loss = combined_loss(pred, hr_img, eps=train_cfg["charb_eps"])
        psnr = compute_psnr(pred, hr_img)
        bs = lr_img.size(0)
        total_loss += loss.item() * bs
        total_psnr += psnr.sum().item()
        n += bs
    return {{
        f"{{split_name}}_loss": total_loss / max(1, n),
        f"{{split_name}}_psnr": total_psnr / max(1, n),
    }}


def load_checkpoint(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


def load_weights_only(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu"):
    ckpt = load_checkpoint(model, checkpoint_path, map_location=map_location)
    if "ema_shadow" in ckpt:
        for name, param in model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name])
    return ckpt


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, metrics: dict[str, object], best_metric: float, best_epoch: int, model_cfg: dict[str, object], train_cfg: dict[str, object], data_cfg: dict[str, object], amp_policy: dict[str, object], stage_name: str, ema: EMA | None = None, scaler=None) -> None:
    state = {{
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "data_cfg": data_cfg,
        "amp_policy": amp_policy,
        "stage_name": stage_name,
    }}
    if ema is not None:
        state["ema_shadow"] = ema.shadow
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def should_run_train_eval(epoch_num: int, train_cfg: dict[str, object]) -> bool:
    interval = train_cfg.get("train_eval_interval", 5)
    if epoch_num == train_cfg["epochs"]:
        return True
    if interval > 0 and epoch_num % interval == 0:
        return True
    checkpoint_interval = train_cfg.get("checkpoint_interval", 10)
    return checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0


def fit_stage(model: nn.Module, train_loader: DataLoader, train_eval_loader: DataLoader, eval_loaders: dict[str, DataLoader], output_dir: Path, model_cfg: dict[str, object], train_cfg: dict[str, object], data_cfg: dict[str, object], device: torch.device, amp_policy: dict[str, object], stage_name: str, channels_last: bool = True, init_checkpoint_path: Path | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    last_ckpt_path = output_dir / "last.pt"
    best_ckpt_path = output_dir / "best.pt"
    selection_metric = train_cfg["selection_metric"]

    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = optimizer_with_fallback(model, train_cfg)
    ema = EMA(model, decay=train_cfg["ema_decay"])
    scaler = make_grad_scaler(amp_policy)
    if init_checkpoint_path is not None:
        load_weights_only(model, init_checkpoint_path, map_location=device)
        ema = EMA(model, decay=train_cfg["ema_decay"])

    metrics_path.write_text("")
    best_metric = float("-inf")
    best_epoch = -1
    history = []
    for epoch in range(train_cfg["epochs"]):
        epoch_num = epoch + 1
        epoch_lr = lr_for_epoch(epoch, train_cfg["epochs"], train_cfg["lr"], train_cfg["warmup_epochs"], train_cfg["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = epoch_lr
        start_time = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, train_cfg, device, amp_policy, ema=ema, scaler=scaler, channels_last=channels_last)
        ema.apply_shadow(model)
        train_eval_metrics = {{}}
        if should_run_train_eval(epoch_num, train_cfg):
            train_eval_metrics = evaluate_loader(model, train_eval_loader, train_cfg, device, amp_policy, "train_eval", channels_last=channels_last)
        split_metrics = {{}}
        for split_name, loader in eval_loaders.items():
            split_metrics.update(evaluate_loader(model, loader, train_cfg, device, amp_policy, split_name, channels_last=channels_last))
        ema.restore(model)
        seconds = round(time.time() - start_time, 1)
        row = {{"epoch": epoch_num, "lr": epoch_lr, "stage": stage_name, **train_metrics, **train_eval_metrics, **split_metrics, "seconds": seconds, "selection_metric": selection_metric}}
        row["selection_metric_value"] = row[selection_metric]
        history.append(row)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\\n")
        if row[selection_metric] > best_metric:
            best_metric = row[selection_metric]
            best_epoch = epoch_num
            save_checkpoint(best_ckpt_path, model, optimizer, epoch_num, row, best_metric, best_epoch, model_cfg, train_cfg, data_cfg, amp_policy, stage_name, ema=ema, scaler=scaler)
        save_checkpoint(last_ckpt_path, model, optimizer, epoch_num, row, best_metric, best_epoch, model_cfg, train_cfg, data_cfg, amp_policy, stage_name, ema=ema, scaler=scaler)
        print(f"{{stage_name}} epoch {{epoch_num}}: paired={{row.get('paired_val_psnr')}} combined={{row.get('combined_val_psnr')}}")
    return history


def export_to_onnx(build_model, model_cfg: dict[str, object], checkpoint_path: Path, onnx_path: Path, data_cfg: dict[str, object], device: torch.device, prepare_export_model=None, verify: bool = False, sample_loader: DataLoader | None = None):
    model = build_model(**model_cfg).to(device)
    load_weights_only(model, checkpoint_path, map_location=device)
    if prepare_export_model is not None:
        model = prepare_export_model(model)
    model.eval()
    dummy = torch.randn(1, 3, data_cfg["eval_size"], data_cfg["eval_size"], device=device)
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, export_params=True, opset_version=13, do_constant_folding=True, input_names=["input"], output_names=["output"])
    except TypeError:
        torch.onnx.export(model, dummy, str(onnx_path), export_params=True, opset_version=13, do_constant_folding=True, input_names=["input"], output_names=["output"])
    if verify and ort is not None and sample_loader is not None:
        sample_lr, _, _, _ = next(iter(sample_loader))
        sample_lr = sample_lr[:1].to(device)
        with torch.no_grad():
            torch_out = model(sample_lr).cpu().numpy()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(["output"], {{"input": sample_lr.cpu().numpy()}})[0]
        diff = abs(torch_out - ort_out)
        print(f"Parity: max_diff={{diff.max():.8f}}, mean_diff={{diff.mean():.8f}}")


{model_source}

DEFAULT_MODEL_ID = "{default_model_id}"
PRETRAIN_MIX = os.environ.get("LAB2_PRETRAIN_MIX", "{default_mix}")
BATCH_SIZE = env_int("LAB2_BATCH_SIZE", 4)
NUM_WORKERS = env_int("LAB2_NUM_WORKERS", 0)
SEED = env_int("LAB2_SEED", 255)
STAGE1_EPOCHS = env_int("LAB2_STAGE1_EPOCHS", 40)
STAGE2_EPOCHS = env_int("LAB2_STAGE2_EPOCHS", 20)
RUN_DIAGNOSTICS = env_flag("LAB2_RUN_DIAGNOSTICS", True)
RUN_ONNX_EXPORT = env_flag("LAB2_RUN_ONNX_EXPORT", True)
USE_AMP = env_flag("LAB2_USE_AMP", True)
CHANNELS_LAST = env_flag("LAB2_CHANNELS_LAST", True)
DATA_ROOT = Path(os.environ.get("LAB2_DATA_ROOT", str(Path.cwd() / "Data")))
OUTPUT_DIR = Path(os.environ.get("LAB2_OUTPUT_DIR", str(Path.cwd() / "runs" / f"phase6_full_{{DEFAULT_MODEL_ID}}_{{PRETRAIN_MIX}}")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = configure_runtime()
amp_policy = choose_amp_policy(device)
if not USE_AMP:
    amp_policy = {{"enabled": False, "dtype": None, "use_scaler": False, "label": "fp32"}}
channels_last = bool(CHANNELS_LAST and device.type == "cuda")

data_cfg = default_data_cfg()
stage1_cfg = default_stage_cfg("stage1_pretrain", BATCH_SIZE, STAGE1_EPOCHS, SEED)
stage2_cfg = default_stage_cfg("stage2_finetune", BATCH_SIZE, STAGE2_EPOCHS, SEED)

model_for_check = build_model(**MODEL_CFG)
assert_npu_compatible(model_for_check)
print(f"Model: {{DEFAULT_MODEL_ID}} -> {{model_for_check.__class__.__name__}}")
print(f"Parameters: {{count_parameters(model_for_check):,}}")
print(f"NPU ops: {{summarize_npu_ops(model_for_check)}}")
print(json.dumps({{"data_cfg": data_cfg, "stage1_cfg": stage1_cfg, "stage2_cfg": stage2_cfg, "pretrain_mix": PRETRAIN_MIX}}, indent=2))

stage1_bundle = build_data_bundle(DATA_ROOT, data_cfg, BATCH_SIZE, NUM_WORKERS, device, SEED, PRETRAIN_MIX, "stage1_pretrain")
fit_stage(
    model=build_model(**MODEL_CFG),
    train_loader=stage1_bundle["train_loader"],
    train_eval_loader=stage1_bundle["train_eval_loader"],
    eval_loaders={{
        "paired_val": stage1_bundle["paired_val_loader"],
        "combined_val": stage1_bundle["combined_val_loader"],
        "coco_val": stage1_bundle["coco_val_loader"],
        "imagenet_val": stage1_bundle["imagenet_val_loader"],
    }},
    output_dir=OUTPUT_DIR / "stage1_pretrain",
    model_cfg=MODEL_CFG,
    train_cfg=stage1_cfg,
    data_cfg=data_cfg,
    device=device,
    amp_policy=amp_policy,
    stage_name="stage1_pretrain",
    channels_last=channels_last,
)

stage2_bundle = build_data_bundle(DATA_ROOT, data_cfg, BATCH_SIZE, NUM_WORKERS, device, SEED, PRETRAIN_MIX, "stage2_finetune")
fit_stage(
    model=build_model(**MODEL_CFG),
    train_loader=stage2_bundle["train_loader"],
    train_eval_loader=stage2_bundle["train_eval_loader"],
    eval_loaders={{
        "paired_val": stage2_bundle["paired_val_loader"],
        "combined_val": stage2_bundle["combined_val_loader"],
        "coco_val": stage2_bundle["coco_val_loader"],
        "imagenet_val": stage2_bundle["imagenet_val_loader"],
    }},
    output_dir=OUTPUT_DIR / "stage2_finetune",
    model_cfg=MODEL_CFG,
    train_cfg=stage2_cfg,
    data_cfg=data_cfg,
    device=device,
    amp_policy=amp_policy,
    stage_name="stage2_finetune",
    channels_last=channels_last,
    init_checkpoint_path=OUTPUT_DIR / "stage1_pretrain" / "best.pt",
)

if RUN_DIAGNOSTICS:
    best_ckpt = OUTPUT_DIR / "stage2_finetune" / "best.pt"
    diag_model = build_model(**MODEL_CFG).to(device)
    load_weights_only(diag_model, best_ckpt, map_location=device)
    if prepare_export_model is not None:
        diag_model = prepare_export_model(diag_model)
    print("Diagnostics ready from:", best_ckpt)

if RUN_ONNX_EXPORT:
    export_to_onnx(
        build_model=build_model,
        model_cfg=MODEL_CFG,
        checkpoint_path=OUTPUT_DIR / "stage2_finetune" / "best.pt",
        onnx_path=OUTPUT_DIR / "stage2_finetune" / "best.onnx",
        data_cfg=data_cfg,
        device=device,
        prepare_export_model=prepare_export_model,
        verify=True,
        sample_loader=stage2_bundle["paired_val_loader"],
    )
    print("ONNX export complete.")
"""


PORTABLE_MODEL_SOURCES = {
    "wide_se": """
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.act = nn.PReLU(num_parameters=mid)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        weights = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * weights


class SEResBlock(nn.Module):
    def __init__(self, channels, reduction=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(self.dropout(out))
        return x + out


class WideSEResNetSR(nn.Module):
    def __init__(self, num_blocks=16, channels=64, reduction=4, dropout=0.08, max_drop_path=0.10):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(*[SEResBlock(channels, reduction, dropout, drops[i]) for i in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


MODEL_CFG = {"num_blocks": 16, "channels": 64, "reduction": 4, "dropout": 0.08, "max_drop_path": 0.10}


def build_model(**model_cfg):
    return WideSEResNetSR(**model_cfg)


prepare_export_model = None
""",
    "dsdan": """
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.act = nn.PReLU(num_parameters=mid)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        weights = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * weights


class SpatialGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        return x * self.gate(self.bn(self.dw(x)))


class DSDABlock(nn.Module):
    def __init__(self, channels, reduction=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dw1 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU(num_parameters=channels)
        self.pw1 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act2 = nn.PReLU(num_parameters=channels)
        self.dw2 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.pw2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.sa = SpatialGate(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act1(self.bn1(self.dw1(x)))
        out = self.act2(self.bn2(self.pw1(out)))
        out = self.bn3(self.dw2(out))
        out = self.bn4(self.pw2(out))
        out = self.se(out)
        out = self.sa(out)
        out = self.drop_path(self.dropout(out))
        return x + out


class DSDANSR(nn.Module):
    def __init__(self, num_blocks=12, channels=128, reduction=4, dropout=0.08, max_drop_path=0.08):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(*[DSDABlock(channels, reduction, dropout, drops[i]) for i in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


MODEL_CFG = {"num_blocks": 12, "channels": 128, "reduction": 4, "dropout": 0.08, "max_drop_path": 0.08}


def build_model(**model_cfg):
    return DSDANSR(**model_cfg)


prepare_export_model = None
""",
    "repconv": """
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel = conv.weight
    bias = conv.bias
    if bias is None:
        bias = torch.zeros(conv.out_channels, device=kernel.device, dtype=kernel.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = bn.bias + (bias - bn.running_mean) * (bn.weight / std)
    return fused_kernel, fused_bias


def _fuse_identity_bn(bn: nn.BatchNorm2d, channels: int, device: torch.device, dtype: torch.dtype):
    kernel = torch.zeros((channels, channels, 3, 3), device=device, dtype=dtype)
    diag = torch.arange(channels, device=device)
    kernel[diag, diag, 1, 1] = 1.0
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = bn.bias - bn.running_mean * (bn.weight / std)
    return fused_kernel, fused_bias


def _pad_1x1_to_3x3(kernel: torch.Tensor):
    return torch.nn.functional.pad(kernel, [1, 1, 1, 1])


def _make_conv_from_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel, bias = _fuse_conv_bn_pair(conv, bn)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True).to(device=kernel.device, dtype=kernel.dtype)
    fused.weight.data.copy_(kernel)
    fused.bias.data.copy_(bias)
    return fused


class ConvBN(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class RepConvBN(nn.Module):
    def __init__(self, channels, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        else:
            self.rbr_dense = ConvBN(channels, 3)
            self.rbr_1x1 = ConvBN(channels, 1)
            self.rbr_identity = nn.BatchNorm2d(channels)
            self.reparam = None

    def forward(self, x):
        if self.reparam is not None:
            return self.reparam(x)
        return self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)

    def get_equivalent_kernel_bias(self):
        if self.reparam is not None:
            return self.reparam.weight, self.reparam.bias
        dense_kernel, dense_bias = _fuse_conv_bn_pair(self.rbr_dense.conv, self.rbr_dense.bn)
        one_kernel, one_bias = _fuse_conv_bn_pair(self.rbr_1x1.conv, self.rbr_1x1.bn)
        id_kernel, id_bias = _fuse_identity_bn(self.rbr_identity, self.channels, dense_kernel.device, dense_kernel.dtype)
        return dense_kernel + _pad_1x1_to_3x3(one_kernel) + id_kernel, dense_bias + one_bias + id_bias

    def switch_to_deploy(self):
        if self.reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=True).to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)
        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.deploy = True


class RepResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.rep = RepConvBN(channels)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act(self.rep(x))
        out = self.bn(self.conv(out))
        out = self.drop_path(self.dropout(out))
        return x + out

    def switch_to_deploy(self):
        self.rep.switch_to_deploy()
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = _make_conv_from_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class RepConvSR(nn.Module):
    def __init__(self, num_blocks=12, channels=96, dropout=0.06, max_drop_path=0.08):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_act = nn.PReLU(num_parameters=channels)
        self.body = nn.Sequential(*[RepResidualBlock(channels, dropout=dropout, drop_path=drops[i]) for i in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        feat = self.stem_act(self.stem_bn(self.stem_conv(x)))
        return x + self.tail(self.body(feat))

    def switch_to_deploy(self):
        if isinstance(self.stem_bn, nn.BatchNorm2d):
            self.stem_conv = _make_conv_from_bn(self.stem_conv, self.stem_bn)
            self.stem_bn = nn.Identity()
        for block in self.body:
            block.switch_to_deploy()
        return self


MODEL_CFG = {"num_blocks": 12, "channels": 96, "dropout": 0.06, "max_drop_path": 0.08}


def build_model(**model_cfg):
    return RepConvSR(**model_cfg)


def prepare_export_model(model):
    model.switch_to_deploy()
    return model
""",
    "large_kernel_dw": """
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


class LargeKernelDWBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, expansion=2, dropout=0.0, drop_path=0.0):
        super().__init__()
        hidden = channels * expansion
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU(num_parameters=channels)
        self.pw_expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.PReLU(num_parameters=hidden)
        self.pw_project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act1(self.bn1(self.dw(x)))
        out = self.act2(self.bn2(self.pw_expand(out)))
        out = self.bn3(self.pw_project(out))
        out = self.drop_path(self.dropout(out))
        return x + out


class LargeKernelDWSR(nn.Module):
    def __init__(self, num_blocks=14, channels=96, expansion=2, kernels=(7, 11), dropout=0.04, max_drop_path=0.06):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        kernel_schedule = [kernels[i % len(kernels)] for i in range(num_blocks)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(*[
            LargeKernelDWBlock(channels, kernel_size=kernel_schedule[i], expansion=expansion, dropout=dropout, drop_path=drops[i])
            for i in range(num_blocks)
        ])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


MODEL_CFG = {"num_blocks": 14, "channels": 96, "expansion": 2, "kernels": (7, 11), "dropout": 0.04, "max_drop_path": 0.06}


def build_model(**model_cfg):
    return LargeKernelDWSR(**model_cfg)


prepare_export_model = None
""",
    "large_kernel_se": """
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, mid, 1, bias=True)
        self.act = nn.PReLU(num_parameters=mid)
        self.fc2 = nn.Conv2d(mid, channels, 1, bias=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        weights = self.gate(self.fc2(self.act(self.fc1(self.pool(x)))))
        return x * weights


class LargeKernelSEBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, expansion=2, reduction=4, dropout=0.0, drop_path=0.0):
        super().__init__()
        hidden = channels * expansion
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU(num_parameters=channels)
        self.pw_expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.PReLU(num_parameters=hidden)
        self.pw_project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, reduction)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act1(self.bn1(self.dw(x)))
        out = self.act2(self.bn2(self.pw_expand(out)))
        out = self.se(self.bn3(self.pw_project(out)))
        out = self.drop_path(self.dropout(out))
        return x + out


class LargeKernelSESR(nn.Module):
    def __init__(self, num_blocks=14, channels=96, expansion=2, kernels=(7, 11), reduction=4, dropout=0.04, max_drop_path=0.06):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        kernel_schedule = [kernels[i % len(kernels)] for i in range(num_blocks)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(num_parameters=channels),
        )
        self.body = nn.Sequential(*[
            LargeKernelSEBlock(channels, kernel_size=kernel_schedule[i], expansion=expansion, reduction=reduction, dropout=dropout, drop_path=drops[i])
            for i in range(num_blocks)
        ])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        return x + self.tail(self.body(self.stem(x)))


MODEL_CFG = {"num_blocks": 14, "channels": 96, "expansion": 2, "kernels": (7, 11), "reduction": 4, "dropout": 0.04, "max_drop_path": 0.06}


def build_model(**model_cfg):
    return LargeKernelSESR(**model_cfg)


prepare_export_model = None
""",
    "hybrid_rep_large_kernel": """
class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if not self.training or self.drop_prob <= 0.0:
            return x
        keep = 1.0 - self.drop_prob
        mask = keep + torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * mask / keep


def _fuse_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel = conv.weight
    bias = conv.bias
    if bias is None:
        bias = torch.zeros(conv.out_channels, device=kernel.device, dtype=kernel.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = bn.bias + (bias - bn.running_mean) * (bn.weight / std)
    return fused_kernel, fused_bias


def _fuse_identity_bn(bn: nn.BatchNorm2d, channels: int, device: torch.device, dtype: torch.dtype):
    kernel = torch.zeros((channels, channels, 3, 3), device=device, dtype=dtype)
    diag = torch.arange(channels, device=device)
    kernel[diag, diag, 1, 1] = 1.0
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).reshape(-1, 1, 1, 1)
    fused_kernel = kernel * scale
    fused_bias = bn.bias - bn.running_mean * (bn.weight / std)
    return fused_kernel, fused_bias


def _pad_1x1_to_3x3(kernel: torch.Tensor):
    return torch.nn.functional.pad(kernel, [1, 1, 1, 1])


def _make_conv_from_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    kernel, bias = _fuse_conv_bn_pair(conv, bn)
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups, bias=True).to(device=kernel.device, dtype=kernel.dtype)
    fused.weight.data.copy_(kernel)
    fused.bias.data.copy_(bias)
    return fused


class ConvBN(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class RepConvBN(nn.Module):
    def __init__(self, channels, deploy=False):
        super().__init__()
        self.channels = channels
        self.deploy = deploy
        if deploy:
            self.reparam = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        else:
            self.rbr_dense = ConvBN(channels, 3)
            self.rbr_1x1 = ConvBN(channels, 1)
            self.rbr_identity = nn.BatchNorm2d(channels)
            self.reparam = None

    def forward(self, x):
        if self.reparam is not None:
            return self.reparam(x)
        return self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)

    def get_equivalent_kernel_bias(self):
        if self.reparam is not None:
            return self.reparam.weight, self.reparam.bias
        dense_kernel, dense_bias = _fuse_conv_bn_pair(self.rbr_dense.conv, self.rbr_dense.bn)
        one_kernel, one_bias = _fuse_conv_bn_pair(self.rbr_1x1.conv, self.rbr_1x1.bn)
        id_kernel, id_bias = _fuse_identity_bn(self.rbr_identity, self.channels, dense_kernel.device, dense_kernel.dtype)
        return dense_kernel + _pad_1x1_to_3x3(one_kernel) + id_kernel, dense_bias + one_bias + id_bias

    def switch_to_deploy(self):
        if self.reparam is not None:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.reparam = nn.Conv2d(self.channels, self.channels, 3, padding=1, bias=True).to(device=kernel.device, dtype=kernel.dtype)
        self.reparam.weight.data.copy_(kernel)
        self.reparam.bias.data.copy_(bias)
        del self.rbr_dense
        del self.rbr_1x1
        del self.rbr_identity
        self.deploy = True


class RepResidualBlock(nn.Module):
    def __init__(self, channels, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.rep = RepConvBN(channels)
        self.act = nn.PReLU(num_parameters=channels)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act(self.rep(x))
        out = self.bn(self.conv(out))
        out = self.drop_path(self.dropout(out))
        return x + out

    def switch_to_deploy(self):
        self.rep.switch_to_deploy()
        if isinstance(self.bn, nn.BatchNorm2d):
            self.conv = _make_conv_from_bn(self.conv, self.bn)
            self.bn = nn.Identity()


class LargeKernelDWBlock(nn.Module):
    def __init__(self, channels, kernel_size=7, expansion=2, dropout=0.0, drop_path=0.0):
        super().__init__()
        hidden = channels * expansion
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act1 = nn.PReLU(num_parameters=channels)
        self.pw_expand = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.act2 = nn.PReLU(num_parameters=hidden)
        self.pw_project = nn.Conv2d(hidden, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path)

    def forward(self, x):
        out = self.act1(self.bn1(self.dw(x)))
        out = self.act2(self.bn2(self.pw_expand(out)))
        out = self.bn3(self.pw_project(out))
        out = self.drop_path(self.dropout(out))
        return x + out


class HybridRepLargeKernelBlock(nn.Module):
    def __init__(self, channels, block_index: int, kernels=(7, 11), expansion=2, dropout=0.05, drop_path=0.0):
        super().__init__()
        if block_index % 2 == 0:
            self.block = RepResidualBlock(channels, dropout=dropout, drop_path=drop_path)
        else:
            kernel = kernels[(block_index // 2) % len(kernels)]
            self.block = LargeKernelDWBlock(channels, kernel_size=kernel, expansion=expansion, dropout=dropout, drop_path=drop_path)

    def forward(self, x):
        return self.block(x)

    def switch_to_deploy(self):
        if hasattr(self.block, "switch_to_deploy"):
            self.block.switch_to_deploy()


class HybridRepLargeKernelSR(nn.Module):
    def __init__(self, num_blocks=12, channels=96, kernels=(7, 11), expansion=2, dropout=0.05, max_drop_path=0.08):
        super().__init__()
        drops = torch.linspace(0.0, max_drop_path, num_blocks).tolist()
        self.stem_conv = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)
        self.stem_act = nn.PReLU(num_parameters=channels)
        self.body = nn.Sequential(*[
            HybridRepLargeKernelBlock(channels=channels, block_index=i, kernels=kernels, expansion=expansion, dropout=dropout, drop_path=drops[i])
            for i in range(num_blocks)
        ])
        self.tail = nn.Conv2d(channels, 3, 3, padding=1)

    def forward(self, x):
        feat = self.stem_act(self.stem_bn(self.stem_conv(x)))
        return x + self.tail(self.body(feat))

    def switch_to_deploy(self):
        if isinstance(self.stem_bn, nn.BatchNorm2d):
            self.stem_conv = _make_conv_from_bn(self.stem_conv, self.stem_bn)
            self.stem_bn = nn.Identity()
        for block in self.body:
            if hasattr(block, "switch_to_deploy"):
                block.switch_to_deploy()
        return self


MODEL_CFG = {"num_blocks": 12, "channels": 96, "kernels": (7, 11), "expansion": 2, "dropout": 0.05, "max_drop_path": 0.08}


def build_model(**model_cfg):
    return HybridRepLargeKernelSR(**model_cfg)


def prepare_export_model(model):
    model.switch_to_deploy()
    return model
""",
}


def create_portable_notebook(model_id: str, pretrain_mix: str, output_path: Path, stage1_epochs: int = 40, stage2_epochs: int = 20) -> Path:
    if model_id not in PORTABLE_MODEL_SOURCES:
        raise KeyError(f"Portable notebook source missing for {model_id}")
    code = portable_runtime_source(PORTABLE_MODEL_SOURCES[model_id], model_id, pretrain_mix)
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# Phase 6 Full Run: {model_id}\n",
                    "\n",
                    "Standalone notebook for Phase 6 full training. This notebook does not import repo-local helpers.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [PORTABLE_NOTEBOOK_SETUP],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"import os\n"
                    f"os.environ.setdefault('LAB2_PRETRAIN_MIX', '{pretrain_mix}')\n"
                    f"os.environ.setdefault('LAB2_STAGE1_EPOCHS', '{stage1_epochs}')\n"
                    f"os.environ.setdefault('LAB2_STAGE2_EPOCHS', '{stage2_epochs}')\n"
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(notebook, indent=2))
    return output_path
