from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable
import hashlib
import io
import json
import math
import os
import random
import shutil
import subprocess
import tarfile
import time
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
TO_TENSOR = transforms.ToTensor()
BICUBIC = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
FORBIDDEN_TYPES = (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.LayerNorm, nn.GroupNorm)

try:
    import google.colab  # type: ignore

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


def default_data_cfg() -> dict[str, Any]:
    return {
        "train_patch_size": 224,
        "eval_size": 256,
        "random_scale_pad": 32,
        "cutout_prob": 0.35,
        "cutout_ratio": 0.18,
        "lr_noise_prob": 0.30,
        "lr_noise_std": 0.015,
        "lr_blur_prob": 0.70,
        "jpeg_prob": 0.50,
        "jpeg_quality_min": 40,
        "jpeg_quality_max": 90,
        "downsample_scales": (2, 3, 4),
        "imagenet_train_limit": 6000,
        "imagenet_val_limit": 300,
        "train_eval_subset_size": 128,
    }


def default_train_cfg(batch_size: int) -> dict[str, Any]:
    return {
        "epochs": 80,
        "batch_size": batch_size,
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "warmup_epochs": 5,
        "min_lr_ratio": 0.05,
        "checkpoint_interval": 10,
        "train_eval_interval": 5,
        "seed": 255,
        "early_stop_patience": 15,
        "grad_clip_norm": 1.0,
        "ema_decay": 0.999,
        "charb_eps": 1e-6,
    }


def default_calibration_cfg(seed: int) -> dict[str, Any]:
    return {"num_samples": 128, "seed": seed, "output_subdir": "calibration"}


def configure_runtime() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        except TypeError:
            pass
        except RuntimeError:
            pass
    return AdamW(model.parameters(), **kwargs)


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


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def ensure_tar_extracted(archive_path: Path, dest_root: Path) -> None:
    if not archive_path.exists():
        return
    dest_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:*") as tf:
        tf.extractall(dest_root)


def ensure_unzipped(zip_path: Path, extracted_dir: Path) -> Path:
    if extracted_dir.exists():
        return extracted_dir
    if not zip_path.exists():
        return extracted_dir
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(zip_path.parent)
    return extracted_dir


def stage_drive_data_to_local(src_root: Path, dest_root: Path) -> Path:
    if dest_root.exists():
        return dest_root
    dest_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_root, dest_root)
    return dest_root


def resolve_lab2_workspace(output_subdir: str) -> dict[str, Path | bool]:
    data_override = os.environ.get("LAB2_DATA_ROOT")
    output_override = os.environ.get("LAB2_OUTPUT_DIR")

    if not IN_COLAB:
        data_root = Path(data_override) if data_override else REPO_ROOT / "Data"
        output_dir = Path(output_override) if output_override else MODULE_DIR / "runs" / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "repo_root": REPO_ROOT,
            "data_root": data_root,
            "output_dir": output_dir,
            "workspace_root": REPO_ROOT,
            "in_colab": False,
        }

    from google.colab import drive  # type: ignore

    drive.mount("/content/drive", force_remount=False)
    mydrive_root = Path("/content/drive/MyDrive")
    course_lab_root = first_existing(
        mydrive_root / "Data 255 Class Spring 2026" / "Lab 2",
        mydrive_root / "Data 255 Spring 2026" / "Lab 2",
    )
    fallback_sync_root = first_existing(
        mydrive_root / "Lab-2-colab",
        mydrive_root / "Colab Notebooks" / "Lab-2-colab",
    )
    sync_root = course_lab_root if course_lab_root.exists() else fallback_sync_root
    sync_root.mkdir(parents=True, exist_ok=True)

    project_root = Path(f"/content/{output_subdir}_workspace")
    local_data_root = project_root / "Data"
    if not local_data_root.exists():
        archive = first_existing(
            sync_root / "lab2_phase5_data.tar",
            sync_root / "lab2_phase5_data.tar.gz",
            sync_root / "lab2_colab_data.tar",
            sync_root / "lab2_colab_data.tar.gz",
        )
        drive_data_root = sync_root / "Data"
        if archive.exists():
            ensure_tar_extracted(archive, project_root)
        elif drive_data_root.exists():
            stage_drive_data_to_local(drive_data_root, local_data_root)
        else:
            raise FileNotFoundError(
                f"Missing data for Colab. Expected Data or data tarball under {sync_root}"
            )

    data_root = Path(data_override) if data_override else local_data_root
    output_dir = Path(output_override) if output_override else sync_root / "runs" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": sync_root,
        "data_root": data_root,
        "output_dir": output_dir,
        "workspace_root": project_root,
        "in_colab": True,
    }


def resolve_image_root(data_root: Path) -> Path:
    candidates = [data_root / "course_files_export", data_root / "ImageNet"]
    for candidate in candidates:
        if (candidate / "imagenet_train20.txt").exists() and (candidate / "imagenet_val20.txt").exists():
            return candidate
    return candidates[0]


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
    missing = 0
    for filename, class_id in rows:
        synset = filename.split("_")[0]
        path = (root / synset / filename) if split == "train" else (root / filename)
        if not path.exists():
            missing += 1
            continue
        records.append(
            {
                "path": path,
                "stem": path.stem,
                "class_id": class_id,
                "split": split,
                "synset": synset,
            }
        )
    if missing:
        print(f"WARNING: skipped {missing} missing ImageNet {split} files")
    return records


def seeded_rng(key: str) -> random.Random:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def take_manifest_subset(records: list[Any], limit: int | None, seed: int) -> list[Any]:
    if limit is None or limit >= len(records):
        return list(records)
    rng = random.Random(seed)
    items = list(records)
    rng.shuffle(items)
    return items[:limit]


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
        lr_img = lr_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 1.2)))
    scale = rng.choice(cfg["downsample_scales"])
    small = (max(1, hr_img.width // scale), max(1, hr_img.height // scale))
    lr_img = lr_img.resize(small, resample=BICUBIC).resize(hr_img.size, resample=BICUBIC)
    if rng.random() < cfg["jpeg_prob"]:
        lr_img = jpeg_roundtrip(lr_img, rng.randint(cfg["jpeg_quality_min"], cfg["jpeg_quality_max"]))
    return lr_img


def apply_tensor_regularization(lr_t: torch.Tensor, rng: random.Random, cfg: dict[str, Any], train: bool) -> torch.Tensor:
    if not train:
        return lr_t
    if rng.random() < cfg["lr_noise_prob"]:
        lr_t = (lr_t + torch.randn_like(lr_t) * cfg["lr_noise_std"]).clamp(0.0, 1.0)
    if rng.random() < cfg["cutout_prob"]:
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


class ImageNetSyntheticSRDataset(Dataset):
    def __init__(self, records, train: bool, data_cfg: dict[str, Any], source_name: str, seed: int):
        self.records = records
        self.train = train
        self.data_cfg = data_cfg
        self.source_name = source_name
        self.seed = seed

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        hr_img = Image.open(record["path"]).convert("RGB")
        rng = random.Random(self.seed + idx) if self.train else seeded_rng(f"{self.source_name}:{record['stem']}")
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
        return lr_t, hr_t, record["stem"], self.source_name


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


def build_data_bundle(
    workspace: dict[str, Path | bool],
    data_cfg: dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    data_root = Path(workspace["data_root"])
    course_export_root = data_root / "course_files_export"
    legacy_image_root = data_root / "ImageNet"

    hr_train_root = data_root / "HR_train"
    lr_train_root = data_root / "LR_train"
    hr_val_dir = first_existing(data_root / "HR_val", data_root / "val" / "HR_val")
    lr_val_dir = first_existing(data_root / "LR_val", data_root / "val" / "LR_val")
    imagenet_train_list = first_existing(
        course_export_root / "imagenet_train20.txt",
        legacy_image_root / "imagenet_train20.txt",
    )
    imagenet_val_list = first_existing(
        course_export_root / "imagenet_val20.txt",
        legacy_image_root / "imagenet_val20.txt",
    )
    imagenet_train_root = ensure_unzipped(
        course_export_root / "imagenet_train20.zip",
        first_existing(course_export_root / "imagenet_train20a", legacy_image_root / "imagenet_train20a"),
    )
    imagenet_val_root = ensure_unzipped(
        course_export_root / "imagenet_val20.zip",
        first_existing(course_export_root / "imagenet_val20", legacy_image_root / "imagenet_val20"),
    )

    required_paths = [
        hr_train_root,
        lr_train_root,
        hr_val_dir,
        lr_val_dir,
        imagenet_train_root,
        imagenet_val_root,
        imagenet_train_list,
        imagenet_val_list,
    ]
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required data paths under {data_root}: {joined}")

    train_pairs = collect_paired_by_subfolder(lr_train_root, hr_train_root)
    val_pairs = collect_paired_flat(lr_val_dir, hr_val_dir)
    if not train_pairs:
        raise FileNotFoundError(
            f"No paired training PNGs found under {hr_train_root} and {lr_train_root}."
        )
    if not val_pairs:
        raise FileNotFoundError(
            f"No paired validation PNGs found under {hr_val_dir} and {lr_val_dir}."
        )

    imagenet_train_rows = read_imagenet_manifest(imagenet_train_list)
    imagenet_val_rows = read_imagenet_manifest(imagenet_val_list)
    imagenet_train_records = collect_imagenet_records(imagenet_train_rows, imagenet_train_root, split="train")
    imagenet_val_records = collect_imagenet_records(imagenet_val_rows, imagenet_val_root, split="val")

    imagenet_train_used = take_manifest_subset(imagenet_train_records, data_cfg["imagenet_train_limit"], seed)
    imagenet_val_used = take_manifest_subset(imagenet_val_records, data_cfg["imagenet_val_limit"], seed)

    paired_train_dataset = PairedSRDataset(train_pairs, train=True, data_cfg=data_cfg, source_name="paired_train", seed=seed)
    paired_train_eval_dataset = PairedSRDataset(train_pairs, train=False, data_cfg=data_cfg, source_name="paired_train", seed=seed)
    paired_val_dataset = PairedSRDataset(val_pairs, train=False, data_cfg=data_cfg, source_name="paired_val", seed=seed)
    imagenet_train_dataset = ImageNetSyntheticSRDataset(imagenet_train_used, train=True, data_cfg=data_cfg, source_name="imagenet_train", seed=seed)
    imagenet_train_eval_dataset = ImageNetSyntheticSRDataset(imagenet_train_used, train=False, data_cfg=data_cfg, source_name="imagenet_train", seed=seed)
    imagenet_val_dataset = ImageNetSyntheticSRDataset(imagenet_val_used, train=False, data_cfg=data_cfg, source_name="imagenet_val", seed=seed)

    train_dataset = ConcatDataset([paired_train_dataset, imagenet_train_dataset])
    train_eval_dataset = ConcatDataset([paired_train_eval_dataset, imagenet_train_eval_dataset])
    val_dataset = ConcatDataset([paired_val_dataset, imagenet_val_dataset])

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs(num_workers, pin_memory),
    )
    train_eval_loader = make_fixed_subset_loader(
        train_eval_dataset,
        data_cfg["train_eval_subset_size"],
        batch_size,
        seed,
        num_workers,
        pin_memory,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    paired_val_loader = DataLoader(paired_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))
    imagenet_val_loader = DataLoader(imagenet_val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs(num_workers, pin_memory))

    return {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "imagenet_train_records": imagenet_train_records,
        "imagenet_val_records": imagenet_val_records,
        "train_dataset": train_dataset,
        "train_eval_dataset": train_eval_dataset,
        "val_dataset": val_dataset,
        "train_loader": train_loader,
        "train_eval_loader": train_eval_loader,
        "val_loader": val_loader,
        "paired_val_loader": paired_val_loader,
        "imagenet_val_loader": imagenet_val_loader,
        "calibration_datasets": {
            "paired_train": paired_train_eval_dataset,
            "imagenet_train": imagenet_train_eval_dataset,
        },
    }


def print_data_summary(bundle: dict[str, Any]) -> None:
    lr_batch, hr_batch, _, _ = next(iter(bundle["train_loader"]))
    print(f"Combined train: {len(bundle['train_dataset'])}")
    print(f"Combined val:   {len(bundle['val_dataset'])}")
    print(f"Train-eval subset: {len(bundle['train_eval_loader'].dataset)}")
    print(f"Batch shapes: LR={tuple(lr_batch.shape)}, HR={tuple(hr_batch.shape)}")


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
        elif isinstance(module, nn.InstanceNorm2d):
            ops["InstanceNorm"] += 1
        elif isinstance(module, nn.PReLU):
            ops["PReLU"] += 1
        elif isinstance(module, nn.Hardsigmoid):
            ops["HardSigmoid"] += 1
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            ops["GlobalAvgPool"] += 1
        elif isinstance(module, nn.Mish):
            ops["Mish"] += 1
        elif isinstance(module, nn.Hardswish):
            ops["HardSwish"] += 1
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
        self.shadow = {}
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
            if param.requires_grad:
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
    return {f"{split_name}_loss": total_loss / max(1, n), f"{split_name}_psnr": total_psnr / max(1, n)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    best_psnr: float,
    best_epoch: int,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    amp_policy: dict[str, Any],
    ema: EMA | None = None,
    scaler=None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "best_psnr": best_psnr,
        "best_epoch": best_epoch,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "data_cfg": data_cfg,
        "amp_policy": amp_policy,
    }
    if ema is not None:
        state["ema_shadow"] = ema.shadow
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


def load_history(metrics_path: Path) -> list[dict[str, Any]]:
    if not metrics_path.exists():
        return []
    return [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]


def load_checkpoint(model: nn.Module, checkpoint_path: Path, map_location: str | torch.device = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return ckpt


def should_run_train_eval(epoch_num: int, train_cfg: dict[str, Any]) -> bool:
    interval = train_cfg.get("train_eval_interval", 5)
    if epoch_num == train_cfg["epochs"]:
        return True
    if interval > 0 and epoch_num % interval == 0:
        return True
    checkpoint_interval = train_cfg.get("checkpoint_interval", 10)
    return checkpoint_interval > 0 and epoch_num % checkpoint_interval == 0


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Path,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    device: torch.device,
    amp_policy: dict[str, Any],
    channels_last: bool = True,
    resume: bool = True,
) -> list[dict[str, Any]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    last_ckpt_path = output_dir / "last.pt"
    best_ckpt_path = output_dir / "best.pt"

    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    optimizer = optimizer_with_fallback(model, train_cfg)
    ema = EMA(model, decay=train_cfg["ema_decay"])
    scaler = make_grad_scaler(amp_policy)
    start_epoch, best_psnr, best_epoch = 0, float("-inf"), -1
    history = []

    if resume and last_ckpt_path.exists():
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        best_psnr = ckpt.get("best_psnr", float("-inf"))
        best_epoch = ckpt.get("best_epoch", -1)
        history = load_history(metrics_path)
        if "ema_shadow" in ckpt:
            ema.shadow = ckpt["ema_shadow"]
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if history and best_epoch < 0:
            best_epoch = int(max(history, key=lambda row: row["val_psnr"])["epoch"])
        print(f"RESUMING from epoch {start_epoch}/{train_cfg['epochs']}")
        print(f"  Best PSNR so far: {best_psnr:.3f} dB (epoch {best_epoch})")
    else:
        set_seed(train_cfg["seed"])
        metrics_path.write_text("")
        print("FRESH START")

    if start_epoch >= train_cfg["epochs"]:
        print(f"Already trained {start_epoch} epochs. Increase epochs to continue.")
        return history

    print(f"\n{'epoch':>5} | {'lr':>8} | {'train_loss':>10} {'train_online':>12} {'train_eval':>11} | {'val_psnr':>8} | {'best':>8} | {'time':>7}")
    print("-" * 98)

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
        run_train_eval = should_run_train_eval(epoch_num, train_cfg)
        train_eval_metrics: dict[str, Any] = {}
        if run_train_eval:
            train_eval_metrics = evaluate_loader(
                model,
                train_eval_loader,
                train_cfg,
                device,
                amp_policy,
                "train_eval",
                channels_last=channels_last,
            )
        val_metrics = evaluate_loader(
            model,
            val_loader,
            train_cfg,
            device,
            amp_policy,
            "val",
            channels_last=channels_last,
        )
        ema.restore(model)

        seconds = round(time.time() - start_time, 1)
        row: dict[str, Any] = {"epoch": epoch_num, "lr": epoch_lr, **train_metrics, **val_metrics, "seconds": seconds}
        row.update(train_eval_metrics)

        history.append(row)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

        val_psnr = row["val_psnr"]
        improved = val_psnr > best_psnr
        if improved:
            best_psnr = val_psnr
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
            best_psnr,
            best_epoch,
            model_cfg,
            train_cfg,
            data_cfg,
            amp_policy,
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
                best_psnr,
                best_epoch,
                model_cfg,
                train_cfg,
                data_cfg,
                amp_policy,
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
                best_psnr,
                best_epoch,
                model_cfg,
                train_cfg,
                data_cfg,
                amp_policy,
                ema=ema,
                scaler=scaler,
            )

        train_eval_str = f"{row['train_eval_psnr']:.3f}" if "train_eval_psnr" in row else "     -"
        mark = "*" if improved else " "
        print(
            f"{epoch_num:5d} | {epoch_lr:8.2e} | {row['train_loss']:10.6f} {row['train_psnr_online']:12.3f} {train_eval_str:>11} | "
            f"{row['val_psnr']:8.3f} | {best_psnr:8.3f} | {seconds:6.1f}s {mark}"
        )

        if should_stop:
            print(f"Early stopping after {train_cfg['early_stop_patience']} epochs without improvement.")
            break

    print(f"\nTraining complete. Best epoch {best_epoch} with val_psnr={best_psnr:.3f} dB")
    print(f"Checkpoints: {output_dir}")
    return history


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
    print(
        f"  p10={torch.quantile(psnrs, 0.1):.3f} | "
        f"p50={torch.quantile(psnrs, 0.5):.3f} | "
        f"p90={torch.quantile(psnrs, 0.9):.3f}"
    )
    for row in sorted(records, key=lambda item: item["pred_psnr"])[:3]:
        print(f"  hardest: {row['source']}:{row['name']} -> {row['pred_psnr']:.3f} dB")


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
    ckpt = load_checkpoint(diag_model, best_ckpt, map_location=device)
    if "ema_shadow" in ckpt:
        for name, param in diag_model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name])
    if prepare_export_model is not None:
        diag_model = prepare_export_model(diag_model)
    summarize_records("train_eval", collect_psnr_records(diag_model, data_bundle["train_eval_loader"], device, max_items=128))
    summarize_records("paired_val", collect_psnr_records(diag_model, data_bundle["paired_val_loader"], device))
    summarize_records("imagenet_val", collect_psnr_records(diag_model, data_bundle["imagenet_val_loader"], device))
    summarize_records("combined_val", collect_psnr_records(diag_model, data_bundle["val_loader"], device))


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
    ckpt = load_checkpoint(export_model, checkpoint_path, map_location=device)
    if "ema_shadow" in ckpt:
        for name, param in export_model.named_parameters():
            if name in ckpt["ema_shadow"]:
                param.data.copy_(ckpt["ema_shadow"][name])
        print("Loaded EMA weights for export")
    if prepare_export_model is not None:
        export_model = prepare_export_model(export_model)
    export_model.eval()

    if "metrics" in ckpt:
        print(
            f"Checkpoint epoch {ckpt['metrics'].get('epoch', '?')}, "
            f"val_psnr={ckpt['metrics'].get('val_psnr', '?'):.3f} dB"
        )

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

    print(f"Exported to {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB)")
    if onnx is not None:
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        print("ONNX checker: passed")

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


def slugify_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in cleaned.split("_") if part)[:80] or "sample"


@torch.no_grad()
def score_lr_tensor(lr_t: torch.Tensor) -> dict[str, float]:
    gray = lr_t.float().mean(dim=0)
    brightness = float(gray.mean().item())
    contrast = float(gray.std().item())
    grad_x = gray[:, 1:] - gray[:, :-1]
    grad_y = gray[1:, :] - gray[:-1, :]
    texture = float(0.5 * (grad_x.abs().mean().item() + grad_y.abs().mean().item()))
    return {"brightness": brightness, "contrast": contrast, "texture": texture}


@torch.no_grad()
def collect_calibration_candidates(dataset_map: dict[str, Dataset]) -> list[dict[str, Any]]:
    records = []
    for dataset_key, dataset in dataset_map.items():
        for dataset_index in range(len(dataset)):
            lr_t, _, name, source = dataset[dataset_index]
            records.append(
                {"dataset_key": dataset_key, "dataset_index": dataset_index, "name": name, "source": source, **score_lr_tensor(lr_t)}
            )
    return records


def assign_tertile_bins(records: list[dict[str, Any]], metric: str) -> None:
    vals = torch.tensor([row[metric] for row in records], dtype=torch.float32)
    lo, hi = torch.quantile(vals, 1 / 3).item(), torch.quantile(vals, 2 / 3).item()
    for row in records:
        row[f"{metric}_bin"] = "low" if row[metric] <= lo else ("high" if row[metric] >= hi else "mid")


def select_diverse_calibration_subset(records: list[dict[str, Any]], num_samples: int, seed: int) -> list[dict[str, Any]]:
    if not records:
        return []
    tagged = [dict(row) for row in records]
    assign_tertile_bins(tagged, "brightness")
    assign_tertile_bins(tagged, "texture")
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in tagged:
        buckets[(row["source"], row["brightness_bin"], row["texture_bin"])].append(row)
    keys = sorted(buckets)
    for key in keys:
        rng.shuffle(buckets[key])

    target = min(num_samples, len(tagged))
    selected, seen = [], set()
    while len(selected) < target:
        progress = False
        for key in keys:
            while buckets[key]:
                row = buckets[key].pop()
                row_id = (row["dataset_key"], row["dataset_index"])
                if row_id in seen:
                    continue
                seen.add(row_id)
                selected.append(row)
                progress = True
                break
            if len(selected) >= target:
                break
        if not progress:
            break
    return selected


def export_calibration_artifacts(records: list[dict[str, Any]], dataset_map: dict[str, Dataset], output_dir: Path, cfg: dict[str, Any]) -> Path:
    cal_dir = Path(output_dir) / cfg["output_subdir"]
    cal_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    batch = []
    for row in records:
        lr_t, _, name, source = dataset_map[row["dataset_key"]][row["dataset_index"]]
        batch.append(lr_t)
        img_path = cal_dir / f"{len(manifest):03d}_{slugify_name(source)}_{slugify_name(name)}.png"
        transforms.ToPILImage()(lr_t).save(img_path)
        manifest.append(
            {
                "dataset_key": row["dataset_key"],
                "dataset_index": row["dataset_index"],
                "name": name,
                "source": source,
                "image_path": str(img_path),
                "brightness": row["brightness"],
                "contrast": row["contrast"],
                "texture": row["texture"],
            }
        )
    tensor_path = cal_dir / "calibration_inputs.pt"
    torch.save(torch.stack(batch), tensor_path)
    summary = {
        "num_samples": len(manifest),
        "brightness_mean": sum(row["brightness"] for row in records) / max(1, len(records)),
        "texture_mean": sum(row["texture"] for row in records) / max(1, len(records)),
    }
    (cal_dir / "manifest.json").write_text(json.dumps({"config": cfg, "summary": summary, "samples": manifest}, indent=2))
    print(f"Calibration: {len(records)} samples exported to {cal_dir}")
    print(f"Calibration manifest: {cal_dir / 'manifest.json'}")
    print(f"Calibration tensor batch: {tensor_path}")
    return cal_dir


def export_default_calibration(data_bundle: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> Path:
    candidates = collect_calibration_candidates(data_bundle["calibration_datasets"])
    selected = select_diverse_calibration_subset(candidates, cfg["num_samples"], cfg["seed"])
    return export_calibration_artifacts(selected, data_bundle["calibration_datasets"], output_dir, cfg)


def print_runtime_summary(device: torch.device, amp_policy: dict[str, Any], batch_size: int, num_workers: int, channels_last: bool) -> None:
    print(f"Using device: {device}")
    print(f"AMP policy: {amp_policy['label']}")
    print(f"Batch size: {batch_size} | Num workers: {num_workers}")
    print(f"Channels last: {channels_last and device.type == 'cuda'}")
