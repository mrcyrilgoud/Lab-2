from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = "lab2-phase6-screening"
DATA_VOLUME_NAME = os.environ.get("MODAL_LAB2_DATA_VOLUME", "lab2-phase4b-data")
RUNS_VOLUME_NAME = os.environ.get("MODAL_LAB2_RUNS_VOLUME", "lab2-phase6-runs")

LOCAL_PHASE6_DIR = Path(__file__).resolve().parent
LOCAL_NOTEBOOK = LOCAL_PHASE6_DIR / "lab2_phase6a_screening_matrix.ipynb"
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_PHASE6_DIR = REMOTE_PROJECT_DIR / "Lab 2 Phase 6"
REMOTE_NOTEBOOK = REMOTE_PHASE6_DIR / LOCAL_NOTEBOOK.name
REMOTE_RUNS_ROOT = Path("/mnt/runs") / "phase6_screening"
REMOTE_DATA_ROOT = Path("/mnt/data") / "Data"

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        add_python="3.11",
    )
    .pip_install(
        "torchvision==0.20.1",
        "onnx==1.17.0",
        "onnxruntime==1.20.1",
        "nbformat==5.10.4",
        "nbclient==0.10.2",
        "ipykernel==6.29.5",
    )
    .add_local_dir(LOCAL_PHASE6_DIR, str(REMOTE_PHASE6_DIR), copy=True)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 4,
    volumes={"/mnt/data": data_volume, "/mnt/runs": runs_volume},
)
def prepare_coco_assets():
    import sys

    sys.path.insert(0, str(REMOTE_PHASE6_DIR))
    from phase6_screening_common import current_modal_identifiers, stage_coco2017

    data_root = REMOTE_DATA_ROOT
    info = stage_coco2017(data_root, download_missing=True)
    data_volume.commit()
    return {
        "mode": "prepare_coco",
        "data_root": str(data_root),
        "data_volume_name": DATA_VOLUME_NAME,
        "runs_volume_name": RUNS_VOLUME_NAME,
        "coco_info": info,
        "modal_identifiers": current_modal_identifiers(),
    }


@app.function(
    image=image,
    gpu=["L40S", "A100"],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 24,
    volumes={"/mnt/data": data_volume, "/mnt/runs": runs_volume},
)
def run_screening_stage(
    model_id: str,
    mix_id: str,
    stage_name: str,
    epochs: int,
    batch_size: int = 4,
    num_workers: int = 2,
    resume_training: bool = True,
    run_diagnostics: bool = True,
    run_onnx_export: bool = False,
    seed: int = 255,
):
    import nbformat
    import sys
    import torch
    from nbclient import NotebookClient

    sys.path.insert(0, str(REMOTE_PHASE6_DIR))
    from phase6_screening_common import current_modal_identifiers, stage_output_dir, summarize_stage_run

    os.environ["LAB2_DATA_ROOT"] = str(REMOTE_DATA_ROOT)
    remote_output_dir = stage_output_dir(REMOTE_RUNS_ROOT, model_id, mix_id, stage_name, seed=seed)
    remote_output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["LAB2_OUTPUT_DIR"] = str(remote_output_dir)
    os.environ["LAB2_MODEL_ID"] = model_id
    os.environ["LAB2_PRETRAIN_MIX"] = mix_id
    os.environ["LAB2_STAGE"] = stage_name
    os.environ["LAB2_STAGE_EPOCHS"] = str(epochs)
    os.environ["LAB2_BATCH_SIZE"] = str(batch_size)
    os.environ["LAB2_NUM_WORKERS"] = str(num_workers)
    os.environ["LAB2_RESUME_TRAINING"] = "1" if resume_training else "0"
    os.environ["LAB2_RUN_DIAGNOSTICS"] = "1" if run_diagnostics else "0"
    os.environ["LAB2_RUN_ONNX_EXPORT"] = "1" if run_onnx_export else "0"
    os.environ["LAB2_SEED"] = str(seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if stage_name == "stage2_finetune":
        os.environ["LAB2_INIT_CHECKPOINT"] = str(stage_output_dir(REMOTE_RUNS_ROOT, model_id, mix_id, "stage1_pretrain", seed=seed) / "best.pt")
    else:
        os.environ.pop("LAB2_INIT_CHECKPOINT", None)

    executed_path = remote_output_dir / f"executed_{stage_name}_e{epochs}.ipynb"
    with REMOTE_NOTEBOOK.open() as f:
        notebook = nbformat.read(f, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REMOTE_PHASE6_DIR)}},
    )
    executed = client.execute()
    with executed_path.open("w") as f:
        nbformat.write(executed, f)

    summary = summarize_stage_run(remote_output_dir)
    summary.update(
        {
            "mode": "run_stage",
            "model_id": model_id,
            "mix_id": mix_id,
            "stage_name": stage_name,
            "epochs": epochs,
            "seed": seed,
            "executed_notebook": str(executed_path),
            "output_dir": str(remote_output_dir),
            "data_volume_name": DATA_VOLUME_NAME,
            "runs_volume_name": RUNS_VOLUME_NAME,
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "modal_identifiers": current_modal_identifiers(),
        }
    )

    (remote_output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    runs_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    mode: str = "run_stage",
    model_id: str = "wide_se",
    mix_id: str = "coco_only",
    stage_name: str = "stage1_pretrain",
    epochs: int = 12,
    batch_size: int = 4,
    num_workers: int = 2,
    resume_training: bool = True,
    run_diagnostics: bool = True,
    run_onnx_export: bool = False,
    seed: int = 255,
):
    if mode == "prepare_coco":
        result = prepare_coco_assets.remote()
    elif mode == "run_stage":
        result = run_screening_stage.remote(
            model_id=model_id,
            mix_id=mix_id,
            stage_name=stage_name,
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            resume_training=resume_training,
            run_diagnostics=run_diagnostics,
            run_onnx_export=run_onnx_export,
            seed=seed,
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    print(json.dumps(result, indent=2))
