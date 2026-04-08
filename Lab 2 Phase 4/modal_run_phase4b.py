from __future__ import annotations

import json
import os
from pathlib import Path

import modal


APP_NAME = "lab2-phase4b-dsdan"
DATA_VOLUME_NAME = os.environ.get("MODAL_LAB2_DATA_VOLUME", "lab2-phase4b-data")
RUNS_VOLUME_NAME = os.environ.get("MODAL_LAB2_RUNS_VOLUME", "lab2-phase4b-runs")

LOCAL_NOTEBOOK = Path(__file__).with_name("lab2_phase4b_dw_attention_net.ipynb")
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_NOTEBOOK = REMOTE_PROJECT_DIR / LOCAL_NOTEBOOK.name
REMOTE_RUNS_ROOT = Path("/mnt/runs")

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
    .add_local_file(LOCAL_NOTEBOOK, str(REMOTE_NOTEBOOK), copy=True)
)

app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_volume = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=["A100", "L40S"],
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 24,
    volumes={"/mnt/data": data_volume, "/mnt/runs": runs_volume},
)
def run_notebook(
    epochs: int = 1,
    run_smoke_test: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    resume_training: bool = False,
    output_subdir: str = "phase4b_dsdan",
    run_postprocessing: bool = True,
):
    import nbformat
    from nbclient import NotebookClient

    os.environ["LAB2_DATA_ROOT"] = "/mnt/data"
    remote_output_dir = REMOTE_RUNS_ROOT / output_subdir
    os.environ["LAB2_OUTPUT_DIR"] = str(remote_output_dir)
    os.environ["LAB2_EPOCHS"] = str(epochs)
    os.environ["LAB2_RUN_SMOKE_TEST"] = "1" if run_smoke_test else "0"
    os.environ["LAB2_BATCH_SIZE"] = str(batch_size)
    os.environ["LAB2_NUM_WORKERS"] = str(num_workers)
    os.environ["LAB2_RESUME_TRAINING"] = "1" if resume_training else "0"
    os.environ["LAB2_RUN_DIAGNOSTICS"] = "1" if run_postprocessing else "0"
    os.environ["LAB2_RUN_ONNX_EXPORT"] = "1" if run_postprocessing else "0"
    os.environ["LAB2_RUN_CALIBRATION_EXPORT"] = "1" if run_postprocessing else "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    remote_output_dir.mkdir(parents=True, exist_ok=True)
    executed_path = remote_output_dir / f"executed_e{epochs}_smoke{int(run_smoke_test)}.ipynb"

    with REMOTE_NOTEBOOK.open() as f:
        notebook = nbformat.read(f, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=None,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(REMOTE_PROJECT_DIR)}},
    )
    executed = client.execute()

    with executed_path.open("w") as f:
        nbformat.write(executed, f)

    metrics_path = remote_output_dir / "metrics.jsonl"
    best_ckpt = remote_output_dir / "best.pt"
    best_onnx = remote_output_dir / "best.onnx"
    calibration_manifest = remote_output_dir / "calibration" / "manifest.json"

    metrics = []
    if metrics_path.exists():
        metrics = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]

    summary = {
        "executed_notebook": str(executed_path),
        "output_dir": str(remote_output_dir),
        "epochs": epochs,
        "run_smoke_test": run_smoke_test,
        "run_postprocessing": run_postprocessing,
        "metrics_rows": len(metrics),
        "last_metric": metrics[-1] if metrics else None,
        "best_ckpt_exists": best_ckpt.exists(),
        "best_onnx_exists": best_onnx.exists(),
        "calibration_manifest_exists": calibration_manifest.exists(),
    }

    runs_volume.commit()
    return summary


@app.local_entrypoint()
def main(
    epochs: int = 1,
    run_smoke_test: bool = False,
    batch_size: int = 1,
    num_workers: int = 0,
    resume_training: bool = False,
    output_subdir: str = "phase4b_dsdan",
    run_postprocessing: bool = True,
):
    result = run_notebook.remote(
        epochs=epochs,
        run_smoke_test=run_smoke_test,
        batch_size=batch_size,
        num_workers=num_workers,
        resume_training=resume_training,
        output_subdir=output_subdir,
        run_postprocessing=run_postprocessing,
    )
    print(json.dumps(result, indent=2))
