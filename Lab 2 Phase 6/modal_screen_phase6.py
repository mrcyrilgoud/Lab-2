from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from phase6_screening_common import (
    DEFAULT_TIE_MARGIN,
    MIX_ORDER,
    MODEL_ORDER,
    PROMOTION_COUNT,
    REPO_ROOT,
    SCREENING_STAGE_ORDER,
    build_config_summary,
    build_leaderboard_rows,
    default_screening_state,
    get_model_spec,
    load_screening_state,
    write_leaderboard,
    write_screening_state,
)


PHASE6_DIR = Path(__file__).resolve().parent
RUNNER_PATH = PHASE6_DIR / "modal_run_phase6_screening.py"
LOCAL_SCREENING_ROOT = REPO_ROOT / "runs" / "phase6_screening"
STATE_PATH = LOCAL_SCREENING_ROOT / "screening_state.json"
LEADERBOARD_PATH = LOCAL_SCREENING_ROOT / "leaderboard.json"
DATA_VOLUME_NAME = "lab2-phase4b-data"


def parse_json_from_stdout(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stdout[start : end + 1])
    raise ValueError(f"Unable to parse JSON from subprocess output:\n{stdout}")


def run_modal_command(args: list[str]) -> dict[str, Any]:
    cmd = ["modal", "run", "-q", str(RUNNER_PATH), *args]
    print(" ".join(cmd))
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    if completed.stderr.strip():
        print(completed.stderr.strip())
    payload = parse_json_from_stdout(completed.stdout)
    print(json.dumps(payload, indent=2))
    return payload


def modal_volume_path_exists(volume_name: str, path: str) -> bool:
    cmd = ["modal", "volume", "ls", "--json", volume_name, path]
    completed = subprocess.run(cmd, text=True, capture_output=True)
    return completed.returncode == 0


def local_stage_dir(model_id: str, mix_id: str, stage_name: str, seed: int) -> Path:
    if seed == 255:
        return LOCAL_SCREENING_ROOT / model_id / mix_id / stage_name
    return LOCAL_SCREENING_ROOT / model_id / mix_id / f"{stage_name}_seed{seed}"


def write_local_summary(model_id: str, mix_id: str, stage_name: str, seed: int, payload: dict[str, Any]) -> Path:
    path = local_stage_dir(model_id, mix_id, stage_name, seed) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def ensure_coco_prep(state: dict[str, Any]) -> dict[str, Any]:
    train_ready = modal_volume_path_exists(DATA_VOLUME_NAME, "/Data/course_files_export/coco2017/train2017")
    val_ready = modal_volume_path_exists(DATA_VOLUME_NAME, "/Data/course_files_export/coco2017/val2017")
    if state.get("coco_prep_completed") and train_ready and val_ready:
        return state
    payload = run_modal_command(["--mode", "prepare_coco"])
    state["coco_prep_completed"] = True
    state["coco_prep_summary"] = payload
    write_screening_state(STATE_PATH, state)
    return state


def screening_epochs(stage_name: str) -> int:
    return 12 if stage_name == "stage1_pretrain" else 8


def run_stage(model_id: str, mix_id: str, stage_name: str, seed: int, batch_size: int, num_workers: int) -> dict[str, Any]:
    payload = run_modal_command(
        [
            "--mode",
            "run_stage",
            "--model-id",
            model_id,
            "--mix-id",
            mix_id,
            "--stage-name",
            stage_name,
            "--epochs",
            str(screening_epochs(stage_name)),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(num_workers),
            "--resume-training",
            "--seed",
            str(seed),
            "--run-diagnostics" if stage_name == "stage2_finetune" else "--no-run-diagnostics",
            "--run-onnx-export" if stage_name == "stage2_finetune" else "--no-run-onnx-export",
        ]
    )
    write_local_summary(model_id, mix_id, stage_name, seed, payload)
    return payload


def collect_all_config_summaries(state: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rows = []
    for model_id, per_model in state.get("model_summaries", {}).items():
        spec = get_model_spec(model_id)
        for mix_id in per_model.get("mixes", {}):
            rows.append(build_config_summary(LOCAL_SCREENING_ROOT, model_id, mix_id, spec, seed=seed))
    return rows


def average_summary_rows(primary: dict[str, Any], secondary: dict[str, Any], tiebreak_seed: int) -> dict[str, Any]:
    averaged = dict(primary)
    for key in ["paired_val_psnr", "combined_val_psnr", "coco_val_psnr", "imagenet_val_psnr"]:
        a = primary.get(key)
        b = secondary.get(key)
        if a is not None and b is not None:
            averaged[key] = round((float(a) + float(b)) / 2.0, 4)
    averaged["tiebreak_seed"] = tiebreak_seed
    averaged["tiebreak_source"] = secondary.get("output_dir")
    return averaged


def collect_rankable_rows(state: dict[str, Any], seed: int, include_tiebreak: bool = False) -> list[dict[str, Any]]:
    config_rows = collect_all_config_summaries(state, seed)
    if not include_tiebreak:
        return config_rows
    merged = []
    tiebreak_runs = state.get("tiebreak_runs", {})
    for row in config_rows:
        key = f"{row['model_id']}:{row['mix_id']}"
        info = tiebreak_runs.get(key)
        if not info:
            merged.append(row)
            continue
        spec = get_model_spec(row["model_id"])
        rerun = build_config_summary(LOCAL_SCREENING_ROOT, row["model_id"], row["mix_id"], spec, seed=info["seed"])
        merged.append(average_summary_rows(row, rerun, info["seed"]))
    return merged


def refresh_leaderboard(
    state: dict[str, Any],
    seed: int,
    tie_threshold: float,
    include_tiebreak: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config_rows = collect_rankable_rows(state, seed, include_tiebreak=include_tiebreak)
    leaderboard_rows, meta = build_leaderboard_rows(config_rows, tie_threshold=tie_threshold)
    meta["tiebreak_applied"] = include_tiebreak
    state["leaderboard_meta"] = meta
    write_leaderboard(LEADERBOARD_PATH, leaderboard_rows, meta)
    write_screening_state(STATE_PATH, state)
    return state, leaderboard_rows, meta


def best_mix_for_model(state: dict[str, Any], model_id: str, seed: int) -> dict[str, Any] | None:
    model_summaries = state.get("model_summaries", {}).get(model_id, {}).get("mixes", {})
    rows = []
    for mix_id in model_summaries:
        spec = get_model_spec(model_id)
        rows.append(build_config_summary(LOCAL_SCREENING_ROOT, model_id, mix_id, spec, seed=seed))
    if not rows:
        return None
    rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("paired_val_psnr") or float("-inf")),
            -(row.get("combined_val_psnr") or float("-inf")),
            row.get("params") or float("inf"),
        ),
    )
    return rows[0]


def format_resume_command(args: argparse.Namespace, next_model: str | None = None) -> str:
    cmd = [
        "python3",
        str(Path(__file__).resolve()),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--tie-threshold",
        str(args.tie_threshold),
    ]
    if next_model is not None:
        cmd.extend(["--resume-from-model", next_model])
    if args.models:
        cmd.extend(["--models", ",".join(args.models)])
    if args.mixes:
        cmd.extend(["--mixes", ",".join(args.mixes)])
    return " ".join(cmd)


def run_one_model(state: dict[str, Any], model_id: str, args: argparse.Namespace) -> dict[str, Any]:
    model_state = state.setdefault("model_summaries", {}).setdefault(model_id, {"mixes": {}})
    for mix_id in args.mixes:
        for stage_name in SCREENING_STAGE_ORDER:
            payload = run_stage(model_id, mix_id, stage_name, args.seed, args.batch_size, args.num_workers)
            model_state["mixes"].setdefault(mix_id, {})[stage_name] = payload
        spec = get_model_spec(model_id)
        config_summary = build_config_summary(LOCAL_SCREENING_ROOT, model_id, mix_id, spec, seed=args.seed)
        model_state["mixes"][mix_id]["summary"] = config_summary
        config_dir = LOCAL_SCREENING_ROOT / model_id / mix_id
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "summary.json").write_text(json.dumps(config_summary, indent=2))
    completed_models = state.setdefault("completed_models", [])
    if model_id not in completed_models:
        completed_models.append(model_id)
    write_screening_state(STATE_PATH, state)
    return state


def run_tiebreaks(state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    candidates = state.get("leaderboard_meta", {}).get("near_tie_candidates", [])
    if not candidates:
        return state
    tiebreak_seed = args.seed + 1
    tiebreak_runs = state.setdefault("tiebreak_runs", {})
    for row in candidates:
        model_id = row["model_id"]
        mix_id = row["mix_id"]
        key = f"{model_id}:{mix_id}"
        if key in tiebreak_runs:
            continue
        for stage_name in SCREENING_STAGE_ORDER:
            payload = run_stage(model_id, mix_id, stage_name, tiebreak_seed, args.batch_size, args.num_workers)
            write_local_summary(model_id, mix_id, stage_name, tiebreak_seed, payload)
        tiebreak_runs[key] = {"seed": tiebreak_seed}
        write_screening_state(STATE_PATH, state)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial Modal orchestrator for Phase 6 screening.")
    parser.add_argument("--models", type=lambda s: [item.strip() for item in s.split(",") if item.strip()], default=None)
    parser.add_argument("--resume-from-model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=255)
    parser.add_argument("--tie-threshold", type=float, default=DEFAULT_TIE_MARGIN)
    parser.add_argument("--run-tiebreaks", action="store_true")
    parser.add_argument("--mixes", type=lambda s: [item.strip() for item in s.split(",") if item.strip()], default=MIX_ORDER)
    args = parser.parse_args()

    requested_models = args.models or MODEL_ORDER
    unknown = sorted(set(requested_models) - set(MODEL_ORDER))
    if unknown:
        raise ValueError(f"Unknown model ids: {unknown}")
    unknown_mixes = sorted(set(args.mixes) - set(MIX_ORDER))
    if unknown_mixes:
        raise ValueError(f"Unknown mix ids: {unknown_mixes}")

    LOCAL_SCREENING_ROOT.mkdir(parents=True, exist_ok=True)
    state = load_screening_state(STATE_PATH, LOCAL_SCREENING_ROOT)
    if not state:
        state = default_screening_state(LOCAL_SCREENING_ROOT)
    state = ensure_coco_prep(state)

    if args.resume_from_model and args.resume_from_model not in requested_models:
        raise ValueError(f"--resume-from-model must be one of {requested_models}")

    start_index = requested_models.index(args.resume_from_model) if args.resume_from_model else 0
    for model_id in requested_models[start_index:]:
        if model_id in state.get("completed_models", []):
            continue
        state = run_one_model(state, model_id, args)
        state, leaderboard_rows, meta = refresh_leaderboard(state, args.seed, args.tie_threshold)
        best = best_mix_for_model(state, model_id, args.seed)
        next_model = next((candidate for candidate in requested_models[requested_models.index(model_id) + 1 :] if candidate not in state.get("completed_models", [])), None)

        print("\n=== MODEL SCREENING COMPLETE ===")
        if best is not None:
            print(
                json.dumps(
                    {
                        "model_id": model_id,
                        "best_mix": best["mix_id"],
                        "paired_val_psnr": best.get("paired_val_psnr"),
                        "combined_val_psnr": best.get("combined_val_psnr"),
                        "elapsed_seconds": best.get("elapsed_seconds"),
                        "gpu_type": best.get("gpu_type"),
                        "artifact_dir": best.get("output_dir"),
                        "modal_identifiers": best.get("modal_identifiers"),
                        "leaderboard_path": str(LEADERBOARD_PATH),
                    },
                    indent=2,
                )
            )
        mix_rows = []
        for mix_id in args.mixes:
            spec = get_model_spec(model_id)
            row = build_config_summary(LOCAL_SCREENING_ROOT, model_id, mix_id, spec, seed=args.seed)
            mix_rows.append(
                {
                    "mix_id": mix_id,
                    "paired_val_psnr": row.get("paired_val_psnr"),
                    "combined_val_psnr": row.get("combined_val_psnr"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "gpu_type": row.get("gpu_type"),
                    "artifact_dir": row.get("output_dir"),
                    "modal_identifiers": row.get("modal_identifiers"),
                }
            )
        print("Mix summaries:")
        print(json.dumps(mix_rows, indent=2))
        if leaderboard_rows:
            print("Top leaderboard rows:")
            print(json.dumps(leaderboard_rows[: min(5, len(leaderboard_rows))], indent=2))
        print(f"Resume command: {format_resume_command(args, next_model=next_model)}")
        return

    state, leaderboard_rows, meta = refresh_leaderboard(state, args.seed, args.tie_threshold)
    if meta.get("near_tie_candidates") and args.run_tiebreaks:
        state = run_tiebreaks(state, args)
        state, leaderboard_rows, meta = refresh_leaderboard(state, args.seed, args.tie_threshold, include_tiebreak=True)
    print("\n=== ALL PRIMARY SCREENING COMPLETE ===")
    print(json.dumps({"leaderboard_path": str(LEADERBOARD_PATH), "leaderboard_meta": meta}, indent=2))
    if meta.get("near_tie_candidates") and not args.run_tiebreaks:
        print("Tie-break reruns are required before locking finalists.")
        mix_arg = f" --mixes {','.join(args.mixes)}" if args.mixes else ""
        print(f"python3 {Path(__file__).resolve()} --run-tiebreaks --batch-size {args.batch_size} --num-workers {args.num_workers} --seed {args.seed} --tie-threshold {args.tie_threshold}{mix_arg}")
        return
    print("Generate finalist notebooks with:")
    print(f"python3 {PHASE6_DIR / 'generate_phase6_notebooks.py'} finalists --leaderboard {LEADERBOARD_PATH} --output-dir '{PHASE6_DIR}'")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "", file=sys.stdout)
        print(exc.stderr or "", file=sys.stderr)
        raise
