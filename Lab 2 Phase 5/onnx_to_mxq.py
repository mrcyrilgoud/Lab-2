from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile a Phase 5 ONNX export into MXQ with explicit calibration inputs."
    )
    parser.add_argument("--run-dir", type=Path, help="Run directory containing best.onnx and calibration/")
    parser.add_argument("--onnx", type=Path, help="Explicit ONNX path")
    parser.add_argument("--calibration-dir", type=Path, help="Explicit calibration directory")
    parser.add_argument("--out-mxq", type=Path, help="Output MXQ path")
    parser.add_argument("--compiler-bin", help="Mobilint/Qubee compiler executable")
    parser.add_argument(
        "--compiler-arg",
        action="append",
        default=[],
        help=(
            "Repeated compiler arg. Supports placeholders: {onnx}, {calibration_dir}, "
            "{manifest}, {inputs}, {out_mxq}, {run_dir}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate resolved inputs and write the sidecar log without invoking a compiler.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    if not args.run_dir and not (args.onnx and args.calibration_dir):
        raise SystemExit("Provide --run-dir or both --onnx and --calibration-dir.")

    run_dir = args.run_dir.resolve() if args.run_dir else None
    onnx_path = args.onnx.resolve() if args.onnx else None
    calibration_dir = args.calibration_dir.resolve() if args.calibration_dir else None

    if run_dir is not None:
        if onnx_path is None:
            onnx_path = run_dir / "best.onnx"
        if calibration_dir is None:
            calibration_dir = run_dir / "calibration"

    if onnx_path is None or calibration_dir is None:
        raise SystemExit("Unable to resolve ONNX or calibration directory.")

    if run_dir is None:
        run_dir = onnx_path.parent

    out_mxq = args.out_mxq.resolve() if args.out_mxq else run_dir / f"{onnx_path.stem}.mxq"
    manifest_path = calibration_dir / "manifest.json"
    inputs_path = calibration_dir / "calibration_inputs.pt"

    return {
        "run_dir": run_dir,
        "onnx_path": onnx_path,
        "calibration_dir": calibration_dir,
        "manifest_path": manifest_path,
        "inputs_path": inputs_path,
        "out_mxq": out_mxq,
    }


def validate_paths(paths: dict[str, Path]) -> dict[str, Any]:
    missing = [key for key in ("onnx_path", "manifest_path", "inputs_path") if not paths[key].exists()]
    if missing:
        details = ", ".join(f"{key}={paths[key]}" for key in missing)
        raise FileNotFoundError(f"Missing required compile inputs: {details}")

    manifest = json.loads(paths["manifest_path"].read_text())
    if not isinstance(manifest, dict):
        raise ValueError(f"Calibration manifest must be a JSON object: {paths['manifest_path']}")
    samples = manifest.get("samples", [])
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"Calibration manifest has no samples: {paths['manifest_path']}")

    paths["out_mxq"].parent.mkdir(parents=True, exist_ok=True)
    return {"sample_count": len(samples), "manifest_summary": manifest.get("summary", {})}


def resolve_command(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    if not args.compiler_bin:
        return []
    tokens = {
        "onnx": str(paths["onnx_path"]),
        "calibration_dir": str(paths["calibration_dir"]),
        "manifest": str(paths["manifest_path"]),
        "inputs": str(paths["inputs_path"]),
        "out_mxq": str(paths["out_mxq"]),
        "run_dir": str(paths["run_dir"]),
    }
    resolved_args = [arg.format(**tokens) for arg in args.compiler_arg]
    return [args.compiler_bin, *resolved_args]


def compiler_exists(compiler_bin: str) -> bool:
    expanded = Path(compiler_bin).expanduser()
    return expanded.exists() or shutil.which(compiler_bin) is not None


def sidecar_path(out_mxq: Path) -> Path:
    return out_mxq.with_name(f"{out_mxq.stem}.mxq_build.json")


def build_log(args: argparse.Namespace, paths: dict[str, Path], command: list[str], validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "dry_run": args.dry_run,
        "run_dir": str(paths["run_dir"]),
        "onnx_path": str(paths["onnx_path"]),
        "calibration_dir": str(paths["calibration_dir"]),
        "manifest_path": str(paths["manifest_path"]),
        "inputs_path": str(paths["inputs_path"]),
        "out_mxq": str(paths["out_mxq"]),
        "compiler_bin": args.compiler_bin,
        "compiler_args": args.compiler_arg,
        "resolved_command": command,
        "resolved_command_shell": shlex.join(command) if command else None,
        "validation": validation,
    }


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    validation = validate_paths(paths)
    command = resolve_command(args, paths)
    log_data = build_log(args, paths, command, validation)

    if args.dry_run:
        log_data["status"] = "dry_run"
        log_data["mxq_exists"] = paths["out_mxq"].exists()
        log_path = sidecar_path(paths["out_mxq"])
        log_path.write_text(json.dumps(log_data, indent=2))
        print(f"Dry run validated: {paths['onnx_path']}")
        print(f"Calibration samples: {validation['sample_count']}")
        print(f"Output MXQ path: {paths['out_mxq']}")
        print(f"Sidecar log: {log_path}")
        if command:
            print(f"Resolved command: {shlex.join(command)}")
        else:
            print("No compiler command resolved. Supply --compiler-bin and --compiler-arg for execution.")
        return 0

    if not args.compiler_bin:
        raise SystemExit("Non-dry-run execution requires --compiler-bin.")
    if not compiler_exists(args.compiler_bin):
        raise FileNotFoundError(f"Compiler binary not found: {args.compiler_bin}")

    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    log_data["status"] = "ok" if proc.returncode == 0 else "failed"
    log_data["returncode"] = proc.returncode
    log_data["stdout"] = proc.stdout
    log_data["stderr"] = proc.stderr
    log_data["mxq_exists"] = paths["out_mxq"].exists()

    log_path = sidecar_path(paths["out_mxq"])
    log_path.write_text(json.dumps(log_data, indent=2))

    if proc.stdout:
        print(proc.stdout.rstrip())
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    print(f"Sidecar log: {log_path}")

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
