from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


RUNNER_PATH = Path(__file__).with_name("modal_run_phase4b.py")


def chunk_targets(total_epochs: int, chunk_size: int) -> list[int]:
    targets = []
    current = chunk_size
    while current < total_epochs:
        targets.append(current)
        current += chunk_size
    targets.append(total_epochs)
    return targets


def run_chunk(
    target_epochs: int,
    batch_size: int,
    num_workers: int,
    output_subdir: str,
    resume_training: bool,
    run_postprocessing: bool,
) -> None:
    cmd = [
        "modal",
        "run",
        "-q",
        str(RUNNER_PATH),
        "--epochs",
        str(target_epochs),
        "--no-run-smoke-test",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--output-subdir",
        output_subdir,
    ]

    cmd.append("--resume-training" if resume_training else "--no-resume-training")
    cmd.append("--run-postprocessing" if run_postprocessing else "--no-run-postprocessing")

    print(f"\n=== Starting chunk to epoch {target_epochs} | resume={resume_training} | post={run_postprocessing} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full Phase 4B training on Modal in resumable chunks.")
    parser.add_argument("--total-epochs", type=int, default=80)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-subdir", type=str, default="phase4b_dsdan_full_train")
    args = parser.parse_args()

    targets = chunk_targets(args.total_epochs, args.chunk_size)
    print(f"Chunk targets: {targets}")
    print(f"Runner: {RUNNER_PATH}")

    for idx, target in enumerate(targets):
        run_chunk(
            target_epochs=target,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            output_subdir=args.output_subdir,
            resume_training=(idx > 0),
            run_postprocessing=(idx == len(targets) - 1),
        )

    print("\nAll chunks completed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"\nChunk failed with exit code {exc.returncode}", file=sys.stderr)
        raise
