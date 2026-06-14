"""Run a YOLO training command and auto-resume after abnormal exits."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a YOLO training job and resume it after crashes.")
    parser.add_argument("--yolo", default=".venv/bin/yolo", help="Path to the yolo executable.")
    parser.add_argument("--model", default="yolo26n.pt", help="Initial model checkpoint.")
    parser.add_argument("--data", default="datasets/doubao_synthetic_yolo/data.yaml", help="YOLO data.yaml.")
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--project", default="runs/detect", help="YOLO project directory.")
    parser.add_argument("--name", default="doubao_synth_yolo26n", help="YOLO run name.")
    parser.add_argument("--device", default=None, help="Optional YOLO device, for example 0 or cpu.")
    parser.add_argument("--workers", default=None, help="Optional YOLO workers value.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--max-restarts", type=int, default=20, help="Maximum abnormal-exit restarts.")
    parser.add_argument("--restart-delay", type=int, default=30, help="Seconds to wait before restarting.")
    parser.add_argument(
        "--extra",
        default="",
        help="Extra YOLO args as one string, for example: \"cache=True cos_lr=True\".",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.project) / args.name
    restarts = 0

    while True:
        resume_checkpoint = run_dir / "weights" / "last.pt"
        resume = resume_checkpoint.is_file()
        command = build_command(args, resume_checkpoint if resume else None)

        print("=" * 80, flush=True)
        print(time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
        print("Starting YOLO training command:", flush=True)
        print(" ".join(shlex.quote(part) for part in command), flush=True)
        print("=" * 80, flush=True)

        started = time.time()
        process = subprocess.run(command)
        elapsed = time.time() - started

        if process.returncode == 0:
            print(f"Training finished normally after {elapsed / 60:.1f} minutes.", flush=True)
            return 0

        restarts += 1
        print(
            f"Training exited abnormally with code {process.returncode} after {elapsed / 60:.1f} minutes. "
            f"Restart {restarts}/{args.max_restarts}.",
            flush=True,
        )
        if restarts > args.max_restarts:
            print("Max restarts exceeded; giving up.", flush=True)
            return process.returncode
        if not resume_checkpoint.is_file():
            print(f"Cannot resume because checkpoint does not exist yet: {resume_checkpoint}", flush=True)
            return process.returncode

        print(f"Waiting {args.restart_delay} seconds before resume...", flush=True)
        time.sleep(args.restart_delay)


def build_command(args: argparse.Namespace, resume_checkpoint: Path | None) -> list[str]:
    command = [
        args.yolo,
        "detect",
        "train",
        f"model={resume_checkpoint if resume_checkpoint else args.model}",
        f"data={args.data}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"project={args.project}",
        f"name={args.name}",
        "exist_ok=True",
        f"patience={args.patience}",
    ]
    if resume_checkpoint is not None:
        command.append("resume=True")
    if args.device:
        command.append(f"device={args.device}")
    if args.workers:
        command.append(f"workers={args.workers}")
    if args.extra:
        command.extend(shlex.split(args.extra))
    return command


if __name__ == "__main__":
    sys.exit(main())
