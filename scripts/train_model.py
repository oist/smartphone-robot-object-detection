#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the dataset and train a named detector variant.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["three-class", "robot-merged"],
        default="three-class",
        help="Training variant to prepare and train.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Skip float16 export and only export the default floating-point model.",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Rebuild the Docker image before training.",
    )
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to train.py. Prefix them with --, for example -- --epochs 40",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]

    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "prepare_dataset.py"),
            "--label-mode",
            args.label_mode,
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    service_name = (
        "mediapipe-model-maker-3class"
        if args.label_mode == "three-class"
        else "mediapipe-model-maker-robot-merged"
    )
    command = ["docker", "compose", "run"]
    if args.build:
        command.append("--build")
    command.extend([service_name, "python", "train.py", "--label-mode", args.label_mode])
    if not args.no_fp16:
        command.append("--export-fp16")
    command.extend(train_args)

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
