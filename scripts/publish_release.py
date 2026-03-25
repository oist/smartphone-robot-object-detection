#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and publish a GitHub release for the trained detector.",
    )
    parser.add_argument("--tag", required=True, help="Release tag, for example 2.0.0.")
    parser.add_argument("--target", default="HEAD", help="Target commitish used when creating a new release.")
    parser.add_argument("--repo", default="oist/smartphone_robot_object_detection")
    parser.add_argument("--dataset-archive")
    parser.add_argument("--release-input")
    parser.add_argument("--title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "publish_github_release.py"),
        "--tag",
        args.tag,
        "--target",
        args.target,
        "--repo",
        args.repo,
    ]
    if args.dataset_archive:
        command.extend(["--dataset-archive", args.dataset_archive])
    if args.release_input:
        command.extend(["--release-input", args.release_input])
    if args.title:
        command.extend(["--title", args.title])

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
