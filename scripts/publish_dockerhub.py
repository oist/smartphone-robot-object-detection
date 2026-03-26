#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "topher217/smartphone-robot-object-detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and push the training container image to DockerHub.",
    )
    parser.add_argument("--tag", required=True, help="Version tag to publish, for example 2.0.0.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image repository to publish.")
    parser.add_argument(
        "--skip-latest",
        action="store_true",
        help="Only publish the explicit version tag and skip the latest tag.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    tags = [f"{args.image}:{args.tag}"]
    if not args.skip_latest:
        tags.append(f"{args.image}:latest")

    build_command = ["docker", "build", "-f", "Dockerfile"]
    for tag in tags:
        build_command.extend(["-t", tag])
    build_command.append(".")
    run(build_command)

    for tag in tags:
        run(["docker", "push", tag])

    print(f"Published Docker image tags: {', '.join(tags)}")


if __name__ == "__main__":
    main()
