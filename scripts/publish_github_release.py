#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_DIR = REPO_ROOT / "build" / "release"
DEFAULT_REPO = "oist/smartphone_robot_object_detection"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update a GitHub release from prepared detector assets.",
    )
    parser.add_argument("--tag", required=True, help="Release tag, for example 2.0.0.")
    parser.add_argument("--title", help="Release title. Defaults to 'Smartphone Robot Detector <tag>'.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository in owner/name form.")
    parser.add_argument("--target", default="HEAD", help="Target commitish used when creating a new release.")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE_DIR))
    parser.add_argument("--dataset-archive")
    parser.add_argument("--release-input")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Assume release assets already exist under build/release/<tag>.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def maybe_prepare_assets(args: argparse.Namespace) -> None:
    if args.skip_prepare:
        return

    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "prepare_release_assets.py"),
        "--tag",
        args.tag,
        "--release-dir",
        args.release_dir,
    ]
    if args.dataset_archive:
        command.extend(["--dataset-archive", args.dataset_archive])
    if args.release_input:
        command.extend(["--release-input", args.release_input])

    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    maybe_prepare_assets(args)

    release_dir = Path(args.release_dir) / args.tag
    notes_path = release_dir / "release-notes.md"
    if not notes_path.is_file():
        raise FileNotFoundError(f"Missing release notes: {notes_path}")

    assets = sorted(
        path
        for path in release_dir.iterdir()
        if path.is_file() and path.name != "release-notes.md"
    )
    if not assets:
        raise ValueError(f"No release assets found in {release_dir}")

    metadata_path = release_dir / "release-metadata.json"
    release_title = f"Smartphone Robot Detector {args.tag}"
    if metadata_path.is_file():
        import json

        with metadata_path.open("r", encoding="utf-8") as handle:
            release_title = json.load(handle).get("title", release_title)

    title = args.title or release_title
    view = run_command(["gh", "release", "view", args.tag, "--repo", args.repo])
    if view.returncode == 0:
        upload_command = ["gh", "release", "upload", args.tag, "--repo", args.repo, "--clobber"]
        upload_command.extend(str(path) for path in assets)
        subprocess.run(upload_command, cwd=REPO_ROOT, check=True)
        subprocess.run(
            [
                "gh",
                "release",
                "edit",
                args.tag,
                "--repo",
                args.repo,
                "--title",
                title,
                "--notes-file",
                str(notes_path),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        print(f"Updated GitHub release {args.tag}")
        return

    create_command = [
        "gh",
        "release",
        "create",
        args.tag,
        "--repo",
        args.repo,
        "--target",
        args.target,
        "--title",
        title,
        "--notes-file",
        str(notes_path),
    ]
    create_command.extend(str(path) for path in assets)
    subprocess.run(create_command, cwd=REPO_ROOT, check=True)
    print(f"Created GitHub release {args.tag}")


if __name__ == "__main__":
    main()
