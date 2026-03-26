#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from label_modes import (  # noqa: E402
    DEFAULT_LABEL_MODE,
    LABEL_MODE_CHOICES,
    expected_classes,
    label_mode_display_name,
)


DEFAULT_DOCKER_IMAGE = "topher217/smartphone-robot-object-detection"
DEFAULT_RELEASE_DIR = REPO_ROOT / "build" / "release"
DEFAULT_MODEL_DIR = REPO_ROOT / "exported_model"
DEFAULT_RELEASE_INPUTS_DIR = REPO_ROOT / "release_inputs"
DEFAULT_DATASET_ARCHIVE_NAME = "dataset.zip"
LEGACY_DATASET_ARCHIVE_NAME = "object-detection-dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare versioned GitHub release assets for the trained detector.",
    )
    parser.add_argument("--tag", required=True, help="Release tag, for example 2.0.0.")
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODE_CHOICES,
        default=None,
        help="Override the model variant documented in the release assets.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing model.tflite and optional related outputs.",
    )
    parser.add_argument(
        "--release-dir",
        default=str(DEFAULT_RELEASE_DIR),
        help="Directory where prepared release assets are written.",
    )
    parser.add_argument(
        "--dataset-archive",
        help="Existing dataset archive to reuse. If omitted, dataset.zip is preferred and legacy names are accepted.",
    )
    parser.add_argument(
        "--release-input",
        help="Optional JSON file describing release title, label mode, docker image, and fallback metrics.",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Override the Docker image repository documented in release notes.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset_archive(args: argparse.Namespace, release_dir: Path) -> Path:
    candidates = []
    if args.dataset_archive:
        candidates.append(Path(args.dataset_archive))
    else:
        candidates.extend(
            [
                REPO_ROOT / DEFAULT_DATASET_ARCHIVE_NAME,
                REPO_ROOT / LEGACY_DATASET_ARCHIVE_NAME,
            ]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    dataset_output = release_dir / DEFAULT_DATASET_ARCHIVE_NAME
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "package_dataset.py"),
            "--output",
            str(dataset_output),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return dataset_output


def load_release_input(tag: str, configured_path: str | None) -> dict:
    candidate_paths = []
    if configured_path:
        candidate_paths.append(Path(configured_path))
    candidate_paths.append(DEFAULT_RELEASE_INPUTS_DIR / f"{tag}.json")

    for candidate in candidate_paths:
        if candidate.is_file():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def metric_value(summary: dict | None, release_input: dict, key: str) -> float | None:
    metrics_override = release_input.get("metrics", {})
    if key in metrics_override and metrics_override[key] is not None:
        return metrics_override[key]
    if not summary:
        return None
    metrics = (
        summary.get("validation_metrics", {})
        if key.startswith("validation_")
        else summary.get("test_metrics", {})
    )
    metric_key = {
        "validation_ap": "AP",
        "validation_ap50": "AP50",
        "test_ap": "AP",
        "test_ap50": "AP50",
    }[key]
    return metrics.get(metric_key)


def copy_if_exists(source: Path, destination: Path) -> bool:
    if not source.is_file():
        return False
    shutil.copy2(source, destination)
    return True


def build_release_notes(
    *,
    tag: str,
    label_mode: str,
    docker_image: str,
    metrics: dict[str, float | None],
    asset_names: list[str],
) -> str:
    classes = ", ".join(expected_classes(label_mode))
    lines = [
        f"# Smartphone Robot Detector {tag}",
        "",
        f"- Published model variant: `{label_mode_display_name(label_mode)}`",
        f"- Classes in this release: `{classes}`",
        f"- Docker image: `{docker_image}:{tag}` and `{docker_image}:latest`",
        "",
        "## Included assets",
        "",
    ]
    lines.extend([f"- `{name}`" for name in asset_names])
    lines.extend(["", "## Metrics", ""])

    if metrics["validation_ap"] is not None:
        lines.append(f"- Validation AP: `{metrics['validation_ap']:.4f}`")
    if metrics["validation_ap50"] is not None:
        lines.append(f"- Validation AP50: `{metrics['validation_ap50']:.4f}`")
    if metrics["test_ap"] is not None:
        lines.append(f"- Test AP: `{metrics['test_ap']:.4f}`")
    if metrics["test_ap50"] is not None:
        lines.append(f"- Test AP50: `{metrics['test_ap50']:.4f}`")
    if all(value is None for value in metrics.values()):
        lines.append("- Metrics were not provided to the release asset generator.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- `dataset.zip` is the canonical training-data asset name for this release line.",
            f"- The repository also supports `--label-mode robot-merged` for future training runs.",
        ]
    )
    if label_mode == DEFAULT_LABEL_MODE:
        lines.append(
            "- This release publishes the current 3-class model. The merged-robot variant is supported in the tooling but is only shipped when explicitly trained and published."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_path = model_dir / "model.tflite"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model artifact: {model_path}")

    release_input = load_release_input(args.tag, args.release_input)
    label_mode = args.label_mode or release_input.get("label_mode") or DEFAULT_LABEL_MODE
    docker_image = args.docker_image or release_input.get("docker_image") or DEFAULT_DOCKER_IMAGE

    release_dir = Path(args.release_dir) / args.tag
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True, exist_ok=True)

    dataset_archive = ensure_dataset_archive(args, release_dir)
    dataset_asset_name = DEFAULT_DATASET_ARCHIVE_NAME
    dataset_asset_path = release_dir / dataset_asset_name
    if dataset_archive.resolve() != dataset_asset_path.resolve():
        shutil.copy2(dataset_archive, dataset_asset_path)

    model_asset_base = f"smartphone-robot-detector-{args.tag}-{label_mode_display_name(label_mode)}"
    copied_assets: list[Path] = []

    model_asset_path = release_dir / f"{model_asset_base}.tflite"
    shutil.copy2(model_path, model_asset_path)
    copied_assets.append(model_asset_path)

    fp16_path = model_dir / "model_fp16.tflite"
    if copy_if_exists(fp16_path, release_dir / f"{model_asset_base}-fp16.tflite"):
        copied_assets.append(release_dir / f"{model_asset_base}-fp16.tflite")

    metadata_path = model_dir / "metadata.json"
    if copy_if_exists(metadata_path, release_dir / f"{model_asset_base}.metadata.json"):
        copied_assets.append(release_dir / f"{model_asset_base}.metadata.json")

    training_summary = model_dir / "training_summary.json"
    summary_payload = None
    if training_summary.is_file():
        with training_summary.open("r", encoding="utf-8") as handle:
            summary_payload = json.load(handle)
        summary_asset = release_dir / f"{model_asset_base}.training-summary.json"
        shutil.copy2(training_summary, summary_asset)
        copied_assets.append(summary_asset)

    copied_assets.append(dataset_asset_path)

    metrics = {
        "validation_ap": metric_value(summary_payload, release_input, "validation_ap"),
        "validation_ap50": metric_value(summary_payload, release_input, "validation_ap50"),
        "test_ap": metric_value(summary_payload, release_input, "test_ap"),
        "test_ap50": metric_value(summary_payload, release_input, "test_ap50"),
    }

    manifest_payload = {
        "tag": args.tag,
        "title": release_input.get("title") or f"Smartphone Robot Detector {args.tag}",
        "label_mode": label_mode,
        "label_mode_display_name": label_mode_display_name(label_mode),
        "classes": expected_classes(label_mode),
        "docker_image": docker_image,
        "assets": [path.name for path in copied_assets],
        "metrics": metrics,
    }
    manifest_path = release_dir / "release-metadata.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)
        handle.write("\n")
    copied_assets.append(manifest_path)

    notes_path = release_dir / "release-notes.md"
    notes_path.write_text(
        build_release_notes(
            tag=args.tag,
            label_mode=label_mode,
            docker_image=docker_image,
            metrics=metrics,
            asset_names=[path.name for path in copied_assets],
        ),
        encoding="utf-8",
    )

    checksums_path = release_dir / "sha256sums.txt"
    checksum_lines = [f"{sha256_file(path)}  {path.name}" for path in sorted(copied_assets, key=lambda item: item.name)]
    checksums_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    print(f"Prepared release assets in {release_dir}")


if __name__ == "__main__":
    main()
