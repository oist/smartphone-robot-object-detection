import argparse
import hashlib
import json
import os
import random
import shutil
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from label_modes import (
    DEFAULT_LABEL_MODE,
    LABEL_MODE_CHOICES,
    detect_label_mode,
    expected_classes,
    remap_coco_dataset,
)

DEFAULT_REPO = "oist/smartphone-robot-object-detection"
DEFAULT_IMAGES_DIR = Path("images")
DEFAULT_ANNOTATIONS = Path("annotations/coco_detection.json")
DEFAULT_RELEASE_ASSET = "dataset.zip"
LEGACY_RELEASE_ASSET = "object-detection-dataset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare a COCO dataset for MediaPipe object detector training.",
    )
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="Path to the local source image directory used with a local COCO export.",
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS),
        help="Path to the local COCO annotations JSON export.",
    )
    parser.add_argument(
        "--source-archive",
        help="Path to a local dataset zip. Used when preparing from a packaged dataset archive.",
    )
    parser.add_argument(
        "--release-manifest",
        default="dataset_release.json",
        help="JSON manifest describing the GitHub release asset to download.",
    )
    parser.add_argument(
        "--download-dir",
        default="data/downloads",
        help="Directory where downloaded release assets are stored.",
    )
    parser.add_argument(
        "--extract-dir",
        default="data/raw",
        help="Directory where the downloaded archive is extracted.",
    )
    parser.add_argument(
        "--prepared-dir",
        default="data/prepared",
        help="Directory containing the generated train/validation/test splits.",
    )
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODE_CHOICES,
        default=DEFAULT_LABEL_MODE,
        help="Label variant to prepare. 'robot-merged' collapses robot-front and robot-back into robot.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Create it from dataset_release.example.json or pass --source-archive."
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for key in ("tag",):
        if key not in payload:
            raise ValueError(f"Manifest {manifest_path} is missing required key '{key}'")
    payload.setdefault("repo", DEFAULT_REPO)
    return payload


def github_request(url: str, accept: str) -> urllib.request.Request:
    headers = {
        "Accept": accept,
        "User-Agent": "smartphone-robot-object-detection/mediapipe-migration",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def download_release_asset(manifest: dict, download_dir: Path) -> Path:
    download_dir.mkdir(parents=True, exist_ok=True)
    release_url = (
        f"https://api.github.com/repos/{manifest['repo']}/releases/tags/{manifest['tag']}"
    )
    with urllib.request.urlopen(github_request(release_url, "application/vnd.github+json")) as response:
        release_info = json.load(response)

    configured_asset = manifest.get("asset")
    asset_names = []
    if configured_asset:
        asset_names.append(configured_asset)
        if configured_asset == DEFAULT_RELEASE_ASSET:
            asset_names.append(LEGACY_RELEASE_ASSET)
        elif configured_asset == LEGACY_RELEASE_ASSET:
            asset_names.append(DEFAULT_RELEASE_ASSET)
    else:
        asset_names.extend([DEFAULT_RELEASE_ASSET, LEGACY_RELEASE_ASSET])

    asset = next(
        (
            candidate
            for candidate in release_info.get("assets", [])
            if candidate["name"] in asset_names
        ),
        None,
    )
    if asset is None:
        raise ValueError(
            f"Could not find any dataset asset matching {asset_names} on release tag '{manifest['tag']}'"
        )

    destination = download_dir / asset["name"]
    with urllib.request.urlopen(github_request(asset["url"], "application/octet-stream")) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    expected_sha256 = manifest.get("sha256")
    if expected_sha256:
        actual_sha256 = sha256_file(destination)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {destination}: expected {expected_sha256}, got {actual_sha256}"
            )

    return destination


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = extract_dir / archive_path.stem
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(dataset_root)
    return find_coco_root(dataset_root)


def find_coco_root(extract_root: Path) -> Path:
    candidates = []
    for candidate in [extract_root, *extract_root.rglob("*")]:
        if not candidate.is_dir():
            continue
        if (candidate / "labels.json").is_file() and (candidate / "images").is_dir():
            candidates.append(candidate)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one COCO dataset root under {extract_root}, found {len(candidates)}"
        )
    return candidates[0]


def load_coco_dataset_from_labels(labels_path: Path) -> dict:
    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    categories = [category["name"] for category in sorted(payload["categories"], key=lambda item: item["id"])]
    detect_label_mode(categories)
    return payload


def load_local_coco_dataset(images_dir: Path, annotations_path: Path) -> tuple[dict, Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not annotations_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotations file: {annotations_path}")
    return load_coco_dataset_from_labels(annotations_path), images_dir


def load_archive_coco_dataset(archive_path: Path, extract_dir: Path) -> tuple[dict, Path]:
    dataset_root = extract_archive(archive_path, extract_dir)
    return load_coco_dataset_from_labels(dataset_root / "labels.json"), dataset_root / "images"


def split_images(images: list[dict], seed: int, train_ratio: float, validation_ratio: float) -> dict[str, list[dict]]:
    shuffled = list(images)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    if total < 3:
        raise ValueError("Need at least 3 annotated images to create train/validation/test splits.")

    train_cutoff = int(total * train_ratio)
    validation_cutoff = train_cutoff + int(total * validation_ratio)

    train_images = shuffled[:train_cutoff]
    validation_images = shuffled[train_cutoff:validation_cutoff]
    test_images = shuffled[validation_cutoff:]

    for split_name, split_images_list in {
        "train": train_images,
        "validation": validation_images,
        "test": test_images,
    }.items():
        if not split_images_list:
            raise ValueError(f"Split '{split_name}' is empty. Adjust the split ratios or dataset size.")

    return {
        "train": train_images,
        "validation": validation_images,
        "test": test_images,
    }


def subset_annotations(annotations: list[dict], image_ids: set[int]) -> list[dict]:
    return [annotation for annotation in annotations if annotation["image_id"] in image_ids]


def copy_or_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(
    split_name: str,
    split_images_list: list[dict],
    annotations: list[dict],
    categories: list[dict],
    source_images_dir: Path,
    prepared_dir: Path,
) -> None:
    split_dir = prepared_dir / split_name
    if split_dir.exists():
        shutil.rmtree(split_dir)
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    image_ids = {image["id"] for image in split_images_list}
    split_annotations = subset_annotations(annotations, image_ids)

    for image in split_images_list:
        source_path = source_images_dir / image["file_name"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by COCO labels: {source_path}")
        copy_or_link(source_path, images_dir / image["file_name"])

    split_payload = {
        "images": split_images_list,
        "annotations": split_annotations,
        "categories": categories,
    }
    with (split_dir / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.validation_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError("train/validation/test ratios must sum to 1.0")

    local_images_dir = Path(args.images_dir)
    local_annotations = Path(args.annotations)

    if local_images_dir.is_dir() and local_annotations.is_file():
        dataset, source_images_dir = load_local_coco_dataset(local_images_dir, local_annotations)
    elif args.source_archive:
        archive_path = Path(args.source_archive)
        if not archive_path.is_file():
            raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
        dataset, source_images_dir = load_archive_coco_dataset(archive_path, Path(args.extract_dir))
    else:
        manifest = load_manifest(Path(args.release_manifest))
        archive_path = download_release_asset(manifest, Path(args.download_dir))
        dataset, source_images_dir = load_archive_coco_dataset(archive_path, Path(args.extract_dir))

    annotated_image_ids = {annotation["image_id"] for annotation in dataset["annotations"]}
    annotated_images = [image for image in dataset["images"] if image["id"] in annotated_image_ids]
    if not annotated_images:
        raise ValueError("The COCO dataset contains no annotated images.")

    dataset = remap_coco_dataset(dataset, args.label_mode)
    prepared_categories = [
        category["name"] for category in sorted(dataset["categories"], key=lambda item: item["id"])
    ]
    expected_category_names = expected_classes(args.label_mode)
    if prepared_categories != expected_category_names:
        raise ValueError(
            f"Prepared categories {prepared_categories} do not match expected {expected_category_names}"
        )

    split_map = split_images(
        annotated_images,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )

    prepared_dir = Path(args.prepared_dir)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_images_list in split_map.items():
        write_split(
            split_name=split_name,
            split_images_list=split_images_list,
            annotations=dataset["annotations"],
            categories=dataset["categories"],
            source_images_dir=source_images_dir,
            prepared_dir=prepared_dir,
        )

    print(f"Prepared dataset written to {prepared_dir}")


if __name__ == "__main__":
    main()
