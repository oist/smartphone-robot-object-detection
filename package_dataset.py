import argparse
import json
import shutil
import zipfile
from pathlib import Path

DEFAULT_IMAGES_DIR = Path("images")
DEFAULT_ANNOTATIONS = Path("annotations/coco_detection.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a local COCO dataset export into a release-ready zip archive.",
    )
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="Path to the source image directory.",
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS),
        help="Path to the COCO annotations JSON file.",
    )
    parser.add_argument(
        "--output",
        default="dataset.zip",
        help="Zip file path to create.",
    )
    parser.add_argument(
        "--staging-dir",
        default="data/package_staging",
        help="Temporary staging directory used to normalize archive layout.",
    )
    return parser.parse_args()


def zip_directory(source_dir: Path, destination_zip: Path) -> None:
    with zipfile.ZipFile(destination_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir))


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    labels_path = Path(args.annotations)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotations file: {labels_path}")

    output_path = Path(args.output)
    staging_root = Path(args.staging_dir)
    staging_dataset = staging_root / "dataset-root"

    if staging_root.exists():
        shutil.rmtree(staging_root)

    (staging_dataset / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, staging_dataset / "labels.json")

    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    referenced_filenames = sorted({image["file_name"] for image in payload.get("images", [])})
    for filename in referenced_filenames:
        image_path = images_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image referenced by COCO labels: {image_path}")
        shutil.copy2(image_path, staging_dataset / "images" / image_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    zip_directory(staging_root, output_path)
    shutil.rmtree(staging_root)

    print(f"Packaged dataset archive: {output_path}")


if __name__ == "__main__":
    main()
