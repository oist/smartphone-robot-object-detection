import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package a local COCO dataset export into a release-ready zip archive.",
    )
    parser.add_argument(
        "dataset_root",
        help="Path to a COCO dataset directory containing images/ and labels.json.",
    )
    parser.add_argument(
        "--output",
        default="object-detection-dataset.zip",
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
    dataset_root = Path(args.dataset_root)
    labels_path = dataset_root / "labels.json"
    images_dir = dataset_root / "images"

    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels.json in {dataset_root}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory in {dataset_root}")

    output_path = Path(args.output)
    staging_root = Path(args.staging_dir)
    staging_dataset = staging_root / "dataset-root"

    if staging_root.exists():
        shutil.rmtree(staging_root)

    (staging_dataset / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy2(labels_path, staging_dataset / "labels.json")

    for image_path in sorted(images_dir.iterdir()):
        if image_path.is_file():
            shutil.copy2(image_path, staging_dataset / "images" / image_path.name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    zip_directory(staging_root, output_path)
    shutil.rmtree(staging_root)

    print(f"Packaged dataset archive: {output_path}")


if __name__ == "__main__":
    main()
