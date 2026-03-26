import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_CLASSES = ["puck", "robot-front", "robot-back"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a COCO dataset export for the smartphone robot detector.",
    )
    parser.add_argument(
        "dataset_root",
        help="Path to a COCO dataset directory containing images/ and labels.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    labels_path = dataset_root / "labels.json"
    images_dir = dataset_root / "images"

    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels.json in {dataset_root}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory in {dataset_root}")

    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    categories = payload.get("categories", [])
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])

    category_names = [category["name"] for category in sorted(categories, key=lambda item: item["id"])]
    if category_names != EXPECTED_CLASSES:
        raise ValueError(
            f"Category names {category_names} do not match expected {EXPECTED_CLASSES}"
        )

    image_ids = {image["id"] for image in images}
    category_ids = {category["id"] for category in categories}
    missing_files = []
    annotation_count_by_class = Counter()

    for image in images:
        image_path = images_dir / image["file_name"]
        if not image_path.is_file():
            missing_files.append(str(image_path))

    if missing_files:
        raise FileNotFoundError(
            f"Dataset references {len(missing_files)} missing image files. First missing file: {missing_files[0]}"
        )

    for annotation in annotations:
        if annotation["image_id"] not in image_ids:
            raise ValueError(f"Annotation references unknown image_id={annotation['image_id']}")
        if annotation["category_id"] not in category_ids:
            raise ValueError(f"Annotation references unknown category_id={annotation['category_id']}")
        bbox = annotation.get("bbox")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Annotation {annotation.get('id')} has invalid bbox={bbox}")
        _, _, width, height = bbox
        if width <= 0 or height <= 0:
            raise ValueError(f"Annotation {annotation.get('id')} has non-positive bbox={bbox}")
        annotation_count_by_class[annotation["category_id"]] += 1

    if not annotations:
        raise ValueError("COCO dataset contains no annotations.")

    print(f"Validated dataset: {dataset_root}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    for category in sorted(categories, key=lambda item: item["id"]):
        print(f"{category['name']}: {annotation_count_by_class.get(category['id'], 0)} annotations")


if __name__ == "__main__":
    main()
