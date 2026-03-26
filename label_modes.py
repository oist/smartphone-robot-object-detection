from __future__ import annotations

from copy import deepcopy


LABEL_MODE_THREE_CLASS = "three-class"
LABEL_MODE_ROBOT_MERGED = "robot-merged"
DEFAULT_LABEL_MODE = LABEL_MODE_THREE_CLASS
LABEL_MODE_CHOICES = [LABEL_MODE_THREE_CLASS, LABEL_MODE_ROBOT_MERGED]

THREE_CLASS_NAMES = ["puck", "robot-front", "robot-back"]
ROBOT_MERGED_NAMES = ["puck", "robot"]

_DISPLAY_NAMES = {
    LABEL_MODE_THREE_CLASS: "3-class",
    LABEL_MODE_ROBOT_MERGED: "robot-merged",
}
_FILE_SUFFIXES = {
    LABEL_MODE_THREE_CLASS: "3class",
    LABEL_MODE_ROBOT_MERGED: "robot-merged",
}


def expected_classes(label_mode: str) -> list[str]:
    if label_mode == LABEL_MODE_THREE_CLASS:
        return list(THREE_CLASS_NAMES)
    if label_mode == LABEL_MODE_ROBOT_MERGED:
        return list(ROBOT_MERGED_NAMES)
    raise ValueError(f"Unsupported label mode: {label_mode}")


def label_mode_display_name(label_mode: str) -> str:
    if label_mode not in _DISPLAY_NAMES:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    return _DISPLAY_NAMES[label_mode]


def label_mode_file_suffix(label_mode: str) -> str:
    if label_mode not in _FILE_SUFFIXES:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    return _FILE_SUFFIXES[label_mode]


def detect_label_mode(categories: list[str]) -> str:
    if categories == THREE_CLASS_NAMES:
        return LABEL_MODE_THREE_CLASS
    if categories == ROBOT_MERGED_NAMES:
        return LABEL_MODE_ROBOT_MERGED
    raise ValueError(
        f"Unsupported category set {categories}. Expected {THREE_CLASS_NAMES} or {ROBOT_MERGED_NAMES}."
    )


def _remap_category_name(name: str, label_mode: str) -> str:
    if label_mode == LABEL_MODE_ROBOT_MERGED and name in {"robot-front", "robot-back", "robot"}:
        return "robot"
    return name


def remap_coco_dataset(payload: dict, label_mode: str) -> dict:
    remapped = deepcopy(payload)
    source_categories = [
        category["name"] for category in sorted(remapped["categories"], key=lambda item: item["id"])
    ]
    source_mode = detect_label_mode(source_categories)

    if label_mode == source_mode:
        return remapped
    if label_mode != LABEL_MODE_ROBOT_MERGED:
        raise ValueError(
            f"Cannot derive label mode '{label_mode}' from source categories {source_categories}"
        )

    category_id_map: dict[int, int] = {}
    name_to_new_id: dict[str, int] = {}
    new_categories: list[dict] = []

    for category in sorted(remapped["categories"], key=lambda item: item["id"]):
        new_name = _remap_category_name(category["name"], label_mode)
        if new_name not in name_to_new_id:
            new_id = len(name_to_new_id) + 1
            name_to_new_id[new_name] = new_id
            new_category = dict(category)
            new_category["id"] = new_id
            new_category["name"] = new_name
            new_categories.append(new_category)
        category_id_map[category["id"]] = name_to_new_id[new_name]

    remapped["categories"] = new_categories
    remapped["annotations"] = [
        {
            **annotation,
            "category_id": category_id_map[annotation["category_id"]],
        }
        for annotation in remapped["annotations"]
    ]
    return remapped
