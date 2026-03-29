from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class SelectedDetection:
    class_name: str
    score: float
    origin_x: int
    origin_y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.origin_x + self.width

    @property
    def bottom(self) -> int:
        return self.origin_y + self.height


def _category_name(category) -> str | None:
    name = getattr(category, "category_name", None) or getattr(category, "display_name", None)
    if name is None:
        return None
    return str(name)


def select_top_detections_per_class(detections: Iterable[object]) -> dict[str, SelectedDetection]:
    best_by_class: dict[str, SelectedDetection] = {}

    for detection in detections:
        bounding_box = getattr(detection, "bounding_box", None)
        categories = getattr(detection, "categories", None) or []
        if bounding_box is None:
            continue

        for category in categories:
            class_name = _category_name(category)
            score = getattr(category, "score", None)
            if class_name is None or score is None:
                continue

            selected = SelectedDetection(
                class_name=class_name,
                score=float(score),
                origin_x=int(getattr(bounding_box, "origin_x")),
                origin_y=int(getattr(bounding_box, "origin_y")),
                width=int(getattr(bounding_box, "width")),
                height=int(getattr(bounding_box, "height")),
            )

            current = best_by_class.get(class_name)
            if current is None or selected.score > current.score:
                best_by_class[class_name] = selected

    return best_by_class


def filter_detections_by_score(
    detections_by_class: dict[str, SelectedDetection], *, min_score_exclusive: float
) -> dict[str, SelectedDetection]:
    return {
        class_name: detection
        for class_name, detection in detections_by_class.items()
        if detection.score > min_score_exclusive
    }
