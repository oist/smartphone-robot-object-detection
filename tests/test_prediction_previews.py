import unittest
from types import SimpleNamespace

from prediction_previews import filter_detections_by_score, select_top_detections_per_class


def make_detection(origin_x, origin_y, width, height, *categories):
    return SimpleNamespace(
        bounding_box=SimpleNamespace(
            origin_x=origin_x,
            origin_y=origin_y,
            width=width,
            height=height,
        ),
        categories=[
            SimpleNamespace(category_name=category_name, score=score) for category_name, score in categories
        ],
    )


class PredictionPreviewSelectionTest(unittest.TestCase):
    def test_selects_highest_confidence_detection_for_each_class(self):
        detections = [
            make_detection(10, 20, 100, 120, ("robot", 0.55)),
            make_detection(30, 40, 90, 110, ("puck", 0.72)),
            make_detection(50, 60, 80, 100, ("robot", 0.91)),
        ]

        selected = select_top_detections_per_class(detections)

        self.assertEqual(set(selected), {"robot", "puck"})
        self.assertAlmostEqual(selected["robot"].score, 0.91)
        self.assertEqual((selected["robot"].origin_x, selected["robot"].origin_y), (50, 60))
        self.assertAlmostEqual(selected["puck"].score, 0.72)

    def test_ignores_categories_without_name_or_score(self):
        detections = [
            make_detection(10, 10, 20, 20, ("robot", 0.4)),
            SimpleNamespace(
                bounding_box=SimpleNamespace(origin_x=0, origin_y=0, width=5, height=5),
                categories=[
                    SimpleNamespace(category_name=None, display_name=None, score=0.9),
                    SimpleNamespace(category_name="puck", score=None),
                ],
            ),
        ]

        selected = select_top_detections_per_class(detections)

        self.assertEqual(list(selected), ["robot"])

    def test_filters_out_detections_at_or_below_threshold(self):
        selected = {
            "robot": make_detection(0, 0, 10, 10, ("robot", 0.5)),
            "puck": make_detection(0, 0, 10, 10, ("puck", 0.5001)),
        }

        filtered = filter_detections_by_score(
            {
                class_name: detection.categories[0] and SimpleNamespace(
                    class_name=class_name,
                    score=detection.categories[0].score,
                    origin_x=detection.bounding_box.origin_x,
                    origin_y=detection.bounding_box.origin_y,
                    width=detection.bounding_box.width,
                    height=detection.bounding_box.height,
                )
                for class_name, detection in selected.items()
            },
            min_score_exclusive=0.5,
        )

        self.assertEqual(set(filtered), {"puck"})


if __name__ == "__main__":
    unittest.main()
