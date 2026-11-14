"""Check language selection without changing data or widget output fields."""

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from src.geometry_utils import compute_homography
from src.localization import localize
from src import notebook_widgets


class LocalizationTests(unittest.TestCase):
    def test_default_and_unknown_language_preserve_chinese(self):
        for settings in ({}, {"BASKETBALL_COURSE_LANGUAGE": "unknown"}):
            with patch.dict(os.environ, settings, clear=True):
                self.assertEqual(localize("中文", "English", "日本語"), "中文")

    def test_errors_use_the_selected_language(self):
        for language, message in [("en", "at least 4"), ("ja", "4 組の対応点")]:
            with patch.dict(os.environ, {"BASKETBALL_COURSE_LANGUAGE": language}):
                with self.assertRaisesRegex(ValueError, message):
                    compute_homography([], [])

    def test_pair_widget_translates_controls_and_preserves_json_fields(self):
        image = Path(__file__).resolve().parents[1] / "assets/samples/sample_court_frame.png"
        for language, label in [("en", "Clear pairs"), ("ja", "対応点を消去")]:
            with patch.dict(os.environ, {"BASKETBALL_COURSE_LANGUAGE": language}):
                with patch.object(notebook_widgets, "display") as display:
                    notebook_widgets.show_court_keypoint_pair_tool(
                        image, image, [{"name": "point", "xy": [10, 20]}]
                    )
                html = display.call_args.args[0].data
                self.assertIn(label, html)
                self.assertNotIn("清除 pairs", html)
                self.assertIn("camera_xy", html)
                self.assertIn("bev_xy", html)


if __name__ == "__main__":
    unittest.main()
