import unittest
from unittest.mock import patch

from wune.font_finder import (
    is_japanese_font,
    filter_japanese_fonts,
    detect_system_japanese_fonts,
    _get_linux_japanese_fonts,
)


class TestFontFinder(unittest.TestCase):
    def test_is_japanese_font_by_keyword(self):
        self.assertTrue(is_japanese_font("Noto Sans CJK JP"))
        self.assertTrue(is_japanese_font("Noto Serif CJK JP"))
        self.assertTrue(is_japanese_font("TakaoPGothic"))
        self.assertTrue(is_japanese_font("IPAPGothic"))
        self.assertTrue(is_japanese_font("VL PGothic"))
        self.assertTrue(is_japanese_font("Meiryo"))
        self.assertTrue(is_japanese_font("Yu Gothic"))
        self.assertTrue(is_japanese_font("MS UI Gothic"))
        self.assertTrue(is_japanese_font("Hiragino Sans"))
        self.assertTrue(is_japanese_font("BIZ UDPGothic"))

        self.assertFalse(is_japanese_font("Arial"))
        self.assertFalse(is_japanese_font("Helvetica"))
        self.assertFalse(is_japanese_font("Courier New"))
        self.assertFalse(is_japanese_font("DejaVu Sans"))
        self.assertFalse(is_japanese_font(""))

    def test_is_japanese_font_by_cjk_characters(self):
        self.assertTrue(is_japanese_font("メイリオ"))
        self.assertTrue(is_japanese_font("游ゴシック"))
        self.assertTrue(is_japanese_font("ＭＳ 明朝"))
        self.assertFalse(is_japanese_font("Liberation Sans"))

    def test_is_japanese_font_with_detected_set(self):
        custom_set = {"CustomJaFont", "AnotherFont"}
        self.assertTrue(is_japanese_font("CustomJaFont", ja_font_set=custom_set))
        self.assertFalse(is_japanese_font("SomeWesternFont", ja_font_set=custom_set))

    def test_filter_japanese_fonts(self):
        fonts = [
            "Arial",
            "Noto Sans CJK JP",
            "Courier New",
            "Meiryo",
            "Liberation Serif",
        ]
        filtered = filter_japanese_fonts(fonts)
        self.assertIn("Noto Sans CJK JP", filtered)
        self.assertIn("Meiryo", filtered)
        self.assertNotIn("Arial", filtered)
        self.assertNotIn("Courier New", filtered)
        self.assertNotIn("Liberation Serif", filtered)

    def test_detect_system_japanese_fonts_returns_set(self):
        detected = detect_system_japanese_fonts()
        self.assertIsInstance(detected, set)

    @patch("shutil.which", return_value="/usr/bin/fc-list")
    @patch("subprocess.run")
    def test_get_linux_japanese_fonts(self, mock_run, mock_which):
        mock_run.return_value.stdout = "Noto Sans CJK JP,Noto Sans CJK JP Bold\nTakaoPGothic\n@VerticalFont\n"
        fonts = _get_linux_japanese_fonts()
        self.assertIn("Noto Sans CJK JP", fonts)
        self.assertIn("Noto Sans CJK JP Bold", fonts)
        self.assertIn("TakaoPGothic", fonts)
        self.assertNotIn("@VerticalFont", fonts)


if __name__ == "__main__":
    unittest.main()
