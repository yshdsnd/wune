import unittest
from unittest.mock import patch

from wune.font_finder import (
    is_japanese_font,
    filter_japanese_fonts,
    detect_system_japanese_fonts,
    is_language_filter_supported,
    is_font_for_language,
    filter_fonts_for_language,
    detect_system_fonts_for_language,
    _get_linux_fonts,
)


class TestFontFinder(unittest.TestCase):
    def test_is_language_filter_supported(self):
        self.assertTrue(is_language_filter_supported("ja"))
        self.assertTrue(is_language_filter_supported("zh"))
        self.assertTrue(is_language_filter_supported("ko"))
        self.assertTrue(is_language_filter_supported("ja_JP"))
        self.assertTrue(is_language_filter_supported("zh-CN"))

        self.assertFalse(is_language_filter_supported("en"))
        self.assertFalse(is_language_filter_supported("en_US"))
        self.assertFalse(is_language_filter_supported("fr"))
        self.assertFalse(is_language_filter_supported(""))

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

    def test_is_chinese_font_by_keyword_and_characters(self):
        self.assertTrue(is_font_for_language("Microsoft YaHei", "zh"))
        self.assertTrue(is_font_for_language("PingFang SC", "zh"))
        self.assertTrue(is_font_for_language("SimSun", "zh"))
        self.assertTrue(is_font_for_language("Noto Sans SC", "zh"))
        self.assertTrue(is_font_for_language("微软雅黑", "zh"))
        self.assertFalse(is_font_for_language("Arial", "zh"))
        self.assertFalse(is_font_for_language("Courier New", "zh"))

    def test_is_korean_font_by_keyword_and_characters(self):
        self.assertTrue(is_font_for_language("Malgun Gothic", "ko"))
        self.assertTrue(is_font_for_language("NanumGothic", "ko"))
        self.assertTrue(is_font_for_language("Apple SD Gothic Neo", "ko"))
        self.assertTrue(is_font_for_language("Noto Sans KR", "ko"))
        self.assertTrue(is_font_for_language("맑은 고딕", "ko"))
        self.assertFalse(is_font_for_language("Arial", "ko"))
        self.assertFalse(is_font_for_language("Courier New", "ko"))

    def test_english_allows_all_fonts(self):
        self.assertTrue(is_font_for_language("Arial", "en"))
        self.assertTrue(is_font_for_language("Meiryo", "en"))
        self.assertTrue(is_font_for_language("Courier New", "en"))

    def test_filter_fonts_for_language(self):
        fonts = [
            "Arial",
            "Noto Sans CJK JP",
            "Microsoft YaHei",
            "Malgun Gothic",
            "Courier New",
        ]
        ja_filtered = filter_fonts_for_language(fonts, "ja")
        self.assertIn("Noto Sans CJK JP", ja_filtered)
        self.assertNotIn("Arial", ja_filtered)
        self.assertNotIn("Courier New", ja_filtered)

        zh_filtered = filter_fonts_for_language(fonts, "zh")
        self.assertIn("Microsoft YaHei", zh_filtered)
        self.assertNotIn("Arial", zh_filtered)

        ko_filtered = filter_fonts_for_language(fonts, "ko")
        self.assertIn("Malgun Gothic", ko_filtered)
        self.assertNotIn("Arial", ko_filtered)

        en_filtered = filter_fonts_for_language(fonts, "en")
        self.assertEqual(en_filtered, fonts)

    def test_detect_system_japanese_fonts_returns_set(self):
        detected = detect_system_japanese_fonts()
        self.assertIsInstance(detected, set)

    @patch("shutil.which", return_value="/usr/bin/fc-list")
    @patch("subprocess.run")
    def test_get_linux_fonts(self, mock_run, mock_which):
        mock_run.return_value.stdout = "Noto Sans CJK JP,Noto Sans CJK JP Bold\nTakaoPGothic\n@VerticalFont\n"
        fonts = _get_linux_fonts("ja")
        self.assertIn("Noto Sans CJK JP", fonts)
        self.assertIn("Noto Sans CJK JP Bold", fonts)
        self.assertIn("TakaoPGothic", fonts)
        self.assertNotIn("@VerticalFont", fonts)


if __name__ == "__main__":
    unittest.main()
