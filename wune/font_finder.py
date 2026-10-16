"""Font discovery and language-specific font detection utilities across platforms."""
from __future__ import annotations

import shutil
import subprocess
import sys

SUPPORTED_FILTER_LANGUAGES = ("ja", "zh", "ko")

_JA_KEYWORDS = (
    "cjk",
    "japanese",
    "meiryo",
    "yu gothic",
    "yugothic",
    "ms gothic",
    "msgothic",
    "ms mincho",
    "msmincho",
    "ms ui gothic",
    "yu mincho",
    "yumincho",
    "takao",
    "ipa",
    "vl gothic",
    "vl pgothic",
    "vlgothic",
    "biz ud",
    "biz udp",
    "hiragino",
    "osaka",
    "toppan bunkyu",
    "tsukushi",
    "klee",
    "epson",
    "morisawa",
    "motoya",
    "hanazono",
    "kouzan",
    "noto sans jp",
    "noto serif jp",
    "noto sans cjk",
    "noto serif cjk",
)

_ZH_KEYWORDS = (
    "cjk",
    "simsun",
    "simhei",
    "yahei",
    "pingfang",
    "songti",
    "heiti",
    "kaiti",
    "fangsong",
    "microsoft yahei",
    "microsoft jhenghei",
    "noto sans sc",
    "noto sans tc",
    "noto sans hk",
    "noto serif sc",
    "noto serif tc",
    "noto sans cjk",
    "noto serif cjk",
    "source han sans",
    "source han serif",
)

_KO_KEYWORDS = (
    "cjk",
    "korean",
    "malgun",
    "nanum",
    "gulim",
    "batang",
    "dotum",
    "gungsuh",
    "apple sd gothic neo",
    "noto sans kr",
    "noto serif kr",
    "noto sans cjk",
    "noto serif cjk",
)

_MACOS_JA_FONTS = {
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Hiragino Kaku Gothic Pro",
    "Hiragino Mincho ProN",
    "Hiragino Mincho Pro",
    "Hiragino Maru Gothic ProN",
    "Osaka",
    "YuGothic",
    "YuMincho",
    "Toppan Bunkyu Gothic",
    "Toppan Bunkyu Mincho",
    "Tsukushi A Round Gothic",
    "Tsukushi B Round Gothic",
    "Klee",
}

_MACOS_ZH_FONTS = {
    "PingFang SC",
    "PingFang TC",
    "PingFang HK",
    "Songti SC",
    "Songti TC",
    "Heiti SC",
    "Heiti TC",
    "Kaiti SC",
    "Kaiti TC",
}

_MACOS_KO_FONTS = {
    "Apple SD Gothic Neo",
    "AppleGothic",
    "AppleMyungjo",
    "Nanum Gothic",
    "Nanum Myeongjo",
}


def is_language_filter_supported(language: str) -> bool:
    """Return True if font filtering is supported for the given language code."""
    if not language:
        return False
    normalized = str(language).replace("-", "_").split("_")[0].lower()
    return normalized in SUPPORTED_FILTER_LANGUAGES


def _get_linux_fonts(lang: str) -> set[str]:
    """Query fontconfig for fonts covering the specified language code on Linux."""
    if not shutil.which("fc-list"):
        return set()
    try:
        proc = subprocess.run(
            ["fc-list", f":lang={lang}", "family"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=True,
        )
        fonts = set()
        for line in proc.stdout.splitlines():
            for part in line.split(","):
                name = part.strip()
                if name and not name.startswith("@") and not name.startswith("~"):
                    fonts.add(name)
        return fonts
    except Exception:
        return set()


def _get_windows_fonts(charsets: tuple[int, ...]) -> set[str]:
    """Query GDI for fonts supporting any of the given charset IDs on Windows."""
    if sys.platform != "win32":
        return set()
    matched_fonts: set[str] = set()
    try:
        import ctypes
        import ctypes.wintypes as w

        LF_FACESIZE = 32

        class LOGFONTW(ctypes.Structure):
            _fields_ = [
                ("lfHeight", w.LONG),
                ("lfWidth", w.LONG),
                ("lfEscapement", w.LONG),
                ("lfOrientation", w.LONG),
                ("lfWeight", w.LONG),
                ("lfItalic", w.BYTE),
                ("lfUnderline", w.BYTE),
                ("lfStrikeOut", w.BYTE),
                ("lfCharSet", w.BYTE),
                ("lfOutPrecision", w.BYTE),
                ("lfClipPrecision", w.BYTE),
                ("lfQuality", w.BYTE),
                ("lfPitchAndFamily", w.BYTE),
                ("lfFaceName", w.WCHAR * LF_FACESIZE),
            ]

        class ENUMLOGFONTEXW(ctypes.Structure):
            _fields_ = [
                ("elfLogFont", LOGFONTW),
                ("elfFullName", w.WCHAR * 64),
                ("elfStyle", w.WCHAR * 32),
                ("elfScript", w.WCHAR * 32),
            ]

        FONTENUMPROCW = ctypes.WINFUNCTYPE(
            ctypes.c_int,
            ctypes.POINTER(ENUMLOGFONTEXW),
            ctypes.c_void_p,
            w.DWORD,
            w.LPARAM,
        )

        def callback(lpelfe, lpntme, font_type, lparam):
            lf = lpelfe.contents.elfLogFont
            name = lf.lfFaceName
            if name and not name.startswith("@") and not name.startswith("~"):
                matched_fonts.add(name)
            return 1

        cb = FONTENUMPROCW(callback)
        hdc = ctypes.windll.user32.GetDC(0)
        try:
            for charset in charsets:
                lf_search = LOGFONTW()
                lf_search.lfCharSet = charset
                lf_search.lfFaceName = ""
                ctypes.windll.gdi32.EnumFontFamiliesExW(
                    hdc, ctypes.byref(lf_search), cb, 0, 0
                )
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)
    except Exception:
        pass
    return matched_fonts


_SYSTEM_CACHE: dict[str, set[str]] = {}


def detect_system_fonts_for_language(language: str) -> set[str]:
    """Detect installed font families that support the given language."""
    norm = str(language or "en").replace("-", "_").split("_")[0].lower()
    if norm not in SUPPORTED_FILTER_LANGUAGES:
        return set()

    if norm in _SYSTEM_CACHE:
        return _SYSTEM_CACHE[norm]

    result: set[str] = set()
    if sys.platform == "win32":
        charsets = {
            "ja": (128,),          # SHIFTJIS_CHARSET
            "zh": (134, 136),      # GB2312_CHARSET, CHINESEBIG5_CHARSET
            "ko": (129,),          # HANGEUL_CHARSET
        }.get(norm, ())
        result.update(_get_windows_fonts(charsets))
    elif sys.platform == "darwin":
        mac_fonts = {
            "ja": _MACOS_JA_FONTS,
            "zh": _MACOS_ZH_FONTS,
            "ko": _MACOS_KO_FONTS,
        }.get(norm, set())
        result.update(mac_fonts)
        result.update(_get_linux_fonts(norm))
    else:  # Linux / BSD
        result.update(_get_linux_fonts(norm))

    _SYSTEM_CACHE[norm] = result
    return result


def is_font_for_language(name: str, language: str, detected_set: set[str] | None = None) -> bool:
    """Return True if the font name supports the given language."""
    if not name:
        return False
    norm = str(language or "en").replace("-", "_").split("_")[0].lower()
    if norm not in SUPPORTED_FILTER_LANGUAGES:
        return True  # For Latin/Western languages, all fonts qualify

    if detected_set is None:
        detected_set = detect_system_fonts_for_language(norm)

    if name in detected_set:
        return True

    lower = name.lower()

    if norm == "ja":
        if any(0x3040 <= ord(c) <= 0x9FFF for c in name):
            return True
        return any(kw in lower for kw in _JA_KEYWORDS)

    if norm == "zh":
        if any(0x4E00 <= ord(c) <= 0x9FFF for c in name):
            return True
        return any(kw in lower for kw in _ZH_KEYWORDS)

    if norm == "ko":
        if any(
            0xAC00 <= ord(c) <= 0xD7A3
            or 0x1100 <= ord(c) <= 0x11FF
            or 0x3130 <= ord(c) <= 0x318F
            for c in name
        ):
            return True
        return any(kw in lower for kw in _KO_KEYWORDS)

    return True


def filter_fonts_for_language(available_fonts: list[str], language: str) -> list[str]:
    """Filter the given list of font families down to those supporting the language."""
    norm = str(language or "en").replace("-", "_").split("_")[0].lower()
    if not is_language_filter_supported(norm):
        return list(available_fonts)

    detected = detect_system_fonts_for_language(norm)
    return [f for f in available_fonts if is_font_for_language(f, norm, detected)]


# Aliases for backward compatibility and convenience
def detect_system_japanese_fonts() -> set[str]:
    return detect_system_fonts_for_language("ja")


def is_japanese_font(name: str, ja_font_set: set[str] | None = None) -> bool:
    return is_font_for_language(name, "ja", detected_set=ja_font_set)


def filter_japanese_fonts(available_fonts: list[str]) -> list[str]:
    return filter_fonts_for_language(available_fonts, "ja")
