"""Font discovery and Japanese font detection utilities across platforms."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys

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


def _get_linux_japanese_fonts() -> set[str]:
    """Query fontconfig for fonts covering the Japanese language on Linux."""
    if not shutil.which("fc-list"):
        return set()
    try:
        proc = subprocess.run(
            ["fc-list", ":lang=ja", "family"],
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


def _get_windows_japanese_fonts() -> set[str]:
    """Query GDI for fonts supporting Shift-JIS (charset 128) on Windows."""
    if sys.platform != "win32":
        return set()
    ja_fonts: set[str] = set()
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
                ja_fonts.add(name)
            return 1

        cb = FONTENUMPROCW(callback)
        lf_search = LOGFONTW()
        lf_search.lfCharSet = 128  # SHIFTJIS_CHARSET
        lf_search.lfFaceName = ""

        hdc = ctypes.windll.user32.GetDC(0)
        try:
            ctypes.windll.gdi32.EnumFontFamiliesExW(
                hdc, ctypes.byref(lf_search), cb, 0, 0
            )
        finally:
            ctypes.windll.user32.ReleaseDC(0, hdc)
    except Exception:
        pass
    return ja_fonts


_SYSTEM_JA_CACHE: set[str] | None = None


def detect_system_japanese_fonts() -> set[str]:
    """Detect installed font families that support Japanese on the host system."""
    global _SYSTEM_JA_CACHE
    if _SYSTEM_JA_CACHE is not None:
        return _SYSTEM_JA_CACHE

    result: set[str] = set()
    if sys.platform == "win32":
        result.update(_get_windows_japanese_fonts())
    elif sys.platform == "darwin":
        result.update(_MACOS_JA_FONTS)
        result.update(_get_linux_japanese_fonts())
    else:  # Linux / BSD
        result.update(_get_linux_japanese_fonts())

    _SYSTEM_JA_CACHE = result
    return result


def is_japanese_font(name: str, ja_font_set: set[str] | None = None) -> bool:
    """Return True if the font family name likely supports Japanese glyphs."""
    if not name:
        return False
    if ja_font_set is None:
        ja_font_set = detect_system_japanese_fonts()

    if name in ja_font_set:
        return True

    # Check for CJK / Hiragana / Katakana / Kanji characters in font name
    if any(0x3040 <= ord(c) <= 0x9FFF for c in name):
        return True

    # Check known keywords
    lower = name.lower()
    return any(kw in lower for kw in _JA_KEYWORDS)


def filter_japanese_fonts(available_fonts: list[str]) -> list[str]:
    """Filter the given list of font families down to Japanese-capable ones."""
    ja_set = detect_system_japanese_fonts()
    return [f for f in available_fonts if is_japanese_font(f, ja_set)]
