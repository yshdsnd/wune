"""Read the user's UI locale without changing the process locale."""
import locale
import sys
from functools import lru_cache


@lru_cache(maxsize=1)
def user_locale():
    if sys.platform == "win32":
        try:
            import ctypes
            language_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            language = locale.windows_locale.get(language_id)
            if language:
                return language
        except (AttributeError, OSError):
            pass
    try:
        return locale.getlocale()[0] or "en"
    except (ValueError, TypeError):
        return "en"
