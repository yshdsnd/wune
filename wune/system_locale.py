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
    elif sys.platform == "darwin":
        try:
            import ctypes
            from ctypes import c_void_p, c_char_p

            cf = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            cf.CFLocaleCopyPreferredLanguages.argtypes = []
            cf.CFRelease.restype = None
            cf.CFLocaleCopyPreferredLanguages.restype = c_void_p
            cf.CFArrayGetCount.restype = ctypes.c_long
            cf.CFArrayGetCount.argtypes = [c_void_p]
            cf.CFArrayGetValueAtIndex.restype = c_void_p
            cf.CFArrayGetValueAtIndex.argtypes = [c_void_p, ctypes.c_long]
            cf.CFStringGetCString.restype = ctypes.c_bool
            cf.CFStringGetCString.argtypes = [c_void_p, c_char_p, ctypes.c_long, ctypes.c_uint32]
            cf.CFRelease.argtypes = [c_void_p]

            languages = cf.CFLocaleCopyPreferredLanguages()
            if languages:
                try:
                    if cf.CFArrayGetCount(languages) > 0:
                        first = cf.CFArrayGetValueAtIndex(languages, 0)
                        buf = ctypes.create_string_buffer(64)
                        if cf.CFStringGetCString(first, buf, 64, 0x08000100):  # kCFStringEncodingUTF8
                            val = buf.value.decode("utf-8")
                            if val:
                                return val
                finally:
                    cf.CFRelease(languages)
        except (AttributeError, OSError, ValueError):
            pass
    try:
        return locale.getlocale()[0] or "en"
    except (ValueError, TypeError):
        return "en"
