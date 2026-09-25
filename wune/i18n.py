"""Message catalogs independent of display, audio, and process-global locale."""
from functools import lru_cache
import json
from pathlib import Path

CATALOG_DIR = Path(__file__).with_name("locales")


@lru_cache(maxsize=None)
def catalog(language):
    try:
        data = json.loads((CATALOG_DIR / f"{language}.json").read_text(encoding="utf-8"))
        return {key: value for key, value in data.items() if isinstance(value, str)}
    except (OSError, ValueError, AttributeError):
        return {}


@lru_cache(maxsize=1)
def languages():
    return ("auto", *sorted(path.stem for path in CATALOG_DIR.glob("*.json")))


def resolve_language(language="auto"):
    if language == "auto":
        from .system_locale import user_locale
        language = user_locale()
    language = str(language or "en").replace("-", "_").split("_")[0].lower()
    return language if language != "auto" and language in languages() else "en"


class Translator:
    def __init__(self, language="auto"):
        self.language = resolve_language(language)

    def __call__(self, key, **values):
        fallback = catalog("en").get(key, key)
        template = catalog(self.language).get(key, fallback)
        try:
            return template.format(**values)
        except (KeyError, ValueError, IndexError, AttributeError, TypeError):
            try:
                return fallback.format(**values)
            except (KeyError, ValueError, IndexError, AttributeError, TypeError):
                return fallback

    def error(self, error):
        return self(error.key, **error.values) if isinstance(error, MessageError) else str(error)


class MessageError(ValueError):
    def __init__(self, key, **values):
        self.key, self.values = key, values
        super().__init__(Translator("en")(key, **values))
