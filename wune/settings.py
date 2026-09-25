"""Versioned user preferences; no display or audio side effects."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import warnings

from .config import Config
from .i18n import languages
from .presets import PRESETS, get_preset
from .ballistics import MOTION_LIMITS, valid_motion


CHOICES = {
    "language": languages(),
    "initial_preset": (None, *(p.name for p in PRESETS)),
    "gauge_style": ("flat", "box"),
    "led_shape": ("rectangle", "rounded", "ellipse"),
    "spectrum_orientation": ("frequency_horizontal", "frequency_vertical"),
    "channel_layout": ("vertical", "horizontal"),
    "info_position": ("top", "bottom"),
}
PREFERENCES = (*CHOICES, "led_aspect_ratio", "bars", "channels", "info_enabled", "limit_to_20khz", *MOTION_LIMITS)


def settings_path():
    base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    return base / "Wune" / "settings.json"


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def valid_preference(key, value):
    if key in MOTION_LIMITS:
        return valid_motion(key, value)
    if key in CHOICES:
        return value in CHOICES[key]
    if key == "led_aspect_ratio":
        return type(value) in (int, float) and math.isfinite(value) and 0.25 <= value <= 8
    if key == "bars":
        return integer(value, 1, 256)
    if key == "channels":
        return integer(value, 1, 2)
    return key in ("info_enabled", "limit_to_20khz") and type(value) is bool


class SettingsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else settings_path()
        self.writable = True
        self._document = {}
        self.user_presets = {}

    def load(self, defaults):
        self.writable = True
        self._document = {}
        self.user_presets = {}
        cfg = deepcopy(defaults)
        geometry = {}
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(document, dict) or type(document.get("version")) is not int or document["version"] not in (1, 2):
                raise ValueError("Unsupported settings format/version")
            if not isinstance(document.get("window", {}), dict) or not isinstance(document.get("appearance", {}), dict):
                raise ValueError("Invalid settings sections")
        except FileNotFoundError:
            return cfg, geometry
        except (OSError, ValueError) as error:
            self.writable = False
            warnings.warn(f"Cannot load {self.path}: {error}. Using defaults; file will not be overwritten.", RuntimeWarning)
            return cfg, geometry
        self._document = document
        from .appearance import decode_preset
        library = document.get("user_themes", {})
        if not isinstance(library, dict):
            warnings.warn("Ignoring invalid user_themes", RuntimeWarning)
            library = {}
        for name, data in library.items():
            try:
                self.user_presets[name] = decode_preset(name, data)
            except ValueError as error:
                warnings.warn(f"Ignoring user theme {name!r}: {error}", RuntimeWarning)
        for key, value in document.get("appearance", {}).items():
            if key in PREFERENCES:
                if valid_preference(key, value) or (key == "initial_preset" and isinstance(value, str) and value in self.user_presets):
                    setattr(cfg, key, value)
                else:
                    warnings.warn(f"Ignoring invalid setting: {key}", RuntimeWarning)
        for key, value in document.get("window", {}).items():
            bounds = (64, 16384) if key in ("width", "height") else (-131072, 131072)
            if key in ("width", "height", "x", "y") and integer(value, *bounds):
                geometry[key] = value
        # A named theme supplies colors only; v1 bundled style is migrated once.
        if cfg.initial_preset is not None:
            preset = self.user_presets[cfg.initial_preset] if cfg.initial_preset in self.user_presets else get_preset(cfg.initial_preset)
            cfg.theme = preset.theme
            if document["version"] == 1:
                # v1 named presets overrode explicit appearance fields at startup.
                # Reproduce that visible style, then save it independently in v2.
                legacy = dict(gauge_style="flat", led_shape="rounded", led_aspect_ratio=2.0)
                if cfg.initial_preset in self.user_presets:
                    legacy.update({key: value for key, value in library[cfg.initial_preset].items()
                                   if key in legacy and valid_preference(key, value)})
                elif cfg.initial_preset in ("BLUE", "CLASSIC BOX"):
                    legacy.update(gauge_style="box", led_shape="rectangle")
                for key, value in legacy.items():
                    setattr(cfg, key, value)
        elif "theme" in document.get("appearance", {}):
            try:
                cfg.theme = decode_preset("Custom colors", {"theme": document["appearance"]["theme"]}).theme
            except ValueError as error:
                warnings.warn(f"Ignoring custom colors: {error}", RuntimeWarning)
        return cfg, geometry

    def save(self, cfg, size, position, preset_name):
        if not self.writable:
            return False
        document = deepcopy(self._document)
        document["version"] = 2
        appearance = document.setdefault("appearance", {})
        appearance.update({key: getattr(cfg, key) for key in PREFERENCES})
        appearance["initial_preset"] = None if preset_name == "CUSTOM" else preset_name
        from .appearance import encode_preset
        from dataclasses import asdict
        appearance["theme"] = asdict(cfg.theme)
        document["user_themes"] = {name: encode_preset(preset) for name, preset in self.user_presets.items()}
        document.setdefault("window", {}).update(width=int(size[0]), height=int(size[1]),
                                                  x=int(position[0]), y=int(position[1]))
        temporary = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix="settings-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._document = document
            return True
        except (OSError, ValueError) as error:
            warnings.warn(f"Cannot save {self.path}: {error}", RuntimeWarning)
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError as error:
                    warnings.warn(f"Cannot remove temporary settings file: {error}", RuntimeWarning)

    def reset(self):
        self.path.unlink(missing_ok=True)
        self.writable = True
        self._document = {}
        self.user_presets = {}
        return Config(), {}
