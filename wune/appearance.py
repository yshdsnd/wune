"""Editable appearance snapshots, independent of either GUI toolkit."""
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
import math

from .colors import Theme
from .i18n import MessageError
from .config import Config
from .presets import PRESETS, VisualPreset, get_preset
from .ballistics import MOTION_LIMITS, valid_motion


BUILTINS = {p.name: p for p in PRESETS}
LAYOUT_FIELDS = ("language", "spectrum_orientation", "channel_layout", "info_enabled", "info_position", "limit_to_20khz")
STYLE_FIELDS = ("gauge_style", "led_shape", "led_aspect_ratio")
COLOR_FIELDS = tuple(f.name for f in fields(Theme) if not f.name.startswith("th_"))


def validate_name(name):
    if not isinstance(name, str) or not name.strip() or len(name) > 40:
        raise MessageError("error.name_length")
    name = name.strip()
    if name in BUILTINS or name == "CUSTOM" or any(ord(c) < 32 for c in name):
        raise MessageError("error.name_reserved")
    return name


def decode_preset(name, data):
    """Validate persisted colors before they can reach pygame."""
    validate_name(name)
    if not isinstance(data, dict) or not isinstance(data.get("theme", {}), dict):
        raise ValueError("Invalid user theme")
    values = asdict(Theme())
    for key, value in data.get("theme", {}).items():
        if key in COLOR_FIELDS:
            if not isinstance(value, (list, tuple)) or len(value) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in value):
                raise ValueError(f"Invalid color: {key}")
            values[key] = tuple(value)
        elif key in ("th_yellow", "th_red"):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("Invalid color threshold")
            values[key] = value
    if not 0 <= values["th_yellow"] <= values["th_red"] <= 1:
        raise ValueError("Invalid color thresholds")
    return VisualPreset(name, Theme(**values))


def encode_preset(preset):
    data = asdict(preset)
    del data["name"]
    return data


@dataclass
class AppearanceState:
    layout: dict
    preset: VisualPreset
    user_presets: dict
    motion: dict
    style: dict

    @classmethod
    def capture(cls, cfg, name, user_presets):
        return cls({key: getattr(cfg, key) for key in LAYOUT_FIELDS},
                   VisualPreset(name, cfg.theme),
                   deepcopy(user_presets), {key: getattr(cfg, key) for key in MOTION_LIMITS},
                   {key: getattr(cfg, key) for key in STYLE_FIELDS})

    def apply(self, cfg):
        for key, value in self.layout.items():
            setattr(cfg, key, value)
        cfg.theme = self.preset.theme
        for key in STYLE_FIELDS:
            setattr(cfg, key, self.style[key])
        cfg.initial_preset = None if self.preset.name == "CUSTOM" else self.preset.name
        for key, value in self.motion.items():
            setattr(cfg, key, value)


class AppearanceDraft:
    def __init__(self, state):
        self.state = deepcopy(state)

    def snapshot(self):
        return deepcopy(self.state)

    def names(self):
        names = [*BUILTINS, *self.state.user_presets]
        if self.state.preset.name == "CUSTOM":
            names.append("CUSTOM")
        return names

    def select(self, name):
        if name != "CUSTOM":
            self.state.preset = BUILTINS[name] if name in BUILTINS else self.state.user_presets[name]

    def available_name(self, base):
        name, index = base[:35], 2
        while name in self.names():
            name = f"{base[:35]} {index}"
            index += 1
        return name

    def create(self, name, source=None):
        name = validate_name(name)
        if name in self.names():
            raise MessageError("error.name_exists")
        self.state.preset = replace(source or get_preset("CLASSIC"), name=name)
        self.state.user_presets[name] = self.state.preset

    def rename(self, name):
        old = self.state.preset.name
        if old not in self.state.user_presets:
            raise MessageError("error.builtin_rename")
        if name == old:
            return
        self.create(name, self.state.preset)
        del self.state.user_presets[old]

    def delete(self):
        name = self.state.preset.name
        if name not in self.state.user_presets:
            raise MessageError("error.builtin_delete")
        del self.state.user_presets[name]
        self.select("CLASSIC")

    def edit(self, **changes):
        preset = replace(self.state.preset, **changes)
        if preset == self.state.preset:
            return
        if preset.name not in self.state.user_presets:
            preset = replace(preset, name=self.available_name(preset.name + " copy"))
        # Reuse file validation for input from spinboxes and the color picker.
        preset = decode_preset(preset.name, encode_preset(preset))
        self.state.preset = preset
        self.state.user_presets[preset.name] = preset

    def edit_style(self, **changes):
        from .settings import valid_preference
        if any(key not in STYLE_FIELDS or not valid_preference(key, value) for key, value in changes.items()):
            raise MessageError("error.style")
        self.state.style.update(changes)

    def reset(self):
        # Reset visible preferences without deleting the user's theme library.
        language = self.state.layout["language"]
        motion = self.state.motion
        self.state = AppearanceState.capture(Config(), "CLASSIC", self.state.user_presets)
        self.state.motion = motion
        self.state.layout["language"] = language

    def edit_motion(self, values):
        if any(key not in MOTION_LIMITS or not valid_motion(key, value) for key, value in values.items()):
            raise MessageError("error.motion")
        self.state.motion.update(values)

    def reset_motion(self):
        defaults = Config()
        self.state.motion = {key: getattr(defaults, key) for key in MOTION_LIMITS}
