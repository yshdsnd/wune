"""Window size presets and registry."""
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WindowPreset:
    id: str
    name_key: str
    width: int
    height: int
    description: str = ""
    leds_per_bar: int | None = None
    led_aspect_ratio: float | None = None
    bar_gap: int | None = None
    adaptive_fill: bool | None = None
    is_custom: bool = False

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)


BUILTIN_WINDOW_PRESETS: tuple[WindowPreset, ...] = (
    WindowPreset(
        id="compact",
        name_key="presets.window.compact",
        width=960,
        height=540,
        description="Compact / Sub-display (960 × 540)",
    ),
    WindowPreset(
        id="standard",
        name_key="presets.window.standard",
        width=1280,
        height=800,
        description="Standard (1280 × 800)",
    ),
    WindowPreset(
        id="fhd",
        name_key="presets.window.fhd",
        width=1920,
        height=1080,
        description="Full HD (1920 × 1080)",
    ),
    WindowPreset(
        id="ultrawide",
        name_key="presets.window.ultrawide",
        width=2560,
        height=1080,
        description="Ultrawide 21:9 (2560 × 1080)",
    ),
    WindowPreset(
        id="wqhd",
        name_key="presets.window.wqhd",
        width=2560,
        height=1440,
        description="WQHD (2560 × 1440)",
    ),
    WindowPreset(
        id="4k",
        name_key="presets.window.4k",
        width=3840,
        height=2160,
        description="4K UHD (3840 × 2160)",
    ),
)


def get_all_window_presets(user_presets: dict | None = None) -> list[WindowPreset]:
    """Return all window presets including built-ins and any custom user presets."""
    presets = list(BUILTIN_WINDOW_PRESETS)
    if user_presets:
        for preset_id, data in user_presets.items():
            if isinstance(data, dict) and "width" in data and "height" in data:
                presets.append(WindowPreset(
                    id=preset_id,
                    name_key=data.get("name_key", preset_id),
                    width=int(data["width"]),
                    height=int(data["height"]),
                    description=data.get("description", f"{data['width']} × {data['height']}"),
                    leds_per_bar=data.get("leds_per_bar"),
                    led_aspect_ratio=data.get("led_aspect_ratio"),
                    bar_gap=data.get("bar_gap"),
                    adaptive_fill=data.get("adaptive_fill"),
                    is_custom=True,
                ))
    return presets


def find_window_preset(preset_id: str, user_presets: dict | None = None) -> WindowPreset | None:
    for preset in get_all_window_presets(user_presets):
        if preset.id == preset_id:
            return preset
    return None
