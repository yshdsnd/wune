"""Named appearances; no audio, layout or ballistics settings."""
from dataclasses import dataclass, replace
from .colors import Theme


@dataclass(frozen=True)
class VisualPreset:
    name: str
    theme: Theme
    gauge_style: str
    led_shape: str = "rounded"
    led_aspect_ratio: float = 2.0


PRESETS = (
    VisualPreset("CLASSIC", Theme(), "flat"),
    VisualPreset("BLUE", replace(Theme(), background=(18, 24, 38),
        green_on=(70, 190, 255), yellow_on=(170, 130, 255), red_on=(255, 110, 170),
        green_off=(10, 30, 45), yellow_off=(25, 18, 40), red_off=(40, 15, 25),
        peak=(255, 220, 130), badge_text=(100, 200, 255),
        badge_background=(20, 45, 70)), "box", "rectangle"),
    VisualPreset("AMBER", replace(Theme(), background=(24, 18, 10),
        green_on=(255, 170, 45), yellow_on=(255, 210, 90), red_on=(255, 90, 45),
        green_off=(35, 22, 8), yellow_off=(40, 30, 12), red_off=(40, 16, 8),
        peak=(255, 240, 190), badge_text=(255, 190, 70),
        badge_glow=(40, 25, 10), badge_background=(65, 40, 15)), "flat"),
    VisualPreset("CLASSIC BOX", Theme(), "box", "rectangle"),
)


def get_preset(name):
    for preset in PRESETS:
        if preset.name == name:
            return preset
    raise ValueError(f"Unknown visual preset: {name!r}")
