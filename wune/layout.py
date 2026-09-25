"""Spectrum geometry only: resizing never changes band or LED counts."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SpectrumLayout:
    plots: tuple
    bar_width: int
    bar_gap: int
    led_height: int
    info_rect: tuple | None
    led_gap: int


def _dimensions(cfg):
    if cfg.spectrum_orientation not in ("frequency_horizontal", "frequency_vertical"):
        raise ValueError("spectrum_orientation must be frequency_horizontal or frequency_vertical")
    if cfg.channel_layout not in ("vertical", "horizontal"):
        raise ValueError("channel_layout must be vertical or horizontal")
    if cfg.channels < 1 or cfg.bars < 1 or cfg.leds_per_bar < 1:
        raise ValueError("channels, bars and leds_per_bar must be positive")
    if cfg.led_gap < 0 or cfg.bar_gap < 0 or cfg.channel_gap < 0:
        raise ValueError("layout gaps must be non-negative")
    cols = cfg.channels if cfg.channel_layout == "horizontal" else 1
    rows = 1 if cfg.channel_layout == "horizontal" else cfg.channels
    margin = max(16, cfg.margin_tb)
    left = max(40, cfg.margin_lr)
    header = max(44, cfg.header_reserved)
    scale = max(30, cfg.scale_reserved) if cfg.show_freq_scale else 0
    if cfg.spectrum_orientation == "frequency_vertical":
        left = max(72, cfg.margin_lr)
        scale = max(30, cfg.scale_reserved) if cfg.show_db_scale else 0
    info = max(28, cfg.info_height) + 12 if cfg.info_enabled else 0
    return cols, rows, margin, left, header, scale, info


def grid_counts(cfg):
    if cfg.spectrum_orientation == "frequency_vertical":
        return cfg.leds_per_bar, cfg.bars
    return cfg.bars, cfg.leds_per_bar


def minimum_window_size(cfg):
    cols, rows, margin, left, header, scale, info = _dimensions(cfg)
    ratio = cfg.led_aspect_ratio
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("led_aspect_ratio must be finite and positive")
    h = max(3, cfg.min_led_height, math.ceil(3 / ratio))
    w = max(3, round(h * ratio))
    gap = max(1, round(h / 4))
    nx, ny = grid_counts(cfg)
    plot_w = nx * w + (nx - 1) * gap
    plot_h = ny * h + (ny - 1) * gap
    width = cols * (left + plot_w + 16) + (cols - 1) * cfg.channel_gap
    height = 2 * margin + 40 + info + rows * (header + plot_h + scale) + (rows - 1) * cfg.channel_gap
    return max(480, width), height


def clamp_window_size(size, cfg):
    return tuple(max(int(value), limit) for value, limit in zip(size, minimum_window_size(cfg)))


def fit_window_size(size, cfg):
    """Fit every orientation to its grid; resizing the window controls scale.

    One common integer LED height determines widths and gaps, so the grid
    grows in pixel steps without changing its design or band count.
    """
    size = clamp_window_size(size, cfg)
    layout = calculate_layout(size, cfg)
    cols, rows, margin, left, header, scale, info = _dimensions(cfg)
    _, _, plot_w, plot_h = layout.plots[0]
    width = cols * (left + plot_w + 16) + (cols - 1) * cfg.channel_gap
    height = 2 * margin + 40 + info + rows * (header + plot_h + scale) + (rows - 1) * cfg.channel_gap
    return clamp_window_size((width, height), cfg)


def calculate_layout(size, cfg):
    width, height = size
    if tuple(size) != clamp_window_size(size, cfg):
        raise ValueError("Drawable area is smaller than the minimum spectrum layout")
    cols, rows, margin, left, header, scale, info = _dimensions(cfg)
    top = margin + 40
    bottom = height - margin
    info_rect = None
    if info:
        if cfg.info_position == "top":
            info_rect = (16, top, width - 32, info - 12)
            top += info
        else:
            bottom -= info
            info_rect = (16, bottom + 12, width - 32, info - 12)
    cell_w = (width - (cols - 1) * cfg.channel_gap) // cols
    cell_h = (bottom - top - (rows - 1) * cfg.channel_gap) // rows
    available_w = cell_w - left - 16
    available_h = cell_h - header - scale
    # Scale the complete dense grid, not LEDs independently inside stretched cells.
    ratio = cfg.led_aspect_ratio
    nx, ny = grid_counts(cfg)
    led_h = int(min(available_w / (nx * ratio), available_h / ny))
    while True:
        bar_w = max(1, round(led_h * ratio))
        gap = max(1, round(led_h / 4))
        plot_w = nx * bar_w + (nx - 1) * gap
        plot_h = ny * led_h + (ny - 1) * gap
        if plot_w <= available_w and plot_h <= available_h:
            break
        led_h -= 1
    plots = []
    packed_w = left + plot_w + 16
    packed_h = header + plot_h + scale
    group_x = (width - (cols * packed_w + (cols-1)*cfg.channel_gap)) // 2
    group_y = top + (bottom-top - (rows*packed_h + (rows-1)*cfg.channel_gap)) // 2
    for ch in range(cfg.channels):
        col, row = (ch, 0) if cols > 1 else (0, ch)
        x = group_x + col * (packed_w + cfg.channel_gap) + left
        y = group_y + row * (packed_h + cfg.channel_gap) + header
        plots.append((x, y, plot_w, plot_h))
    return SpectrumLayout(tuple(plots), bar_w, gap, led_h, info_rect, gap)
