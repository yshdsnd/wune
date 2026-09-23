"""Spectrum geometry only: resizing never changes band or LED counts."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpectrumLayout:
    plots: tuple
    bar_width: int
    bar_gap: int
    led_height: int
    info_rect: tuple | None


def _dimensions(cfg):
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
    info = max(28, cfg.info_height) + 12 if cfg.info_enabled else 0
    return cols, rows, margin, left, header, scale, info


def minimum_window_size(cfg):
    cols, rows, margin, left, header, scale, info = _dimensions(cfg)
    min_gap = min(1, cfg.bar_gap)
    plot_w = cfg.bars * 3 + (cfg.bars - 1) * min_gap
    plot_h = cfg.leds_per_bar * max(3, cfg.min_led_height) + (cfg.leds_per_bar - 1) * cfg.led_gap
    width = cols * (left + plot_w + 16) + (cols - 1) * cfg.channel_gap
    height = 2 * margin + 40 + info + rows * (header + plot_h + scale) + (rows - 1) * cfg.channel_gap
    return max(480, width), height


def clamp_window_size(size, cfg):
    return tuple(max(int(value), limit) for value, limit in zip(size, minimum_window_size(cfg)))


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
    gap = min(cfg.bar_gap, max(1, available_w // (cfg.bars * 5)))
    bar_w = (available_w - (cfg.bars - 1) * gap) // cfg.bars
    plot_w = cfg.bars * bar_w + (cfg.bars - 1) * gap
    led_h = (cell_h - header - scale - (cfg.leds_per_bar - 1) * cfg.led_gap) // cfg.leds_per_bar
    plot_h = cfg.leds_per_bar * led_h + (cfg.leds_per_bar - 1) * cfg.led_gap
    plots = []
    for ch in range(cfg.channels):
        col, row = (ch, 0) if cols > 1 else (0, ch)
        x = col * (cell_w + cfg.channel_gap) + left
        y = top + row * (cell_h + cfg.channel_gap) + cell_h - scale - plot_h
        plots.append((x, y, plot_w, plot_h))
    return SpectrumLayout(tuple(plots), bar_w, gap, led_h, info_rect)
