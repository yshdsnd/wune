"""Spectrum geometry only: resizing never changes band or LED counts."""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class SpectrumLayout:
    plots: tuple
    bar_width: int
    bar_gap: int
    led_height: int
    info_rect: tuple | None
    led_gap: int
    now_playing_rect: tuple | None = None
    ui_scale: float = 1.0


def calculate_ui_scale(size) -> float:
    """Calculate UI scaling factor relative to 1280x800 baseline.

    Clamped to [1.0, 3.0] to preserve legibility on small windows and prevent
    oversized elements on high-resolution/ultrawide displays.
    """
    width, height = size
    ratio = min(width / 1280.0, height / 800.0)
    return max(1.0, min(3.0, ratio))


def info_bar_base_height(cfg) -> int:
    font_size = getattr(cfg, "info_font_size", 14)
    needed = round(font_size * 1.5) + 7
    custom_height = getattr(cfg, "info_height", 28)
    if custom_height != 28:
        return max(20, custom_height, needed)
    return max(20, needed)


def _dimensions(cfg):
    if cfg.spectrum_orientation not in ("frequency_horizontal", "frequency_vertical"):
        raise ValueError("spectrum_orientation must be frequency_horizontal or frequency_vertical")
    if cfg.channel_layout not in ("vertical", "horizontal"):
        raise ValueError("channel_layout must be vertical or horizontal")
    if cfg.display_channels < 1 or cfg.bars < 1 or cfg.leds_per_bar < 1:
        raise ValueError("channels, bars and leds_per_bar must be positive")
    if cfg.led_gap < 0 or cfg.bar_gap < 0 or cfg.channel_gap < 0:
        raise ValueError("layout gaps must be non-negative")
    cols = cfg.display_channels if cfg.channel_layout == "horizontal" else 1
    rows = 1 if cfg.channel_layout == "horizontal" else cfg.display_channels
    margin = max(16, cfg.margin_tb)
    left = max(50, cfg.margin_lr + 10)
    header = max(44, cfg.header_reserved)
    scale = max(30, cfg.scale_reserved) if cfg.show_freq_scale else 0
    if cfg.spectrum_orientation == "frequency_vertical":
        left = max(72, cfg.margin_lr)
        scale = max(30, cfg.scale_reserved) if cfg.show_db_scale else 0
    base_info = info_bar_base_height(cfg)
    info = base_info + 12 if cfg.info_enabled else 0
    now_playing = base_info + 8 if cfg.show_now_playing else 0
    return cols, rows, margin, left, header, scale, info, now_playing


def grid_counts(cfg):
    if cfg.spectrum_orientation == "frequency_vertical":
        return cfg.leds_per_bar, cfg.bars
    return cfg.bars, cfg.leds_per_bar


def minimum_window_size(cfg):
    cols, rows, margin, left, header, scale, info, now_playing = _dimensions(cfg)
    ratio = cfg.led_aspect_ratio
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("led_aspect_ratio must be finite and positive")
    h = max(3, cfg.min_led_height, math.ceil(3 / ratio))
    w = max(3, round(h * ratio))
    led_gap = max(1, round((h - 0.01) / 4))
    bar_gap = getattr(cfg, "bar_gap", 2)
    if cfg.spectrum_orientation == "frequency_vertical":
        x_gap, y_gap = led_gap, bar_gap
    else:
        x_gap, y_gap = bar_gap, led_gap
    nx, ny = grid_counts(cfg)
    plot_w = nx * w + (nx - 1) * x_gap
    plot_h = ny * h + (ny - 1) * y_gap
    width = cols * (left + plot_w + 16) + (cols - 1) * cfg.channel_gap
    height = 2 * margin + 40 + info + now_playing + rows * (header + plot_h + scale) + (rows - 1) * cfg.channel_gap
    return max(480, width), height


def clamp_window_size(size, cfg):
    return tuple(max(int(value), limit) for value, limit in zip(size, minimum_window_size(cfg)))


def _vertical_overhead(size, cfg, scale: float) -> tuple[int, int]:
    """Calculate the safe top boundary and bottom clearance margin for plots."""
    width, height = size
    cols, rows, margin, left, header, scale_reserved, info, now_playing = _dimensions(cfg)
    auto_scale = getattr(cfg, "auto_scale_fonts", True)
    bar_scale = scale if auto_scale else 1.0
    bar_h = max(20, round(info_bar_base_height(cfg) * bar_scale))
    gap = max(8, round(8 * scale))
    header_extra = max(0, round(46 * (scale - 1.0)))
    scale_extra = max(0, round(28 * (scale - 1.0)))
    top = margin + 40
    top_bar_bottom = 0
    if now_playing:
        np_y = max(top, round(54 * scale) + 4)
        top_bar_bottom = np_y + bar_h
    if info and cfg.info_position == "top":
        info_top = (top_bar_bottom + max(6, round(6 * scale))) if top_bar_bottom > 0 else max(top, round(44 * scale) + 4)
        top_bar_bottom = info_top + bar_h
    min_group_y = (top_bar_bottom + gap + header_extra) if top_bar_bottom > 0 else margin + 40
    bottom_extra = (gap + scale_extra + bar_h + margin) if (info and cfg.info_position == "bottom") else margin
    return min_group_y, bottom_extra


def fit_window_size(size, cfg):
    """Fit every orientation to its grid; resizing the window controls scale.

    One common integer LED height determines widths and gaps, so the grid
    grows in pixel steps without changing its design or band count.
    """
    size = clamp_window_size(size, cfg)
    layout = calculate_layout(size, cfg)
    cols, rows, margin, left, header, scale, info, now_playing = _dimensions(cfg)
    _, _, plot_w, plot_h = layout.plots[0]
    width = cols * (left + plot_w + 16) + (cols - 1) * cfg.channel_gap
    total_h = rows * (header + plot_h + scale) + (rows - 1) * cfg.channel_gap
    base_height = 2 * margin + 40 + info + now_playing + total_h
    h = base_height
    for _ in range(5):
        cur_scale = calculate_ui_scale((width, h))
        min_gy, bot_extra = _vertical_overhead((width, h), cfg, cur_scale)
        needed_h = max(base_height, min_gy + bot_extra + total_h)
        if needed_h == h:
            break
        h = needed_h
    return clamp_window_size((width, h), cfg)


def channel_mode_window_size(size, previous_cfg, cfg):
    """Keep LED scale when adding/removing the second spectrum.

    Fitting two plots inside the old single-plot window would shrink both
    axes on every round trip. Instead, rebuild the window around the same
    LED height, including the new number of rows/columns and their margins.
    """
    layout = calculate_layout(clamp_window_size(size, previous_cfg), previous_cfg)
    led_h = layout.led_height
    led_w = max(1, round(led_h * cfg.led_aspect_ratio))
    led_gap = layout.led_gap
    bar_gap = getattr(cfg, "bar_gap", 2)
    if cfg.spectrum_orientation == "frequency_vertical":
        x_gap, y_gap = led_gap, bar_gap
    else:
        x_gap, y_gap = bar_gap, led_gap
    nx, ny = grid_counts(cfg)
    plot_w = nx * led_w + (nx - 1) * x_gap
    plot_h = ny * led_h + (ny - 1) * y_gap
    cols, rows, margin, left, header, scale, info, now_playing = _dimensions(cfg)
    width = cols * (left + plot_w + 16) + (cols - 1) * cfg.channel_gap
    total_h = rows * (header + plot_h + scale) + (rows - 1) * cfg.channel_gap
    base_height = 2 * margin + 40 + info + now_playing + total_h
    h = base_height
    for _ in range(5):
        cur_scale = calculate_ui_scale((width, h))
        min_gy, bot_extra = _vertical_overhead((width, h), cfg, cur_scale)
        needed_h = max(base_height, min_gy + bot_extra + total_h)
        if needed_h == h:
            break
        h = needed_h
    return clamp_window_size((width, h), cfg)


def layout_has_clearance(layout, cfg, size) -> bool:
    """Check if the layout fits within the window without colliding with info bars or labels."""
    width, height = size
    scale = getattr(layout, "ui_scale", 1.0)
    gap = max(4, round(4 * scale))

    # Top boundary check (Now playing bar / top info bar)
    top_limit = 0
    if layout.now_playing_rect:
        top_limit = max(top_limit, layout.now_playing_rect[1] + layout.now_playing_rect[3])
    if layout.info_rect and cfg.info_position == "top":
        top_limit = max(top_limit, layout.info_rect[1] + layout.info_rect[3])

    if top_limit > 0:
        db_unit_offset = max(6, round(cfg.db_unit_offset * scale))
        unit_h = max(10, round(12 * scale))
        lr_h = max(12, round(16 * scale))
        top_label_y = layout.plots[0][1] - unit_h - db_unit_offset - lr_h - max(2, round(2 * scale))
        if top_label_y < top_limit + gap:
            return False

    # Bottom boundary check (device info bar / bottom margin)
    bottom_limit = height - max(16, cfg.margin_tb)
    if layout.info_rect and cfg.info_position == "bottom":
        bottom_limit = min(bottom_limit, layout.info_rect[1])

    p_last = layout.plots[-1]
    p_last_bottom = p_last[1] + p_last[3]
    show_bottom_scale = cfg.show_db_scale if cfg.spectrum_orientation == "frequency_vertical" else cfg.show_freq_scale
    if show_bottom_scale:
        text_pad = max(6, round(8 * scale))
        scale_font_h = max(10, round(14 * scale))
        labels_bottom = p_last_bottom + max(3, round(4 * scale)) + text_pad + scale_font_h
    else:
        labels_bottom = p_last_bottom

    if bottom_limit < labels_bottom + gap:
        return False

    return True


def calculate_optimal_leds_per_bar(size, cfg, min_leds: int = 20, max_leds: int = 100) -> int:
    """Calculate the optimal leds_per_bar to minimize blank space in the window.

    Finds the segment count that maximizes the utilized spectrum area (minimizing
    blank space) within the current window dimensions.
    """
    # Clamp against minimum possible window size (at leds_per_bar=10) rather than
    # current cfg.leds_per_bar so that a large current segment count does not
    # inflate the window size when shrinking.
    min_cfg = replace(cfg, leds_per_bar=10)
    size = clamp_window_size(size, min_cfg)
    best_k = getattr(cfg, "leds_per_bar", 20)
    best_area = -1
    for search_min in (min_leds, 10):
        for k in range(search_min, max_leds + 1):
            c = replace(cfg, leds_per_bar=k)
            if tuple(size) != clamp_window_size(size, c):
                continue
            try:
                layout = calculate_layout(size, c)
            except ValueError:
                continue
            if not layout_has_clearance(layout, c, size):
                continue
            area = layout.plots[0][2] * layout.plots[0][3]
            if area > best_area:
                best_area = area
                best_k = k
        if best_area > 0:
            break
    return max(10, min(max_leds, best_k))


def calculate_layout(size, cfg):
    width, height = size
    if tuple(size) != clamp_window_size(size, cfg):
        raise ValueError("Drawable area is smaller than the minimum spectrum layout")
    scale = calculate_ui_scale(size)
    cols, rows, margin, left, header, scale_reserved, info, now_playing = _dimensions(cfg)
    auto_scale = getattr(cfg, "auto_scale_fonts", True)
    bar_scale = scale if auto_scale else 1.0
    bar_h = max(20, round(info_bar_base_height(cfg) * bar_scale))

    top = margin + 40
    bottom = height - margin

    now_playing_rect = None
    if now_playing:
        np_y = max(top, round(54 * scale) + 4)
        now_playing_rect = (16, np_y, width - 32, bar_h)
        top += now_playing

    info_rect = None
    if info:
        if cfg.info_position == "top":
            info_top = (now_playing_rect[1] + now_playing_rect[3] + max(6, round(6 * scale))) if now_playing_rect else max(top, round(44 * scale) + 4)
            info_rect = (16, info_top, width - 32, bar_h)
            top += info
        else:
            info_rect = (16, height - margin - bar_h, width - 32, bar_h)
            bottom -= info

    min_group_y, bottom_extra = _vertical_overhead(size, cfg, scale)
    max_plot_bottom = height - bottom_extra

    cell_w = (width - (cols - 1) * cfg.channel_gap) // cols
    cell_h = (bottom - top - (rows - 1) * cfg.channel_gap) // rows
    real_cell_h = (max_plot_bottom - min_group_y - (rows - 1) * cfg.channel_gap) // rows
    available_w = cell_w - left - 16
    available_h = min(cell_h - header - scale_reserved, max(0, real_cell_h - header - scale_reserved))
    # Scale the complete dense grid, not LEDs independently inside stretched cells.
    ratio = cfg.led_aspect_ratio
    nx, ny = grid_counts(cfg)
    bar_gap = getattr(cfg, "bar_gap", 2)
    led_h = int(min(available_w / (nx * ratio), available_h / ny))
    while True:
        bar_w = max(1, round(led_h * ratio))
        led_gap = max(1, round((led_h - 0.01) / 4))
        if cfg.spectrum_orientation == "frequency_vertical":
            x_gap, y_gap = led_gap, bar_gap
        else:
            x_gap, y_gap = bar_gap, led_gap
        plot_w = nx * bar_w + (nx - 1) * x_gap
        plot_h = ny * led_h + (ny - 1) * y_gap
        if plot_w <= available_w and plot_h <= available_h:
            break
        led_h -= 1

    plots = []
    packed_w = left + plot_w + 16
    packed_h = header + plot_h + scale_reserved
    total_w = cols * packed_w + (cols - 1) * cfg.channel_gap
    total_h = rows * packed_h + (rows - 1) * cfg.channel_gap

    # Group centering with safety margins to prevent overlapping with info bars or window edges
    min_group_x = max(0, round(32 * (scale - 1.0)))
    group_x = max(min_group_x, (width - total_w) // 2)
    group_y = min_group_y + max(0, (max_plot_bottom - min_group_y - total_h)) // 2

    for ch in range(cfg.display_channels):
        col, row = (ch, 0) if cols > 1 else (0, ch)
        x = group_x + col * (packed_w + cfg.channel_gap) + left
        y = group_y + row * (packed_h + cfg.channel_gap) + header
        plots.append((x, y, plot_w, plot_h))
    return SpectrumLayout(tuple(plots), bar_w, bar_gap, led_h, info_rect, led_gap, now_playing_rect, scale)
