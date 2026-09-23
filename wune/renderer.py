# renderer.py
# 描画ルーチン

import math

import numpy as np
import pygame as pg
from typing import Tuple
from .config import Config
from .layout import calculate_layout
from .presets import PRESETS, get_preset

# ==========================
# 描画系
# ==========================
class LedBarRenderer:
    def __init__(self, surf: pg.Surface, cfg: Config):
        self.surf = surf
        self.cfg = cfg
        self.preset_name = "CUSTOM"
        if cfg.gauge_style not in ("flat", "box"):
            raise ValueError("gauge_style must be flat or box")
        self.channels = cfg.channels
        self._layout = None
        self._led_cache = {}
        # ピーク情報を (ch, bar) で持つ
        self.peak_pos = np.zeros((self.channels, self.cfg.bars), dtype=np.float32)
        self.peak_hold = np.zeros((self.channels, self.cfg.bars), dtype=np.int32)

        # チャンネルラベル用フォント（任意）
        self.font_channel = pg.font.SysFont("Bahnschrift", 16, bold=True)

        # 透明サーフェス（残像用）
        self.trail = None

        # フォント
        # SysFont picks one installed font; it does not fill missing glyphs
        # from other fonts. Prefer Japanese-capable fonts for endpoint names.
        self.font_small = pg.font.SysFont(
            "Meiryo,Yu Gothic UI,Yu Gothic,MS Gothic,"
            "Noto Sans CJK JP,Noto Sans JP,Segoe UI", 15
        )
        self.font_badge = pg.font.SysFont("Bahnschrift", 18, bold=True)
        self.font_logo = pg.font.SysFont("OCR A Extended, OCR A, Consolas", 16)
        self.font_scale = pg.font.SysFont("Consolas, Segoe UI", 12)

        # 表示用インフォテキスト（外部からセット）
        self.info_text = ""
        self.resize(surf)

    def resize(self, surf):
        """Refresh geometry/surfaces without resetting levels, peaks or presets."""
        layout = calculate_layout(surf.get_size(), self.cfg)
        size_changed = self.trail is None or self.trail.get_size() != surf.get_size()
        self.surf = surf
        self.width, self.height = surf.get_size()
        if not size_changed and layout == self._layout:
            return
        self._layout = layout
        self._led_cache.clear()
        self.plots = [pg.Rect(rect) for rect in layout.plots]
        self.bar_w, self.bar_gap = layout.bar_width, layout.bar_gap
        self.led_h = layout.led_height
        self.led_gap = layout.led_gap
        self.ch_y0 = [rect.y for rect in self.plots]
        self.ch_h = self.plots[0].height
        self.trail = pg.Surface(surf.get_size(), pg.SRCALPHA)

    def apply_preset(self, name):
        preset = get_preset(name)
        self.cfg.theme = preset.theme
        self.cfg.gauge_style = preset.gauge_style
        self.cfg.led_shape = preset.led_shape
        self.cfg.led_aspect_ratio = preset.led_aspect_ratio
        self.preset_name = preset.name

    def next_preset(self):
        names = [preset.name for preset in PRESETS]
        index = names.index(self.preset_name) if self.preset_name in names else -1
        self.apply_preset(names[(index + 1) % len(names)])

    def badge_rect(self):
        if not self.cfg.show_badge:
            return None
        width, height = self.font_badge.size(self.preset_name)
        return pg.Rect(self.width - width - 16 - 24, 14, width + 16, height + 8)

    def badge_contains(self, pos):
        rect = self.badge_rect()
        return rect is not None and rect.collidepoint(pos)



    def draw_panel(self):
        self.surf.fill(self.cfg.theme.background)
        # 枠線
        pg.draw.rect(self.surf, self.cfg.theme.border, (8, 8, self.width-16, self.height-16), 2, border_radius=10)
        # ロゴ
        logo = self.font_logo.render("SPECTRA-LED 90", True, self.cfg.theme.logo_text)
        self.surf.blit(logo, (self.cfg.margin_lr, 16))
        # バッジ（GROOVEなど）
        if self.cfg.show_badge:
            text = self.font_badge.render(self.preset_name, True, self.cfg.theme.badge_text)
            tw, th = text.get_size()
            pad = 8
            bx, by = self.badge_rect().topleft
            # グロー風
            pg.draw.rect(self.surf, self.cfg.theme.badge_glow, (bx-2, by-2, tw+pad*2+4, th+pad+4), border_radius=10)
            pg.draw.rect(self.surf, self.cfg.theme.badge_background, (bx, by, tw+pad*2, th+pad), border_radius=10)
            self.surf.blit(text, (bx+pad, by+2))
        # 入力スペックのインフォバー
        if self.cfg.info_enabled:
            bar_rect = pg.Rect(self._layout.info_rect)
            ih = bar_rect.height
            pg.draw.rect(self.surf, self.cfg.theme.info_background, bar_rect, border_radius=8)
            pg.draw.rect(self.surf, self.cfg.theme.info_border, bar_rect, width=1, border_radius=8)
            info_surf = self.font_small.render(self._fit_text(self.info_text, self.font_small, bar_rect.width - 20), True, self.cfg.theme.info_text)
            self.surf.blit(info_surf, (bar_rect.x + 10, bar_rect.y + (ih - info_surf.get_height())//2))


    def update_peaks(self, level_leds: np.ndarray):
        # level_leds: shape (ch, bars)
        for ch in range(self.channels):
            for i, lvl in enumerate(level_leds[ch]):
                if lvl > self.peak_pos[ch, i]:
                    self.peak_pos[ch, i] = lvl
                    self.peak_hold[ch, i] = self.cfg.peak_hold_frames
                else:
                    if self.peak_hold[ch, i] > 0:
                        self.peak_hold[ch, i] -= 1
                    else:
                        self.peak_pos[ch, i] = max(
                            0.0, self.peak_pos[ch, i] - self.cfg.peak_fall_per_frame * self.cfg.leds_per_bar
                        )

    def _freq_to_bar(self, f_hz: float) -> int:
        """対数スケールで周波数→バー番号へ概算マッピング"""
        fmin = max(1.0, self.cfg.min_freq_hz)
        fmax = max(fmin * 1.01, self.cfg.max_freq_hz)
        pos = (math.log10(f_hz) - math.log10(fmin)) / (math.log10(fmax) - math.log10(fmin))
        idx = int(round(pos * (self.cfg.bars - 1)))
        return max(0, min(self.cfg.bars - 1, idx))

    @staticmethod
    def _fit_text(text, font, width):
        if font.size(text)[0] <= width:
            return text
        while text and font.size(text + "…")[0] > width:
            text = text[:-1]
        return text + "…" if text else ""

    def _fmt_freq_label(self, f, with_unit=False):
        label = f"{f / 1000:g}k" if f >= 1000 else f"{f:g}"
        return label + ("Hz" if with_unit else "")

    def draw_freq_scale(self):
        if not self.cfg.show_freq_scale:
            return
        for plot in self.plots:
            base_y = plot.bottom + 4
            occupied = []
            if self.cfg.show_freq_edge_labels:
                for freq, right in ((self.cfg.min_freq_hz, False), (self.cfg.max_freq_hz, True)):
                    label = self._fmt_freq_label(freq, with_unit=right)
                    image = self.font_scale.render(label, True, self.cfg.theme.edge_text)
                    rect = image.get_rect(topleft=(plot.x, base_y + 8))
                    if right:
                        rect.right = plot.right
                    self.surf.blit(image, rect)
                    occupied.append(rect.inflate(8, 0))
            for freq in self.cfg.scale_ticks_hz:
                if not self.cfg.min_freq_hz < freq < self.cfg.max_freq_hz:
                    continue
                x = plot.x + self._freq_to_bar(freq) * (self.bar_w + self.bar_gap) + self.bar_w // 2
                pg.draw.line(self.surf, self.cfg.theme.scale_line, (x, base_y), (x, base_y + 5))
                image = self.font_scale.render(self._fmt_freq_label(freq), True, self.cfg.theme.scale_text)
                rect = image.get_rect(midtop=(x, base_y + 8))
                if rect.left < plot.left or rect.right > plot.right or any(rect.colliderect(other) for other in occupied):
                    continue
                self.surf.blit(image, rect)
                occupied.append(rect.inflate(8, 0))

    def draw_db_labels_ch(self, ch: int):
        if not self.cfg.show_db_scale:
            return
        x_right = self.plots[ch].x - self.cfg.db_label_pad
        y0 = self.ch_y0[ch]
        ch_h = self.ch_h

        denom = (self.cfg.db_max - self.cfg.db_min) or 1.0
        rng = np.arange(self.cfg.db_max, self.cfg.db_min - 0.1, -self.cfg.db_step)
        for db in rng:
            ratio = (db - self.cfg.db_min) / denom
            y = int(y0 + ch_h - ratio * ch_h)
            label = f"{int(db)}"
            ts = self.font_scale.render(label, True, self.cfg.theme.db_text)
            self.surf.blit(ts, (x_right - ts.get_width(), y - ts.get_height() // 2))

        # dB の位置を先に決める（各段の上端から少し下げる）
        unit = self.font_scale.render("dB", True, self.cfg.theme.unit_text)
        unit_x = x_right - unit.get_width()
        unit_y = y0 - unit.get_height() - self.cfg.db_unit_offset  # ←ここはお好みのマージン

        # L/R は dB より“さらに上”に置く
        label = "L" if ch == 0 else ("R" if ch == 1 else f"Ch{ch+1}")
        ts_lr = self.font_channel.render(label, True, self.cfg.theme.edge_text)
        lr_x  = x_right - ts_lr.get_width()
        lr_y  = unit_y - ts_lr.get_height() - 2  # ← dBの上に来る

        # 描画
        self.surf.blit(ts_lr, (lr_x,  lr_y))
        self.surf.blit(unit,  (unit_x, unit_y))

    def draw(self, levels: np.ndarray):
        self.resize(self.surf)
        # バックパネル等
        self.draw_panel()

        # 残像を薄く塗る
        self.trail.fill((*self.cfg.theme.overlay, self.cfg.afterglow_alpha))
        self.surf.blit(self.trail, (0, 0))

        # レベルをLED段数へ変換
        level_leds = levels * self.cfg.leds_per_bar
        self.update_peaks(level_leds)

        for ch in range(self.channels):
            y0 = self.ch_y0[ch]           # この段の“下端”基準
            ch_h = self.ch_h
            
            for b in range(self.cfg.bars):
                x = self.plots[ch].x + b * (self.bar_w + self.bar_gap)
                # 下から上へLEDを描く
                lit = float(level_leds[ch, b])
                for j in range(self.cfg.leds_per_bar):
                    led_ratio = (j + 0.5) / self.cfg.leds_per_bar  # このLEDの高さ割合
                    on_color, off_color = self.cfg.theme.choose_color(led_ratio)
                    y = y0 + ch_h - (j+1) * (self.led_h + self.led_gap) + self.led_gap
                    rect = pg.Rect(x, y, self.bar_w, self.led_h)
                    on = (j < lit)
                    self.draw_led(rect, on_color if on else off_color, on)

                # ピークマーカー（ホールド位置を使う）
                peak = float(self.peak_pos[ch, b])
                if peak > 0:
                    top_index = min(self.cfg.leds_per_bar - 1, max(0, math.ceil(peak) - 1))

                    # トップLED矩形（外枠）
                    led_y = y0 + ch_h - (top_index + 1) * (self.led_h + self.led_gap) + self.led_gap
                    led_rect = self.led_rect(pg.Rect(x, led_y, self.bar_w, self.led_h))

                    # トップLEDの"inner"を算出（draw_ledのパディングと揃える）
                    pad = 1 if (led_rect.w < 6 or led_rect.h < 6) else 2
                    inner = led_rect.inflate(-pad, -pad)

                    # まず“上のスリット”に描けるか判定（= gap >= 1）
                    if self.led_gap >= 1:
                        # トップLEDの上端の1px上（= ギャップ内の最下段）に白線を置く
                        y_gap = led_rect.top - 1

                        # 白線の横幅はバー幅ではなく inner に合わせる
                        overhang = 1  # ← 左右対称に"ちょい出し"したいときは 1〜2 に
                        x_line = inner.left - overhang
                        w_line = inner.width + overhang * 2

                        # バー外枠にクランプ（左右対称を崩さない）
                        x_min = led_rect.left
                        x_max = led_rect.right - 1
                        if x_line < x_min:
                            shift = x_min - x_line
                            x_line = x_min
                            w_line = max(1, w_line - shift)
                        if x_line + w_line - 1 > x_max:
                            w_line = max(1, x_max - x_line + 1)

                        pm_rect = pg.Rect(x_line, y_gap, w_line, 1)

                        s = pg.Surface((pm_rect.w, pm_rect.h), pg.SRCALPHA)
                        s.fill((*self.cfg.theme.peak, 255))
                        self.surf.blit(s, pm_rect)
                    else:
                        # フォールバック：LED内側に“カットアウト→白”で視認性確保
                        if inner.w > 0 and inner.h > 0:
                            y_line = max(inner.top, min(inner.bottom - 1, inner.top))
                            cut = pg.Rect(inner.left, y_line, inner.width, 1)
                            pg.draw.rect(self.surf, self.cfg.theme.peak_cutout, cut)         # 暗線で下地を断つ
                            pg.draw.rect(self.surf, self.cfg.theme.peak, cut)    # その上に白

            # dBラベル（段ごと）
            self.draw_db_labels_ch(ch)

        self.draw_freq_scale()

    def led_rect(self, cell: pg.Rect) -> pg.Rect:
        """Fit a style's proportions inside a layout cell (nearest pixel)."""
        ratio = self.cfg.led_aspect_ratio
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("led_aspect_ratio must be finite and positive")
        if self.cfg.led_shape not in ("rectangle", "rounded", "ellipse"):
            raise ValueError("led_shape must be rectangle, rounded or ellipse")
        width = min(cell.width, cell.height * ratio)
        height = width / ratio
        rect = pg.Rect(0, 0, max(1, round(width)), max(1, round(height)))
        rect.center = cell.center
        return rect

    def draw_led(self, rect: pg.Rect, color: Tuple[int, int, int], on: bool):
        rect = self.led_rect(rect)
        # Rasterize one reference design, then scale all its details together.
        if min(rect.size) < 3:
            pg.draw.rect(self.surf, color, rect)
            return
        key = (rect.size, tuple(color), on, self.cfg.gauge_style, self.cfg.led_shape,
               self.cfg.led_aspect_ratio, repr(self.cfg.theme))
        tile = self._led_cache.get(key)
        if tile is None:
            original = self.surf
            reference = pg.Surface((max(3, round(20 * self.cfg.led_aspect_ratio)), 20), pg.SRCALPHA)
            try:
                self.surf = reference
                self._draw_led_design(reference.get_rect(), color, on)
            finally:
                self.surf = original
            tile = pg.transform.smoothscale(reference, rect.size)
            if len(self._led_cache) >= 256:
                self._led_cache.clear()
            self._led_cache[key] = tile
        self.surf.blit(tile, rect)

    def _draw_led_design(self, rect, color, on):
        if self.cfg.led_shape == "ellipse":
            pg.draw.ellipse(self.surf, color, rect)
            return
        if self.cfg.gauge_style == "box" and self.cfg.led_shape == "rectangle":
            self.draw_box_led(rect, color, on)
            return
        radius = round(min(rect.size) * 0.25) if self.cfg.led_shape == "rounded" else 0
        # ベース（枠）
        pg.draw.rect(self.surf, self.cfg.theme.led_border, rect, border_radius=radius)

        # 小さいLEDでも内側が消えないように、padを自動で絞る
        # 高さ2pxなら pad=0、3〜5pxなら pad=1、それ以上は2
        if rect.height <= 2:
            pad = 0
        elif rect.height <= 5:
            pad = 1
        else:
            pad = 2
        inner = rect.inflate(-pad, -pad)

        # もし内側がゼロ/マイナスになったら、外枠のまま塗る簡易パス
        if inner.width <= 0 or inner.height <= 0:
            pg.draw.rect(self.surf, color, rect,
                        border_radius=max(0, radius - 1))
            return

        # 内側の塗り
        pg.draw.rect(self.surf, color, inner,
                    border_radius=max(0, radius - 1))

        if on:
            # 小さいときはグロスを抑える/描かない
            if inner.height >= 6 and inner.width > 2:
                gloss_h = max(2, min(inner.height - 2, int(inner.height * 0.35)))
                gloss = pg.Rect(inner.x + 1, inner.y + 1, inner.width - 2, gloss_h)
                s = pg.Surface(gloss.size, pg.SRCALPHA)
                s.fill((*self.cfg.theme.highlight, 45))
                self.surf.blit(s, gloss)
            # Leave a colored face between the two outline edges.
            if inner.height >= 4 and inner.width > 2:
                pg.draw.rect(self.surf, self.cfg.theme.led_outline, inner, width=1,
                            border_radius=max(0, radius - 1))

    def draw_box_led(self, rect: pg.Rect, color: Tuple[int, int, int], on: bool):
        """Bevel inside the supplied geometry; no level or peak calculations."""
        if rect.width <= 0 or rect.height <= 0:
            return
        pg.draw.rect(self.surf, color, rect)
        if rect.width < 3 or rect.height < 3:
            return  # Preserve the face color on LEDs too small for a bevel.
        strength = 0.45 if on else 0.15
        light = tuple(round(c + (h - c) * strength)
                      for c, h in zip(color, self.cfg.theme.highlight))
        dark = tuple(round(c + (h - c) * 0.55)
                     for c, h in zip(color, self.cfg.theme.shadow))
        pg.draw.line(self.surf, light, rect.topleft, (rect.right - 1, rect.top))
        pg.draw.line(self.surf, light, rect.topleft, (rect.left, rect.bottom - 1))
        pg.draw.line(self.surf, dark, (rect.left, rect.bottom - 1),
                     (rect.right - 1, rect.bottom - 1))
        pg.draw.line(self.surf, dark, (rect.right - 1, rect.top),
                     (rect.right - 1, rect.bottom - 1))

    def draw_pause_overlay(self):
        # 画面中央に "PAUSED" を半透明で表示
        overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
        overlay.fill((*self.cfg.theme.overlay, 100))
        text = self.font_badge.render("PAUSED", True, self.cfg.theme.pause_text)
        tw, th = text.get_size()
        overlay.blit(text, ((self.width - tw)//2, (self.height - th)//2))
        self.surf.blit(overlay, (0, 0))
