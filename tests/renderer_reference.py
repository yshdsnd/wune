'''Pre-optimization draw path from main 19b53b9, retained as a pixel oracle.'''
import math
import numpy as np
import pygame as pg
from typing import Tuple
from wune.renderer import LedBarRenderer

class ReferenceRenderer(LedBarRenderer):
    def draw(self, levels: np.ndarray, dt=None):
        self.resize(self.surf)
        # バックパネル等
        self.draw_panel()

        # 残像を薄く塗る
        self.trail.fill((*self.cfg.theme.overlay, self.cfg.afterglow_alpha))
        self.surf.blit(self.trail, (0, 0))

        # レベルをLED段数へ変換
        level_leds = levels * self.cfg.leds_per_bar
        self.update_peaks(level_leds, dt)

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
                    rect = self.cell_rect(ch, b, j)
                    on = (j < lit)
                    self.draw_led(rect, on_color if on else off_color, on)

                # ピークマーカー（ホールド位置を使う）
                peak = float(self.peak_pos[ch, b])
                if peak > 0:
                    top_index = min(self.cfg.leds_per_bar - 1, max(0, math.ceil(peak) - 1))

                    if self.cfg.spectrum_orientation == "frequency_vertical":
                        led_rect = self.cell_rect(ch, b, top_index)
                        marker = pg.Rect(led_rect.right, led_rect.top, 1, led_rect.height)
                        pg.draw.rect(self.surf, self.cfg.theme.peak, marker)
                        continue

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


