"""
ウネウネLEDスペアナ v0.1
- ステップ1〜3を実装：バー描画 / 色分け / ピークホールド
- 90年代ミニコンポ風の見た目を意識したLEDセグメント描画
- いまは『それっぽく動く』ための疑似スペクトラム（擬似ノイズ＋波）

依存: pygame, numpy
  pip install pygame numpy

実行: python speana.py
終了: ESC / Q / 閉じるボタン
フルスクリーン切替: F11
一時停止: Space

※ pygame の Python 3.13 サポートは環境により差異がある可能性があります。
   もしインストールでコケる場合は 3.12 を一時的に利用してください。
"""
from __future__ import annotations
import math
import random
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygame as pg

# ==========================
# 設定群（好みに合わせて調整）
# ==========================
@dataclass
class Config:
    width: int = 1280
    height: int = 480
    fps: int = 60
    bars: int = 64                # バー本数
    leds_per_bar: int = 32        # 縦のLED個数
    led_gap: int = 2              # LEDの隙間(px)
    margin_lr: int = 40           # 左右マージン
    margin_tb: int = 60           # 上下マージン（インフォバー分を少し広めに）
    bar_gap: int = 6              # バー同士の隙間
    corner_radius: int = 4        # LED角の丸み

    # インフォバー（入力仕様）
    info_enabled: bool = True
    info_height: int = 28
    info_position: str = "bottom"    # "top" or "bottom"

    # 色分けしきい値（割合）
    th_yellow: float = 0.60
    th_red: float = 0.85

    # レベル挙動
    attack: float = 0.30          # 上昇スピード（0〜1）
    release: float = 0.08         # 下降スピード（0〜1）

    # ピークホールド
    peak_hold_frames: int = 18    # 頂点を保持するフレーム数
    peak_fall_per_frame: float = 0.9 / 32  # 1フレームごとの落下量（LED個数比率）

    # ほんのり残像（画面に黒を薄く重ねる）
    afterglow_alpha: int = 35     # 0で残像無し、値が大きいほど早く消える（0〜255）

    # 90s風ラベルなど
    show_badge: bool = True
    badge_text: str = "GROOVE"
    
    # 周波数スケール（表示用）
    show_freq_scale: bool = True
    min_freq_hz: float = 20.0
    max_freq_hz: float = 48000.0
    scale_ticks_hz: tuple = (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000, 48000)
    show_freq_edge_labels: bool = True  # 左端/右端に最小・最大の周波数ラベルを描く

    # 縦軸（dB）ラベル
    show_db_scale: bool = True
    db_min: float = -60.0   # 下端
    db_max: float = 0.0     # 上端
    db_step: float = 10.0   # 間隔
    db_label_pad: int = 6   # ラベルの右端とバー領域の間隔(px)
    db_unit_offset: int = 10   # dB単位ラベルを上にずらす量

CFG = Config()

# LEDカラー定義（オン/オフ）
GREEN_ON  = ( 80, 255, 110)
YELLOW_ON = (255, 240,  90)
RED_ON    = (255,  80,  80)

GREEN_OFF  = ( 10,  28,  12)
YELLOW_OFF = ( 30,  28,  10)
RED_OFF    = ( 28,  10,  10)

BORDER_DARK = (12, 12, 12)
PANEL_BG    = (6, 6, 10)
GLASS_WHITE = (255, 255, 255)

# ==========================
# ユーティリティ
# ==========================
def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def choose_color(level_ratio: float) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """レベル位置に応じてLEDカラー（オン/オフ）を返す"""
    if level_ratio >= CFG.th_red:
        return RED_ON, RED_OFF
    elif level_ratio >= CFG.th_yellow:
        return YELLOW_ON, YELLOW_OFF
    else:
        return GREEN_ON, GREEN_OFF


# ==========================
# スペクトラムの『それっぽい』生成器
# ==========================
class FakeSpectrum:
    """雰囲気重視のフェイクスペクトラム発生器。
    複数の波＋ノイズを合成して、帯域方向にウネウネさせる。
    """
    def __init__(self, bars: int):
        self.bars = bars
        self.t = 0.0
        # 個体差（位相・速度・オフセット）
        rng = np.random.default_rng()
        self.phase1 = rng.uniform(0, math.tau)
        self.phase2 = rng.uniform(0, math.tau)
        self.speed1 = rng.uniform(0.4, 0.8)
        self.speed2 = rng.uniform(0.8, 1.6)
        self.band_phase = rng.uniform(0.1, 0.35)
        self.noise = rng.normal(0.0, 0.12, size=bars)
        self.smooth = np.zeros(bars, dtype=np.float32)

    def step(self, dt: float) -> np.ndarray:
        self.t += dt
        i = np.arange(self.bars, dtype=np.float32)
        # 2種の正弦波でうねりを作り、帯域方向に位相差
        w1 = 0.5 * (1.0 + np.sin(self.t * self.speed1 + i * self.band_phase + self.phase1))
        w2 = 0.5 * (1.0 + np.sin(self.t * self.speed2 - i * (self.band_phase*0.7) + self.phase2))
        base = 0.55 * w1 + 0.45 * w2
        # 緩やかな時間ノイズ
        self.noise = 0.98 * self.noise + 0.02 * np.random.normal(0.0, 0.9, size=self.bars)
        n = 0.07 * self.noise
        raw = clamp_array(base + n, 0.0, 1.0)
        # ゆっくり平滑化（縦方向の残像とは別）
        self.smooth = 0.85 * self.smooth + 0.15 * raw
        return self.smooth


def clamp_array(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


# ==========================
# 描画系
# ==========================
class LedBarRenderer:
    def __init__(self, surf: pg.Surface, cfg: Config):
        self.surf = surf
        self.cfg = cfg
        # バー配置の算出
        inner_w = cfg.width - cfg.margin_lr * 2
        self.bar_w = (inner_w - (cfg.bars - 1) * cfg.bar_gap) // cfg.bars
        self.bar_x0 = cfg.margin_lr
        self.bar_y0 = cfg.margin_tb
        self.bar_h = cfg.height - cfg.margin_tb * 2
        # LEDサイズ（高さ方向に等分）
        total_gap = (cfg.leds_per_bar - 1) * cfg.led_gap
        self.led_h = (self.bar_h - total_gap) // cfg.leds_per_bar
        # ピーク情報
        self.peak_pos = np.zeros(cfg.bars, dtype=np.float32)  # 0〜leds_per_bar
        self.peak_hold = np.zeros(cfg.bars, dtype=np.int32)

        # 透明サーフェス（残像用）
        self.trail = pg.Surface((cfg.width, cfg.height), pg.SRCALPHA)

        # フォント
        self.font_small = pg.font.SysFont("Segoe UI", 15)
        self.font_badge = pg.font.SysFont("Bahnschrift", 18, bold=True)
        self.font_logo = pg.font.SysFont("OCR A Extended, OCR A, Consolas", 16)
        self.font_scale = pg.font.SysFont("Consolas, Segoe UI", 12)

        # 表示用インフォテキスト（外部からセット）
        self.info_text = "INPUT: N/A | 48.0 kHz | 24-bit | Stereo | API: N/A"


    def draw_panel(self):
        self.surf.fill(PANEL_BG)
        # 枠線
        pg.draw.rect(self.surf, BORDER_DARK, (8, 8, self.cfg.width-16, self.cfg.height-16), 2, border_radius=10)
        # ロゴ
        logo = self.font_logo.render("SPECTRA-LED 90", True, (120, 120, 120))
        self.surf.blit(logo, (self.cfg.margin_lr, 16))
        # バッジ（GROOVEなど）
        if self.cfg.show_badge:
            text = self.font_badge.render(self.cfg.badge_text, True, (14, 230, 180))
            tw, th = text.get_size()
            pad = 8
            bx = self.cfg.width - tw - pad*2 - 24
            by = 14
            # グロー風
            pg.draw.rect(self.surf, (10, 40, 36), (bx-2, by-2, tw+pad*2+4, th+pad+4), border_radius=10)
            pg.draw.rect(self.surf, (30, 90, 80), (bx, by, tw+pad*2, th+pad), border_radius=10)
            self.surf.blit(text, (bx+pad, by+2))
        # 入力スペックのインフォバー
        if self.cfg.info_enabled:
            ih = self.cfg.info_height
            if self.cfg.info_position == "top":
                y = self.cfg.margin_tb - ih - 8
            else:
                y = self.cfg.height - self.cfg.margin_tb + 28
            bar_rect = pg.Rect(16, y, self.cfg.width-32, ih)
            pg.draw.rect(self.surf, (12, 22, 26), bar_rect, border_radius=8)
            pg.draw.rect(self.surf, (20, 40, 44), bar_rect, width=1, border_radius=8)
            info_surf = self.font_small.render(self.info_text, True, (190, 220, 220))
            self.surf.blit(info_surf, (bar_rect.x + 10, bar_rect.y + (ih - info_surf.get_height())//2))


    def update_peaks(self, level_leds: np.ndarray):
        # level_leds: 各バーの点灯LED数（floatでもOK）
        for i, lvl in enumerate(level_leds):
            if lvl > self.peak_pos[i]:
                self.peak_pos[i] = lvl
                self.peak_hold[i] = self.cfg.peak_hold_frames
            else:
                if self.peak_hold[i] > 0:
                    self.peak_hold[i] -= 1
                else:
                    self.peak_pos[i] = max(0.0, self.peak_pos[i] - self.cfg.peak_fall_per_frame * self.cfg.leds_per_bar)

    def _freq_to_bar(self, f_hz: float) -> int:
        """対数スケールで周波数→バー番号へ概算マッピング"""
        fmin = max(1.0, self.cfg.min_freq_hz)
        fmax = max(fmin * 1.01, self.cfg.max_freq_hz)
        import math
        pos = (math.log10(f_hz) - math.log10(fmin)) / (math.log10(fmax) - math.log10(fmin))
        idx = int(round(pos * (self.cfg.bars - 1)))
        return max(0, min(self.cfg.bars - 1, idx))

    def _fmt_freq_label(self, f: float, with_unit: bool = False) -> str:
        """周波数ラベル表示をいい感じに整形"""
        if f >= 1000:
            v = f / 1000.0
            s = f"{int(v)}k" if abs(v - int(v)) < 1e-6 else f"{v:.1f}k".rstrip("0").rstrip(".") + "k"
            return s + ("Hz" if with_unit else "")
        else:
            s = f"{f:.1f}" if abs(f - int(f)) > 1e-6 else f"{int(f)}"
            return s + ("Hz" if with_unit else "")

    def draw_freq_scale(self):
        """バー列の下に目盛り線とラベルを描く"""
        if not self.cfg.show_freq_scale:
            return
        base_y = self.bar_y0 + self.bar_h + 4   # バーの直下
        EPS = 1e-6
        for f in self.cfg.scale_ticks_hz:
            # 端ラベルと重複するのを避ける
            if abs(f - self.cfg.min_freq_hz) < EPS or abs(f - self.cfg.max_freq_hz) < EPS:
                continue
            b = self._freq_to_bar(f)
            x = self.bar_x0 + b * (self.bar_w + self.cfg.bar_gap) + self.bar_w // 2
            # 目盛り
            pg.draw.line(self.surf, (70, 90, 100), (x, base_y), (x, base_y + 6), 1)

            if f >= 1000:
                # 1k, 2k, 4k, 16k, 32k, 48k 表記
                if f % 1000 == 0:
                    label = f"{int(f/1000)}k"
                else:
                    label = f"{f/1000:.1f}k".rstrip("0").rstrip(".")  # 1.6k など
            else:
                # 31.5, 63 など。整数ならそのまま、小数点ありは 1桁
                label = f"{f:.1f}" if (f != int(f)) else f"{int(f)}"

            ts = self.font_scale.render(label, True, (150, 180, 190))
            self.surf.blit(ts, (x - ts.get_width()//2, base_y + 8))

        # 端ラベル（min/max）を明示
        if self.cfg.show_freq_edge_labels:
            # 左端（min）: 単位なし（例: "20"）
            x_left = self.bar_x0
            min_label = self._fmt_freq_label(self.cfg.min_freq_hz, with_unit=False)
            ts_min = self.font_scale.render(min_label, True, (180, 200, 210))
            self.surf.blit(ts_min, (x_left, base_y + 8))

            # 右端（max）:
            #  数値＋k まではバー中心下にセンタリング、単位 "Hz" は右にはみ出し
            right_bar_center = (
                self.bar_x0
                + (self.cfg.bars - 1) * (self.bar_w + self.cfg.bar_gap)
                + self.bar_w // 2
            )

            fmax = self.cfg.max_freq_hz
            if fmax >= 1000:
                v = fmax / 1000.0
                # 例: 48000 -> "48k", 1600 -> "1.6k"
                numk = f"{int(v)}k" if abs(v - int(v)) < 1e-6 else f"{v:.1f}".rstrip("0").rstrip(".") + "k"
            else:
                numk = f"{int(fmax)}" if abs(fmax - int(fmax)) < 1e-6 else f"{fmax:.1f}".rstrip("0").rstrip(".")

            ts_numk = self.font_scale.render(numk, True, (180, 200, 210))
            # 「48k」をバー中心にセンタリング
            self.surf.blit(ts_numk, (right_bar_center - ts_numk.get_width() // 2, base_y + 8))

            # 単位は "Hz" だけ、数字の右に少し間を空けて配置（はみ出してOK）
            ts_unit = self.font_scale.render("Hz", True, (150, 180, 190))
            gap_px = 2
            self.surf.blit(
                ts_unit,
                (right_bar_center + ts_numk.get_width() // 2 + gap_px, base_y + 8)
            )

    def draw_db_labels(self):
        if not self.cfg.show_db_scale:
            return
        # ラベルはバー領域の左外。右端をバーに揃える（右寄せ）
        x_right = self.bar_x0 - self.cfg.db_label_pad
        # 上端から下端へ（0, -10, -20, ...）
        rng = np.arange(self.cfg.db_max, self.cfg.db_min - 0.1, -self.cfg.db_step)
        denom = (self.cfg.db_max - self.cfg.db_min) if (self.cfg.db_max != self.cfg.db_min) else 1.0

        for db in rng:
            # dB値 -> 縦位置（LED段数に合わせた等間隔でOK）
            ratio = (db - self.cfg.db_min) / denom  # 0..1
            y = int(self.bar_y0 + self.bar_h - ratio * self.bar_h)

            label = f"{int(db)}"  # -10, -20 ... なので整数表示で十分
            ts = self.font_scale.render(label, True, (160, 180, 190))
            # 右寄せして、バー領域から少し離す
            self.surf.blit(ts, (x_right - ts.get_width(), y - ts.get_height() // 2))

        # 単位 "dB" は左上に小さく
        unit = self.font_scale.render("dB", True, (190, 210, 210))
        self.surf.blit(unit, (x_right - unit.get_width(), self.bar_y0 - unit.get_height() - self.cfg.db_unit_offset))

    def draw(self, levels: np.ndarray):
        # 残像を薄く塗る
        self.trail.fill((0, 0, 0, self.cfg.afterglow_alpha))
        self.surf.blit(self.trail, (0, 0))

        # バックパネル等
        self.draw_panel()

        # レベルをLED段数へ変換
        level_leds = levels * self.cfg.leds_per_bar
        self.update_peaks(levels * self.cfg.leds_per_bar)

        for b in range(self.cfg.bars):
            x = self.bar_x0 + b * (self.bar_w + self.cfg.bar_gap)
            # 下から上へLEDを描く
            lit = level_leds[b]
            for j in range(self.cfg.leds_per_bar):
                led_ratio = (j + 0.5) / self.cfg.leds_per_bar  # このLEDの高さ割合
                on_color, off_color = choose_color(led_ratio)
                y = self.bar_y0 + self.bar_h - (j+1) * (self.led_h + self.cfg.led_gap) + self.cfg.led_gap
                rect = pg.Rect(x, y, self.bar_w, self.led_h)
                on = (j < lit)
                self.draw_led(rect, on_color if on else off_color, on)

            # ピークマーカー（細いハイライト）
            # これまでは floor(int) になっていて 1段下に描かれていた。
            # 上端に合わせるため、ceilして“点灯中の最上段LED”の上端に配置する。
            lit_count = level_leds[b]  # 点灯LED数（float）
            if lit_count > 0:
                from math import ceil
                top_index = min(self.cfg.leds_per_bar - 1, max(0, ceil(lit_count) - 1))  # 最上段LEDのインデックス
                y = self.bar_y0 + self.bar_h - (top_index + 1) * (self.led_h + self.cfg.led_gap) + self.cfg.led_gap
                pm_rect = pg.Rect(x, y, self.bar_w, max(2, self.led_h // 5))
                pg.draw.rect(self.surf, (240, 240, 240), pm_rect, border_radius=min(2, self.cfg.corner_radius))

        self.draw_freq_scale()
        self.draw_db_labels()

    def draw_led(self, rect: pg.Rect, color: Tuple[int, int, int], on: bool):
        # ベース（枠）
        pg.draw.rect(self.surf, (18, 18, 20), rect, border_radius=self.cfg.corner_radius)
        inner = rect.inflate(-2, -2)
        # 内側の塗り
        pg.draw.rect(self.surf, color, inner, border_radius=self.cfg.corner_radius-1)
        if on:
            # ガラス光沢：上部に薄い白
            gloss_h = max(2, inner.height // 5)
            gloss = pg.Rect(inner.x+2, inner.y+2, inner.width-4, gloss_h)
            s = pg.Surface(gloss.size, pg.SRCALPHA)
            s.fill((*GLASS_WHITE, 60))
            self.surf.blit(s, gloss)
            # 内側の縁に微妙な暗さで立体感
            pg.draw.rect(self.surf, (0,0,0), inner, width=1, border_radius=self.cfg.corner_radius-1)
        else:
            # 消灯時は少しだけ縁を明るくして“存在感”
            pg.draw.rect(self.surf, (24,24,28), inner, width=1, border_radius=self.cfg.corner_radius-1)

    def draw_pause_overlay(self):
        # 画面中央に "PAUSED" を半透明で表示
        overlay = pg.Surface((self.cfg.width, self.cfg.height), pg.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        text = self.font_badge.render("PAUSED", True, (255, 255, 255))
        tw, th = text.get_size()
        overlay.blit(text, ((self.cfg.width - tw)//2, (self.cfg.height - th)//2))
        self.surf.blit(overlay, (0, 0))


# ==========================
# メインループ
# ==========================
class App:
    def __init__(self, cfg: Config):
        pg.init()
        pg.display.set_caption("WuneWune LED Speana v0.1")
        self.cfg = cfg
        self.screen = pg.display.set_mode((cfg.width, cfg.height), pg.RESIZABLE)
        self.clock = pg.time.Clock()
        self.renderer = LedBarRenderer(self.screen, cfg)
        self.spectrum = FakeSpectrum(cfg.bars)
        self.running = True
        self.paused = False
        # 入力仕様のプレースホルダ（後でWASAPI実装時に更新）
        self.audio_spec = {
            "samplerate": 48000,
            "bit_depth": 24,
            "channels": 2,
            "api": "N/A",
            "device": "N/A",
            "loopback": False,
            "latency_ms": None,
        }
        self.update_info_text()


    def toggle_fullscreen(self):
        flags = self.screen.get_flags()
        if flags & pg.FULLSCREEN:
            self.screen = pg.display.set_mode((self.cfg.width, self.cfg.height), pg.RESIZABLE)
        else:
            self.screen = pg.display.set_mode((self.cfg.width, self.cfg.height), pg.FULLSCREEN)
        self.renderer.surf = self.screen

    def handle_event(self, e: pg.event.Event):
        if e.type == pg.QUIT:
            self.running = False
        elif e.type == pg.KEYDOWN:
            if e.key in (pg.K_ESCAPE, pg.K_q):
                self.running = False
            elif e.key == pg.K_F11:
                self.toggle_fullscreen()
            elif e.key == pg.K_SPACE:
                self.paused = not self.paused
            elif e.key == pg.K_i:
                # インフォバー表示のトグル
                self.cfg.info_enabled = not self.cfg.info_enabled
            elif e.key == pg.K_l:
                # ループバックON/OFF（ダミー）
                self.audio_spec["loopback"] = not self.audio_spec["loopback"]
                self.update_info_text()
        elif e.type == pg.MOUSEBUTTONDOWN:
            # どこをクリックしても一時停止/再開をトグル
            self.paused = not self.paused

    def update_info_text(self):
        sr = self.audio_spec["samplerate"]
        bd = self.audio_spec["bit_depth"]
        ch = self.audio_spec["channels"]
        api = self.audio_spec["api"]
        dev = self.audio_spec["device"]
        loop = "ON" if self.audio_spec["loopback"] else "OFF"
        sr_k = f"{sr/1000:.1f} kHz" if isinstance(sr, (int, float)) else "N/A"
        bits = f"{bd}-bit" if bd else "-"
        chs = {1: "Mono", 2: "Stereo"}.get(ch, f"{ch}ch") if ch else "-"
        info = f"INPUT: {dev}  |  {sr_k}  |  {bits}  |  {chs}  |  Loopback: {loop}  |  API: {api}"
        self.renderer.info_text = info

    def run(self):
        while self.running:
            dt = self.clock.tick(self.cfg.fps) / 1000.0
            for e in pg.event.get():
                self.handle_event(e)

            if not self.paused:
                levels = self.spectrum.step(dt)  # 0〜1
            else:
                levels = self.spectrum.smooth

            self.renderer.draw(levels)
            if self.paused:
                self.renderer.draw_pause_overlay()
            pg.display.flip()
        pg.quit()


if __name__ == "__main__":
    try:
        App(CFG).run()
    except Exception as ex:
        print("Error:", ex)
        pg.quit()
        sys.exit(1)
