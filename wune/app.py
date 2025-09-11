# wune/app.py
import pygame as pg
from .renderer import LedBarRenderer
from .spectrum_fake import FakeSpectrum
from .config import Config

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
        self.spectrum = FakeSpectrum(cfg.bars, cfg.channels)

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
        self.audio_spec["channels"] = cfg.channels
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
