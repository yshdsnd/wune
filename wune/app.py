"""Application lifecycle and pygame event loop."""

import numpy as np
import pygame as pg

from .config import Config
from .layout import clamp_window_size, fit_window_size
from .renderer import LedBarRenderer
from .spectrum_audio import AudioSpectrum


class App:
    def __init__(self, cfg: Config):
        pg.init()
        pg.display.set_caption("WuneWune LED Speana v0.1")
        self.cfg = cfg
        self._fullscreen = False
        self._windowed_size = fit_window_size((cfg.width, cfg.height), cfg)
        self.screen = pg.display.set_mode(self._windowed_size, pg.RESIZABLE)
        self.clock = pg.time.Clock()
        self.renderer = LedBarRenderer(self.screen, cfg)
        if cfg.initial_preset is not None:
            self.renderer.apply_preset(cfg.initial_preset)
        # Audio errors must remain visible rather than silently showing fake data.
        self.spectrum = AudioSpectrum(cfg, cfg.bars, cfg.channels)
        self.spectrum.set_range(cfg.min_freq_hz, cfg.spectrum_upper_hz(self.spectrum.sr))
        cfg.max_freq_hz = self.spectrum.fmax

        self.running = True
        self.paused = False
        self.levels = np.zeros((cfg.channels, cfg.bars), dtype=np.float32)
        self.update_info_text()

    def toggle_fullscreen(self):
        if self._fullscreen:
            self.screen = pg.display.set_mode(fit_window_size(self._windowed_size, self.cfg), pg.RESIZABLE)
            self._fullscreen = False
        else:
            self._windowed_size = self.screen.get_size()
            self.screen = pg.display.set_mode((0, 0), pg.FULLSCREEN)
            self._fullscreen = True
            if self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
                self.screen = pg.display.set_mode(fit_window_size(self._windowed_size, self.cfg), pg.RESIZABLE)
                self._fullscreen = False
        self.renderer.resize(self.screen)

    def resize_window(self, size):
        if self._fullscreen and self.screen.get_size() != clamp_window_size(self.screen.get_size(), self.cfg):
            self.toggle_fullscreen()
            return
        if not self._fullscreen:
            size = fit_window_size(size, self.cfg)
            if self.screen.get_size() != size:
                self.screen = pg.display.set_mode(size, pg.RESIZABLE)
            self._windowed_size = size
        self.renderer.resize(self.screen)

    def handle_event(self, event: pg.event.Event):
        if event.type == pg.QUIT:
            self.running = False
        elif event.type == pg.VIDEORESIZE:
            self.resize_window(event.size)
        elif event.type == pg.WINDOWSIZECHANGED:
            # pygame 2 updates the display Surface when the native window resizes.
            self.resize_window(self.screen.get_size())
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1 and self.renderer.badge_contains(event.pos):
                self.renderer.next_preset()
        elif event.type == pg.KEYDOWN:
            if event.key in (pg.K_ESCAPE, pg.K_q):
                self.running = False
            elif event.key == pg.K_F11:
                self.toggle_fullscreen()
            elif event.key == pg.K_SPACE:
                self.paused = not self.paused
            elif event.key == pg.K_i:
                self.cfg.info_enabled = not self.cfg.info_enabled
                self.resize_window(self.screen.get_size())
            elif event.key == pg.K_t:
                self.renderer.next_preset()

    def update_info_text(self):
        # float32 describes the transferred samples, not the device's ADC bit depth.
        spectrum = self.spectrum
        device = spectrum.device if spectrum.device is not None else "Default output"
        self.renderer.info_text = (
            f"LOOPBACK:{device} | {spectrum.sr / 1000:.1f} kHz | float32 | "
            f"{'GATED' if spectrum.gated else 'LIVE'} RMS={spectrum.last_rms:.1e}"
        )

    def run(self):
        try:
            while self.running:
                dt = self.clock.tick(self.cfg.fps) / 1000.0
                for event in pg.event.get():
                    self.handle_event(event)
                if not self.running:
                    break

                if not self.paused:
                    self.levels = self.spectrum.step(dt)
                # Pause reuses the last levels and freezes peak timers.
                self.update_info_text()
                self.renderer.draw(self.levels, dt=0.0 if self.paused else dt)
                if self.paused:
                    self.renderer.draw_pause_overlay()
                pg.display.flip()
        finally:
            try:
                self.spectrum.close()
            finally:
                pg.quit()
