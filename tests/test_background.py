from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pygame as pg

from wune.background import ImageBackground
from wune.config import Config
from wune.appearance import AppearanceDraft, AppearanceState
from wune.settings import SettingsStore


class BackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "画像.png"
        image = pg.Surface((40, 20), pg.SRCALPHA)
        image.fill((220, 30, 60))
        pg.image.save(image, str(self.path))
        self.cfg = Config(background_mode="image", background_path=str(self.path))

    def test_fit_fill_resize_and_original_untouched(self):
        before = self.path.read_bytes()
        background = ImageBackground()
        surface = pg.Surface((100, 100))
        background.draw(surface, self.cfg)
        self.assertEqual(surface.get_at((50, 5))[:3], self.cfg.theme.background)
        self.assertEqual(surface.get_at((50, 50))[:3], (220, 30, 60))
        self.cfg.background_sizing = "fill"
        background.draw(surface, self.cfg)
        self.assertEqual(surface.get_at((0, 0))[:3], (220, 30, 60))
        surface = pg.Surface((200, 60))
        background.draw(surface, self.cfg)
        self.assertEqual(background.scaled.get_size(), (200, 60))
        self.assertEqual(self.path.read_bytes(), before)

    def test_cache_and_missing_corrupt_files_fall_back(self):
        background = ImageBackground()
        surface = pg.Surface((100, 100))
        with patch("wune.background.pg.image.load", wraps=pg.image.load) as load, \
             patch("wune.background.pg.transform.smoothscale", wraps=pg.transform.smoothscale) as scale:
            background.draw(surface, self.cfg)
            background.draw(surface, self.cfg)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(scale.call_count, 1)
        for content in (None, b"not an image"):
            if content is None:
                self.path.unlink()
            else:
                self.path.write_bytes(content)
            background = ImageBackground()
            with patch("wune.background.pg.image.load", wraps=pg.image.load) as load:
                background.draw(surface, self.cfg)
                background.draw(surface, self.cfg)
                self.assertEqual(load.call_count, 1)
            self.assertEqual(surface.get_at((50, 50))[:3], self.cfg.theme.background)

    def test_palette_png_alpha_and_jpeg(self):
        surface = pg.Surface((100, 100))
        palette = pg.Surface((10, 10), depth=8)
        palette.fill((0, 255, 0))
        pg.image.save(palette, str(self.path))
        ImageBackground().draw(surface, self.cfg)
        transparent = pg.Surface((10, 10), pg.SRCALPHA)
        transparent.fill((255, 0, 0, 0))
        pg.image.save(transparent, str(self.path))
        ImageBackground().draw(surface, self.cfg)
        self.assertEqual(surface.get_at((50, 50))[:3], self.cfg.theme.background)
        jpg = self.path.with_suffix(".jpg")
        pg.image.save(palette, str(jpg))
        cfg = replace(self.cfg, background_path=str(jpg))
        background = ImageBackground()
        background.draw(surface, cfg)
        self.assertIsNotNone(background.source)

    def test_solid_never_reads_image(self):
        with patch("wune.background.pg.image.load") as load:
            ImageBackground().draw(pg.Surface((50, 50)), replace(self.cfg, background_mode="solid"))
            load.assert_not_called()

    def test_roundtrip_reset_and_theme_independence(self):
        store = SettingsStore(Path(self.temp.name) / "settings.json")
        self.assertTrue(store.save(self.cfg, (800, 600), (0, 0), "CLASSIC"))
        loaded, _ = store.load(Config())
        self.assertEqual(loaded.background_path, str(self.path))
        self.assertEqual(loaded.background_mode, "image")
        draft = AppearanceDraft(AppearanceState.capture(loaded, "CLASSIC", {}))
        before = draft.snapshot()
        draft.select("BLUE")
        self.assertEqual(draft.state.background, before.background)
        draft.reset()
        self.assertEqual(draft.state.background["background_mode"], "solid")
        self.assertEqual(before.background["background_mode"], "image")
        before.apply(loaded)
        self.assertEqual(loaded.background_path, str(self.path))

    def test_legacy_and_invalid_settings(self):
        path = Path(self.temp.name) / "settings.json"
        path.write_text(json.dumps({"version": 2, "appearance": {}}))
        cfg, _ = SettingsStore(path).load(Config())
        self.assertEqual(cfg.background_mode, "solid")
        path.write_text(json.dumps({"version": 2, "appearance": {
            "background_mode": "bad", "background_path": 123, "background_sizing": "bad"}}))
        with self.assertWarns(RuntimeWarning):
            cfg, _ = SettingsStore(path).load(Config())
        self.assertEqual((cfg.background_mode, cfg.background_path, cfg.background_sizing), ("solid", "", "fit"))
