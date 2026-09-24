from dataclasses import asdict
import json
from pathlib import Path
import shutil
import uuid
import unittest
from unittest.mock import patch

from wune.config import Config
from wune.settings import SettingsStore
from wune.window_geometry import restore_geometry
from wune.layout import minimum_window_size


class SettingsTests(unittest.TestCase):
    def setUp(self):
        directory = (Path.cwd()/f"test-settings-{uuid.uuid4().hex}").resolve()
        self.assertEqual(directory.parent, Path.cwd().resolve())
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory)
        self.path = directory/"Wune"/"settings.json"
        self.store = SettingsStore(self.path)

    def write(self, value):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(value), encoding="utf-8")

    def test_first_launch_copies_defaults_without_creating_file(self):
        defaults = Config()
        cfg, window = self.store.load(defaults)
        cfg.width = 42
        self.assertEqual(defaults.width, 1280)
        self.assertEqual(window, {})
        self.assertFalse(self.path.exists())

    def test_round_trip_geometry_preset_and_orientation(self):
        cfg = Config(spectrum_orientation="frequency_vertical", channel_layout="horizontal", bars=32)
        self.assertTrue(self.store.save(cfg, (916, 504), (-1400, 120), "BLUE"))
        loaded, window = SettingsStore(self.path).load(Config())
        self.assertEqual(window, dict(width=916, height=504, x=-1400, y=120))
        self.assertEqual(loaded.initial_preset, "BLUE")
        self.assertEqual(loaded.gauge_style, "box")
        self.assertEqual(loaded.spectrum_orientation, "frequency_vertical")
        self.assertEqual(loaded.channel_layout, "horizontal")
        self.assertEqual(loaded.bars, 32)

    def test_custom_style_round_trip(self):
        cfg = Config(initial_preset=None, led_shape="ellipse", led_aspect_ratio=1.5)
        self.store.save(cfg, (1000, 700), (50, 60), "CUSTOM")
        loaded, _ = self.store.load(Config())
        self.assertIsNone(loaded.initial_preset)
        self.assertEqual(loaded.led_shape, "ellipse")
        self.assertEqual(loaded.led_aspect_ratio, 1.5)

    def test_invalid_values_do_not_reach_layout(self):
        self.write({"version": 1, "appearance": {"bars": True, "channels": 0,
                    "led_aspect_ratio": float('nan'), "initial_preset": "missing",
                    "spectrum_orientation": "bad", "info_enabled": "yes"},
                    "window": {"width": -1, "height": 10**10, "x": "left", "y": False}})
        with self.assertWarns(RuntimeWarning):
            cfg, geometry = self.store.load(Config())
        self.assertEqual(cfg.bars, 64)
        self.assertEqual(cfg.channels, 2)
        self.assertEqual(cfg.led_aspect_ratio, 2)
        self.assertEqual(geometry, {})

    def test_corrupt_or_future_version_is_not_overwritten(self):
        for text in ('{broken', '{"version": 2}', '[]', '{"version":1,"window":[]}'):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(text, encoding="utf-8")
            store = SettingsStore(self.path)
            with self.assertWarns(RuntimeWarning):
                cfg, _ = store.load(Config())
            self.assertFalse(store.save(cfg, (800, 600), (0, 0), "CLASSIC"))
            self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_reset_restores_factory_defaults_and_removes_only_settings(self):
        self.write({"version": 1, "appearance": {"bars": 32}})
        other = self.path.parent/"keep.txt"
        other.write_text("keep")
        cfg, geometry = self.store.reset()
        self.assertEqual(asdict(cfg), asdict(Config()))
        self.assertEqual(geometry, {})
        self.assertFalse(self.path.exists())
        self.assertTrue(other.exists())

    def test_atomic_write_failure_preserves_previous_file(self):
        self.write({"version": 1})
        before = self.path.read_bytes()
        with patch("wune.settings.os.replace", side_effect=PermissionError("locked")):
            with self.assertWarns(RuntimeWarning):
                self.assertFalse(self.store.save(Config(), (800, 600), (0, 0), "CLASSIC"))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_unknown_fields_survive_save_for_future_ui(self):
        self.write({"version": 1, "user_themes": {"mine": {}}, "appearance": {"future": 123}})
        cfg, _ = self.store.load(Config())
        self.store.save(cfg, (800, 600), (30, 40), "CLASSIC")
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document["appearance"]["future"], 123)
        self.assertEqual(document["user_themes"], {"mine": {}})


class GeometryTests(unittest.TestCase):
    def test_valid_negative_monitor_position_is_preserved(self):
        cfg = Config()
        size, position = restore_geometry(cfg, dict(width=800, height=600, x=-1500, y=100),
                                          [(0, 0, 1920, 1040), (-1920, 0, 1920, 1080)])
        self.assertEqual(size, (800, 600))
        self.assertEqual(position, (-1500, 100))

    def test_removed_monitor_and_oversized_window_are_recovered(self):
        cfg = Config()
        size, position = restore_geometry(cfg, dict(width=8000, height=4000, x=10000, y=-5000),
                                          [(0, 0, 1920, 1040)])
        self.assertGreaterEqual(position[0], 0)
        self.assertGreaterEqual(position[1], 40)
        self.assertLessEqual(position[0]+size[0], 1920)
        self.assertLessEqual(position[1]+size[1], 1040)

    def test_tiny_screen_keeps_title_accessible_and_layout_minimum(self):
        cfg = Config()
        size, pos = restore_geometry(cfg, {}, [(0, 0, 320, 240)])
        self.assertTrue(all(v >= bound for v, bound in zip(size, minimum_window_size(cfg))))
        self.assertEqual(pos, (16, 40))

    def test_transposed_fit_stays_on_selected_screen(self):
        cfg = Config(bars=32, spectrum_orientation="frequency_vertical", channel_layout="horizontal")
        size, pos = restore_geometry(cfg, dict(width=960, height=1800, x=100, y=100), [(0, 0, 1920, 1040)])
        self.assertEqual(size, (916, 504))
        self.assertEqual(pos, (100, 100))
