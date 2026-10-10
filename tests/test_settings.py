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
        self.assertEqual(cfg.peak_hold_ms, 500)
        cfg.width = 42
        self.assertEqual(defaults.width, 1280)
        self.assertEqual(window, {})
        self.assertFalse(self.path.exists())

    def test_exit_confirmation_preference_survives_restart_and_reenable(self):
        for enabled in (False, True):
            self.assertTrue(self.store.save(Config(confirm_keyboard_exit=enabled), (1280, 800), (0, 0), "CLASSIC"))
            loaded, _ = SettingsStore(self.path).load(Config())
            self.assertEqual(loaded.confirm_keyboard_exit, enabled)

    def test_now_playing_preference_survives_restart_and_reenable(self):
        for enabled in (False, True):
            self.assertTrue(self.store.save(Config(show_now_playing=enabled), (1280, 800), (0, 0), "CLASSIC"))
            loaded, _ = SettingsStore(self.path).load(Config())
            self.assertEqual(loaded.show_now_playing, enabled)

    def test_font_preferences_survive_restart(self):
        for font_size, auto_scale, info_size in ((10, False, 10), (14, True, 14), (20, False, 18), (24, True, 24)):
            with self.subTest(font_size=font_size, auto_scale=auto_scale, info_size=info_size):
                cfg = Config(label_font_size=font_size, auto_scale_fonts=auto_scale, info_font_size=info_size)
                self.assertTrue(self.store.save(cfg, (1280, 800), (0, 0), "CLASSIC"))
                loaded, _ = SettingsStore(self.path).load(Config())
                self.assertEqual(loaded.label_font_size, font_size)
                self.assertEqual(loaded.auto_scale_fonts, auto_scale)
                self.assertEqual(loaded.info_font_size, info_size)


    def test_saved_peak_hold_values_survive_restart(self):
        for hold_ms in (0, 120, 200, 875.5, 5000):
            with self.subTest(hold_ms=hold_ms):
                self.assertTrue(self.store.save(Config(peak_hold_ms=hold_ms), (1280, 800), (0, 0), "CLASSIC"))
                loaded, _ = SettingsStore(self.path).load(Config())
                self.assertEqual(loaded.peak_hold_ms, hold_ms)

    def test_round_trip_geometry_preset_and_orientation(self):
        cfg = Config(spectrum_orientation="frequency_vertical", channel_layout="horizontal", bars=32)
        self.assertTrue(self.store.save(cfg, (916, 504), (-1400, 120), "BLUE"))
        loaded, window = SettingsStore(self.path).load(Config())
        self.assertEqual(window, dict(width=916, height=504, x=-1400, y=120))
        self.assertEqual(loaded.initial_preset, "BLUE")
        self.assertEqual(loaded.gauge_style, "flat")
        self.assertEqual(loaded.spectrum_orientation, "frequency_vertical")
        self.assertEqual(loaded.channel_layout, "horizontal")
        self.assertEqual(loaded.bars, 32)

    def test_fullscreen_monitor_round_trip_and_windowed_exit_clears_it(self):
        self.assertTrue(self.store.save(Config(), (900, 600), (-900, 50), "CLASSIC",
                                        fullscreen=True, fullscreen_monitor="monitor-a"))
        _, window = SettingsStore(self.path).load(Config())
        self.assertTrue(window["fullscreen"])
        self.assertEqual(window["fullscreen_monitor"], "monitor-a")
        self.assertEqual((window["width"], window["height"]), (900, 600))
        self.store.save(Config(), (900, 600), (-900, 50), "CLASSIC")
        _, window = SettingsStore(self.path).load(Config())
        self.assertNotIn("fullscreen", window)
        self.assertIsNone(json.loads(self.path.read_text())["window"]["fullscreen_monitor"])

    def test_invalid_fullscreen_metadata_does_not_request_restoration(self):
        for flag, identity in ((1, "monitor-a"), ("true", "monitor-a"), (True, None),
                               (True, ""), (True, "x" * 1025), (True, "bad\x00id")):
            self.write({"version": 2, "window": {"fullscreen": flag, "fullscreen_monitor": identity}})
            _, window = self.store.load(Config())
            self.assertNotIn("fullscreen", window)

    def test_custom_style_round_trip(self):
        cfg = Config(initial_preset=None, led_shape="ellipse", led_aspect_ratio=1.5, leds_per_bar=40)
        self.store.save(cfg, (1000, 700), (50, 60), "CUSTOM")
        loaded, _ = self.store.load(Config())
        self.assertIsNone(loaded.initial_preset)
        self.assertEqual(loaded.led_shape, "ellipse")
        self.assertEqual(loaded.led_aspect_ratio, 1.5)
        self.assertEqual(loaded.leds_per_bar, 40)

    def test_invalid_values_do_not_reach_layout(self):
        self.write({"version": 1, "appearance": {"bars": True, "channels": 0,
                    "led_aspect_ratio": float('nan'), "leds_per_bar": 999, "initial_preset": "missing",
                    "spectrum_orientation": "bad", "info_enabled": "yes",
                    "label_font_size": 99, "info_font_size": 99, "auto_scale_fonts": "yes"},
                    "window": {"width": -1, "height": 10**10, "x": "left", "y": False}})
        with self.assertWarns(RuntimeWarning):
            cfg, geometry = self.store.load(Config())
        self.assertEqual(cfg.bars, 64)
        self.assertEqual(cfg.channels, 2)
        self.assertEqual(cfg.led_aspect_ratio, 2)
        self.assertEqual(cfg.leds_per_bar, 20)
        self.assertEqual(cfg.label_font_size, 14)
        self.assertEqual(cfg.info_font_size, 14)
        self.assertEqual(cfg.auto_scale_fonts, True)
        self.assertEqual(geometry, {})

    def test_corrupt_or_future_version_is_not_overwritten(self):
        for text in ('{broken', '{"version": 3}', '[]', '{"version":1,"window":[]}'):
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
        self.write({"version": 1, "future_section": {"mine": {}}, "appearance": {"future": 123}})
        cfg, _ = self.store.load(Config())
        self.store.save(cfg, (800, 600), (30, 40), "CLASSIC")
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document["appearance"]["future"], 123)
        self.assertEqual(document["future_section"], {"mine": {}})


class GeometryTests(unittest.TestCase):
    def test_valid_negative_monitor_position_is_preserved(self):
        cfg = Config()
        size, position = restore_geometry(cfg, dict(width=800, height=600, x=-1500, y=100),
                                          [(0, 0, 1920, 1040), (-1920, 0, 1920, 1080)])
        self.assertEqual(size, (800, 600))
        self.assertEqual(position, (-1500, 100))
        cfg_no_np = Config(show_now_playing=False)
        size_no_np, _ = restore_geometry(cfg_no_np, dict(width=800, height=600, x=-1500, y=100),
                                         [(0, 0, 1920, 1040), (-1920, 0, 1920, 1080)])
        self.assertEqual(size_no_np, (800, 600))

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
        self.assertEqual(size, (960, 984))
        self.assertEqual(pos, (100, 40))
        cfg_no_np = Config(bars=32, spectrum_orientation="frequency_vertical", channel_layout="horizontal", show_now_playing=False)
        size_no_np, pos_no_np = restore_geometry(cfg_no_np, dict(width=960, height=1800, x=100, y=100), [(0, 0, 1920, 1040)])
        self.assertEqual(size_no_np, (960, 984))
        self.assertEqual(pos_no_np, (100, 40))
