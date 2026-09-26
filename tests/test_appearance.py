from dataclasses import replace
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from wune.appearance import AppearanceDraft, AppearanceState, decode_preset, encode_preset
from wune.colors import Theme
from wune.config import Config
from wune.presets import get_preset
from wune.settings import SettingsStore


class AppearanceTests(unittest.TestCase):
    def test_style_and_layout_edits_never_grow_theme_library(self):
        draft = AppearanceDraft(AppearanceState.capture(Config(), "CLASSIC", {}))
        for name in ("CLASSIC", "BLUE", "AMBER", "CLASSIC BOX"):
            draft.select(name)
            for ratio in (1.0, 2.8, 1.5):
                draft.edit_style(gauge_style="box", led_shape="ellipse", led_aspect_ratio=ratio)
                draft.state.layout["channel_layout"] = "horizontal"
                self.assertEqual(draft.state.preset.name, name)
                self.assertEqual(draft.state.user_presets, {})
            draft.select("BLUE")
            self.assertEqual(draft.state.style, dict(gauge_style="box", led_shape="ellipse", led_aspect_ratio=1.5))
        before = draft.snapshot()
        draft.create("mine")
        draft.create("duplicate", draft.state.preset)
        draft.rename("renamed")
        draft.delete()
        self.assertEqual(draft.state.style, before.style)
        self.assertEqual(draft.state.layout, before.layout)

    def test_same_color_is_noop_and_repeated_colors_reuse_one_copy(self):
        draft = AppearanceDraft(AppearanceState.capture(Config(), "CLASSIC", {}))
        draft.edit(theme=draft.state.preset.theme)
        self.assertEqual(draft.state.user_presets, {})
        for color in ((1, 2, 3), (4, 5, 6)):
            draft.edit(theme=replace(draft.state.preset.theme, peak=color))
            draft.edit_style(led_shape="ellipse")
        self.assertEqual(list(draft.state.user_presets), ["CLASSIC copy"])
        self.assertEqual(set(encode_preset(draft.state.preset)), {"theme"})

    def setUp(self):
        self.original = AppearanceState.capture(Config(), "CLASSIC", {})
        self.draft = AppearanceDraft(self.original)

    def test_builtin_edit_creates_one_copy_without_modifying_defaults(self):
        theme = replace(Theme(), green_on=(10, 20, 30))
        self.draft.edit(theme=theme)
        self.draft.edit_style(led_shape="ellipse", led_aspect_ratio=1.5)
        self.assertEqual(list(self.draft.state.user_presets), ["CLASSIC copy"])
        self.assertEqual(get_preset("CLASSIC").theme, Theme())
        self.assertEqual(self.original.user_presets, {})
        self.assertEqual(self.draft.state.preset.theme.green_on, (10, 20, 30))

    def test_create_duplicate_rename_delete_and_select(self):
        self.draft.create("青", get_preset("BLUE"))
        self.draft.create("青2", self.draft.state.preset)
        self.draft.rename("空")
        self.assertNotIn("青2", self.draft.names())
        self.assertEqual(self.draft.state.preset.theme, get_preset("BLUE").theme)
        self.draft.delete()
        self.assertEqual(self.draft.state.preset.name, "CLASSIC")
        self.draft.select("青")
        self.assertEqual(self.draft.state.style["gauge_style"], "flat")

    def test_reserved_duplicate_blank_and_control_names_are_rejected(self):
        self.draft.create("mine")
        for name in ("CLASSIC", "CUSTOM", "", "  ", "mine", "a"*41, "bad\nname"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.draft.create(name)

    def test_builtins_cannot_be_deleted_or_renamed(self):
        with self.assertRaises(ValueError):
            self.draft.delete()
        with self.assertRaises(ValueError):
            self.draft.rename("renamed")

    def test_invalid_edit_does_not_create_copy_or_change_state(self):
        for value in (float("nan"), 0, 100, True):
            with self.assertRaises(ValueError):
                self.draft.edit_style(led_aspect_ratio=value)
        self.assertEqual(self.draft.snapshot(), self.original)

    def test_colors_and_thresholds_are_validated_before_rendering(self):
        for theme in ({"green_on": [0, 256, 0]}, {"background": [True, 0, 0]},
                      {"peak": "red"}, {"th_yellow": 0.9, "th_red": 0.3},
                      {"th_red": float("inf")}):
            with self.assertRaises(ValueError):
                decode_preset("mine", {"theme": theme})

    def test_reset_is_reversible_and_retains_user_library(self):
        self.draft.create("mine", get_preset("BLUE"))
        before = self.draft.snapshot()
        self.draft.state.layout["channel_layout"] = "horizontal"
        self.draft.reset()
        self.assertEqual(self.draft.state.preset, get_preset("CLASSIC"))
        self.assertIn("mine", self.draft.names())
        self.assertEqual(before.preset.name, "mine")

    def test_apply_does_not_change_audio_or_ballistics_or_band_count(self):
        cfg = Config(sample_rate=96000, bars=32, vis_attack_ms=12)
        self.draft = AppearanceDraft(AppearanceState.capture(cfg, "CLASSIC", {}))
        self.draft.state.layout["spectrum_orientation"] = "frequency_vertical"
        self.draft.select("AMBER")
        self.draft.state.apply(cfg)
        self.assertEqual((cfg.sample_rate, cfg.bars, cfg.vis_attack_ms), (96000, 32, 12))
        self.assertEqual(cfg.theme, get_preset("AMBER").theme)


class UserThemePersistenceTests(unittest.TestCase):
    def test_v1_selected_style_migrates_once_then_stays_independent(self):
        for name, legacy in (("BLUE", dict(gauge_style="box", led_shape="rectangle", led_aspect_ratio=2.0)),
                             ("mine", dict(gauge_style="flat", led_shape="ellipse", led_aspect_ratio=1.75))):
            with self.subTest(name=name):
                self.path.write_text(json.dumps({"version": 1,
                    "appearance": {"initial_preset": name, "led_aspect_ratio": 7},
                    "user_themes": {"mine": {"theme": {"peak": [1, 2, 3]}, **legacy}}}), encoding="utf-8")
                store = SettingsStore(self.path)
                cfg, _ = store.load(Config())
                self.assertEqual({key: getattr(cfg, key) for key in legacy}, legacy)
                cfg.led_aspect_ratio = 3.0
                self.assertTrue(store.save(cfg, (1000, 700), (0, 0), name))
                document = json.loads(self.path.read_text(encoding="utf-8"))
                self.assertEqual(document["version"], 2)
                self.assertEqual(set(document["user_themes"]["mine"]), {"theme"})
                actual, _ = SettingsStore(self.path).load(Config())
                self.assertEqual(actual.led_aspect_ratio, 3.0)
                self.assertEqual(actual.led_shape, legacy["led_shape"])

    def setUp(self):
        directory = (Path.cwd()/f"test-settings-{uuid.uuid4().hex}").resolve()
        self.assertEqual(directory.parent, Path.cwd().resolve())
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory)
        self.path = directory/"settings.json"

    def test_user_colors_style_and_selection_survive_relaunch(self):
        store = SettingsStore(self.path)
        draft = AppearanceDraft(AppearanceState.capture(Config(), "CLASSIC", {}))
        draft.create("夜の青", get_preset("BLUE"))
        draft.edit(theme=replace(draft.state.preset.theme, peak=(123, 45, 67)))
        draft.edit_style(led_aspect_ratio=1.75)
        cfg = Config()
        draft.state.apply(cfg)
        store.user_presets = draft.state.user_presets
        self.assertTrue(store.save(cfg, (1000, 700), (-1000, 50), "夜の青"))
        reopened = SettingsStore(self.path)
        actual, _ = reopened.load(Config())
        self.assertEqual(actual.initial_preset, "夜の青")
        self.assertEqual(actual.theme.peak, (123, 45, 67))
        self.assertEqual(actual.led_aspect_ratio, 1.75)
        self.assertEqual(reopened.user_presets, store.user_presets)

    def test_custom_unnamed_colors_survive_relaunch(self):
        cfg = Config(initial_preset=None, theme=replace(Theme(), background=(1, 2, 3)))
        SettingsStore(self.path).save(cfg, (800, 600), (0, 0), "CUSTOM")
        actual, _ = SettingsStore(self.path).load(Config())
        self.assertEqual(actual.theme, cfg.theme)

    def test_bad_theme_and_missing_selection_fall_back_safely(self):
        self.path.write_text(json.dumps({"version": 1, "appearance": {"initial_preset": "broken"},
                                        "user_themes": {"broken": {"theme": {"green_on": [-1, 0, 0]}}}}))
        with self.assertWarns(RuntimeWarning):
            actual, _ = SettingsStore(self.path).load(Config())
        self.assertEqual(actual.initial_preset, "CLASSIC")
        self.assertEqual(actual.theme, Theme())

    def test_save_failure_does_not_overwrite_user_theme_library(self):
        store = SettingsStore(self.path)
        store.user_presets = {"mine": decode_preset("mine", {})}
        store.save(Config(), (800, 600), (0, 0), "CLASSIC")
        before = self.path.read_bytes()
        store.user_presets = {}
        with patch("wune.settings.os.replace", side_effect=PermissionError("locked")), self.assertWarns(RuntimeWarning):
            self.assertFalse(store.save(Config(), (800, 600), (0, 0), "CLASSIC"))
        self.assertEqual(self.path.read_bytes(), before)

    def test_motion_round_trip_and_legacy_defaults(self):
        cfg = Config(vis_attack_ms=25, vis_release_ms=300, peak_hold_ms=0, peak_fall_per_second=7.5)
        SettingsStore(self.path).save(cfg, (800, 600), (0, 0), "CLASSIC")
        actual, _ = SettingsStore(self.path).load(Config())
        self.assertEqual((actual.vis_attack_ms, actual.vis_release_ms, actual.peak_hold_ms,
                          actual.peak_fall_per_second), (25, 300, 0, 7.5))
        self.path.write_text('{"version": 1, "appearance": {"initial_preset": "BLUE"}}')
        actual, _ = SettingsStore(self.path).load(Config())
        self.assertEqual((actual.vis_attack_ms, actual.vis_release_ms, actual.peak_hold_ms,
                          actual.peak_fall_per_second), (5, 120, 500, 2.5))

    def test_invalid_persisted_motion_uses_defaults(self):
        self.path.write_text(json.dumps({"version": 1, "appearance": {
            "vis_attack_ms": 0, "vis_release_ms": float('nan'),
            "peak_hold_ms": True, "peak_fall_per_second": 21}}))
        with self.assertWarns(RuntimeWarning):
            cfg, _ = SettingsStore(self.path).load(Config())
        self.assertEqual((cfg.vis_attack_ms, cfg.vis_release_ms, cfg.peak_hold_ms,
                          cfg.peak_fall_per_second), (5, 120, 500, 2.5))

    def test_frequency_cap_round_trip_and_old_settings(self):
        cfg = Config(limit_to_20khz=True)
        SettingsStore(self.path).save(cfg, (800, 600), (0, 0), "CLASSIC")
        actual, _ = SettingsStore(self.path).load(Config())
        self.assertTrue(actual.limit_to_20khz)
        self.assertEqual(actual.spectrum_upper_hz(96000), 20000)
        self.path.write_text('{"version": 1}')
        actual, _ = SettingsStore(self.path).load(Config())
        self.assertFalse(actual.limit_to_20khz)
        self.path.write_text('{"version": 1, "appearance": {"limit_to_20khz": "false"}}')
        with self.assertWarns(RuntimeWarning):
            actual, _ = SettingsStore(self.path).load(Config())
        self.assertFalse(actual.limit_to_20khz)
