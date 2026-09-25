from pathlib import Path
from queue import Queue
import shutil
import string
import unittest
from unittest.mock import patch
import uuid

from wune.i18n import Translator, catalog, resolve_language
from wune.appearance import AppearanceDraft, AppearanceState, MessageError
from wune.config import Config
from wune.settings import SettingsStore


class TranslationTests(unittest.TestCase):
    def test_catalogs_have_matching_keys_and_placeholders(self):
        en, ja = catalog("en"), catalog("ja")
        self.assertGreater(len(en), 80)
        self.assertEqual(set(en), set(ja))
        for key in en:
            placeholders = lambda text: {field for _, field, _, _ in string.Formatter().parse(text) if field}
            self.assertEqual(placeholders(en[key]), placeholders(ja[key]), key)

    def test_system_language_and_unsupported_locale_fallback(self):
        for system, expected in (("ja_JP", "ja"), ("en-US", "en"), ("de_DE", "en"), (None, "en")):
            with patch("wune.system_locale.user_locale", return_value=system):
                self.assertEqual(resolve_language(), expected)
                self.assertEqual(resolve_language("ja"), "ja")

    def test_missing_keys_and_bad_translation_fall_back_safely(self):
        english = {"greeting": "Hello {name}"}
        for translated in ({}, {"greeting": "{unknown}"}):
            with patch("wune.i18n.catalog", side_effect=lambda code: english if code == "en" else translated):
                self.assertEqual(Translator("ja")("greeting", name="Wune"), "Hello Wune")
                self.assertEqual(Translator("ja")("missing"), "missing")

    def test_validation_errors_use_callers_language(self):
        draft = AppearanceDraft(AppearanceState.capture(Config(), "CLASSIC", {}))
        with self.assertRaises(MessageError) as raised:
            draft.create("")
        self.assertIn("theme name", Translator("en").error(raised.exception))
        self.assertIn("テーマ名", Translator("ja").error(raised.exception))

    def test_language_survives_save_and_legacy_defaults(self):
        directory = (Path.cwd() / f"test-settings-{uuid.uuid4().hex}").resolve()
        self.assertEqual(directory.parent, Path.cwd().resolve())
        directory.mkdir()
        self.addCleanup(shutil.rmtree, directory)
        path = directory / "settings.json"
        for language in ("auto", "en", "ja"):
            store = SettingsStore(path)
            self.assertTrue(store.save(Config(language=language), (1000, 700), (0, 0), "CLASSIC"))
            cfg, _ = SettingsStore(path).load(Config())
            self.assertEqual(cfg.language, language)
        path.write_text('{"version": 1}', encoding="utf-8")
        self.assertEqual(SettingsStore(path).load(Config())[0].language, "auto")

    def test_third_language_only_needs_a_catalog(self):
        with patch("wune.i18n.languages", return_value=("auto", "en", "ja", "fr")):
            with patch("wune.i18n.catalog", side_effect=lambda code: {"hello": "Bonjour"} if code == "fr" else {"hello": "Hello"}):
                self.assertEqual(Translator("fr_FR")("hello"), "Bonjour")


class LocalizedDialogTests(unittest.TestCase):
    def test_name_and_confirmation_prompts_use_localized_buttons(self):
        import tkinter as tk
        from tkinter import ttk
        from wune.localized_dialogs import Prompt
        root = tk.Tk()
        root.withdraw()
        try:
            for language, cancel in (("en", "Cancel"), ("ja", "キャンセル")):
                for initial in (None, "夜空"):
                    seen = []
                    def finish():
                        for child in root.winfo_children():
                            if isinstance(child, Prompt):
                                def visit(widget):
                                    if isinstance(widget, ttk.Button):
                                        seen.append(widget.cget("text"))
                                    for nested in widget.winfo_children():
                                        visit(nested)
                                visit(child)
                                child.ok()
                    root.after(30, finish)
                    prompt = Prompt(root, Translator(language), "Test", "Test", initial)
                    self.assertIn(cancel, seen)
                    self.assertEqual(prompt.result, True if initial is None else "夜空")
        finally:
            root.destroy()

    def test_both_languages_render_controls_and_keep_pending_values(self):
        import tkinter as tk
        from wune.settings_dialog import _Dialog
        for language, title, motion in (("en", "Display settings", "Motion"), ("ja", "表示設定", "動作")):
            root = tk.Tk()
            root.withdraw()
            try:
                draft = AppearanceDraft(AppearanceState.capture(Config(language=language), "CLASSIC", {}))
                events = Queue()
                dialog = _Dialog(root, draft, "settings.json", events, Queue())
                root.update_idletasks()
                self.assertIn(title, root.title())
                self.assertEqual(dialog.notebook.tab(2, "text"), motion)
                dialog.ratio.set("1.5")
                dialog.set_ratio()
                dialog.layout("language", "ja" if language == "en" else "en")
                self.assertEqual(dialog.ratio.get(), "1.5")
                self.assertEqual(draft.state.user_presets, {})
                self.assertIn(title, root.title())  # Deliberately applies on reopen.
                dialog.reset()
                self.assertNotEqual(draft.state.layout["language"], language)
            finally:
                root.destroy()
