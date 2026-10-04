import argparse
import importlib.util
from pathlib import Path
import unittest


class PackagingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "tools" / "build_windows.py"
        spec = importlib.util.spec_from_file_location("build_windows", path)
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    def test_version_is_safe_for_archive_names(self):
        for value in ("1.0.0", "1.0.0-rc.1", "0.0.0-dev"):
            self.assertEqual(self.builder.validate_version(value), value)
        for value in ("../release", "v1.0.0", "1.0", "1.0.0/x", "1.0.0;exit", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                self.builder.validate_version(value)

    def test_source_python_cannot_masquerade_as_frozen_smoke_test(self):
        from wune.package_smoke import run
        with self.assertRaisesRegex(RuntimeError, "packaged Wune.exe"):
            run(Path("must-not-be-created.json"))

    def test_macos_version_is_safe_for_archive_names(self):
        path = Path(__file__).resolve().parents[1] / "tools" / "build_macos.py"
        spec = importlib.util.spec_from_file_location("build_macos", path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        for value in ("1.0.0", "1.0.0-rc.1", "0.0.0-dev"):
            self.assertEqual(builder.validate_version(value), value)
        for value in ("../release", "v1.0.0", "1.0", "1.0.0/x", "1.0.0;exit", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                builder.validate_version(value)

    def test_platform_installation_documents_exist_and_linked(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("INSTALL_WINDOWS.md", "INSTALL_WINDOWS.en.md",
                     "INSTALL_MACOS.md", "INSTALL_MACOS.en.md"):
            doc = root / name
            self.assertTrue(doc.is_file(), f"{name} must exist")
            self.assertGreater(len(doc.read_text(encoding="utf-8").strip()), 100,
                               f"{name} must not be empty")

        readme_ja = (root / "README.md").read_text(encoding="utf-8")
        readme_en = (root / "README.en.md").read_text(encoding="utf-8")
        self.assertIn("INSTALL_WINDOWS.md", readme_ja)
        self.assertIn("INSTALL_MACOS.md", readme_ja)
        self.assertIn("INSTALL_WINDOWS.en.md", readme_en)
        self.assertIn("INSTALL_MACOS.en.md", readme_en)

    def test_packaging_scripts_bundle_platform_specific_install_docs(self):
        root = Path(__file__).resolve().parents[1]
        win_script = (root / "tools" / "build_windows.py").read_text(encoding="utf-8")
        mac_script = (root / "tools" / "build_macos.py").read_text(encoding="utf-8")

        self.assertIn("INSTALL_WINDOWS.md", win_script)
        self.assertIn("INSTALL.md", win_script)
        self.assertNotIn("INSTALL_MACOS.md", win_script)

        self.assertIn("INSTALL_MACOS.md", mac_script)
        self.assertIn("INSTALL.md", mac_script)
        self.assertNotIn("INSTALL_WINDOWS.md", mac_script)
