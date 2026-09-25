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
