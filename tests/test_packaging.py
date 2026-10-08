import argparse
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


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

    def test_windows_smoke_test_reraises_sac_error_when_retries_exhausted(self):
        sac_error = OSError("Smart App Control blocked execution")
        sac_error.winerror = 4551
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = Path(tmp_dir) / "bundle"
            bundle.mkdir()
            (bundle / "Wune.exe").touch()
            with patch.dict("os.environ", {"SYSTEMROOT": r"C:\Windows"}), \
                 patch.object(self.builder.subprocess, "run", side_effect=sac_error) as mock_run, \
                 patch.object(self.builder, "try_self_sign") as mock_sign:
                with self.assertRaises(OSError) as cm:
                    self.builder.smoke_test(bundle)
                self.assertIs(cm.exception, sac_error)
                self.assertEqual(mock_run.call_count, 5)
                self.assertEqual(mock_sign.call_count, 4)

    def mac_builder(self):
        path = Path(__file__).resolve().parents[1] / "tools" / "build_macos.py"
        spec = importlib.util.spec_from_file_location("build_macos", path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        return builder

    def test_mac_build_rejects_unsupported_architecture_before_writing(self):
        builder = self.mac_builder()
        with patch.object(builder.sys, "platform", "darwin"), \
             patch.object(builder.platform, "machine", return_value="x86_64"), \
             patch.object(builder.subprocess, "run") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                builder.main(["--version", "1.1.0"])
            run.assert_not_called()

    def test_mac_rebuilds_native_helper_for_target_architecture(self):
        builder = self.mac_builder()
        with patch.object(builder.subprocess, "run") as run:
            builder.build_audio_helper()
        command = run.call_args.args[0]
        self.assertIn("arm64", command)
        self.assertIn("-mmacosx-version-min=14.2", command)
        self.assertIn(str(builder.ROOT / "wune" / "tap_backend.m"), command)
        self.assertTrue(run.call_args.kwargs["check"])

    def test_mac_version_validation(self):
        builder = self.mac_builder()
        self.assertEqual(builder.validate_version("1.1.0-rc.1"), "1.1.0-rc.1")
        with self.assertRaises(argparse.ArgumentTypeError):
            builder.validate_version("../release")

    def test_platform_installation_documents_are_linked(self):
        root = Path(__file__).resolve().parents[1]
        for suffix, readme in ((".md", "README.md"), (".en.md", "README.en.md")):
            for platform in ("WINDOWS", "MACOS"):
                name = "INSTALL_" + platform + suffix
                self.assertGreater(len((root / name).read_text(encoding="utf-8")), 100)
                self.assertIn(name, (root / readme).read_text(encoding="utf-8"))
