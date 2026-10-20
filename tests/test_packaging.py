import argparse
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch


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
            wune_calls = []
            def fake_run(cmd, *args, **kwargs):
                if isinstance(cmd, (list, tuple)) and any(str(p).endswith("Wune.exe") for p in cmd):
                    wune_calls.append(cmd)
                    raise sac_error
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch.dict("os.environ", {"SYSTEMROOT": r"C:\Windows"}), \
                 patch.object(self.builder.subprocess, "run", side_effect=fake_run), \
                 patch.object(self.builder, "try_self_sign") as mock_sign:
                with self.assertRaises(OSError) as cm:
                    self.builder.smoke_test(bundle)
                self.assertIs(cm.exception, sac_error)
                self.assertEqual(len(wune_calls), 5)
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
            for platform in ("WINDOWS", "MACOS", "LINUX"):
                name = "INSTALL_" + platform + suffix
                self.assertGreater(len((root / name).read_text(encoding="utf-8")), 100)
                self.assertIn(name, (root / readme).read_text(encoding="utf-8"))

    def test_packaging_scripts_bundle_platform_specific_install_docs_only(self):
        root = Path(__file__).resolve().parents[1]
        win_script = (root / "tools" / "build_windows.py").read_text(encoding="utf-8")
        mac_script = (root / "tools" / "build_macos.py").read_text(encoding="utf-8")

        self.assertIn("INSTALL_WINDOWS.md", win_script)
        self.assertIn("INSTALL.md", win_script)
        self.assertNotIn("INSTALL_MACOS.md", win_script)
        self.assertNotIn("INSTALL_MACOS.en.md", win_script)
        self.assertNotIn("INSTALL_LINUX.md", win_script)
        self.assertNotIn("INSTALL_LINUX.en.md", win_script)

        self.assertIn("INSTALL_MACOS.md", mac_script)
        self.assertIn("INSTALL.md", mac_script)
        self.assertNotIn("INSTALL_WINDOWS.md", mac_script)
        self.assertNotIn("INSTALL_WINDOWS.en.md", mac_script)
        self.assertNotIn("INSTALL_LINUX.md", mac_script)
        self.assertNotIn("INSTALL_LINUX.en.md", mac_script)

        linux_script = (root / "tools" / "build_linux.py").read_text(encoding="utf-8")
        self.assertIn("INSTALL_LINUX.md", linux_script)
        self.assertIn("INSTALL.md", linux_script)
        self.assertNotIn("INSTALL_WINDOWS.md", linux_script)
        self.assertNotIn("INSTALL_WINDOWS.en.md", linux_script)
        self.assertNotIn("INSTALL_MACOS.md", linux_script)
        self.assertNotIn("INSTALL_MACOS.en.md", linux_script)

        # Bundles must replace INSTALL_<OS>.md with INSTALL.md, not keep duplicate original filenames
        self.assertNotIn('bundle / "INSTALL_WINDOWS.md"', win_script)
        self.assertNotIn('bundle / name', win_script)
        self.assertNotIn('bundle_dir / "INSTALL_MACOS.md"', mac_script)
        self.assertNotIn('bundle_dir / name', mac_script)
        self.assertNotIn('bundle / "INSTALL_LINUX.md"', linux_script)

    def linux_builder(self):
        path = Path(__file__).resolve().parents[1] / "tools" / "build_linux.py"
        spec = importlib.util.spec_from_file_location("build_linux", path)
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        return builder

    def test_linux_build_rejects_unsupported_architecture_before_writing(self):
        builder = self.linux_builder()
        with patch.object(builder.sys, "platform", "linux"), \
             patch.object(builder.platform, "machine", return_value="armv7l"), \
             patch.object(builder.subprocess, "run") as run, patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                builder.main(["--version", "1.1.0"])
            build_calls = [
                c for c in run.call_args_list
                if c.args and c.args[0] and (
                    c.args[0][0] == sys.executable or "PyInstaller" in str(c.args[0])
                )
            ]
            self.assertEqual(build_calls, [])

    def test_linux_version_validation(self):
        builder = self.linux_builder()
        self.assertEqual(builder.validate_version("1.1.0"), "1.1.0")
        self.assertEqual(builder.validate_version("1.1.0-rc.1"), "1.1.0-rc.1")
        with self.assertRaises(argparse.ArgumentTypeError):
            builder.validate_version("v1.1.0")

    def test_linux_desktop_file_creation(self):
        builder = self.linux_builder()
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir)
            builder.create_desktop_file(target)
            desktop_file = target / "wune.desktop"
            self.assertTrue(desktop_file.is_file())
            content = desktop_file.read_text(encoding="utf-8")
            self.assertIn("[Desktop Entry]", content)
            self.assertIn("Name=Wune", content)
            self.assertIn("Exec=wune", content)
            self.assertIn("Icon=wune", content)

    def test_linux_rpm_creation_when_rpmbuild_missing_skips(self):
        builder = self.linux_builder()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = Path(tmp_dir) / "bundle"
            bundle.mkdir()
            output = Path(tmp_dir) / "dist"
            output.mkdir()
            with patch.object(builder.shutil, "which", return_value=None):
                result = builder.create_rpm_package(bundle, output, "1.2.0")
                self.assertIsNone(result)

    def test_linux_rpm_creation_with_mock_rpmbuild(self):
        builder = self.linux_builder()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = Path(tmp_dir) / "bundle"
            bundle.mkdir()
            (bundle / "Wune").touch()
            (bundle / "wune.desktop").touch()
            (bundle / "wune.png").touch()
            output = Path(tmp_dir) / "dist"
            output.mkdir()

            rpm_calls = []
            def fake_rpmbuild(cmd, *args, **kwargs):
                if isinstance(cmd, (list, tuple)) and cmd[0] == "rpmbuild":
                    rpm_calls.append(cmd)
                    spec_path = Path(cmd[2])
                    spec_text = spec_path.read_text(encoding="utf-8")
                    self.assertIn("License:        BSD-2-Clause", spec_text)
                    self.assertIn("Requires:       pulseaudio-libs", spec_text)
                    self.assertIn("/usr/bin/wune", spec_text)
                    self.assertIn("/usr/bin/Wune", spec_text)
                    topdir = spec_path.parents[1]
                    rpm_dir = topdir / "RPMS" / "x86_64"
                    rpm_dir.mkdir(parents=True, exist_ok=True)
                    (rpm_dir / "wune-1.2.0-1.x86_64.rpm").write_bytes(b"dummy_rpm_content")
                    return MagicMock(returncode=0)
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch.object(builder.shutil, "which", return_value="/usr/bin/rpmbuild"), \
                 patch.object(builder.subprocess, "run", side_effect=fake_rpmbuild):
                result = builder.create_rpm_package(bundle, output, "1.2.0")
                self.assertIsNotNone(result)
                self.assertTrue(result.is_file())
                self.assertEqual(result.name, "Wune-v1.2.0-1.x86_64.rpm")
                self.assertTrue(result.with_suffix(".rpm.sha256").is_file())
                self.assertEqual(len(rpm_calls), 1)

    def test_linux_deb_creation_when_dpkg_deb_missing_skips(self):
        builder = self.linux_builder()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = Path(tmp_dir) / "bundle"
            bundle.mkdir()
            output = Path(tmp_dir) / "dist"
            output.mkdir()
            with patch.object(builder.shutil, "which", return_value=None):
                result = builder.create_deb_package(bundle, output, "1.2.0")
                self.assertIsNone(result)

    def test_linux_deb_creation_with_mock_dpkg_deb(self):
        builder = self.linux_builder()
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = Path(tmp_dir) / "bundle"
            bundle.mkdir()
            (bundle / "Wune").touch()
            (bundle / "wune.desktop").touch()
            (bundle / "wune.png").touch()
            output = Path(tmp_dir) / "dist"
            output.mkdir()

            deb_calls = []
            def fake_dpkg_deb(cmd, *args, **kwargs):
                if isinstance(cmd, (list, tuple)) and cmd[0] == "dpkg-deb":
                    deb_calls.append(cmd)
                    pkg_root = Path(cmd[3])
                    control_text = (pkg_root / "DEBIAN" / "control").read_text(encoding="utf-8")
                    self.assertIn("Depends: libpulse0", control_text)
                    wune_bin = pkg_root / "usr" / "bin" / "wune"
                    wune_upper = pkg_root / "usr" / "bin" / "Wune"
                    self.assertTrue(os.path.lexists(wune_bin))
                    self.assertTrue(os.path.lexists(wune_upper))
                    if wune_bin.is_symlink():
                        self.assertEqual(os.readlink(wune_bin), "/opt/wune/Wune")
                    if wune_upper.is_symlink():
                        self.assertEqual(os.readlink(wune_upper), "/opt/wune/Wune")
                    target_deb = Path(cmd[4])
                    target_deb.write_bytes(b"dummy_deb_content")
                    return MagicMock(returncode=0)
                return MagicMock(returncode=0, stdout="", stderr="")

            with patch.object(builder.shutil, "which", return_value="/usr/bin/dpkg-deb"), \
                 patch.object(builder.subprocess, "run", side_effect=fake_dpkg_deb):
                result = builder.create_deb_package(bundle, output, "1.2.0-rc.1")
                self.assertIsNotNone(result)
                self.assertTrue(result.is_file())
                self.assertEqual(result.name, "wune_1.2.0~rc.1_amd64.deb")
                self.assertTrue(result.with_suffix(".deb.sha256").is_file())
                self.assertEqual(len(deb_calls), 1)

    def test_linux_licenses_collect_notices(self):
        try:
            from tools.linux_licenses import collect_notices
        except ModuleNotFoundError:
            from linux_licenses import collect_notices
        with tempfile.TemporaryDirectory() as tmp_dir:
            bundle = Path(tmp_dir) / "bundle"
            bundle.mkdir()
            (bundle / "Wune").touch()
            inputs_file = Path(tmp_dir) / "inputs.json"
            inputs_file.write_text("[]", encoding="utf-8")
            inventory = collect_notices(bundle, inputs_file)
            self.assertEqual(inventory["platform"], "Linux")
            self.assertTrue((bundle / "licenses" / "inventory.json").is_file())
            self.assertTrue((bundle / "licenses" / "provenance.json").is_file())


