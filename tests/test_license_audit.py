import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import license_audit as audit


class LicenseAuditTests(unittest.TestCase):
    def test_namespace_placeholder_is_not_treated_as_a_file(self):
        self.assertTrue(audit.is_namespace_entry("-", "PYMODULE"))
        for source, kind in (("-", "BINARY"), ("-", "DATA"), ("missing.py", "PYMODULE")):
            self.assertFalse(audit.is_namespace_entry(source, kind))

    def test_notice_detection_keeps_nested_and_embedded_notices(self):
        for name in ("numpy/licenses/src/dragon4_LICENSE.txt", "setuptools/config/NOTICE",
                     "vendor/pkg.dist-info/licenses/any-name.txt", "docs/LGPL.txt"):
            self.assertTrue(audit.is_notice(Path(name)), name)
        for name in ("pip/main.py", "license_audit.py", "foo/licensed_module.py"):
            self.assertFalse(audit.is_notice(Path(name)), name)

    def test_missing_and_empty_notices_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "LICENSE"
            for exists in (False, True):
                if exists:
                    source.touch()
                with self.assertRaisesRegex(RuntimeError, "Missing or empty"):
                    audit.copy_required(source, Path(directory) / "output")

    def test_final_zip_rejects_missing_or_changed_notices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "Wune"
            (bundle / "licenses").mkdir(parents=True)
            (bundle / "LICENSE").write_text("Wune license")
            (bundle / "licenses" / "NOTICE").write_text("upstream notice")
            archive = root / "release.zip"
            for content in (None, "wrong", "upstream notice"):
                with zipfile.ZipFile(archive, "w") as zipped:
                    zipped.write(bundle / "LICENSE", "Wune/LICENSE")
                    if content is not None:
                        zipped.writestr("Wune/licenses/NOTICE", content)
                if content == "upstream notice":
                    audit.verify_archive(archive, bundle)
                else:
                    with self.assertRaises((KeyError, RuntimeError)):
                        audit.verify_archive(archive, bundle)

    def test_source_checksum_mismatch_stops_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supplement = root / "supplement"
            supplement.mkdir()
            (supplement / "sources.json").write_text(json.dumps([
                {"filename": "source.zip", "url": "https://invalid.example", "sha256": "0" * 64}]))
            cache = root / "cache"
            cache.mkdir()
            (cache / "source.zip").write_bytes(b"changed upstream source")
            bundle = root / "Wune"
            (bundle / "licenses").mkdir(parents=True)
            with patch.object(audit, "SUPPLEMENT", supplement):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    audit.collect_sources(bundle, cache)

    def test_unknown_dll_or_font_stops_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "licenses").mkdir()
            (bundle / "licenses/native-libraries.json").write_text("{}")
            (bundle / "_internal").mkdir()
            for name in ("surprise.dll", "unknown.ttf"):
                file = bundle / "_internal" / name
                file.write_bytes(b"unknown component")
                with self.assertRaisesRegex(RuntimeError, "Unreviewed native"):
                    audit.audit_native_files(bundle, {})
                file.unlink()

    def test_native_changes_and_missing_notice_stop_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "licenses").mkdir()
            (bundle / "_internal").mkdir()
            file = bundle / "_internal/SDL2.dll"
            file.write_bytes(b"reviewed library")
            rule = {"component": "SDL2", "sha256": audit.sha256(file),
                    "notices": ["licenses/SDL-LICENSE"]}
            (bundle / "licenses/native-libraries.json").write_text(json.dumps({"sdl2.dll": rule}))
            with self.assertRaisesRegex(RuntimeError, "Missing notice"):
                audit.audit_native_files(bundle, {})
            (bundle / "licenses/SDL-LICENSE").write_text("upstream license")
            self.assertEqual(len(audit.audit_native_files(bundle, {})), 1)
            file.write_bytes(b"updated library needing review")
            with self.assertRaisesRegex(RuntimeError, "Native library changed"):
                audit.audit_native_files(bundle, {})

    def test_checked_in_native_rules_reference_real_notice_texts(self):
        rules = json.loads((audit.SUPPLEMENT / "native-libraries.json").read_text())
        self.assertIn("freesansbold.ttf", rules)
        self.assertIn("sdl2_ttf.dll", rules)
        for name, rule in rules.items():
            self.assertEqual(len(rule["sha256"]), 64, name)
            self.assertTrue(rule["notices"], name)
            for notice in rule["notices"]:
                self.assertGreater((audit.SUPPLEMENT.parent / notice).stat().st_size, 0, name)
        for record in json.loads((audit.SUPPLEMENT / "provenance.json").read_text()):
            self.assertEqual(audit.sha256(audit.SUPPLEMENT / record["path"]), record["sha256"])

    def test_audit_native_files_recognizes_winrt_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "licenses").mkdir()
            (bundle / "licenses/native-libraries.json").write_text("{}")
            (bundle / "_internal" / "winrt").mkdir(parents=True)
            notice = bundle / "licenses/pywinrt/LICENSE.txt"
            notice.parent.mkdir(parents=True)
            notice.write_text("pywinrt notice")
            components = {
                "winrt-Windows.Media.Control": {"notices": ["licenses/pywinrt/LICENSE.txt"]},
                "winrt-runtime": {"notices": ["licenses/pywinrt/LICENSE.txt"]},
            }
            pyd1 = bundle / "_internal/winrt/_winrt_windows_media_control.cp313-win_amd64.pyd"
            pyd1.write_bytes(b"media control binary")
            pyd2 = bundle / "_internal/winrt/_winrt.cp313-win_amd64.pyd"
            pyd2.write_bytes(b"runtime binary")
            results = audit.audit_native_files(bundle, components)
            self.assertEqual(len(results), 2)
            self.assertEqual({r["component"] for r in results}, {"winrt-Windows.Media.Control", "winrt-runtime"})

    def test_audit_native_files_recognizes_msvcp140(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            (bundle / "licenses").mkdir()
            (bundle / "licenses/native-libraries.json").write_text("{}")
            (bundle / "_internal").mkdir()
            (bundle / "licenses/Python").mkdir(parents=True)
            (bundle / "licenses/Python/LICENSE.txt").write_text("python license")
            (bundle / "licenses/README.md").write_text("licenses readme")
            dll = bundle / "_internal/msvcp140.dll"
            dll.write_bytes(b"msvcp140 binary")
            results = audit.audit_native_files(bundle, {})
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["component"], "Microsoft runtime")


if __name__ == "__main__":
    unittest.main()
