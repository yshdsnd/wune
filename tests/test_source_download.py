import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.license_audit import download_source


class SourceDownloadTests(unittest.TestCase):
    def test_failed_primary_uses_mirror_with_same_digest(self):
        data = b"reviewed source"
        spec = {"url": "https://primary.example/a", "mirrors": ["https://mirror.example/a"],
                "sha256": hashlib.sha256(data).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source.zip"
            with patch("urllib.request.urlopen", side_effect=[OSError("timeout"), io.BytesIO(data)]) as fetch:
                download_source(spec, target)
            self.assertEqual(target.read_bytes(), data)
            self.assertEqual(fetch.call_args_list[1].args[0], spec["mirrors"][0])
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_changed_bytes_never_enter_cache(self):
        spec = {"url": "https://primary.example/a", "mirrors": ["https://mirror.example/a"], "sha256": "0" * 64}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source.zip"
            with patch("urllib.request.urlopen", return_value=io.BytesIO(b"changed")) as fetch:
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    download_source(spec, target)
            fetch.assert_called_once()
            self.assertFalse(target.exists())

    def test_unreachable_sources_fail_without_partial_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "source.zip"
            with patch("urllib.request.urlopen", side_effect=OSError("offline")):
                with self.assertRaisesRegex(RuntimeError, "Cannot download"):
                    download_source({"url": "https://example.com/source", "sha256": "0" * 64}, target)
            self.assertFalse(target.exists())
