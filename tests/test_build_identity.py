import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

from wune import build_identity as identity


class BuildIdentityTests(unittest.TestCase):
    commit = "5ab996c" + "0" * 33

    def test_release_and_prerelease_metadata(self):
        for version in ("1.0.0", "1.0.0-rc.1"):
            self.assertEqual(identity.format_identity({"release_version": version, "commit": self.commit}),
                             f"Wune v{version}")

    def test_development_archive_version_is_not_a_release(self):
        self.assertEqual(identity.format_identity({"version": "1.0.0", "commit": self.commit}),
                         "Wune dev (5ab996c)")

    def test_invalid_or_missing_metadata_has_safe_fallback(self):
        for data in (None, [], {}, {"release_version": "invalid", "commit": "bad"},
                     {"release_version": 123, "commit": True}):
            self.assertEqual(identity.format_identity(data), "Wune dev (unknown)")

    def test_source_execution_uses_checkout_commit(self):
        with patch.object(identity.sys, "frozen", False, create=True), \
             patch.object(identity, "source_commit", return_value=self.commit):
            self.assertEqual(identity.application_identity(), "Wune dev (5ab996c)")

    def test_frozen_build_reads_metadata_beside_exe_without_git(self):
        for data, expected in (({"release_version": "1.0.0"}, "Wune v1.0.0"),
                               ({"commit": self.commit}, "Wune dev (5ab996c)")):
            with patch.object(identity.sys, "frozen", True, create=True), \
                 patch.object(identity.sys, "executable", str(Path("relocated/Wune.exe").resolve())), \
                 patch.object(Path, "read_text", autospec=True, return_value=json.dumps(data)) as read, \
                 patch.object(identity, "source_commit") as git:
                self.assertEqual(identity.application_identity(), expected)
                self.assertEqual(read.call_args.args[0], Path("relocated/build-info.json").resolve())
                git.assert_not_called()

    def test_missing_or_corrupt_frozen_metadata_never_uses_host_git(self):
        for error in (FileNotFoundError(), ValueError("bad JSON")):
            with patch.object(identity.sys, "frozen", True, create=True), \
                 patch.object(Path, "read_text", side_effect=error), \
                 patch.object(identity, "source_commit") as git:
                self.assertEqual(identity.application_identity(), "Wune dev (unknown)")
                git.assert_not_called()

    def test_git_resolution_and_unrelated_parent_repository(self):
        root = Path.cwd()
        with patch.object(subprocess, "check_output", side_effect=[str(root).encode(), self.commit.encode()]):
            self.assertEqual(identity.source_commit(root), self.commit)
        with patch.object(subprocess, "check_output", return_value=str(root.parent).encode()) as run:
            self.assertIsNone(identity.source_commit(root))
            self.assertEqual(run.call_count, 1)

    def test_git_unavailable_failure_and_timeout(self):
        for error in (FileNotFoundError(), subprocess.CalledProcessError(128, "git"),
                      subprocess.TimeoutExpired("git", 3)):
            with patch.object(subprocess, "check_output", side_effect=error):
                self.assertIsNone(identity.source_commit(Path.cwd()))

    def test_localized_titles_share_identity(self):
        with patch.object(identity, "application_identity", return_value="Wune v1.0.0"):
            self.assertEqual(identity.window_title("en"), "Wune v1.0.0 — F2: Settings")
            self.assertEqual(identity.window_title("ja"), "Wune v1.0.0 — F2: 設定")
