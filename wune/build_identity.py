"""Shared application identity for source checkouts and standalone bundles."""
import json
from pathlib import Path
import re
import subprocess
import sys


VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"


def source_commit(root):
    try:
        # Do not accidentally identify a containing, unrelated repository.
        top = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=root,
                                      stderr=subprocess.DEVNULL, timeout=3,
                                      creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if Path(top.decode().strip()).resolve() != Path(root).resolve():
            return None
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                        stderr=subprocess.DEVNULL, timeout=3,
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).decode().strip()
        return value if re.fullmatch(r"[0-9a-f]{40,64}", value) else None
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None


def format_identity(data):
    if not isinstance(data, dict):
        return "Wune dev (unknown)"
    version = data.get("release_version")
    if isinstance(version, str) and re.fullmatch(VERSION_PATTERN, version):
        return f"Wune v{version}"
    commit = data.get("commit")
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40,64}", commit):
        return f"Wune dev ({commit[:7]})"
    return "Wune dev (unknown)"


def application_identity():
    if getattr(sys, "frozen", False):
        try:
            data = json.loads((Path(sys.executable).parent / "build-info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        # A packaged build must never pick up Git metadata from the user's cwd.
        return format_identity(data)
    return format_identity({"commit": source_commit(Path(__file__).resolve().parents[1])})


def window_title(language):
    from .i18n import Translator
    return Translator(language)("app.title", identity=application_identity())
