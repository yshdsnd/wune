"""Shared application identity for source checkouts and standalone bundles."""
import json
from pathlib import Path
import re
import subprocess
import sys


VERSION_PATTERN = r"[0-9]+\.[0-9]+\.[0-9]+(?:[._-][0-9A-Za-z]+)*"


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


def installed_package_version():
    """Retrieve version metadata when running from an installed system/Python package."""
    # 1. Package-internal build-info.json (installed by OS ports/packages)
    pkg_info = Path(__file__).resolve().parent / "build-info.json"
    if pkg_info.is_file():
        try:
            data = json.loads(pkg_info.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                ver = data.get("release_version")
                if isinstance(ver, str) and re.fullmatch(VERSION_PATTERN, ver):
                    return ver
        except (OSError, ValueError):
            pass

    # 2. FreeBSD pkg query (exact package version from pkg DB)
    if sys.platform.startswith("freebsd"):
        try:
            val = subprocess.check_output(
                ["pkg", "query", "%v", "wune"],
                stderr=subprocess.DEVNULL, timeout=2,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            ).decode().strip()
            if val and re.fullmatch(VERSION_PATTERN, val):
                return val
        except (OSError, subprocess.SubprocessError, UnicodeError):
            pass

    # 3. Python package metadata (pip/setuptools)
    try:
        from importlib.metadata import version
        val = version("wune")
        if val and re.fullmatch(VERSION_PATTERN, val):
            return val
    except Exception:
        pass

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

    commit = source_commit(Path(__file__).resolve().parents[1])
    if commit:
        return format_identity({"commit": commit})

    pkg_version = installed_package_version()
    if pkg_version:
        return format_identity({"release_version": pkg_version})

    return format_identity(None)


def window_title(language):
    from .i18n import Translator
    return Translator(language)("app.title", identity=application_identity())
