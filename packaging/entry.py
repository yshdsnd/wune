"""Console-free executable entry; source main.py remains available."""
import os
from pathlib import Path
import sys
import traceback


def launch():
    # PyInstaller windowed builds have no stdout/stderr. Keep diagnostics per user.
    from wune.settings import settings_path
    log = settings_path().with_name("Wune.log")
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        stream = log.open("w", encoding="utf-8", buffering=1)
    except OSError:
        stream = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stderr = stream
    try:
        if len(sys.argv) == 3 and sys.argv[1] == "--package-smoke-test":
            from wune.package_smoke import run
            run(Path(sys.argv[2]))
        else:
            from main import main
            main()
        return 0
    except Exception:
        traceback.print_exc()
        stream.flush()
        if "--package-smoke-test" not in sys.argv:
            # Also works if pygame/Tk failed before the application opened.
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                None, f"Wune could not start. / Wuneを起動できませんでした。\n\n{log}\n\n{traceback.format_exc()}",
                "Wune", 0x10)
        return 1
    finally:
        stream.flush()


if __name__ == "__main__":
    raise SystemExit(launch())
