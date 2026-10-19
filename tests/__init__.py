# tests/__init__.py
import sys

if sys.platform.startswith("freebsd"):
    from wune.soundcard_compat import prepare_soundcard
    prepare_soundcard()
