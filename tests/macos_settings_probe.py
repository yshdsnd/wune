"""Real spawn/Tk lifecycle check, isolated from the unittest process."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == '__main__':
    from wune.appearance import AppearanceState
    from wune.config import Config
    from wune.settings_dialog import SettingsDialog
    for commit in (False, True):
        dialog = SettingsDialog(AppearanceState.capture(Config(), 'CLASSIC', {}), 'unused.json')
        try:
            assert dialog.events.get(timeout=20) == ('ready', None)
            if commit:
                dialog.reply(True, close=True)
                assert dialog.events.get(timeout=10) == ('closed', None)
        finally:
            dialog.close()
        assert not dialog.thread.is_alive()
