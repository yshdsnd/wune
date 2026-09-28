# Wune

[日本語](README.md) / **English**

Wune is an LED audio spectrum analyzer for Windows, inspired by compact stereo systems from the 1980s and 1990s.
Watch the left and right channels respond while listening to music through your usual speakers or headphones.
Customize the colors, LED shapes, layout, and response to the beat in Settings.

![Stereo display with the CLASSIC theme](docs/images/wune-classic.png)

Display example rendered by the current application using demo data; it is not a measurement of audio hardware.

## Download

Download the Windows ZIP from **[GitHub Releases](https://github.com/yshdsnd/wune/releases)**.
Look for **Wune-vX.Y.Z-win64.zip**, where X.Y.Z is the version number.
The “Source code” downloads automatically provided by GitHub are not the packaged application.

**The v1.0 public release is currently being prepared.** If no release ZIP is available yet, follow these instructions once it is published.

## Get started

1. Right-click the ZIP and choose **Extract All** to extract it to a folder of your choice.
2. Open the **Wune** folder inside the extracted files.
3. Play audio through your speakers or headphones using your usual music app.
4. Double-click **Wune.exe**.

**You do not need to install Python or run Wune as an administrator for normal use.**
Extract the entire ZIP rather than launching the application from inside it.
The **_internal** folder must remain next to Wune.exe. Do not move the EXE on its own.
To launch Wune from your desktop, create a shortcut to Wune.exe.

Wune captures audio from the default Windows output device selected at startup. It does not play audio or change your output device.
The information bar shows the device name, capture sample rate, and channel count.
Restart Wune after changing the output device or its sample rate in Windows.

### Update or uninstall

To update, close Wune, extract the new ZIP into a separate folder, and launch it.
Settings are stored in your user profile and carry over when you change the application folder.
After checking that the new version works, delete the old application folder and update any shortcuts.

To uninstall, close Wune and delete the extracted application folder and any shortcuts.
If you also want to remove settings and logs, delete their folder described below.

## Controls

Click the main window before using keyboard shortcuts.

| Action | Function |
| --- | --- |
| F2 | Open Settings |
| F11 | Toggle fullscreen / windowed mode |
| Space | Pause / resume the display; music playback continues |
| T, or left-click the theme name at the top right | Switch to the next color theme |
| I | Show / hide the output information bar |
| Esc, Q, or the close button | Quit |
| Drag the window border | Resize the display |

Window dimensions are constrained by a minimum size and aspect-ratio adjustments to preserve LED shapes and spacing.
Width and height cannot be changed completely independently.
Pressing F2 in fullscreen mode returns to a normal window.
While Settings is open, theme changes through T or the theme badge, and information-bar changes through I, are disabled.
Esc or the close button in the Settings window cancels settings changes rather than quitting the application.

## Settings and customization

Press **F2** to open Settings. Preview changes in the main display while music continues to play.

| Tab | Main settings |
| --- | --- |
| Layout and LEDs | Language, frequency / level orientation, stacked or side-by-side L/R channels, LED rendering / shape / aspect ratio, information bar, and a 20 kHz display limit |
| Themes and colors | Select, create, duplicate, rename, or delete color themes; edit individual colors |
| Motion | Attack time, release time, peak hold time, and peak fall speed |

“LED aspect ratio (width/height)” controls the shape: higher values make LEDs wider relative to their height. Resize the window to change the overall display size.
Themes store colors only. LED shapes, layout, and motion are separate settings.
Editing a built-in theme creates a user copy, preserving the original colors.

| Button | Behavior |
| --- | --- |
| Save | Write the current changes to disk immediately and close Settings |
| Apply | Confirm changes and keep Settings open; write them to disk on normal exit |
| Cancel / close / Esc | Restore the last applied state, or the state before opening Settings if nothing has been applied |
| Reset display | Preview the default display settings and colors; preserve language, motion settings, and the user theme list |
| Reset motion only | Preview defaults for the four settings on the Motion tab only |

On exit, Wune saves the window size and position and the confirmed display, theme, and motion settings, then restores them at the next launch.
If you quit with Settings still open, unapplied previews are not saved.

### Switch between English and Japanese

Open F2 → **Layout and LEDs**（配置・LED）→ **Language**（表示言語）and select
System / Auto, English, or 日本語. Automatic selection falls back to English for unsupported system languages.
The main display updates immediately. To update the Settings window, use Save to close it, or Apply and then close it, and reopen it with F2.
The standard Windows color picker follows the Windows display language.

## Settings location and reset

Paste the following into the File Explorer address bar:

~~~text
%LOCALAPPDATA%\Wune
~~~

| File | Contents |
| --- | --- |
| settings.json | Display settings, user themes, and the normal window's position and size |
| Wune.log | Startup and runtime log for the packaged EXE; overwritten at each launch |

The Settings window also shows the save path at the bottom. To back up your settings, close Wune and copy settings.json.

To reset everything, close Wune, **move settings.json to another location**, then launch Wune.
This also resets user themes and window placement. To restore your backup, close Wune and put the file back.
Corrupt files or unsupported settings formats are left untouched while Wune starts with defaults, so this reset method can also help when settings are not being saved.

To reset from the command line, open PowerShell in the folder containing Wune.exe.
This deletes the existing settings file; back it up first if needed.

~~~powershell
.\Wune.exe --reset-settings
~~~

## Supported environment and limitations

- **Tested: Windows 11 (64-bit), using the x64 package.**
- An output device capable of stereo audio playback is required.
- Windows 10 and Windows on ARM have not been tested. No packages are available for 32-bit Windows, macOS, Linux, or FreeBSD.
- Audio is captured from the default Windows output device through WASAPI loopback. Microphone input and direct audio-file loading are not supported.
- Output-device and sample-rate changes are not followed automatically while Wune is running.
- Capture uses channels 0 and 1. Full surround downmixing and mono-only devices are not supported.
- The Settings window does not include audio-device selection or band-count controls.
- At higher sample rates, the display range extends up to 40 kHz. Use “Limit display to 20 kHz (keep capture rate)” to focus on the audible range.
- Some scale labels are omitted in narrow layouts. Levels are per-band indications, not sound-pressure or true-peak measurements.
- Changes to DPI or display configuration while running have not been tested.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| Wune will not start or closes immediately | Extract the entire ZIP and check that _internal is next to the EXE. Check any error message and Wune.log |
| The spectrum does not move | Resume with Space if paused. Check that audio is playing and that the information bar names the same output device used by your playback app |
| The display stops after switching to headphones or another output | Check the Windows output device and restart Wune |
| F2 or other shortcuts do not work | Click the main window first. Check whether Settings is open behind another window |
| Settings revert or are not saved | Use Save, or Apply followed by a normal exit. Check the save path and log; back up and reset settings if necessary |
| Only the Settings window stays in the previous language | Save or apply the language, then close and reopen Settings |

For unresolved problems or feature suggestions, please use **[GitHub Issues](https://github.com/yshdsnd/wune/issues)**.
For bug reports, include:

- Wune version (ZIP filename or the bundled build-info.json) and Windows version
- Audio output device name, sample rate shown in the information bar, and steps to reproduce
- Expected and actual behavior, with screenshots if useful
- Any error message and Wune.log captured immediately after the problem

Remove usernames or other information you do not want to share publicly from logs and screenshots.

## Development / running from source

For normal use, choose the ZIP described above.
See the **[developer guide (Japanese)](docs/development.md)** for Python setup, running from source, development settings, and tests.
See the **[packaging guide (Japanese)](docs/packaging.md)** for building the distribution ZIP and checks before publication.

## Credits and license

Wune uses NumPy, pygame / SDL, SoundCard / CFFI, and Python / Tcl / Tk, and is packaged with PyInstaller.
The distribution ZIP includes third-party licenses and notices in the licenses folder.
Its inventory.json lists the actual bundled components and DLLs, and sources/ contains the corresponding sources for rebuilding.
See the **[bundled software notes (Japanese)](packaging/licenses/README.md)** for third-party provenance and redistribution terms.
The custom icon's provenance and creation notes are in the **[icon documentation](https://github.com/yshdsnd/wune/blob/main/wune/assets/README.md)**.

Wune itself is provided under the **[BSD 2-Clause License](LICENSE)**.
Bundled third-party software retains its own licenses; Wune's license does not replace them.
