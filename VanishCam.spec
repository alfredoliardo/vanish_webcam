"""PyInstaller one-file build for the webcam app (Windows and macOS)."""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs


root = Path(SPECPATH).resolve()
icon_file = root / "assets" / "vanish_icon.ico"
resources = [
    (root / "hand_landmarker.task", "."),
    (root / "models" / "gesture_recognizer.task", "models"),
    (root / "models" / "selfie_segmenter_landscape.tflite", "models"),
    (root / "models" / "blaze_face_short_range.tflite", "models"),
    (root / "assets" / "substitution_log.png", "assets"),
    (root / "assets" / "vanish.MP3", "assets"),
    (root / "assets" / "goku-teleport-sound.mp3", "assets"),
]
# The .ico icon is Windows-only; macOS builds a console binary without one.
use_icon = sys.platform == "win32"
missing = [str(path) for path, _ in resources if not path.is_file()]
if use_icon and not icon_file.is_file():
    missing.append(str(icon_file))
if missing:
    raise SystemExit("File necessari alla build mancanti:\n" + "\n".join(missing))

a = Analysis(
    [str(root / "vanish_cam.py")],
    pathex=[str(root)],
    binaries=collect_dynamic_libs("mediapipe"),
    datas=[(str(path), destination) for path, destination in resources],
    hiddenimports=(["mediapipe.tasks.c"] if sys.platform == "darwin"
                   else ["pygame.mixer", "mediapipe.tasks.c"]),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VanishCam",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=str(icon_file) if use_icon else None,
)
