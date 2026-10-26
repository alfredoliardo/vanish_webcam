#!/bin/bash
# Crea dist/VanishCam, un eseguibile unico per macOS (stessa architettura del Mac che lo compila).
cd "$(dirname "$0")" || exit 1

VENV_DIR=".venv-build"
VENV_PY="$VENV_DIR/bin/python"

fail() {
  echo
  echo "Build non riuscita. Controlla i messaggi qui sopra."
  read -r -p "Premi INVIO per chiudere..."
  exit 1
}

if [ ! -x "$VENV_PY" ]; then
  PYTHON=""
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON" ]; then
    echo "Installa Python 3 per macOS prima di creare l'eseguibile."
    fail
  fi
  "$PYTHON" -m venv "$VENV_DIR" || fail
fi

if ! "$VENV_PY" -c "import PyInstaller, cv2, mediapipe, numpy, pyvirtualcam" >/dev/null 2>&1; then
  echo "Installo le dipendenze di build..."
  "$VENV_PY" -m pip install -r requirements.txt "pyinstaller>=6,<7" || fail
fi

if [ ! -f "models/gesture_recognizer.task" ] || [ ! -f "models/selfie_segmenter_landscape.tflite" ] \
   || [ ! -f "models/blaze_face_short_range.tflite" ]; then
  echo "Scarico i modelli MediaPipe necessari..."
  "$VENV_PY" -c "from vanish_cam import download_models; download_models()" || fail
fi

"$VENV_PY" -m PyInstaller --noconfirm --distpath dist --workpath build VanishCam.spec || fail
"dist/VanishCam" --self-test || fail

echo
echo "Pronto: dist/VanishCam"
exit 0
