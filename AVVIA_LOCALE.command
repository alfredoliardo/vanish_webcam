#!/bin/bash
# Avvio per macOS: doppio clic dal Finder oppure ./AVVIA_LOCALE.command dal Terminale.
cd "$(dirname "$0")" || exit 1

VENV_DIR=".venv-local"
VENV_PY="$VENV_DIR/bin/python"

fail() {
  echo
  echo "Impossibile avviare Vanish Lab. Leggi README.md per le soluzioni comuni."
  read -r -p "Premi INVIO per chiudere..."
  exit 1
}

if [ ! -x "$VENV_PY" ]; then
  echo "[Vanish Lab] Creo l'ambiente Python locale..."
  PYTHON=""
  for candidate in python3.12 python3.11 python3.13 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
  if [ -z "$PYTHON" ]; then
    echo "Python 3 non trovato. Installalo da https://www.python.org/downloads/macos/ oppure con 'brew install python@3.12'."
    fail
  fi
  "$PYTHON" -m venv "$VENV_DIR" || fail
fi

if ! "$VENV_PY" -c "import cv2, mediapipe, numpy, pyvirtualcam" >/dev/null 2>&1; then
  echo "[Vanish Lab] Installo i componenti necessari. Serve internet solo questa volta..."
  "$VENV_PY" -m pip install --upgrade pip || fail
  "$VENV_PY" -m pip install -r requirements.txt || fail
fi

echo "[Vanish Lab] Avvio la webcam..."
"$VENV_PY" vanish_cam.py "$@" || fail
exit 0
