"""Forehead two-finger pose and comic speed-line teleport for Goku mode."""

from __future__ import annotations

import cv2
import numpy as np


GOKU_HOLD_SECONDS = 1.0
GOKU_EFFECT_SECONDS = 0.95
# Moment the person disappears (or appears again, in reverse).
GOKU_SWAP_SECONDS = 0.42
# Short detection gaps do not break the pose countdown.
GOKU_GRACE_SECONDS = 0.25


def forehead_from_face(keypoints: np.ndarray, box_width: float) -> tuple[np.ndarray, float]:
    """Estimate the forehead centre from BlazeFace eyes and mouth keypoints."""
    eyes = keypoints[:2].mean(axis=0)
    mouth = keypoints[3]
    # Moving from the mouth to the eyes and as far again lands on the forehead,
    # also when the head is tilted.
    up = eyes - mouth
    return eyes + up * 0.8, max(box_width, float(np.linalg.norm(up)) * 2.2)


def two_finger_salute(points: np.ndarray, forehead: np.ndarray, face_size: float) -> bool:
    """Index and middle straight and together, ring and little finger folded,
    fingertips against the forehead (image coordinates in pixels)."""
    if points.shape != (21, 2) or not np.isfinite(points).all() or face_size <= 0:
        return False
    wrist = points[0]

    def reach(index: int) -> float:
        return float(np.linalg.norm(points[index] - wrist))

    palm = max(reach(9), 1e-6)
    extended = reach(8) > reach(6) * 1.08 and reach(12) > reach(10) * 1.08
    folded = reach(16) < reach(14) * 1.12 and reach(20) < reach(18) * 1.12
    width = max(float(np.linalg.norm(points[5] - points[9])), palm * 0.15)
    together = float(np.linalg.norm(points[8] - points[12])) < width * 1.6
    touch = min(float(np.linalg.norm(points[i] - forehead)) for i in (7, 8, 11, 12))
    return extended and folded and together and touch < face_size * 0.55


def make_speed_lines(width: int, seed: int = 4417) -> np.ndarray:
    """Columns, thickness, speed, phase, dash period and reveal threshold."""
    rng = np.random.default_rng(seed)
    count = max(12, width // 5)
    lines = np.empty((count, 6), dtype=np.float32)
    lines[:, 0] = rng.uniform(0, width, count)
    lines[:, 1] = rng.choice((1, 2, 2, 3, 4, 6), count)
    lines[:, 2] = rng.uniform(1.5, 4.5, count)
    lines[:, 3] = rng.uniform(0, 1, count)
    lines[:, 4] = rng.uniform(0.25, 0.9, count)
    lines[:, 5] = rng.uniform(0.0, 0.95, count)
    return lines


def _line_map(lines: np.ndarray, left: int, top: int, width: int, height: int,
              frame_height: int, age: float, strength: float) -> np.ndarray:
    columns = lines[:, 0].astype(np.int32) - left
    visible = (columns >= 0) & (columns < width) & (lines[:, 5] < strength)
    if not np.any(visible):
        return np.zeros((height, width), dtype=np.float32)
    chosen = lines[visible]
    columns = columns[visible]
    rows = (np.arange(top, top + height, dtype=np.float32) / frame_height)[:, None]
    # Long streaks slide vertically and close their gaps as the energy builds.
    cycle = np.mod(rows / chosen[:, 4] + chosen[:, 2] * age + chosen[:, 3], 1.0)
    dashes = (cycle < 0.45 + 0.5 * strength).astype(np.float32)
    result = np.zeros((height, width), dtype=np.float32)
    for offset in range(int(chosen[:, 1].max())):
        thick = chosen[:, 1] > offset
        target = np.clip(columns[thick] + offset, 0, width - 1)
        result[:, target] = np.maximum(result[:, target], dashes[:, thick])
    return result


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray | None,
                    age: float, lines: np.ndarray, appearing: bool = False) -> np.ndarray:
    """Vertical scratch streaks over the silhouette, then the swap; reversed when appearing.

    Like the anime instant transmission, the body itself gets smeared into
    vertical strips crossed by thin dark and light lines.
    """
    if mask is None:
        return (frame if appearing else background).copy()
    swap = GOKU_SWAP_SECONDS
    if age < swap:
        strength = (age / swap) ** 1.3
    else:
        strength = max(0.0, 1.0 - (age - swap) / (GOKU_EFFECT_SECONDS - swap))
    if appearing:
        visible = min(1.0, max(0.0, (age - swap) / 0.08))
    else:
        visible = 1.0 - min(1.0, max(0.0, (age - swap) / 0.06))

    person = mask * visible
    output = (frame.astype(np.float32) * person[:, :, None]
              + background.astype(np.float32) * (1.0 - person[:, :, None])).astype(np.uint8)
    if strength <= 0.01:
        return output

    points = cv2.findNonZero((mask > 0.3).astype(np.uint8))
    if points is None:
        return output
    x, y, w, h = cv2.boundingRect(points)
    frame_height, frame_width = mask.shape
    pad_x, pad_y = max(6, int(w * 0.06)), max(6, int(h * 0.12))
    left, top = max(0, x - pad_x), max(0, y - pad_y)
    right, bottom = min(frame_width, x + w + pad_x), min(frame_height, y + h + pad_y)
    width, height = right - left, bottom - top

    # Streaks spill above and below the body, barely sideways.
    region = cv2.dilate((mask[top:bottom, left:right] > 0.3).astype(np.float32),
                        cv2.getStructuringElement(cv2.MORPH_RECT, (3, max(3, pad_y))))
    region = cv2.GaussianBlur(region, (0, 0), sigmaX=1.5, sigmaY=12.0)

    # The body smeared vertically: always the live camera image, so the
    # strips keep the person's colours even after the swap.
    source = frame[top:bottom, left:right]
    smear = cv2.blur(source, (1, max(3, int(height * (0.10 + 0.30 * strength))))).astype(np.float32)
    streaks = _line_map(lines, left, top, width, height, frame_height, age, strength)
    ink = _line_map(lines[::4], left, top, width, height, frame_height,
                    age * 1.4 + 0.37, strength * 0.8)
    shine = _line_map(lines[1::6], left, top, width, height, frame_height,
                      age * 0.8 + 0.61, strength * 0.6)
    flash = float(np.exp(-((age - swap) / 0.045) ** 2)) * 0.35
    flash_map = cv2.GaussianBlur(mask[top:bottom, left:right], (0, 0), sigmaX=8.0) * flash

    area = output[top:bottom, left:right].astype(np.float32)
    # A light overall vertical blur, then the strongly smeared strips.
    body = region[:, :, None] * (0.45 * strength * visible)
    area = area * (1.0 - body) + smear * body
    alpha = np.clip(streaks * region * strength, 0, 1)[:, :, None]
    area = area * (1.0 - alpha) + (smear * 0.8) * alpha
    for layer, opacity, color in (
        (ink, 0.8, (28, 22, 24)),  # Dark ink scratches (BGR).
        (shine, 0.55, (235, 240, 245)),
        (flash_map, 1.0, (250, 250, 255)),
    ):
        alpha = np.clip(layer * (region if layer is not flash_map else 1.0) * opacity
                        * (strength if layer is not flash_map else 1.0), 0, 1)[:, :, None]
        area = area * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha
    output[top:bottom, left:right] = np.clip(area, 0, 255).astype(np.uint8)
    return output
