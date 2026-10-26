"""Silhouette that breaks into small fragments carried away by the wind."""

from __future__ import annotations

import cv2
import numpy as np


PARTICLES_SECONDS = 2.1
# Fragments leave the body over this window, sweeping against the wind.
DETACH_SECONDS = 1.0
FLIGHT_SECONDS = PARTICLES_SECONDS - DETACH_SECONDS


class ParticleBurst:
    """Captures the silhouette once, then animates it without new segmentation."""

    def __init__(self, frame: np.ndarray, mask: np.ndarray, seed: int = 2207):
        height, width = mask.shape
        self.cell = max(3, round(min(height, width) / 110))
        self.snapshot = frame.copy()
        self.mask = mask.astype(np.float32)
        self.left = self.top = 0
        self.alive_shape = (0, 0)
        self.count = 0
        points = cv2.findNonZero((mask > 0.4).astype(np.uint8))
        if points is None:
            return
        x, y, w, h = cv2.boundingRect(points)
        cell = self.cell
        self.left, self.top = x, y
        rows, cols = -(-h // cell), -(-w // cell)
        self.alive_shape = (rows, cols)
        padded = np.zeros((rows * cell, cols * cell), dtype=np.float32)
        padded[:h, :w] = mask[y:y + h, x:x + w]
        coverage = padded.reshape(rows, cell, cols, cell).mean(axis=(1, 3))
        image = np.zeros((rows * cell, cols * cell, 3), dtype=np.float32)
        image[:h, :w] = frame[y:y + h, x:x + w]
        colors = image.reshape(rows, cell, cols, cell, 3).mean(axis=(1, 3))

        grid_y, grid_x = np.nonzero(coverage > 0.35)
        self.grid = (grid_y, grid_x)
        self.count = len(grid_y)
        rng = np.random.default_rng(seed)
        position = grid_x / max(1, cols - 1)
        # Wind blows to the right: the right edge dissolves first.
        self.delay = (1.0 - position) * DETACH_SECONDS * 0.8 + rng.uniform(0, 0.2, self.count)
        self.origin = np.stack((x + grid_x * cell, y + grid_y * cell), axis=1).astype(np.float32)
        scale = min(height, width)
        self.velocity = np.stack((rng.uniform(0.35, 0.9, self.count) * scale,
                                  rng.uniform(-0.35, 0.05, self.count) * scale), axis=1)
        self.gust = rng.uniform(0.4, 1.1, self.count) * scale
        self.wobble = rng.uniform(0, 2 * np.pi, self.count)
        # Slightly brighter, a few sparkling cyan or gold fragments.
        accent = rng.random(self.count)
        tint = np.where((accent > 0.9)[:, None], (255.0, 236.0, 120.0),
                        np.where((accent > 0.82)[:, None], (90.0, 215.0, 255.0), colors[grid_y, grid_x]))
        self.colors = np.clip(colors[grid_y, grid_x] * 0.35 + tint * 0.65 + 18.0, 0, 255)
        offsets = np.mgrid[0:cell, 0:cell].reshape(2, -1)
        self.offset_y, self.offset_x = offsets[0], offsets[1]

    def render(self, background: np.ndarray, age: float) -> np.ndarray:
        output = background.copy()
        if self.count == 0 or age >= PARTICLES_SECONDS:
            return output
        cell = self.cell
        rows, cols = self.alive_shape
        height, width = background.shape[:2]

        # Body parts not yet detached stay in place at full resolution.
        alive = np.zeros((rows, cols), dtype=np.float32)
        attached = self.delay > age
        alive[self.grid[0][attached], self.grid[1][attached]] = 1.0
        bottom = min(height, self.top + rows * cell)
        right = min(width, self.left + cols * cell)
        alive_map = np.repeat(np.repeat(alive, cell, axis=0), cell, axis=1)
        alive_map = alive_map[:bottom - self.top, :right - self.left]
        area = np.s_[self.top:bottom, self.left:right]
        alpha = (self.mask[area] * alive_map)[:, :, None]
        output[area] = (self.snapshot[area].astype(np.float32) * alpha
                        + output[area].astype(np.float32) * (1.0 - alpha)).astype(np.uint8)

        flying = ~attached
        if not np.any(flying):
            return output
        t = age - self.delay[flying]
        progress = np.clip(t / FLIGHT_SECONDS, 0.0, 1.0)
        position = (self.origin[flying] + self.velocity[flying] * t[:, None])
        position[:, 0] += 0.5 * self.gust[flying] * t * t
        position[:, 1] += np.sin(self.wobble[flying] + t * 7.0) * cell * 1.5
        opacity = (1.0 - progress) ** 1.4
        keep = opacity > 0.02
        if not np.any(keep):
            return output
        position, opacity, colors = position[keep], opacity[keep], self.colors[flying][keep]

        ys = position[:, 1].astype(np.int32)[:, None] + self.offset_y
        xs = position[:, 0].astype(np.int32)[:, None] + self.offset_x
        inside = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
        weight = np.broadcast_to(opacity[:, None], ys.shape)[inside][:, None]
        color = np.broadcast_to(colors[:, None, :], ys.shape + (3,))[inside]
        ys, xs = ys[inside], xs[inside]
        current = output[ys, xs].astype(np.float32)
        output[ys, xs] = np.clip(color * weight + current * (1.0 - weight), 0, 255).astype(np.uint8)
        return output
