"""Replaceable vision backends for the RSRCC capability tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np


def connected_components(mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Return 4-connected component masks without a SciPy/OpenCV dependency."""

    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary)
    components: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(binary & ~visited), strict=True):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and binary[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) >= min_area:
            component = np.zeros_like(binary)
            ys, xs = zip(*pixels, strict=True)
            component[np.asarray(ys), np.asarray(xs)] = True
            components.append(component)
    return components


class ChangeBackend(Protocol):
    name: str
    validated_for_rsrcc: bool
    checkpoint_sha256: str | None

    def predict(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Return a float32 HxW change confidence map in [0, 1]."""


class EntityBackend(Protocol):
    name: str
    validated_for_rsrcc: bool
    checkpoint_sha256: str | None

    def predict(self, image: np.ndarray) -> Sequence[Mapping[str, object]]:
        """Return label/mask/score mappings for detected instances."""


@dataclass
class MultiscaleDifferenceBackend:
    """Deterministic baseline; production runs should replace it with Open-CD."""

    name: str = "numpy_multiscale_absdiff_v1"
    validated_for_rsrcc: bool = False
    checkpoint_sha256: str | None = None

    def predict(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        first = before.astype(np.float32) / 255.0
        second = after.astype(np.float32) / 255.0
        fine = np.mean(np.abs(first - second), axis=2)
        # A 3x3 neighborhood maximum keeps small changes while adding context.
        padded = np.pad(fine, 1, mode="edge")
        neighborhoods = [
            padded[dy : dy + fine.shape[0], dx : dx + fine.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
        coarse = np.maximum.reduce(neighborhoods)
        return np.clip(0.65 * fine + 0.35 * coarse, 0.0, 1.0).astype(np.float32)


@dataclass
class CallableChangeBackend:
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray]
    name: str
    validated_for_rsrcc: bool = False
    checkpoint_sha256: str | None = None
    preprocessing_version: str = "caller-defined"

    def predict(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        result = np.asarray(self.predictor(before, after), dtype=np.float32)
        if result.shape != before.shape[:2]:
            raise ValueError("change backend returned an invalid spatial shape")
        if not np.all(np.isfinite(result)):
            raise ValueError("change backend returned non-finite confidence values")
        return np.clip(result, 0.0, 1.0)


@dataclass
class PaletteEntityBackend:
    """Exact/color-tolerant backend used for tests and annotated raster layers."""

    palette: Mapping[str, tuple[int, int, int]]
    tolerance: float = 8.0
    min_area: int = 4
    name: str = "palette_components_v1"
    validated_for_rsrcc: bool = False
    checkpoint_sha256: str | None = None

    def predict(self, image: np.ndarray) -> Sequence[Mapping[str, object]]:
        pixels = image.astype(np.float32)
        predictions: list[Mapping[str, object]] = []
        for label, color in self.palette.items():
            distance = np.linalg.norm(pixels - np.asarray(color, dtype=np.float32), axis=2)
            for component in connected_components(distance <= self.tolerance, self.min_area):
                predictions.append({"label": label, "mask": component, "score": 1.0})
        return predictions


@dataclass
class SpectralHeuristicEntityBackend:
    """Runnable RGB baseline, deliberately marked as not RSRCC-validated."""

    min_area: int = 16
    name: str = "rgb_spectral_heuristic_v1"
    validated_for_rsrcc: bool = False
    checkpoint_sha256: str | None = None

    def predict(self, image: np.ndarray) -> Sequence[Mapping[str, object]]:
        rgb = image.astype(np.int16)
        red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maximum = rgb.max(axis=2)
        minimum = rgb.min(axis=2)
        saturation = maximum - minimum
        intensity = rgb.mean(axis=2)
        masks = {
            "water": (blue > 75) & (blue > red * 1.20) & (blue > green * 1.10),
            "vegetation": (green > 60) & (green > red * 1.15) & (green > blue * 1.05),
            "building": (saturation < 30) & (intensity >= 175),
            "parking_lot": (saturation < 24) & (intensity >= 85) & (intensity < 175),
            "terrain": (red > green * 1.05) & (green > blue * 1.08) & (red > 70),
        }
        predictions: list[Mapping[str, object]] = []
        occupied = np.zeros(image.shape[:2], dtype=bool)
        for label in ("water", "vegetation", "building", "parking_lot", "terrain"):
            candidate = masks[label] & ~occupied
            for component in connected_components(candidate, self.min_area):
                occupied |= component
                predictions.append({"label": label, "mask": component, "score": 0.55})
        return predictions


@dataclass
class CallableEntityBackend:
    predictor: Callable[[np.ndarray], Sequence[Mapping[str, object]]]
    name: str
    validated_for_rsrcc: bool = False
    checkpoint_sha256: str | None = None
    preprocessing_version: str = "caller-defined"

    def predict(self, image: np.ndarray) -> Sequence[Mapping[str, object]]:
        return self.predictor(image)
