"""Executable RSRCC tools backed by immutable artifacts.

The default vision backends are deterministic baselines.  They make every
capability executable and testable, while the backend protocols provide the
boundary for locally trained Open-CD/MMDetection/MMSegmentation adapters.
Default backends are never reported as RSRCC-validated.
"""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .artifacts import ArtifactStore
from .backends import (
    ChangeBackend,
    EntityBackend,
    MultiscaleDifferenceBackend,
    SpectralHeuristicEntityBackend,
    connected_components,
)
from .schemas import (
    EVENT_VOCAB,
    LINEAR_LABELS,
    PAVED_LABELS,
    SMALL_OBJECT_LABELS,
    QuerySpec,
    ToolResponse,
)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0.0, 0.0)
    return (float(xs.mean()), float(ys.mean()))


def _entity_from_prediction(
    prediction: Mapping[str, object], index: int, shape: tuple[int, int]
) -> dict[str, Any]:
    label = str(prediction["label"]).strip().lower().replace(" ", "_")
    mask = np.asarray(prediction["mask"], dtype=bool)
    if mask.shape != shape:
        raise ValueError(f"entity backend returned {mask.shape}, expected {shape}")
    area = int(mask.sum())
    if area == 0:
        raise ValueError("entity backend returned an empty instance mask")
    return {
        "entity_id": f"{label}_{index:04d}",
        "label": label,
        "score": float(prediction.get("score", 1.0)),
        "bbox": _bbox(mask),
        "centroid": _centroid(mask),
        "area_px": area,
        "mask": mask,
    }


def _as_pair(store: ArtifactStore, artifact_id: str) -> tuple[np.ndarray, np.ndarray, str]:
    artifact = store.get(artifact_id)
    if artifact.kind not in {"RGBPair", "AlignedRGBPair"}:
        raise TypeError(f"expected RGBPair or AlignedRGBPair, got {artifact.kind}")
    return artifact.data["before"], artifact.data["after"], artifact.kind


def _shift_without_wrap(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    result = np.zeros_like(image)
    height, width = image.shape[:2]
    src_x0 = max(0, -dx)
    src_x1 = min(width, width - dx)
    src_y0 = max(0, -dy)
    src_y1 = min(height, height - dy)
    dst_x0 = max(0, dx)
    dst_x1 = min(width, width + dx)
    dst_y0 = max(0, dy)
    dst_y1 = min(height, height + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        result[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return result


def _scale_about_center(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("scale must be positive")
    height, width = image.shape[:2]
    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    resized = np.asarray(
        Image.fromarray(image).resize((scaled_width, scaled_height), Image.Resampling.BILINEAR)
    )
    result = np.zeros_like(image)
    if scaled_width >= width:
        src_x0 = (scaled_width - width) // 2
        src_x1 = src_x0 + width
        dst_x0, dst_x1 = 0, width
    else:
        src_x0, src_x1 = 0, scaled_width
        dst_x0 = (width - scaled_width) // 2
        dst_x1 = dst_x0 + scaled_width
    if scaled_height >= height:
        src_y0 = (scaled_height - height) // 2
        src_y1 = src_y0 + height
        dst_y0, dst_y1 = 0, height
    else:
        src_y0, src_y1 = 0, scaled_height
        dst_y0 = (height - scaled_height) // 2
        dst_y1 = dst_y0 + scaled_height
    result[dst_y0:dst_y1, dst_x0:dst_x1] = resized[src_y0:src_y1, src_x0:src_x1]
    return result


def _phase_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[int, int, float]:
    first = reference.astype(np.float32).mean(axis=2)
    second = moving.astype(np.float32).mean(axis=2)
    first -= first.mean()
    second -= second.mean()
    spectrum = np.fft.fft2(first) * np.conj(np.fft.fft2(second))
    magnitude = np.abs(spectrum)
    spectrum /= np.maximum(magnitude, 1e-8)
    correlation = np.abs(np.fft.ifft2(spectrum))
    peak_y, peak_x = np.unravel_index(np.argmax(correlation), correlation.shape)
    height, width = correlation.shape
    dx = int(peak_x if peak_x <= width // 2 else peak_x - width)
    dy = int(peak_y if peak_y <= height // 2 else peak_y - height)
    confidence = float(correlation[peak_y, peak_x] / max(float(correlation.mean()), 1e-8))
    return dx, dy, confidence


def _bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return float(intersection / union) if union else 0.0


def _bbox_gap(first: Sequence[int], second: Sequence[int]) -> float:
    dx = max(first[0] - second[2], second[0] - first[2], 0)
    dy = max(first[1] - second[3], second[1] - first[3], 0)
    return float(math.hypot(dx, dy))


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / union if union else 0.0


def _region_accepts(region: str | None, centroid: Sequence[float], shape: Sequence[int]) -> bool:
    if not region:
        return True
    x, y = centroid
    height, width = shape
    horizontal = "left" if x < width / 3 else "right" if x > 2 * width / 3 else "center"
    vertical = "top" if y < height / 3 else "bottom" if y > 2 * height / 3 else "center"
    normalized = region.lower().replace("-", "").replace("_", "").replace(" ", "")
    aliases = {
        "northwest": ("top", "left"),
        "upperleft": ("top", "left"),
        "northeast": ("top", "right"),
        "upperright": ("top", "right"),
        "southwest": ("bottom", "left"),
        "lowerleft": ("bottom", "left"),
        "southeast": ("bottom", "right"),
        "lowerright": ("bottom", "right"),
        "north": ("top", None),
        "upper": ("top", None),
        "south": ("bottom", None),
        "lower": ("bottom", None),
        "east": (None, "right"),
        "west": (None, "left"),
        "center": ("center", "center"),
        "central": ("center", "center"),
    }
    expected = aliases.get(normalized)
    if expected is None:
        raise ValueError(f"unsupported spatial region: {region}")
    return (expected[0] is None or expected[0] == vertical) and (
        expected[1] is None or expected[1] == horizontal
    )


def _event_for_added(label: str) -> str:
    if label in {"building", "house", "structure", "road", "sidewalk", "driveway"}:
        return "constructed"
    if label in {"vegetation", "tree", "shrub"}:
        return "planted"
    if label in SMALL_OBJECT_LABELS:
        return "object_added"
    return "appeared"


def _event_for_removed(label: str) -> str:
    if label in {"building", "house", "structure"}:
        return "demolished"
    if label in {"vegetation", "tree", "shrub"}:
        return "vegetation_loss"
    if label in {"water", "pond", "lake", "swimming_pool"}:
        return "water_decrease"
    if label in SMALL_OBJECT_LABELS:
        return "object_removed"
    return "removed"


def _backend_info(backend: ChangeBackend | EntityBackend) -> dict[str, Any]:
    """Produce auditable model metadata without requiring a specific framework."""

    validated = bool(backend.validated_for_rsrcc)
    checkpoint = getattr(backend, "checkpoint_sha256", None)
    if validated and not checkpoint:
        raise ValueError(
            f"backend {backend.name!r} is marked RSRCC-validated without a checkpoint hash"
        )
    if checkpoint is not None and (
        len(checkpoint) != 64 or any(character not in "0123456789abcdefABCDEF" for character in checkpoint)
    ):
        raise ValueError(f"backend {backend.name!r} has an invalid SHA-256 checkpoint digest")
    return {
        "name": backend.name,
        "validated_for_rsrcc": validated,
        "checkpoint_sha256": checkpoint,
        "preprocessing_version": getattr(backend, "preprocessing_version", None),
    }


class RsrccTools:
    """All capability-level tools declared by ``rsrcc-tool-registry.md``."""

    def __init__(
        self,
        store: ArtifactStore | None = None,
        *,
        change_backend: ChangeBackend | None = None,
        entity_backend: EntityBackend | None = None,
    ) -> None:
        # ArtifactStore implements ``__len__``; an injected empty store is
        # therefore false-y and must not be replaced via ``store or ...``.
        self.store = store if store is not None else ArtifactStore()
        self.change_backend = change_backend or MultiscaleDifferenceBackend()
        self.entity_backend = entity_backend or SpectralHeuristicEntityBackend()

    def _resolve_entity_data(
        self,
        *,
        entity_instances_id: str | None,
        scene_layer_id: str | None,
        rgb_pair_id: str | None,
        time: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve specialized-parser inputs from instances, a scene layer, or RGB."""

        supplied = sum(
            value is not None for value in (entity_instances_id, scene_layer_id, rgb_pair_id)
        )
        if supplied != 1:
            raise ValueError(
                "provide exactly one of entity_instances_id, scene_layer_id, or rgb_pair_id"
            )
        if entity_instances_id is not None:
            data = self.store.get(
                entity_instances_id, expected_kind="EntityInstances"
            ).data
            return data, {"source_instances": entity_instances_id}
        if scene_layer_id is not None:
            scene = self.store.get(scene_layer_id, expected_kind="SceneLayer").data
            inverse_codes = {code: label for label, code in scene["label_codes"].items()}
            predictions: list[dict[str, Any]] = []
            for code, label in inverse_codes.items():
                for component in connected_components(scene["labels"] == code, 1):
                    predictions.append({"label": label, "mask": component, "score": 1.0})
            instances = [
                _entity_from_prediction(prediction, index, tuple(scene["shape"]))
                for index, prediction in enumerate(predictions)
            ]
            return {
                "instances": instances,
                "shape": scene["shape"],
                "time": self.store.get(scene_layer_id).metadata.get("time"),
            }, {"source_scene_layer": scene_layer_id}
        if time not in {"before", "after"}:
            raise ValueError("time must be 'before' or 'after' when rgb_pair_id is supplied")
        before, after, _ = _as_pair(self.store, str(rgb_pair_id))
        image = before if time == "before" else after
        predictions = self.entity_backend.predict(image)
        instances = [
            _entity_from_prediction(prediction, index, image.shape[:2])
            for index, prediction in enumerate(predictions)
        ]
        return {
            "instances": instances,
            "shape": list(image.shape[:2]),
            "time": time,
        }, {
            "source_pair": rgb_pair_id,
            "time": time,
            "backend": _backend_info(self.entity_backend),
        }

    def load_rgb_pair(self, img_1: str | Path, img_2: str | Path) -> ToolResponse:
        paths = [Path(img_1).expanduser().resolve(), Path(img_2).expanduser().resolve()]
        arrays: list[np.ndarray] = []
        source: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image.load()
                source.append(
                    {
                        "path": str(path),
                        "format": image.format,
                        "source_mode": image.mode,
                        "size": list(image.size),
                    }
                )
                arrays.append(np.asarray(image.convert("RGB"), dtype=np.uint8))
        artifact_id = self.store.put(
            "RGBPair",
            {"before": arrays[0], "after": arrays[1]},
            metadata={"source": source, "time_order": ["before", "after"]},
        )
        return ToolResponse(
            {"rgb_pair_id": artifact_id},
            {"rgb_pair_id": artifact_id, "shapes": [list(a.shape) for a in arrays]},
        )

    def inspect_rgb_pair(self, rgb_pair_id: str) -> ToolResponse:
        before, after, kind = _as_pair(self.store, rgb_pair_id)
        pair_artifact = self.store.get(rgb_pair_id)
        same_shape = before.shape == after.shape
        report = {
            "pair_kind": kind,
            "before": {
                "shape": list(before.shape),
                "dtype": str(before.dtype),
                "min": int(before.min()),
                "max": int(before.max()),
            },
            "after": {
                "shape": list(after.shape),
                "dtype": str(after.dtype),
                "min": int(after.min()),
                "max": int(after.max()),
            },
            "same_shape": same_shape,
            "source": pair_artifact.metadata.get("source"),
            "is_rgb_uint8": bool(
                before.ndim == 3
                and after.ndim == 3
                and before.shape[2] == after.shape[2] == 3
                and before.dtype == after.dtype == np.uint8
            ),
        }
        report["valid"] = bool(report["same_shape"] and report["is_rgb_uint8"])
        artifact_id = self.store.put(
            "PairMetadataReport", report, metadata={"source_artifact": rgb_pair_id}
        )
        return ToolResponse(
            {"metadata_report_id": artifact_id},
            {"metadata_report_id": artifact_id, "valid": report["valid"]},
        )

    def check_pixel_alignment(
        self,
        rgb_pair_id: str,
        *,
        tolerance_px: float = 1.0,
        scale_tolerance: float = 0.01,
        minimum_confidence: float = 3.0,
        scale_candidates: Sequence[float] = (0.98, 0.99, 1.0, 1.01, 1.02),
        max_correctable_translation_px: float = 16.0,
        max_correctable_scale_delta: float = 0.05,
        minimum_relative_correction_gain: float = 0.05,
        maximum_corrected_residual: float = 0.20,
    ) -> ToolResponse:
        before, after, _ = _as_pair(self.store, rgb_pair_id)
        if before.shape != after.shape:
            raise ValueError("alignment requires equal image shapes")
        if not scale_candidates or any(scale <= 0 for scale in scale_candidates):
            raise ValueError("scale_candidates must contain positive values")
        candidates: list[dict[str, Any]] = []
        height, width = before.shape[:2]
        margin_y = max(1, height // 12)
        margin_x = max(1, width // 12)
        reference_core = before[margin_y : height - margin_y, margin_x : width - margin_x]
        identity_core = after[margin_y : height - margin_y, margin_x : width - margin_x]
        identity_residual = float(
            np.mean(np.abs(reference_core.astype(np.float32) - identity_core.astype(np.float32)))
            / 255.0
        )
        for scale in scale_candidates:
            scaled = _scale_about_center(after, float(scale))
            candidate_dx, candidate_dy, candidate_confidence = _phase_shift(before, scaled)
            corrected = _shift_without_wrap(scaled, candidate_dx, candidate_dy)
            corrected_core = corrected[margin_y : height - margin_y, margin_x : width - margin_x]
            residual = float(
                np.mean(
                    np.abs(reference_core.astype(np.float32) - corrected_core.astype(np.float32))
                )
                / 255.0
            )
            candidates.append(
                {
                    "scale": float(scale),
                    "dx": candidate_dx,
                    "dy": candidate_dy,
                    "confidence": candidate_confidence,
                    "residual": residual,
                }
            )
        best = min(candidates, key=lambda item: (item["residual"], abs(item["scale"] - 1.0)))
        dx, dy = int(best["dx"]), int(best["dy"])
        confidence = float(best["confidence"])
        scale = float(best["scale"])
        magnitude = float(math.hypot(dx, dy))
        correction_gain = identity_residual - float(best["residual"])
        relative_correction_gain = correction_gain / max(identity_residual, 1e-8)
        reliable_correction = bool(
            confidence >= minimum_confidence
            and relative_correction_gain >= minimum_relative_correction_gain
            and best["residual"] <= maximum_corrected_residual
        )
        report = {
            "shift_to_apply_px": {"dx": dx, "dy": dy},
            "scale_to_apply": scale,
            "translation_magnitude_px": magnitude,
            "confidence": confidence,
            "identity_normalized_residual": identity_residual,
            "normalized_residual": best["residual"],
            "correction_gain": correction_gain,
            "relative_correction_gain": relative_correction_gain,
            "reliable_correction": reliable_correction,
            "correctable": bool(
                magnitude <= max_correctable_translation_px
                and abs(scale - 1.0) <= max_correctable_scale_delta
                and reliable_correction
            ),
            "passed": bool(
                magnitude <= tolerance_px
                and abs(scale - 1.0) <= scale_tolerance
                and confidence >= minimum_confidence
            ),
            "method": "centered_scale_search_plus_phase_correlation",
            "limits": {
                "transform": "isotropic_scale_then_translation",
                "tolerance_px": tolerance_px,
                "scale_tolerance": scale_tolerance,
                "scale_candidates": list(scale_candidates),
                "max_correctable_translation_px": max_correctable_translation_px,
                "max_correctable_scale_delta": max_correctable_scale_delta,
                "minimum_relative_correction_gain": minimum_relative_correction_gain,
                "maximum_corrected_residual": maximum_corrected_residual,
            },
        }
        artifact_id = self.store.put(
            "AlignmentReport", report, metadata={"source_artifact": rgb_pair_id}
        )
        return ToolResponse(
            {"alignment_report_id": artifact_id},
            {
                "alignment_report_id": artifact_id,
                "passed": report["passed"],
                "correctable": report["correctable"],
                "translation_magnitude_px": magnitude,
                "scale_to_apply": scale,
                "confidence": confidence,
                "relative_correction_gain": relative_correction_gain,
                "reliable_correction": reliable_correction,
            },
        )

    def correct_pixel_alignment(
        self,
        rgb_pair_id: str,
        alignment_report_id: str,
        *,
        max_correction_px: int = 16,
        max_scale_delta: float = 0.05,
    ) -> ToolResponse:
        before, after, _ = _as_pair(self.store, rgb_pair_id)
        report = self.store.get(alignment_report_id, expected_kind="AlignmentReport").data
        dx = int(report["shift_to_apply_px"]["dx"])
        dy = int(report["shift_to_apply_px"]["dy"])
        scale = float(report.get("scale_to_apply", 1.0))
        if abs(dx) > max_correction_px or abs(dy) > max_correction_px:
            raise ValueError("estimated translation exceeds the configured correction limit")
        if abs(scale - 1.0) > max_scale_delta:
            raise ValueError("estimated scale exceeds the configured correction limit")
        corrected = _shift_without_wrap(_scale_about_center(after, scale), dx, dy)
        artifact_id = self.store.put(
            "AlignedRGBPair",
            {"before": before, "after": corrected},
            metadata={
                "source_pair": rgb_pair_id,
                "alignment_report": alignment_report_id,
                "transform": {
                    "type": "isotropic_scale_then_translation",
                    "scale": scale,
                    "dx": dx,
                    "dy": dy,
                },
            },
        )
        return ToolResponse(
            {"aligned_rgb_pair_id": artifact_id},
            {
                "aligned_rgb_pair_id": artifact_id,
                "applied_scale": scale,
                "applied_shift_px": {"dx": dx, "dy": dy},
            },
        )

    def detect_multiscale_change(
        self,
        rgb_pair_id: str,
        *,
        threshold: float = 0.12,
        min_region_area: int = 9,
    ) -> ToolResponse:
        before, after, _ = _as_pair(self.store, rgb_pair_id)
        if before.shape != after.shape:
            raise ValueError("change detection requires equal image shapes")
        confidence = np.asarray(self.change_backend.predict(before, after), dtype=np.float32)
        if confidence.shape != before.shape[:2]:
            raise ValueError("change backend returned an invalid confidence-map shape")
        mask = confidence >= threshold
        components = connected_components(mask, min_region_area)
        filtered = np.zeros_like(mask)
        regions: list[dict[str, Any]] = []
        for index, component in enumerate(components):
            filtered |= component
            regions.append(
                {
                    "region_id": f"change_{index:04d}",
                    "bbox": _bbox(component),
                    "centroid": _centroid(component),
                    "area_px": int(component.sum()),
                    "mean_confidence": float(confidence[component].mean()),
                }
            )
        backend = _backend_info(self.change_backend)
        map_id = self.store.put(
            "ChangeMap",
            {"confidence": confidence, "mask": filtered, "threshold": threshold},
            metadata={"source_pair": rgb_pair_id, "backend": backend},
        )
        regions_id = self.store.put(
            "ChangeRegions",
            {"regions": regions, "shape": list(mask.shape)},
            metadata={"change_map": map_id},
        )
        return ToolResponse(
            {"change_map_id": map_id, "change_regions_id": regions_id},
            {
                "change_map_id": map_id,
                "change_regions_id": regions_id,
                "changed_pixel_ratio": float(filtered.mean()),
                "region_count": len(regions),
                "backend": backend,
            },
        )

    def parse_scene_entities(
        self, rgb_pair_id: str, *, time: str, ontology_version: str = "rsrcc-v1"
    ) -> ToolResponse:
        before, after, _ = _as_pair(self.store, rgb_pair_id)
        if time not in {"before", "after"}:
            raise ValueError("time must be 'before' or 'after'")
        image = before if time == "before" else after
        predictions = self.entity_backend.predict(image)
        instances = [
            _entity_from_prediction(prediction, index, image.shape[:2])
            for index, prediction in enumerate(predictions)
        ]
        label_names = sorted({instance["label"] for instance in instances})
        label_codes = {label: index + 1 for index, label in enumerate(label_names)}
        labels = np.zeros(image.shape[:2], dtype=np.uint16)
        for instance in sorted(instances, key=lambda item: item["score"]):
            labels[instance["mask"]] = label_codes[instance["label"]]
        backend = _backend_info(self.entity_backend)
        scene_id = self.store.put(
            "SceneLayer",
            {"labels": labels, "label_codes": label_codes, "shape": list(labels.shape)},
            metadata={
                "source_pair": rgb_pair_id,
                "time": time,
                "ontology_version": ontology_version,
                "backend": backend,
            },
        )
        instances_id = self.store.put(
            "EntityInstances",
            {"instances": instances, "shape": list(labels.shape), "time": time},
            metadata={"scene_layer": scene_id, "backend": backend},
        )
        counts = Counter(instance["label"] for instance in instances)
        return ToolResponse(
            {"scene_layer_id": scene_id, "entity_instances_id": instances_id},
            {
                "scene_layer_id": scene_id,
                "entity_instances_id": instances_id,
                "counts": dict(counts),
                "backend": backend,
            },
        )

    def extract_linear_infrastructure(
        self,
        entity_instances_id: str | None = None,
        *,
        scene_layer_id: str | None = None,
        rgb_pair_id: str | None = None,
        time: str | None = None,
        labels: Iterable[str] | None = None,
    ) -> ToolResponse:
        data, source_metadata = self._resolve_entity_data(
            entity_instances_id=entity_instances_id,
            scene_layer_id=scene_layer_id,
            rgb_pair_id=rgb_pair_id,
            time=time,
        )
        selected_labels = set(labels or LINEAR_LABELS)
        features: list[dict[str, Any]] = []
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        combined = np.zeros(tuple(data["shape"]), dtype=bool)
        for instance in data["instances"]:
            if instance["label"] not in selected_labels:
                continue
            mask = instance["mask"]
            combined |= mask
            x0, y0, x1, y1 = instance["bbox"]
            width, height = x1 - x0, y1 - y0
            if width >= height:
                points = [(x0, (y0 + y1 - 1) / 2), (x1 - 1, (y0 + y1 - 1) / 2)]
            else:
                points = [((x0 + x1 - 1) / 2, y0), ((x0 + x1 - 1) / 2, y1 - 1)]
            first_id = f"node_{len(nodes):04d}"
            second_id = f"node_{len(nodes) + 1:04d}"
            nodes.extend(
                [
                    {"node_id": first_id, "point": points[0]},
                    {"node_id": second_id, "point": points[1]},
                ]
            )
            length = float(math.dist(points[0], points[1]))
            edge_id = f"edge_{len(edges):04d}"
            edges.append(
                {
                    "edge_id": edge_id,
                    "source": first_id,
                    "target": second_id,
                    "length_px": length,
                    "entity_id": instance["entity_id"],
                }
            )
            features.append(
                {
                    "feature_id": edge_id,
                    "label": instance["label"],
                    "entity_id": instance["entity_id"],
                    "centerline": points,
                    "length_px": length,
                    "bbox": instance["bbox"],
                }
            )
        layer_id = self.store.put(
            "LinearFeatureLayer",
            {"features": features, "mask": combined, "shape": data["shape"]},
            metadata=source_metadata,
        )
        graph_id = self.store.put(
            "NetworkGraph",
            {"nodes": nodes, "edges": edges},
            metadata={"linear_layer": layer_id},
        )
        return ToolResponse(
            {"linear_feature_layer_id": layer_id, "network_graph_id": graph_id},
            {
                "linear_feature_layer_id": layer_id,
                "network_graph_id": graph_id,
                "feature_count": len(features),
                "total_length_px": sum(edge["length_px"] for edge in edges),
            },
        )

    def parse_paved_and_parking_areas(
        self,
        entity_instances_id: str | None = None,
        *,
        scene_layer_id: str | None = None,
        rgb_pair_id: str | None = None,
        time: str | None = None,
        labels: Iterable[str] | None = None,
    ) -> ToolResponse:
        data, source_metadata = self._resolve_entity_data(
            entity_instances_id=entity_instances_id,
            scene_layer_id=scene_layer_id,
            rgb_pair_id=rgb_pair_id,
            time=time,
        )
        selected_labels = set(labels or PAVED_LABELS)
        areas: list[dict[str, Any]] = []
        mask = np.zeros(tuple(data["shape"]), dtype=bool)
        for instance in data["instances"]:
            if instance["label"] in selected_labels:
                mask |= instance["mask"]
                areas.append(
                    {
                        "area_id": instance["entity_id"],
                        "label": instance["label"],
                        "bbox": instance["bbox"],
                        "area_px": instance["area_px"],
                    }
                )
        layer_id = self.store.put(
            "PavedAreaLayer",
            {"areas": areas, "mask": mask, "shape": data["shape"]},
            metadata=source_metadata,
        )
        layout = {
            "parking_area_count": sum(a["label"] in {"parking", "parking_lot"} for a in areas),
            "estimated_stall_count": None,
            "stall_estimate_status": "not_available_without_marking_detector",
        }
        layout_id = self.store.put(
            "ParkingLayout", layout, metadata={"paved_area_layer": layer_id}
        )
        return ToolResponse(
            {"paved_area_layer_id": layer_id, "parking_layout_id": layout_id},
            {
                "paved_area_layer_id": layer_id,
                "parking_layout_id": layout_id,
                "area_count": len(areas),
                "area_px": int(mask.sum()),
            },
        )

    def detect_small_aux_objects(
        self,
        entity_instances_id: str | None = None,
        *,
        scene_layer_id: str | None = None,
        rgb_pair_id: str | None = None,
        time: str | None = None,
        roi_id: str | None = None,
        labels: Iterable[str] | None = None,
    ) -> ToolResponse:
        data, source_metadata = self._resolve_entity_data(
            entity_instances_id=entity_instances_id,
            scene_layer_id=scene_layer_id,
            rgb_pair_id=rgb_pair_id,
            time=time,
        )
        selected_labels = set(labels or SMALL_OBJECT_LABELS)
        roi_mask = np.ones(tuple(data["shape"]), dtype=bool)
        if roi_id is not None:
            roi_mask = self.store.get(roi_id, expected_kind="QueryROI").data["mask"]
        objects: list[dict[str, Any]] = []
        mask = np.zeros(tuple(data["shape"]), dtype=bool)
        for instance in data["instances"]:
            overlap = np.logical_and(instance["mask"], roi_mask)
            if instance["label"] in selected_labels and overlap.any():
                mask |= overlap
                objects.append(
                    {
                        "object_id": instance["entity_id"],
                        "label": instance["label"],
                        "bbox": instance["bbox"],
                        "score": instance["score"],
                    }
                )
        counts = Counter(item["label"] for item in objects)
        layer_id = self.store.put(
            "SmallObjectLayer",
            {"objects": objects, "mask": mask, "shape": data["shape"]},
            metadata={**source_metadata, "roi": roi_id},
        )
        counts_id = self.store.put(
            "ObjectCounts", {"counts": dict(counts), "total": len(objects)},
            metadata={"small_object_layer": layer_id},
        )
        return ToolResponse(
            {"small_object_layer_id": layer_id, "object_counts_id": counts_id},
            {
                "small_object_layer_id": layer_id,
                "object_counts_id": counts_id,
                "counts": dict(counts),
            },
        )

    def match_entities_across_time(
        self,
        before_instances_id: str,
        after_instances_id: str,
        *,
        min_iou: float = 0.05,
        max_centroid_distance_px: float = 32.0,
    ) -> ToolResponse:
        before = self.store.get(before_instances_id, expected_kind="EntityInstances").data
        after = self.store.get(after_instances_id, expected_kind="EntityInstances").data
        if before["shape"] != after["shape"]:
            raise ValueError("entity layers must share a spatial shape")
        candidates: list[tuple[float, int, int, float, float]] = []
        for first_index, first in enumerate(before["instances"]):
            for second_index, second in enumerate(after["instances"]):
                if first["label"] != second["label"]:
                    continue
                distance = math.dist(first["centroid"], second["centroid"])
                bbox_iou = _bbox_iou(first["bbox"], second["bbox"])
                if bbox_iou >= min_iou or distance <= max_centroid_distance_px:
                    # Full-resolution mask IoU is only useful after a cheap
                    # spatial filter.  Computing it for every same-class pair
                    # made real scenes quadratic in both entities and pixels.
                    iou = _mask_iou(first["mask"], second["mask"]) if bbox_iou > 0 else 0.0
                    distance_score = max(0.0, 1.0 - distance / max(max_centroid_distance_px, 1e-6))
                    score = 0.75 * iou + 0.25 * distance_score
                    candidates.append((score, first_index, second_index, iou, distance))
        used_before: set[int] = set()
        used_after: set[int] = set()
        matches: list[dict[str, Any]] = []
        for score, first_index, second_index, iou, distance in sorted(candidates, reverse=True):
            if first_index in used_before or second_index in used_after:
                continue
            used_before.add(first_index)
            used_after.add(second_index)
            matches.append(
                {
                    "before_id": before["instances"][first_index]["entity_id"],
                    "after_id": after["instances"][second_index]["entity_id"],
                    "label": before["instances"][first_index]["label"],
                    "score": score,
                    "iou": iou,
                    "centroid_distance_px": distance,
                }
            )
        removed = [
            {"entity_id": item["entity_id"], "label": item["label"]}
            for index, item in enumerate(before["instances"])
            if index not in used_before
        ]
        added = [
            {"entity_id": item["entity_id"], "label": item["label"]}
            for index, item in enumerate(after["instances"])
            if index not in used_after
        ]
        # Record overlap-based split/merge candidates in addition to 1:1 matches.
        transitions: list[dict[str, Any]] = []
        for first in before["instances"]:
            overlaps = [
                second["entity_id"]
                for second in after["instances"]
                if second["label"] == first["label"]
                and _bbox_iou(first["bbox"], second["bbox"]) > 0
                and _mask_iou(first["mask"], second["mask"]) > 0
            ]
            if len(overlaps) > 1:
                transitions.append({"type": "split", "source": first["entity_id"], "targets": overlaps})
        for second in after["instances"]:
            overlaps = [
                first["entity_id"]
                for first in before["instances"]
                if first["label"] == second["label"]
                and _bbox_iou(first["bbox"], second["bbox"]) > 0
                and _mask_iou(first["mask"], second["mask"]) > 0
            ]
            if len(overlaps) > 1:
                transitions.append({"type": "merge", "sources": overlaps, "target": second["entity_id"]})
        result = {
            "matches": matches,
            "added": added,
            "removed": removed,
            "transitions": transitions,
            "before_instances_id": before_instances_id,
            "after_instances_id": after_instances_id,
        }
        artifact_id = self.store.put("TemporalEntityMatches", result)
        return ToolResponse(
            {"temporal_entity_matches_id": artifact_id},
            {
                "temporal_entity_matches_id": artifact_id,
                "matched_count": len(matches),
                "added_count": len(added),
                "removed_count": len(removed),
                "transition_count": len(transitions),
            },
        )

    def compare_entity_attributes(
        self, temporal_entity_matches_id: str, rgb_pair_id: str
    ) -> ToolResponse:
        matches = self.store.get(
            temporal_entity_matches_id, expected_kind="TemporalEntityMatches"
        ).data
        before_data = self.store.get(
            matches["before_instances_id"], expected_kind="EntityInstances"
        ).data
        after_data = self.store.get(
            matches["after_instances_id"], expected_kind="EntityInstances"
        ).data
        before_image, after_image, _ = _as_pair(self.store, rgb_pair_id)
        before_by_id = {item["entity_id"]: item for item in before_data["instances"]}
        after_by_id = {item["entity_id"]: item for item in after_data["instances"]}
        deltas: list[dict[str, Any]] = []
        for match in matches["matches"]:
            first = before_by_id[match["before_id"]]
            second = after_by_id[match["after_id"]]
            first_color = before_image[first["mask"]].mean(axis=0)
            second_color = after_image[second["mask"]].mean(axis=0)
            area_delta = second["area_px"] - first["area_px"]
            deltas.append(
                {
                    "before_id": first["entity_id"],
                    "after_id": second["entity_id"],
                    "label": first["label"],
                    "before_area_px": first["area_px"],
                    "after_area_px": second["area_px"],
                    "area_delta_px": area_delta,
                    "area_ratio": float(second["area_px"] / first["area_px"]),
                    "centroid_shift_px": float(math.dist(first["centroid"], second["centroid"])),
                    "mean_rgb_before": first_color.tolist(),
                    "mean_rgb_after": second_color.tolist(),
                    "color_distance": float(np.linalg.norm(second_color - first_color)),
                }
            )
        artifact_id = self.store.put(
            "AttributeDeltaTable",
            {"deltas": deltas},
            metadata={"matches": temporal_entity_matches_id, "source_pair": rgb_pair_id},
        )
        return ToolResponse(
            {"attribute_delta_table_id": artifact_id},
            {"attribute_delta_table_id": artifact_id, "row_count": len(deltas)},
        )

    def build_scene_graph(
        self,
        entity_instances_ids: Sequence[str],
        *,
        linear_feature_layer_ids: Sequence[str] = (),
        small_object_layer_ids: Sequence[str] = (),
        near_distance_px: float = 24.0,
        adjacency_distance_px: float = 2.0,
    ) -> ToolResponse:
        if not entity_instances_ids:
            raise ValueError("at least one entity layer is required")
        nodes: list[dict[str, Any]] = []
        shape: list[int] | None = None
        for artifact_id in entity_instances_ids:
            data = self.store.get(artifact_id, expected_kind="EntityInstances").data
            if shape is None:
                shape = data["shape"]
            elif shape != data["shape"]:
                raise ValueError("all scene layers must share a spatial shape")
            for instance in data["instances"]:
                nodes.append(
                    {
                        "entity_id": instance["entity_id"],
                        "label": instance["label"],
                        "bbox": instance["bbox"],
                        "centroid": instance["centroid"],
                        "area_px": instance["area_px"],
                        "source_instances_id": artifact_id,
                    }
                )
        existing_ids = {node["entity_id"] for node in nodes}
        for artifact_id in linear_feature_layer_ids:
            layer = self.store.get(artifact_id, expected_kind="LinearFeatureLayer").data
            if shape is None:
                shape = layer["shape"]
            elif shape != layer["shape"]:
                raise ValueError("all scene layers must share a spatial shape")
            for feature in layer["features"]:
                entity_id = feature["entity_id"]
                if entity_id in existing_ids:
                    continue
                bbox = feature["bbox"]
                nodes.append(
                    {
                        "entity_id": entity_id,
                        "label": feature["label"],
                        "bbox": bbox,
                        "centroid": ((bbox[0] + bbox[2] - 1) / 2, (bbox[1] + bbox[3] - 1) / 2),
                        "area_px": int(layer["mask"].sum()),
                        "source_instances_id": artifact_id,
                    }
                )
                existing_ids.add(entity_id)
        for artifact_id in small_object_layer_ids:
            layer = self.store.get(artifact_id, expected_kind="SmallObjectLayer").data
            if shape is None:
                shape = layer["shape"]
            elif shape != layer["shape"]:
                raise ValueError("all scene layers must share a spatial shape")
            for item in layer["objects"]:
                entity_id = item["object_id"]
                if entity_id in existing_ids:
                    continue
                bbox = item["bbox"]
                nodes.append(
                    {
                        "entity_id": entity_id,
                        "label": item["label"],
                        "bbox": bbox,
                        "centroid": ((bbox[0] + bbox[2] - 1) / 2, (bbox[1] + bbox[3] - 1) / 2),
                        "area_px": max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])),
                        "source_instances_id": artifact_id,
                    }
                )
                existing_ids.add(entity_id)
        # Disambiguate IDs if before/after layers are combined.
        id_counts = Counter(node["entity_id"] for node in nodes)
        for node in nodes:
            if id_counts[node["entity_id"]] > 1:
                suffix = self.store.get(node["source_instances_id"]).data.get("time", "layer")
                node["entity_id"] = f"{suffix}:{node['entity_id']}"
        edges: list[dict[str, str]] = []
        for first in nodes:
            for second in nodes:
                if first["entity_id"] == second["entity_id"]:
                    continue
                fx, fy = first["centroid"]
                sx, sy = second["centroid"]
                tolerance = max(1.0, math.sqrt(min(first["area_px"], second["area_px"])) * 0.15)
                relations: list[str] = []
                if fy < sy - tolerance:
                    relations.append("north_of")
                if fy > sy + tolerance:
                    relations.append("south_of")
                if fx < sx - tolerance:
                    relations.append("west_of")
                if fx > sx + tolerance:
                    relations.append("east_of")
                gap = _bbox_gap(first["bbox"], second["bbox"])
                if gap <= near_distance_px:
                    relations.append("near")
                if gap <= adjacency_distance_px:
                    relations.append("adjacent_to")
                fb, sb = first["bbox"], second["bbox"]
                if fb[0] >= sb[0] and fb[1] >= sb[1] and fb[2] <= sb[2] and fb[3] <= sb[3]:
                    relations.append("inside")
                for relation in relations:
                    edges.append(
                        {"source": first["entity_id"], "relation": relation, "target": second["entity_id"]}
                    )
        artifact_id = self.store.put(
            "SceneGraph", {"nodes": nodes, "edges": edges, "shape": shape},
            metadata={
                "source_layers": list(entity_instances_ids),
                "linear_layers": list(linear_feature_layer_ids),
                "small_object_layers": list(small_object_layer_ids),
            },
        )
        return ToolResponse(
            {"scene_graph_id": artifact_id},
            {"scene_graph_id": artifact_id, "node_count": len(nodes), "edge_count": len(edges)},
        )

    def resolve_spatial_query(
        self, query: QuerySpec | Mapping[str, Any], scene_graph_id: str
    ) -> ToolResponse:
        spec = QuerySpec.from_value(query)
        graph = self.store.get(scene_graph_id, expected_kind="SceneGraph").data
        nodes = list(graph["nodes"])
        unconstrained = not (spec.labels or spec.region or spec.relations or spec.ordinal)
        labels = {label.lower().replace(" ", "_") for label in spec.labels}
        if labels:
            nodes = [node for node in nodes if node["label"] in labels]
        nodes = [
            node
            for node in nodes
            if _region_accepts(spec.region, node["centroid"], graph["shape"])
        ]
        edge_set = {
            (edge["source"], edge["relation"], edge["target"]) for edge in graph["edges"]
        }
        node_by_id = {node["entity_id"]: node for node in graph["nodes"]}
        for relation in spec.relations:
            relation_name = relation["relation"]
            target_ids: set[str] = set()
            if relation.get("target_id"):
                target_ids.add(relation["target_id"])
            if relation.get("target_label"):
                target_label = relation["target_label"].lower().replace(" ", "_")
                target_ids.update(
                    node_id for node_id, node in node_by_id.items() if node["label"] == target_label
                )
            nodes = [
                node
                for node in nodes
                if any((node["entity_id"], relation_name, target) in edge_set for target in target_ids)
            ]
        if spec.ordinal and nodes:
            ordinal = spec.ordinal.lower()
            key_and_reverse = {
                "leftmost": (lambda item: item["centroid"][0], False),
                "rightmost": (lambda item: item["centroid"][0], True),
                "topmost": (lambda item: item["centroid"][1], False),
                "bottommost": (lambda item: item["centroid"][1], True),
                "largest": (lambda item: item["area_px"], True),
                "smallest": (lambda item: item["area_px"], False),
            }
            if ordinal not in key_and_reverse:
                raise ValueError(f"unsupported ordinal: {spec.ordinal}")
            key, reverse = key_and_reverse[ordinal]
            nodes = [sorted(nodes, key=key, reverse=reverse)[0]]
        if unconstrained:
            mask = np.ones(tuple(graph["shape"]), dtype=bool)
            roi_bbox = (0, 0, int(graph["shape"][1]), int(graph["shape"][0]))
        else:
            mask = np.zeros(tuple(graph["shape"]), dtype=bool)
            for node in nodes:
                x0, y0, x1, y1 = node["bbox"]
                mask[y0:y1, x0:x1] = True
            roi_bbox = _bbox(mask) if nodes else None
        result = {
            "target_ids": [node["entity_id"] for node in nodes],
            "bbox": roi_bbox,
            "mask": mask,
            "query": {
                "labels": spec.labels,
                "region": spec.region,
                "relations": spec.relations,
                "ordinal": spec.ordinal,
            },
            "shape": graph["shape"],
            "full_scene": unconstrained,
        }
        artifact_id = self.store.put(
            "QueryROI", result, metadata={"scene_graph": scene_graph_id}
        )
        return ToolResponse(
            {"query_roi_id": artifact_id},
            {"query_roi_id": artifact_id, "target_ids": result["target_ids"], "bbox": roi_bbox},
        )

    def infer_change_event(
        self,
        query_roi_id: str,
        temporal_entity_matches_id: str,
        *,
        attribute_delta_table_id: str | None = None,
        change_map_id: str | None = None,
        global_change_report_id: str | None = None,
        area_change_threshold: float = 0.18,
        color_change_threshold: float = 35.0,
    ) -> ToolResponse:
        roi = self.store.get(query_roi_id, expected_kind="QueryROI").data
        matching = self.store.get(
            temporal_entity_matches_id, expected_kind="TemporalEntityMatches"
        ).data
        target_ids = set(roi["target_ids"])
        # Scene-graph IDs may be time-prefixed when both layers were used.
        target_ids |= {target.split(":", 1)[-1] for target in target_ids}
        events: list[dict[str, Any]] = []
        if global_change_report_id is not None:
            global_report = self.store.get(
                global_change_report_id, expected_kind="GlobalChangeReport"
            ).data
            events.append(
                {
                    "event": global_report["state"],
                    "entity_id": None,
                    "label": "scene",
                    "basis": "global_change_report",
                    "confidence": 0.92,
                }
            )
        for item in matching["added"]:
            if not target_ids or item["entity_id"] in target_ids:
                events.append(
                    {
                        "event": _event_for_added(item["label"]),
                        "entity_id": item["entity_id"],
                        "label": item["label"],
                        "basis": "unmatched_after_entity",
                        "confidence": 0.85,
                    }
                )
        for item in matching["removed"]:
            if not target_ids or item["entity_id"] in target_ids:
                events.append(
                    {
                        "event": _event_for_removed(item["label"]),
                        "entity_id": item["entity_id"],
                        "label": item["label"],
                        "basis": "unmatched_before_entity",
                        "confidence": 0.85,
                    }
                )
        if attribute_delta_table_id is not None:
            deltas = self.store.get(
                attribute_delta_table_id, expected_kind="AttributeDeltaTable"
            ).data["deltas"]
            for delta in deltas:
                if target_ids and not ({delta["before_id"], delta["after_id"]} & target_ids):
                    continue
                ratio_delta = delta["area_ratio"] - 1.0
                label = delta["label"]
                event: str | None = None
                if ratio_delta >= area_change_threshold:
                    if label in {"vegetation", "tree", "shrub"}:
                        event = "grown"
                    elif label in {"water", "pond", "lake"}:
                        event = "water_increase"
                    else:
                        event = "expanded"
                elif ratio_delta <= -area_change_threshold:
                    if label in {"vegetation", "tree", "shrub"}:
                        event = "vegetation_loss"
                    elif label in {"water", "pond", "lake"}:
                        event = "water_decrease"
                    else:
                        event = "narrowed"
                elif delta["color_distance"] >= color_change_threshold:
                    event = "roof_color_changed" if label == "building" else "repainted"
                if event:
                    events.append(
                        {
                            "event": event,
                            "entity_id": delta["after_id"],
                            "label": label,
                            "basis": "attribute_delta",
                            "confidence": 0.80,
                        }
                    )
        if not events and change_map_id is not None:
            change = self.store.get(change_map_id, expected_kind="ChangeMap").data["mask"]
            roi_mask = roi["mask"] if roi["mask"].any() else np.ones_like(change)
            ratio = float(np.logical_and(change, roi_mask).sum() / max(int(roi_mask.sum()), 1))
            events.append(
                {
                    "event": "local_change" if ratio >= 0.01 else "unchanged",
                    "entity_id": None,
                    "label": None,
                    "basis": "roi_change_ratio",
                    "confidence": min(0.95, 0.55 + ratio),
                }
            )
        if not events:
            events.append(
                {
                    "event": "unchanged",
                    "entity_id": None,
                    "label": None,
                    "basis": "no_supported_entity_change",
                    "confidence": 0.6,
                }
            )
        for event in events:
            if event["event"] not in EVENT_VOCAB:
                raise RuntimeError(f"event outside the controlled vocabulary: {event['event']}")
        primary = max(events, key=lambda item: item["confidence"])
        result = {
            "primary_event": primary["event"],
            "events": events,
            "query_roi_id": query_roi_id,
            "source_artifacts": {
                "matches": temporal_entity_matches_id,
                "attributes": attribute_delta_table_id,
                "change_map": change_map_id,
                "global_change_report": global_change_report_id,
            },
        }
        artifact_id = self.store.put("EventEvidence", result)
        return ToolResponse(
            {"event_evidence_id": artifact_id},
            {
                "event_evidence_id": artifact_id,
                "primary_event": result["primary_event"],
                "event_count": len(events),
                "events": [
                    {
                        "event": item["event"],
                        "label": item["label"],
                        "confidence": item["confidence"],
                    }
                    for item in events
                ],
            },
        )

    def summarize_roi_change(
        self,
        query_roi_id: str,
        change_map_id: str,
        *,
        before_instances_id: str | None = None,
        after_instances_id: str | None = None,
        network_graph_id: str | None = None,
    ) -> ToolResponse:
        roi = self.store.get(query_roi_id, expected_kind="QueryROI").data
        change = self.store.get(change_map_id, expected_kind="ChangeMap").data
        roi_mask = roi["mask"] if roi["mask"].any() else np.ones_like(change["mask"])
        changed = np.logical_and(change["mask"], roi_mask)
        summary: dict[str, Any] = {
            "roi_pixel_count": int(roi_mask.sum()),
            "changed_pixel_count": int(changed.sum()),
            "changed_pixel_ratio": float(changed.sum() / max(int(roi_mask.sum()), 1)),
            "changed_region_count": len(connected_components(changed, 1)),
        }
        if before_instances_id is not None:
            before = self.store.get(before_instances_id, expected_kind="EntityInstances").data
            summary["before_object_count"] = sum(
                bool(np.logical_and(item["mask"], roi_mask).any()) for item in before["instances"]
            )
        if after_instances_id is not None:
            after = self.store.get(after_instances_id, expected_kind="EntityInstances").data
            summary["after_object_count"] = sum(
                bool(np.logical_and(item["mask"], roi_mask).any()) for item in after["instances"]
            )
        if "before_object_count" in summary and "after_object_count" in summary:
            summary["object_count_delta"] = (
                summary["after_object_count"] - summary["before_object_count"]
            )
        if network_graph_id is not None:
            graph = self.store.get(network_graph_id, expected_kind="NetworkGraph").data
            summary["linear_length_px"] = float(
                sum(edge["length_px"] for edge in graph["edges"])
            )
        artifact_id = self.store.put(
            "ChangeStatistics", summary,
            metadata={"query_roi": query_roi_id, "change_map": change_map_id},
        )
        return ToolResponse(
            {"change_statistics_id": artifact_id},
            {"change_statistics_id": artifact_id, **summary},
        )

    def score_answer_options(
        self, options: Mapping[str, str], event_evidence_id: str
    ) -> ToolResponse:
        if set(options) != {"A", "B", "C", "D"}:
            raise ValueError("RSRCC multiple-choice options must contain exactly A-D")
        evidence = self.store.get(event_evidence_id, expected_kind="EventEvidence").data
        supported_events = {item["event"] for item in evidence["events"]}
        constraint_rules = (
            (
                "addition",
                {"constructed", "appeared", "installed", "developed", "object_added", "planted"},
                ("new", "construct", "built", "developed", "appear", "install", "planted"),
            ),
            (
                "removal",
                {
                    "removed",
                    "demolished",
                    "destroyed",
                    "cleared",
                    "vegetation_loss",
                    "water_decrease",
                    "object_removed",
                },
                ("remove", "disappear", "demolish", "torn down", "destroy", "clear", "cut down", "filled in", "dried"),
            ),
            (
                "growth_or_expansion",
                {
                    "expanded",
                    "enlarged",
                    "extended",
                    "widened",
                    "grown",
                    "vegetation_increase",
                    "water_increase",
                    "count_increased",
                },
                ("expand", "enlarg", "addition", "extend", "widen", "grown", "larger", "mature", "increased", "filled with water"),
            ),
            ("contraction", {"narrowed", "count_decreased", "water_decrease"}, ("narrow", "shrink", "decreased", "dried")),
            ("paving", {"paved", "surface_covered", "developed"}, ("paved", "concrete", "asphalt", "resurfaced")),
            ("appearance", {"roof_color_changed", "repainted", "construction_completed"}, ("roof color", "painted", "repaint", "different color", "roof was replaced")),
            ("replacement", {"replaced", "converted", "developed"}, ("replaced", "converted", "transformed")),
            (
                "unchanged",
                {"unchanged"},
                ("no change", "unchanged", "remained the same", "visual parity", "scene consistency", "stability", "no significant temporal deviation"),
            ),
            ("local_change", {"local_change"}, ("local change", "localized", "changed")),
            ("global_change", {"global_change"}, ("global", "substantial overall", "extensive", "entire scene")),
        )
        rows: list[dict[str, Any]] = []
        for label in "ABCD":
            normalized = options[label].lower()
            support: list[str] = []
            conflicts: list[str] = []
            for constraint, compatible_events, phrases in constraint_rules:
                if any(phrase in normalized for phrase in phrases):
                    matched = sorted(compatible_events.intersection(supported_events))
                    if matched:
                        support.extend(matched)
                    else:
                        conflicts.append(constraint)
            score = 2.0 * len(support) - 0.75 * len(conflicts)
            if evidence["primary_event"] in support:
                score += 1.0
            rows.append(
                {"option": label, "score": score, "support": support, "conflicts": conflicts}
            )
        best = max(rows, key=lambda row: (row["score"], -ord(row["option"])))
        result = {"scores": rows, "best_option": best["option"], "source": event_evidence_id}
        artifact_id = self.store.put("OptionScores", result)
        return ToolResponse(
            {"option_scores_id": artifact_id},
            {
                "option_scores_id": artifact_id,
                "best_option": best["option"],
                "scores": {row["option"]: row["score"] for row in rows},
                "details": {
                    row["option"]: {
                        "support": row["support"],
                        "conflicts": row["conflicts"],
                    }
                    for row in rows
                },
            },
        )

    def assess_global_scene_change(
        self,
        change_map_id: str,
        *,
        unchanged_max_ratio: float = 0.01,
        global_min_ratio: float = 0.30,
    ) -> ToolResponse:
        if not 0 <= unchanged_max_ratio < global_min_ratio <= 1:
            raise ValueError("global-change thresholds must satisfy 0 <= unchanged < global <= 1")
        change = self.store.get(change_map_id, expected_kind="ChangeMap").data
        ratio = float(change["mask"].mean())
        state = (
            "unchanged"
            if ratio <= unchanged_max_ratio
            else "global_change"
            if ratio >= global_min_ratio
            else "local_change"
        )
        result = {
            "state": state,
            "changed_pixel_ratio": ratio,
            "thresholds": {
                "unchanged_max_ratio": unchanged_max_ratio,
                "global_min_ratio": global_min_ratio,
            },
        }
        artifact_id = self.store.put(
            "GlobalChangeReport", result, metadata={"change_map": change_map_id}
        )
        return ToolResponse(
            {"global_change_report_id": artifact_id},
            {"global_change_report_id": artifact_id, **result},
        )

    def render_evidence(
        self,
        rgb_pair_id: str,
        *,
        change_map_id: str | None = None,
        query_roi_id: str | None = None,
        entity_instances_id: str | None = None,
        change_statistics_id: str | None = None,
    ) -> ToolResponse:
        before, after, _ = _as_pair(self.store, rgb_pair_id)
        panels = [Image.fromarray(before), Image.fromarray(after)]
        if change_map_id is not None:
            change = self.store.get(change_map_id, expected_kind="ChangeMap").data["mask"]
            overlay = np.array(after, copy=True)
            overlay[change] = (0.45 * overlay[change] + 0.55 * np.array([255, 0, 0])).astype(
                np.uint8
            )
            panels.append(Image.fromarray(overlay))
        if query_roi_id is not None:
            roi = self.store.get(query_roi_id, expected_kind="QueryROI").data
            if roi["bbox"] is not None:
                for panel in panels:
                    draw = ImageDraw.Draw(panel)
                    x0, y0, x1, y1 = roi["bbox"]
                    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 255, 0), width=2)
        if entity_instances_id is not None:
            instances = self.store.get(
                entity_instances_id, expected_kind="EntityInstances"
            ).data["instances"]
            target_panel = panels[1]
            draw = ImageDraw.Draw(target_panel)
            for instance in instances:
                x0, y0, x1, y1 = instance["bbox"]
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 255, 255), width=1)
                draw.text((x0, max(0, y0 - 9)), instance["label"], fill=(0, 255, 255))
        statistics: dict[str, Any] | None = None
        if change_statistics_id is not None:
            statistics = self.store.get(
                change_statistics_id, expected_kind="ChangeStatistics"
            ).data
        width = sum(panel.width for panel in panels)
        height = max(panel.height for panel in panels)
        canvas = Image.new("RGB", (width, height), (0, 0, 0))
        offset = 0
        for panel in panels:
            canvas.paste(panel, (offset, 0))
            offset += panel.width
        stream = BytesIO()
        canvas.save(stream, format="PNG")
        payload = stream.getvalue()
        artifact_id = self.store.put(
            "EvidenceRender",
            payload,
            metadata={
                "mime_type": "image/png",
                "width": width,
                "height": height,
                "source_pair": rgb_pair_id,
                "change_map": change_map_id,
                "query_roi": query_roi_id,
                "entity_instances": entity_instances_id,
                "change_statistics": change_statistics_id,
                "statistics_summary": statistics,
            },
        )
        return ToolResponse(
            {"evidence_render_id": artifact_id},
            {
                "evidence_render_id": artifact_id,
                "mime_type": "image/png",
                "width": width,
                "height": height,
                "byte_count": len(payload),
            },
        )

    def verify_evidence_consistency(
        self,
        candidate_answer: str,
        answer_format: str,
        event_evidence_id: str,
        *,
        option_scores_id: str | None = None,
        query_roi_id: str | None = None,
        hypothesis_events: Sequence[str] | None = None,
    ) -> ToolResponse:
        evidence = self.store.get(event_evidence_id, expected_kind="EventEvidence").data
        checks: dict[str, bool] = {}
        expected: str | None
        if answer_format == "binary":
            if hypothesis_events:
                invalid_events = set(hypothesis_events) - EVENT_VOCAB
                if invalid_events:
                    raise ValueError(
                        f"binary hypothesis contains uncontrolled events: {sorted(invalid_events)}"
                    )
                observed_events = {item["event"] for item in evidence["events"]}
                expected = "yes" if observed_events.intersection(hypothesis_events) else "no"
                checks["hypothesis_grounded"] = True
            else:
                expected = "no" if evidence["primary_event"] == "unchanged" else "yes"
            checks["answer_supported"] = candidate_answer == expected
        elif answer_format == "multiple_choice":
            if option_scores_id is None:
                raise ValueError("multiple-choice verification requires option_scores_id")
            scores = self.store.get(option_scores_id, expected_kind="OptionScores").data
            expected = scores["best_option"]
            checks["answer_supported"] = candidate_answer == expected
        else:
            raise ValueError("answer_format must be binary or multiple_choice")
        checks["controlled_event"] = all(
            item["event"] in EVENT_VOCAB for item in evidence["events"]
        )
        checks["evidence_present"] = bool(evidence["events"])
        if query_roi_id is not None:
            roi = self.store.get(query_roi_id, expected_kind="QueryROI").data
            checks["spatial_target_resolved"] = bool(roi["target_ids"])
        result = {
            "consistent": all(checks.values()),
            "checks": checks,
            "candidate_answer": candidate_answer,
            "evidence_supported_answer": expected,
            "source_artifacts": {
                "event_evidence": event_evidence_id,
                "option_scores": option_scores_id,
                "query_roi": query_roi_id,
            },
        }
        artifact_id = self.store.put("VerificationReport", result)
        return ToolResponse(
            {"verification_report_id": artifact_id},
            {
                "verification_report_id": artifact_id,
                "consistent": result["consistent"],
                "checks": checks,
            },
        )

    def validate_output_schema(
        self, candidate_answer: str, output_schema: str | Mapping[str, Any]
    ) -> ToolResponse:
        schema_name: str
        if isinstance(output_schema, str):
            schema_name = output_schema
        else:
            schema_name = str(output_schema.get("answer_format") or output_schema.get("name") or "")
            if not schema_name and "enum" in output_schema:
                values = set(output_schema["enum"])
                if values == {"yes", "no"}:
                    schema_name = "binary"
                elif values == {"A", "B", "C", "D"}:
                    schema_name = "multiple_choice"
        if schema_name in {"binary", "yes/no"}:
            allowed = ["yes", "no"]
        elif schema_name in {"multiple_choice", "A-D"}:
            allowed = ["A", "B", "C", "D"]
        else:
            raise ValueError(f"unsupported RSRCC output schema: {schema_name!r}")
        valid = candidate_answer in allowed
        result = {
            "valid": valid,
            "candidate_answer": candidate_answer,
            "allowed_values": allowed,
            "reason": None if valid else "candidate must exactly match one allowed token",
        }
        artifact_id = self.store.put("SchemaValidationReport", result)
        return ToolResponse(
            {"schema_validation_report_id": artifact_id},
            {"schema_validation_report_id": artifact_id, **result},
        )
