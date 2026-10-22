"""Content-addressed artifacts used by the RSRCC tool executor.

Large payloads stay inside the store.  Tool observations expose only artifact
IDs and small, JSON-compatible summaries, which keeps image evidence separate
from the language-model context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
import copy
import json
from pathlib import Path
from threading import RLock
from typing import Any, Mapping

import numpy as np


def _hash_value(hasher: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray")
        hasher.update(str(array.dtype).encode())
        hasher.update(repr(array.shape).encode())
        hasher.update(array.tobytes())
    elif isinstance(value, bytes):
        hasher.update(b"bytes")
        hasher.update(value)
    elif is_dataclass(value):
        _hash_value(hasher, asdict(value))
    elif isinstance(value, Mapping):
        hasher.update(b"mapping")
        for key in sorted(value, key=str):
            _hash_value(hasher, str(key))
            _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(b"sequence")
        for item in value:
            _hash_value(hasher, item)
    elif isinstance(value, Path):
        _hash_value(hasher, str(value))
    else:
        hasher.update(json.dumps(value, sort_keys=True, default=str).encode())


def _immutable_copy(value: Any) -> Any:
    """Return an isolated copy and make arrays read-only."""

    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, dict):
        return {key: _immutable_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_immutable_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_immutable_copy(item) for item in value)
    return copy.deepcopy(value)


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    kind: str
    data: Any
    metadata: Mapping[str, Any]
    digest: str


class ArtifactStore:
    """Thread-safe, immutable, content-addressed in-process artifact store."""

    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._lock = RLock()

    def put(
        self,
        kind: str,
        data: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if not kind or not kind.replace("_", "").isalnum():
            raise ValueError("artifact kind must be a non-empty identifier")
        safe_data = _immutable_copy(data)
        safe_metadata = _immutable_copy(dict(metadata or {}))
        hasher = sha256()
        _hash_value(hasher, kind)
        _hash_value(hasher, safe_data)
        _hash_value(hasher, safe_metadata)
        digest = hasher.hexdigest()
        artifact_id = f"{kind.lower()}_{digest[:16]}"
        artifact = Artifact(artifact_id, kind, safe_data, safe_metadata, digest)
        with self._lock:
            self._artifacts.setdefault(artifact_id, artifact)
        return artifact_id

    def get(self, artifact_id: str, *, expected_kind: str | None = None) -> Artifact:
        with self._lock:
            if artifact_id not in self._artifacts:
                raise KeyError(f"unknown artifact: {artifact_id}")
            artifact = self._artifacts[artifact_id]
        if expected_kind is not None and artifact.kind != expected_kind:
            raise TypeError(
                f"artifact {artifact_id} has kind {artifact.kind}, expected {expected_kind}"
            )
        return Artifact(
            artifact.artifact_id,
            artifact.kind,
            _immutable_copy(artifact.data),
            _immutable_copy(artifact.metadata),
            artifact.digest,
        )

    def describe(self, artifact_id: str) -> dict[str, Any]:
        artifact = self.get(artifact_id)
        return {
            "artifact_id": artifact.artifact_id,
            "kind": artifact.kind,
            "digest": artifact.digest,
            "metadata": dict(artifact.metadata),
        }

    def __contains__(self, artifact_id: object) -> bool:
        with self._lock:
            return artifact_id in self._artifacts

    def __len__(self) -> int:
        with self._lock:
            return len(self._artifacts)

