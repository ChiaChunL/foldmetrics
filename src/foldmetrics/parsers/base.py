"""Parser registry and shared helpers for tool-specific parsers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from foldmetrics.models import Prediction


class ParserError(Exception):
    """A prediction unit exists but could not be parsed."""


@dataclass
class Unit:
    """One scoreable model: a structure file plus its confidence files."""

    tool: str
    name: str
    dir: Path
    files: dict[str, Path] = field(default_factory=dict)
    # parser-specific hints, e.g. which slice of a stacked npz belongs here
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.tool, str(self.dir), self.name)


class ToolParser(ABC):
    """A parser recognizes its own output files in a directory and loads them."""

    tool: str = ""

    @abstractmethod
    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        """Return the scoreable units among ``filenames`` in ``directory``."""

    @abstractmethod
    def load(self, unit: Unit) -> Prediction:
        """Parse one unit into a normalized :class:`Prediction`."""


_REGISTRY: list[ToolParser] = []


def register(cls: type[ToolParser]) -> type[ToolParser]:
    _REGISTRY.append(cls())
    return cls


def registered_parsers() -> list[ToolParser]:
    return list(_REGISTRY)


def parser_names() -> list[str]:
    return [p.tool for p in _REGISTRY]


def get_parser(tool: str) -> ToolParser:
    for p in _REGISTRY:
        if p.tool == tool:
            return p
    raise KeyError(f"no parser registered for tool {tool!r}")


# ---------------------------------------------------------------- shared utils
def as_float(value: Any) -> float | None:
    """Best-effort scalar conversion (handles numpy scalars and 0-d arrays)."""
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ParserError(f"expected a JSON object in {path}")
    return data


def map_pair_matrix(chains: list[str], matrix: Any) -> dict[tuple[str, str], float]:
    """Map an (n_chains, n_chains) matrix onto chain-name pairs."""
    import numpy as np

    out: dict[tuple[str, str], float] = {}
    if matrix is None:
        return out
    arr = np.asarray(matrix, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim != 2 or arr.shape != (len(chains), len(chains)):
        return out
    for i, a in enumerate(chains):
        for j, b in enumerate(chains):
            if a != b:
                out[(a, b)] = float(arr[i, j])
    return out


def map_pair_nested(chains: list[str], nested: Any) -> dict[tuple[str, str], float]:
    """Map ``{key: {key: value}}`` onto chain-name pairs.

    Keys may be chain names directly, or stringified indices into ``chains``
    (Boltz writes ``{"0": {"1": ...}}``).
    """
    out: dict[tuple[str, str], float] = {}
    if not isinstance(nested, dict):
        return out
    chainset = set(chains)

    def to_chain(key: str) -> str | None:
        key = str(key)
        if key in chainset:
            return key
        if key.isdigit() and int(key) < len(chains):
            return chains[int(key)]
        return None

    for k1, row in nested.items():
        a = to_chain(k1)
        if a is None or not isinstance(row, dict):
            continue
        for k2, value in row.items():
            b = to_chain(k2)
            if b is None or a == b:
                continue
            v = as_float(value)
            if v is not None:
                out[(a, b)] = v
    return out
