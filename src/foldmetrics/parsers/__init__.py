"""Prediction discovery: map files on disk to normalized :class:`Prediction` objects."""

from __future__ import annotations

import os
import warnings as _warnings
from collections.abc import Iterable
from pathlib import Path

from foldmetrics.models import Prediction

# Importing the modules registers their parsers.
from foldmetrics.parsers import (  # noqa: E402, F401
    alphafold2,
    alphafold3,
    boltz,
    chai,
    colabfold,
    protenix,
)
from foldmetrics.parsers.base import (
    ParserError,
    ToolParser,
    Unit,
    get_parser,
    parser_names,
    registered_parsers,
)

__all__ = [
    "ParserError",
    "ToolParser",
    "Unit",
    "discover",
    "get_parser",
    "load_predictions",
    "parser_names",
    "registered_parsers",
]

_PRUNE_DIRS = {"__pycache__", "msas", "templates"}


def _walk_dirs(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if not d.startswith(".") and d not in _PRUNE_DIRS
        )
        yield Path(dirpath)


def _units_in_dir(directory: Path, parsers: list[ToolParser]) -> list[Unit]:
    try:
        filenames = sorted(p.name for p in directory.iterdir() if p.is_file())
    except OSError:
        return []
    units: list[Unit] = []
    for parser in parsers:
        units.extend(parser.find_units(directory, filenames))
    return units


def _drop_duplicated_summary_models(units: list[Unit]) -> list[Unit]:
    """Remove models an engine writes twice under one job directory.

    AlphaFold3 copies its best-ranked sample to ``<job>_model.cif`` in the
    job directory while the sample itself also lives in
    ``seed-<S>_sample-<N>/`` below it (verified by identical coordinates).
    Counting both would inflate every per-target aggregate, so the copy is
    dropped whenever the samples it summarises were found as well.
    """
    sample_parents = {
        unit.dir.parent for unit in units if unit.tool == "alphafold3"
    }
    return [
        unit
        for unit in units
        if not (unit.tool == "alphafold3" and unit.dir in sample_parents)
    ]


def discover(
    paths: Iterable[str | Path] | str | Path, tool: str = "auto"
) -> list[Unit]:
    """Find every scoreable prediction under the given files/directories."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if tool == "auto":
        parsers = registered_parsers()
    else:
        parsers = [get_parser(tool)]

    units: list[Unit] = []
    seen: set[tuple[str, str, str]] = set()

    def add(unit: Unit) -> None:
        if unit.key not in seen:
            seen.add(unit.key)
            units.append(unit)

    for raw in paths:
        path = Path(raw)
        if path.is_file():
            for unit in _units_in_dir(path.parent, parsers):
                if any(f.name == path.name for f in unit.files.values()):
                    add(unit)
        elif path.is_dir():
            for directory in _walk_dirs(path):
                for unit in _units_in_dir(directory, parsers):
                    add(unit)
        else:
            raise FileNotFoundError(f"path does not exist: {path}")

    units = _drop_duplicated_summary_models(units)
    return sorted(units, key=lambda u: (str(u.dir), u.tool, u.name))


def load_predictions(
    paths: Iterable[str | Path] | str | Path,
    tool: str = "auto",
    on_error: str = "raise",
) -> list[Prediction]:
    """Discover and parse predictions.

    ``on_error="warn"`` skips units that fail to parse (with a warning)
    instead of raising — useful for large heterogeneous batches.
    """
    predictions: list[Prediction] = []
    for unit in discover(paths, tool=tool):
        try:
            predictions.append(get_parser(unit.tool).load(unit))
        except Exception as exc:
            if on_error == "warn":
                _warnings.warn(
                    f"skipping {unit.tool} unit {unit.name!r} in {unit.dir}: {exc}",
                    stacklevel=2,
                )
            else:
                raise
    return predictions
