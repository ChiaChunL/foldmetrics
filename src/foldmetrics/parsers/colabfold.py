"""Parser for ColabFold (localcolabfold) outputs.

Recognizes ``<job>_scores_rank_NNN_<tag>.json`` paired with the matching
``<job>_relaxed_/_unrelaxed_rank_NNN_<tag>.pdb``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from foldmetrics.models import Prediction
from foldmetrics.parsers.base import ToolParser, Unit, as_float, load_json, register
from foldmetrics.parsers.structure import autoscale_plddt, tokenize_structure

SCORES_RE = re.compile(r"^(?P<base>.+)_scores_rank_(?P<rank>\d+)_(?P<tag>.+)\.json$")

# ColabFold names every file "{jobname}_{kind}_rank_{NNN}_{model_type}...",
# so the job is the prefix. colabfold_batch writes a whole panel flat into
# one output directory, where the directory name says nothing about the
# job -- the file name is the only reliable source.
TARGET_SPLIT_RE = re.compile(r"_(?:(?:un)?relaxed|scores)_rank_\d+_")


def target_from_filename(filename: str) -> str | None:
    """Job name from a ColabFold output file name, or None if not one.

    ColabFold passes job names through its ``safe_filename()`` (anything
    outside ``[A-Za-z0-9_.-]`` becomes ``_``), so names such as ``A__B``
    survive unchanged.
    """
    match = TARGET_SPLIT_RE.search(filename)
    return filename[: match.start()] or None if match else None


@register
class ColabFoldParser(ToolParser):
    tool = "colabfold"

    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        names = set(filenames)
        units: list[Unit] = []
        for fn in filenames:
            m = SCORES_RE.match(fn)
            if not m:
                continue
            structure = None
            for state in ("relaxed", "unrelaxed"):
                candidate = f"{m['base']}_{state}_rank_{m['rank']}_{m['tag']}.pdb"
                if candidate in names:
                    structure = candidate
                    break
            if structure is None:
                continue
            units.append(
                Unit(
                    tool=self.tool,
                    name=f"{m['base']}_rank_{m['rank']}_{m['tag']}",
                    dir=directory,
                    files={"scores": directory / fn, "structure": directory / structure},
                )
            )
        return units

    def load(self, unit: Unit) -> Prediction:
        data = load_json(unit.files["scores"])
        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)

        warnings: list[str] = []
        plddt = data.get("plddt")
        if plddt is not None:
            if len(plddt) == len(tokens):
                for token, value in zip(tokens, plddt, strict=True):
                    token.plddt = float(value)
                    token.cb_plddt = float(value)
            else:
                warnings.append(
                    f"scores JSON has {len(plddt)} pLDDT values for {len(tokens)} tokens; "
                    "using structure B-factors instead"
                )

        pae = data.get("pae", data.get("predicted_aligned_error"))
        pae = np.asarray(pae, dtype=float) if pae is not None else None

        ptm = as_float(data.get("ptm"))
        iptm = as_float(data.get("iptm"))
        ranking = None
        extras: dict = {}
        if ptm is not None and iptm is not None:
            ranking = 0.8 * iptm + 0.2 * ptm
            extras["ranking_score_note"] = "computed as 0.8*ipTM + 0.2*pTM"
        if "max_pae" in data:
            extras["max_pae"] = as_float(data["max_pae"])
        # recent ColabFold embeds its own per chain-pair interface scores
        # (note: its ipSAE uses PAE cutoff 15 Å, not the ipsae.py default 10);
        # preserve them so users can cross-check against our computed columns
        for key in ("ipsae", "pdockq", "pdockq2", "actifptm"):
            if key in data:
                extras[f"colabfold_{key}"] = data[key]

        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            target=target_from_filename(unit.files["structure"].name),
            tokens=tokens,
            pae=pae,
            ptm=ptm,
            iptm=iptm,
            ranking_score=ranking,
            extras=extras,
            warnings=warnings,
        )
