"""Parser for Chai-1 prediction outputs.

Recognizes ``scores.model_idx_N.npz`` paired with ``pred.model_idx_N.cif``.
PAE is read from the scores npz (key ``pae``) or a separate
``pae.model_idx_N.npz`` when present.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from foldmetrics.models import Prediction
from foldmetrics.parsers.base import ToolParser, Unit, register
from foldmetrics.parsers.structure import autoscale_plddt, tokenize_structure

SCORES_RE = re.compile(r"^scores\.model_idx_(?P<idx>\d+)\.npz$")


@register
class ChaiParser(ToolParser):
    tool = "chai"

    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        names = set(filenames)
        units: list[Unit] = []
        for fn in filenames:
            m = SCORES_RE.match(fn)
            if not m:
                continue
            idx = m["idx"]
            structure = None
            for ext in (".cif", ".pdb"):
                candidate = f"pred.model_idx_{idx}{ext}"
                if candidate in names:
                    structure = candidate
                    break
            if structure is None:
                continue
            files = {"scores": directory / fn, "structure": directory / structure}
            pae_file = f"pae.model_idx_{idx}.npz"
            if pae_file in names:
                files["pae"] = directory / pae_file
            units.append(
                Unit(
                    tool=self.tool,
                    name=f"model_idx_{idx}",
                    dir=directory,
                    files=files,
                )
            )
        return units

    def load(self, unit: Unit) -> Prediction:
        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)
        warnings: list[str] = []

        with np.load(unit.files["scores"]) as npz:
            keys = set(npz.files)

            def scalar(key: str) -> float | None:
                if key not in keys:
                    return None
                return float(np.ravel(np.asarray(npz[key]))[0])

            ptm = scalar("ptm")
            iptm = scalar("iptm")
            ranking = scalar("aggregate_score")
            extras: dict = {}
            for key in ("per_chain_ptm", "has_inter_chain_clashes"):
                if key in keys:
                    extras[key] = np.squeeze(np.asarray(npz[key])).tolist()
            pair_matrix = (
                np.squeeze(np.asarray(npz["per_chain_pair_iptm"]))
                if "per_chain_pair_iptm" in keys
                else None
            )
            pae = np.squeeze(np.asarray(npz["pae"], dtype=float)) if "pae" in keys else None

        if pae is None and "pae" in unit.files:
            with np.load(unit.files["pae"]) as npz:
                key = "pae" if "pae" in npz.files else npz.files[0]
                pae = np.squeeze(np.asarray(npz[key], dtype=float))
        if pae is not None and pae.ndim == 3:
            pae = pae[0]
        if pae is None:
            warnings.append(
                "no PAE found in Chai output (rerun with PAE export enabled); "
                "PAE-based metrics unavailable"
            )

        chains: list[str] = []
        for t in tokens:
            if t.chain not in chains:
                chains.append(t.chain)

        from foldmetrics.parsers.base import map_pair_matrix

        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            tokens=tokens,
            pae=pae,
            ptm=ptm,
            iptm=iptm,
            ranking_score=ranking,
            chain_pair_iptm=map_pair_matrix(chains, pair_matrix),
            extras=extras,
            warnings=warnings,
        )
