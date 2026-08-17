"""Parser for Boltz-1/Boltz-2 prediction outputs.

Recognizes, inside a predictions directory::

    <name>_model_N.cif  (or .pdb)
    confidence_<name>_model_N.json
    pae_<name>_model_N.npz          (optional)
    plddt_<name>_model_N.npz        (optional)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from foldmetrics.models import Prediction
from foldmetrics.parsers.base import (
    ToolParser,
    Unit,
    as_float,
    load_json,
    map_pair_nested,
    register,
)
from foldmetrics.parsers.structure import autoscale_plddt, tokenize_structure

CONFIDENCE_RE = re.compile(r"^confidence_(?P<base>.+)_model_(?P<idx>\d+)\.json$")

_EXTRA_KEYS = (
    "confidence_score",
    "complex_plddt",
    "complex_iplddt",
    "complex_pde",
    "complex_ipde",
    "protein_iptm",
    "ligand_iptm",
    "chains_ptm",
)


def _load_npz_array(path: Path, key: str) -> np.ndarray | None:
    with np.load(path) as data:
        if key in data.files:
            return np.squeeze(np.asarray(data[key], dtype=float))
        if len(data.files) == 1:
            return np.squeeze(np.asarray(data[data.files[0]], dtype=float))
    return None


@register
class BoltzParser(ToolParser):
    tool = "boltz"

    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        names = set(filenames)
        units: list[Unit] = []
        for fn in filenames:
            m = CONFIDENCE_RE.match(fn)
            if not m:
                continue
            base, idx = m["base"], m["idx"]
            structure = None
            for ext in (".cif", ".pdb"):
                candidate = f"{base}_model_{idx}{ext}"
                if candidate in names:
                    structure = candidate
                    break
            if structure is None:
                continue
            files = {"confidence": directory / fn, "structure": directory / structure}
            for role, candidate in (
                ("pae", f"pae_{base}_model_{idx}.npz"),
                ("plddt", f"plddt_{base}_model_{idx}.npz"),
            ):
                if candidate in names:
                    files[role] = directory / candidate
            units.append(
                Unit(
                    tool=self.tool,
                    name=f"{base}_model_{idx}",
                    dir=directory,
                    files=files,
                )
            )
        return units

    def load(self, unit: Unit) -> Prediction:
        data = load_json(unit.files["confidence"])
        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)
        warnings: list[str] = []

        if "plddt" in unit.files:
            plddt = _load_npz_array(unit.files["plddt"], "plddt")
            if plddt is not None:
                if 0.0 < float(np.nanmax(plddt)) <= 1.05:
                    plddt = plddt * 100.0
                if plddt.shape == (len(tokens),):
                    for token, value in zip(tokens, plddt, strict=True):
                        token.plddt = float(value)
                        token.cb_plddt = float(value)
                else:
                    warnings.append(
                        f"plddt npz has shape {plddt.shape} for {len(tokens)} tokens; "
                        "using structure B-factors instead"
                    )

        pae = None
        if "pae" in unit.files:
            pae = _load_npz_array(unit.files["pae"], "pae")
            if pae is not None and pae.ndim == 3:
                pae = pae[0]
        else:
            warnings.append("no pae npz found; PAE-based metrics unavailable")

        chains: list[str] = []
        for t in tokens:
            if t.chain not in chains:
                chains.append(t.chain)

        extras = {k: data[k] for k in _EXTRA_KEYS if k in data}
        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            tokens=tokens,
            pae=pae,
            ptm=as_float(data.get("ptm")),
            iptm=as_float(data.get("iptm")),
            ranking_score=as_float(data.get("confidence_score")),
            chain_pair_iptm=map_pair_nested(chains, data.get("pair_chains_iptm")),
            extras=extras,
            warnings=warnings,
        )
