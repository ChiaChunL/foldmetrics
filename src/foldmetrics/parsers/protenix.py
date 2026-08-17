"""Parser for Protenix prediction outputs.

Recognizes ``*summary_confidence*.json`` (singular "confidence", unlike
AlphaFold3's "confidences") paired with the matching ``.cif`` file, e.g.::

    <job>_seed_42_sample_0.cif
    <job>_seed_42_summary_confidence_sample_0.json
    <job>_seed_42_full_data_sample_0.json   (optional, holds the PAE)
"""

from __future__ import annotations

import re
from pathlib import Path

from foldmetrics.models import Prediction
from foldmetrics.parsers.alphafold3 import attach_token_pae
from foldmetrics.parsers.base import (
    ToolParser,
    Unit,
    as_float,
    load_json,
    map_pair_matrix,
    map_pair_nested,
    register,
)
from foldmetrics.parsers.structure import autoscale_plddt, tokenize_structure

SUMMARY_RE = re.compile(r"^(?P<pre>.*?)summary_confidence(?P<post>.*)\.json$")

_SCALAR_EXTRA_KEYS = ("plddt", "gpde", "chain_ptm", "chain_iptm", "chain_plddt")


@register
class ProtenixParser(ToolParser):
    tool = "protenix"

    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        names = set(filenames)
        units: list[Unit] = []
        for fn in filenames:
            m = SUMMARY_RE.match(fn)
            if not m:
                continue
            pre, post = m["pre"], m["post"]
            candidates = [
                f"{pre.rstrip('_')}{post}.cif",
                f"{pre}{post.lstrip('_')}.cif",
                f"{pre.rstrip('_')}{post}.pdb",
            ]
            structure = next((c for c in candidates if c in names), None)
            if structure is None:
                continue
            files = {"summary": directory / fn, "structure": directory / structure}
            full_data = fn.replace("summary_confidence", "full_data")
            if full_data in names:
                files["confidences"] = directory / full_data
            units.append(
                Unit(
                    tool=self.tool,
                    name=Path(structure).stem,
                    dir=directory,
                    files=files,
                )
            )
        return units

    def load(self, unit: Unit) -> Prediction:
        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)
        warnings: list[str] = []

        summary = load_json(unit.files["summary"])
        extras = {k: summary[k] for k in _SCALAR_EXTRA_KEYS if k in summary}

        pae = None
        if "confidences" in unit.files:
            conf = load_json(unit.files["confidences"])
            # Protenix names its PAE "token_pair_pae"; normalize to the
            # AF3-style key so the shared attach logic applies.
            if "pae" not in conf and "token_pair_pae" in conf:
                conf["pae"] = conf["token_pair_pae"]
            tokens, pae = attach_token_pae(tokens, conf, warnings)
            if pae is None and "pae" not in conf:
                warnings.append("full_data JSON has no PAE matrix")
        else:
            warnings.append("no full_data JSON found; PAE unavailable")

        chains: list[str] = []
        for t in tokens:
            if t.chain not in chains:
                chains.append(t.chain)

        raw_pair = summary.get("chain_pair_iptm")
        pair_iptm = map_pair_nested(chains, raw_pair)
        if not pair_iptm:
            pair_iptm = map_pair_matrix(chains, raw_pair)

        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            tokens=tokens,
            pae=pae,
            ptm=as_float(summary.get("ptm")),
            iptm=as_float(summary.get("iptm")),
            ranking_score=as_float(summary.get("ranking_score")),
            chain_pair_iptm=pair_iptm,
            extras=extras,
            warnings=warnings,
        )
