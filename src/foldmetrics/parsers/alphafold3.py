"""Parser for AlphaFold3 outputs (AlphaFold Server downloads and local runs).

Recognized layouts, all requiring a ``*summary_confidences*.json``:

- server: ``fold_<job>_model_N.cif`` + ``fold_<job>_summary_confidences_N.json``
  + ``fold_<job>_full_data_N.json``
- local (top level): ``<job>_model.cif`` + ``<job>_summary_confidences.json``
  + ``<job>_confidences.json``
- local (per sample): ``model.cif`` + ``summary_confidences.json`` + ``confidences.json``
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from foldmetrics.models import Prediction, Token
from foldmetrics.parsers.base import (
    ToolParser,
    Unit,
    as_float,
    load_json,
    map_pair_matrix,
    register,
)
from foldmetrics.parsers.structure import autoscale_plddt, tokenize_structure

MODEL_RE = re.compile(r"^(?:(?P<base>.+)_)?model(?:_(?P<idx>\d+))?\.cif$")

_SUMMARY_EXTRA_KEYS = (
    "chain_ptm",
    "chain_iptm",
    "chain_pair_pae_min",
    "fraction_disordered",
    "has_clash",
    "num_recycles",
)


def reconcile_tokens(
    tokens: list[Token], chain_ids: list[str], res_ids: list[int]
) -> list[Token] | None:
    """Rebuild the token list to follow explicit (chain, residue) token ids.

    Used when our tokenization disagrees with the tool's PAE indexing
    (e.g. glycans or modified residues). Returns None if the ids reference
    residues we did not see.
    """
    groups: dict[tuple[str, int], list[Token]] = {}
    for token in tokens:
        groups.setdefault((token.chain, token.res_id), []).append(token)

    out: list[Token] = []
    counters: dict[tuple[str, int], int] = {}
    for chain, res in zip(chain_ids, res_ids, strict=False):
        key = (str(chain), int(res))
        group = groups.get(key)
        if not group:
            return None
        k = counters.get(key, 0)
        out.append(group[min(k, len(group) - 1)])
        counters[key] = k + 1
    return out


def attach_token_pae(
    pred_tokens: list[Token], data: dict, warnings: list[str]
) -> tuple[list[Token], np.ndarray | None]:
    """Validate a token-level PAE matrix against our tokenization."""
    pae = data.get("pae")
    if pae is None:
        return pred_tokens, None
    pae = np.asarray(pae, dtype=float)
    n = pae.shape[0] if pae.ndim == 2 else -1

    if n == len(pred_tokens):
        chain_ids = data.get("token_chain_ids")
        if chain_ids is not None and len(chain_ids) == n:
            mismatches = sum(
                1 for t, c in zip(pred_tokens, chain_ids, strict=True) if t.chain != str(c)
            )
            if mismatches:
                warnings.append(
                    f"{mismatches}/{n} token chain ids differ from the structure "
                    "(label vs auth chain naming?); keeping PAE"
                )
        return pred_tokens, pae

    chain_ids = data.get("token_chain_ids")
    res_ids = data.get("token_res_ids")
    if chain_ids is not None and res_ids is not None and len(chain_ids) == n:
        rebuilt = reconcile_tokens(pred_tokens, chain_ids, res_ids)
        if rebuilt is not None and len(rebuilt) == n:
            warnings.append(
                f"tokenization mismatch ({len(pred_tokens)} vs {n} tokens); "
                "re-aligned tokens to the tool's token ids"
            )
            return rebuilt, pae

    warnings.append(
        f"PAE has {n} tokens but the structure produced {len(pred_tokens)}; "
        "PAE-based metrics disabled"
    )
    return pred_tokens, None


@register
class AlphaFold3Parser(ToolParser):
    tool = "alphafold3"

    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        names = set(filenames)
        units: list[Unit] = []
        for fn in filenames:
            m = MODEL_RE.match(fn)
            if not m:
                continue
            base, idx = m["base"] or "", m["idx"]
            prefix = f"{base}_" if base else ""
            suffix = f"_{idx}" if idx is not None else ""

            summary = f"{prefix}summary_confidences{suffix}.json"
            if summary not in names:
                continue
            files = {"structure": directory / fn, "summary": directory / summary}
            for conf in (
                f"{prefix}confidences{suffix}.json",
                f"{prefix}full_data{suffix}.json",
            ):
                if conf in names:
                    files["confidences"] = directory / conf
                    break

            if base:
                name = f"{base}_model_{idx}" if idx is not None else base
            else:
                name = directory.name
            units.append(Unit(tool=self.tool, name=name, dir=directory, files=files))
        return units

    def load(self, unit: Unit) -> Prediction:
        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)
        warnings: list[str] = []

        summary = load_json(unit.files["summary"])
        extras = {k: summary[k] for k in _SUMMARY_EXTRA_KEYS if k in summary}

        pae = None
        if "confidences" in unit.files:
            conf = load_json(unit.files["confidences"])
            tokens, pae = attach_token_pae(tokens, conf, warnings)
        else:
            warnings.append("no confidences/full_data JSON found; PAE unavailable")

        chains: list[str] = []
        for t in tokens:
            if t.chain not in chains:
                chains.append(t.chain)

        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            tokens=tokens,
            pae=pae,
            ptm=as_float(summary.get("ptm")),
            iptm=as_float(summary.get("iptm")),
            ranking_score=as_float(summary.get("ranking_score")),
            chain_pair_iptm=map_pair_matrix(chains, summary.get("chain_pair_iptm")),
            extras=extras,
            warnings=warnings,
        )
