"""Parser for AlphaFold2 / AlphaFold-Multimer pipeline outputs.

Two layouts are recognized, both requiring the model structure file:

- pickle layout: ``result_<tag>.pkl`` (+ ``ranking_debug.json``) with
  ``unrelaxed_<tag>.pdb`` / ``relaxed_<tag>.pdb`` / ``ranked_N.pdb``
- JSON layout (as written by common AF2 wrappers): ``iptm_ptm.json`` /
  ``ranking_debug.json`` plus per-model ``confidence_<tag>.json``
  (per-residue pLDDT), ``pae_<tag>.json`` (EBI-style PAE) and
  ``unrelaxed_<tag>.cif`` or ``.pdb``
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from foldmetrics.models import Prediction
from foldmetrics.parsers.base import (
    ParserError,
    ToolParser,
    Unit,
    as_float,
    load_json,
    register,
)
from foldmetrics.parsers.structure import autoscale_plddt, tokenize_structure

RESULT_RE = re.compile(r"^result_(?P<tag>model_.+)\.pkl$")
STRUCTURE_RE = re.compile(r"^(?:unrelaxed|relaxed)_(?P<tag>model_.+)\.(?:pdb|cif)$")


def _apply_plddt(tokens, plddt, warnings: list[str], source: str) -> None:
    if plddt is None:
        return
    plddt = np.asarray(plddt, dtype=float)
    if len(plddt) == len(tokens):
        for token, value in zip(tokens, plddt, strict=True):
            token.plddt = float(value)
            token.cb_plddt = float(value)
    else:
        warnings.append(
            f"{source} has {len(plddt)} pLDDT values for {len(tokens)} tokens; "
            "using structure B-factors instead"
        )


@register
class AlphaFold2Parser(ToolParser):
    tool = "alphafold2"

    def find_units(self, directory: Path, filenames: list[str]) -> list[Unit]:
        names = set(filenames)
        ranking_order: list[str] | None = None
        if "ranking_debug.json" in names:
            try:
                ranking = load_json(directory / "ranking_debug.json")
                order = ranking.get("order")
                if isinstance(order, list):
                    ranking_order = [str(x) for x in order]
            except Exception:
                ranking_order = None

        units: list[Unit] = []
        claimed: set[str] = set()

        # --- pickle layout ---------------------------------------------------
        for fn in filenames:
            m = RESULT_RE.match(fn)
            if not m:
                continue
            tag = m["tag"]
            structure = None
            for candidate in (f"unrelaxed_{tag}.pdb", f"relaxed_{tag}.pdb",
                              f"unrelaxed_{tag}.cif", f"relaxed_{tag}.cif"):
                if candidate in names:
                    structure = candidate
                    break
            if structure is None and ranking_order and tag in ranking_order:
                candidate = f"ranked_{ranking_order.index(tag)}.pdb"
                if candidate in names:
                    structure = candidate
            if structure is None:
                continue
            claimed.add(tag)
            units.append(
                Unit(
                    tool=self.tool,
                    name=tag,
                    dir=directory,
                    files={"result": directory / fn, "structure": directory / structure},
                )
            )

        # --- JSON layout -----------------------------------------------------
        has_scores = "iptm_ptm.json" in names or "ranking_debug.json" in names
        if has_scores:
            for fn in filenames:
                m = STRUCTURE_RE.match(fn)
                if not m or m["tag"] in claimed:
                    continue
                tag = m["tag"]
                pae = f"pae_{tag}.json"
                confidence = f"confidence_{tag}.json"
                if pae not in names and confidence not in names:
                    continue
                files = {"structure": directory / fn}
                if pae in names:
                    files["pae"] = directory / pae
                if confidence in names:
                    files["confidence"] = directory / confidence
                if "iptm_ptm.json" in names:
                    files["scores"] = directory / "iptm_ptm.json"
                elif "ranking_debug.json" in names:
                    files["ranking"] = directory / "ranking_debug.json"
                claimed.add(tag)
                units.append(Unit(tool=self.tool, name=tag, dir=directory, files=files))

        return units

    def load(self, unit: Unit) -> Prediction:
        if "result" in unit.files:
            return self._load_pickle(unit)
        return self._load_json(unit)

    # --------------------------------------------------------------- pickle
    def _load_pickle(self, unit: Unit) -> Prediction:
        with open(unit.files["result"], "rb") as fh:
            try:
                data = pickle.load(fh)
            except Exception as exc:  # noqa: BLE001 - report as parse failure
                raise ParserError(f"cannot unpickle {unit.files['result']}: {exc}") from exc
        if not isinstance(data, dict):
            raise ParserError(f"unexpected pickle content in {unit.files['result']}")

        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)
        warnings: list[str] = []
        _apply_plddt(tokens, data.get("plddt"), warnings, "pickle")

        pae = data.get("predicted_aligned_error")
        pae = np.asarray(pae, dtype=float) if pae is not None else None

        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            tokens=tokens,
            pae=pae,
            ptm=as_float(data.get("ptm")),
            iptm=as_float(data.get("iptm")),
            ranking_score=as_float(data.get("ranking_confidence")),
            warnings=warnings,
        )

    # ----------------------------------------------------------------- JSON
    def _load_json(self, unit: Unit) -> Prediction:
        tokens = tokenize_structure(unit.files["structure"])
        autoscale_plddt(tokens)
        warnings: list[str] = []

        if "confidence" in unit.files:
            conf = load_json(unit.files["confidence"])
            _apply_plddt(tokens, conf.get("confidenceScore"), warnings, "confidence JSON")

        pae = None
        if "pae" in unit.files:
            import json

            with open(unit.files["pae"]) as fh:
                raw = json.load(fh)
            # EBI/AF2 format: [{"predicted_aligned_error": [[...]], ...}]
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                raw = raw[0]
            if isinstance(raw, dict):
                matrix = raw.get("predicted_aligned_error", raw.get("pae"))
                if matrix is not None:
                    pae = np.asarray(matrix, dtype=float)
        else:
            warnings.append("no pae JSON found; PAE-based metrics unavailable")

        ptm = iptm = ranking = None
        if "scores" in unit.files:
            scores = load_json(unit.files["scores"]).get(unit.name)
            if isinstance(scores, dict):
                ptm = as_float(scores.get("ptm"))
                iptm = as_float(scores.get("iptm"))
                ranking = as_float(scores.get("ranking_confidence"))
            else:
                warnings.append(f"model {unit.name!r} not found in iptm_ptm.json")
        elif "ranking" in unit.files:
            ranking_data = load_json(unit.files["ranking"])
            for key in ("iptm+ptm", "plddts"):
                if unit.name in ranking_data.get(key, {}):
                    ranking = as_float(ranking_data[key][unit.name])
                    break

        return Prediction(
            name=unit.name,
            tool=self.tool,
            source=unit.files["structure"],
            tokens=tokens,
            pae=pae,
            ptm=ptm,
            iptm=iptm,
            ranking_score=ranking,
            warnings=warnings,
        )
