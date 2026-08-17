"""Regression tests against the real prediction outputs bundled in examples/data."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from foldmetrics.api import evaluate
from foldmetrics.metrics import find_contacts
from foldmetrics.parsers import discover, get_parser

DATA = Path(__file__).resolve().parent.parent / "examples" / "data"

pytestmark = pytest.mark.skipif(not DATA.exists(), reason="examples/data not present")


def test_detects_all_bundled_tools():
    units = discover(DATA)
    assert sorted(u.tool for u in units) == [
        "alphafold2", "alphafold3", "alphafold3", "boltz", "chai", "colabfold",
    ]


def test_scores_real_outputs():
    df = evaluate(DATA)
    assert len(df) == 6
    assert df["has_pae"].all()
    assert df["warnings"].fillna("").eq("").all()

    barnase = df[df["model"] != "mpro_nirmatrelvir"]
    assert (barnase["iptm"] > 0.90).all()
    assert (barnase["ipsae"] > 0.85).all()
    assert (barnase["pdockq2"] > 0.90).all()

    mpro = df[df["model"] == "mpro_nirmatrelvir"].iloc[0]
    assert math.isnan(mpro["pdockq"])  # its only interface is protein-ligand
    assert mpro["ipsae"] > 0.80  # token-mode ligand ipSAE


def test_colabfold_embedded_score_parity():
    """Our ipSAE/pDockQ/pDockQ2 must reproduce ColabFold's own embedded values.

    ColabFold >= 1.6 computes these with the same constants from its
    full-precision PAE; the JSON stores PAE rounded to 2 decimals, which
    bounds the achievable agreement at ~1e-3.
    """
    from foldmetrics.metrics import ipsae_asym, pdockq, pdockq2_asym

    units = [u for u in discover(DATA) if u.tool == "colabfold"]
    pred = get_parser("colabfold").load(units[0])
    native_ipsae = pred.extras["colabfold_ipsae"]
    native_pdockq = pred.extras["colabfold_pdockq"]
    native_pdockq2 = pred.extras["colabfold_pdockq2"]

    for a, b, key in (("A", "B", "A-B"), ("B", "A", "B-A")):
        ours = ipsae_asym(pred, a, b, pae_cutoff=15.0).value  # ColabFold cutoff
        assert math.isclose(ours, native_ipsae[key], abs_tol=1e-3)
        assert math.isclose(
            pdockq2_asym(pred, a, b), native_pdockq2[key], abs_tol=1e-3
        )
    assert math.isclose(
        pdockq(pred, "A", "B").value, native_pdockq["A-B"], abs_tol=1e-3
    )


def test_barnase_hotspot_contacts():
    units = [u for u in discover(DATA) if u.tool == "alphafold3"
             and "barnase" in u.name]
    pred = get_parser("alphafold3").load(units[0])
    contacts = find_contacts(pred)
    assert len(contacts) > 30
    # the literature hotspots must be among the extracted interface residues
    res_a = {r["res_a"] for r in contacts}
    assert {59, 83, 87, 102} <= res_a
