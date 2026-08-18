from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import make_two_chain_pred

from foldmetrics.metrics import (
    calc_d0,
    calc_d0_array,
    compute_all,
    ipsae_asym,
    lis_asym,
    pair_metrics,
    pdockq,
    pdockq2_asym,
    ptm_transform,
)


# ------------------------------------------------------------------------- d0
def test_calc_d0_scalar_matches_reference_semantics():
    # scalar variant: TM formula only for L > 27 (ipsae.py calc_d0)
    assert calc_d0(27) == 1.0
    expected_28 = 1.24 * (28 - 15) ** (1 / 3) - 1.8
    assert math.isclose(calc_d0(28), expected_28, rel_tol=1e-12)
    assert calc_d0(5) == 1.0
    assert calc_d0(5, nucleic=True) == 2.0
    assert calc_d0(1000, nucleic=True) == 1.24 * (1000 - 15) ** (1 / 3) - 1.8


def test_calc_d0_array_clamps_at_26():
    # array variant: L clamped to >= 26 before the formula (ipsae.py calc_d0_array)
    d5, d26, d27 = calc_d0_array([5, 26, 27])
    assert d5 == d26 == 1.0  # formula at 26 gives 0.958 -> floor 1.0
    assert math.isclose(d27, 1.24 * (27 - 15) ** (1 / 3) - 1.8, rel_tol=1e-12)
    assert calc_d0_array([5], nucleic=True)[0] == 2.0


# --------------------------------------------------------------------- pDockQ
def test_pdockq_expected_value():
    pred = make_two_chain_pred(n_a=40, n_b=40, plddt=80.0)
    result = pdockq(pred, "A", "B")
    assert result.n_pairs == 3 * 40 - 2  # geometry: i touches j in {i-1, i, i+1}
    assert result.n_if_res == 80
    x = 80.0 * math.log10(result.n_pairs)
    expected = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
    assert math.isclose(result.value, expected, rel_tol=1e-12)


def test_pdockq_no_contacts_is_zero():
    pred = make_two_chain_pred(gap=50.0)
    result = pdockq(pred, "A", "B")
    assert result.value == 0.0
    assert result.n_pairs == 0


# -------------------------------------------------------------------- pDockQ2
def test_pdockq2_expected_value():
    pred = make_two_chain_pred(plddt=80.0, pae=5.0)
    mean_ptm = 1.0 / (1.0 + (5.0 / 10.0) ** 2)  # 0.8
    x = 80.0 * mean_ptm
    expected = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
    assert math.isclose(pdockq2_asym(pred, "A", "B"), expected, rel_tol=1e-12)
    assert math.isclose(pdockq2_asym(pred, "B", "A"), expected, rel_tol=1e-12)


def test_pdockq2_without_pae_is_nan():
    pred = make_two_chain_pred(pae=None)
    assert math.isnan(pdockq2_asym(pred, "A", "B"))


# ------------------------------------------------------------------------ LIS
def test_lis_values():
    assert math.isclose(lis_asym(make_two_chain_pred(pae=6.0), "A", "B"), 0.5)
    assert lis_asym(make_two_chain_pred(pae=15.0), "A", "B") == 0.0


# ---------------------------------------------------------------------- ipSAE
def test_ipsae_constant_pae():
    pred = make_two_chain_pred(n_a=40, n_b=40, pae=5.0)
    result = ipsae_asym(pred, "A", "B")
    assert result is not None
    assert result.n0res == 40
    d0 = 1.24 * (40 - 15) ** (1 / 3) - 1.8
    assert math.isclose(result.d0res, d0, rel_tol=1e-12)
    expected = float(ptm_transform(5.0, d0))
    assert math.isclose(result.value, expected, rel_tol=1e-12)


def test_ipsae_all_above_cutoff_is_zero():
    pred = make_two_chain_pred(pae=15.0)
    result = ipsae_asym(pred, "A", "B")
    assert result is not None
    assert result.value == 0.0


def test_ipsae_directionality():
    pred = make_two_chain_pred(pae_ab=4.0, pae_ba=20.0)
    ab = ipsae_asym(pred, "A", "B")
    ba = ipsae_asym(pred, "B", "A")
    assert ab.value > 0.0
    assert ba.value == 0.0
    row = pair_metrics(pred, "A", "B")
    assert math.isclose(row["ipsae"], ab.value, rel_tol=1e-12)


def test_ipsae_nucleic_d0_floor():
    pred = make_two_chain_pred(n_a=10, n_b=10, pae=2.0, kinds=("protein", "rna"))
    result = ipsae_asym(pred, "A", "B")
    # n0res=10 -> clamped to 26 -> formula 0.958 -> nucleic floor 2.0
    assert result.d0res == 2.0
    assert math.isclose(result.value, 1.0 / (1.0 + (2.0 / 2.0) ** 2), rel_tol=1e-12)


# ---------------------------------------------------------------- aggregation
def test_compute_all_summary():
    pred = make_two_chain_pred(n_a=40, n_b=40, plddt=80.0, pae=5.0)
    summary, interfaces = compute_all(pred)
    assert len(interfaces) == 1
    row = interfaces[0]
    assert summary["ipsae"] == row["ipsae"]
    assert summary["pdockq"] == row["pdockq"]
    assert summary["n_interfaces"] == 1
    assert summary["has_pae"] is True
    assert math.isclose(summary["plddt_mean"], 80.0)
    assert math.isclose(summary["iplddt"], 80.0)
    assert row["ipsae_mode"] == "residues"
    # intra-chain PAE == inter-chain PAE == 5 here, so both means are 5
    assert math.isclose(summary["ipae_mean"], 5.0)


def test_pae_shape_mismatch_disables_pae():
    pred = make_two_chain_pred(n_a=10, n_b=10, pae=np.full((5, 5), 3.0))
    assert pred.pae is None
    assert pred.warnings
    summary, _ = compute_all(pred)
    assert math.isnan(summary["ipsae"])
    assert not math.isnan(summary["pdockq"])  # contact-based metrics still work


# ------------------------------------------- pDockQ2: which atom's pLDDT?
def _per_atom_plddt_pred(tmp_path):
    """A two-chain model whose CB pLDDT differs from its backbone pLDDT."""
    from conftest import add_protein_chain, finish_structure, new_structure

    import foldmetrics as fmx

    st, model = new_structure()
    add_protein_chain(model, "A", 20, origin=(0.0, 0.0, 0.0), plddt=90.0, cb_plddt=60.0)
    add_protein_chain(model, "B", 20, origin=(0.0, 5.0, 0.0), plddt=90.0, cb_plddt=60.0)
    path = tmp_path / "per_atom.pdb"
    finish_structure(st, model, path)

    from foldmetrics.parsers.structure import tokenize_structure

    tokens = tokenize_structure(path)
    n = len(tokens)
    return fmx.Prediction(
        name="per_atom", tool="test", source=path, tokens=tokens,
        pae=np.full((n, n), 5.0),
    )


def test_pdockq2_reads_cb_by_default_and_ca_for_zhu2023(tmp_path):
    pred = _per_atom_plddt_pred(tmp_path)
    assert pred.plddt_arr[0] == 90.0  # CA
    assert pred.cb_plddt_arr[0] == 60.0  # CB

    consensus = pdockq2_asym(pred, "A", "B")
    zhu = pdockq2_asym(pred, "A", "B", variant="zhu2023")

    def expected(mean_plddt):
        mean_ptm = 1.0 / (1.0 + (5.0 / 10.0) ** 2)
        return 1.31 / (1 + math.exp(-0.075 * (mean_plddt * mean_ptm - 84.733))) + 0.005

    # the default reads the CB pLDDT (ipsae.py / ColabFold convention)
    assert math.isclose(consensus, expected(60.0), rel_tol=1e-12)
    # the paper's own script reads the CA pLDDT of the scored chain
    assert math.isclose(zhu, expected(90.0), rel_tol=1e-12)
    assert consensus < zhu


def test_pdockq2_rejects_an_unknown_variant():
    pred = make_two_chain_pred(pae=5.0)
    with pytest.raises(ValueError, match="unknown pDockQ2 variant"):
        pdockq2_asym(pred, "A", "B", variant="nope")
