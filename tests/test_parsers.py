from __future__ import annotations

import math

import pytest
from conftest import (
    write_af2_json_dir,
    write_af3_dir,
    write_boltz_dir,
    write_colabfold_dir,
)

from foldmetrics.api import evaluate, evaluate_full
from foldmetrics.parsers import discover, get_parser


# ------------------------------------------------------------------ ColabFold
def test_colabfold_end_to_end(tmp_path):
    directory = write_colabfold_dir(tmp_path / "cf")
    units = discover(directory)
    assert [u.tool for u in units] == ["colabfold"]

    pred = get_parser("colabfold").load(units[0])
    assert pred.n_tokens == 60
    assert pred.has_pae
    assert math.isclose(pred.ptm, 0.78)
    assert math.isclose(pred.iptm, 0.66)
    assert math.isclose(pred.ranking_score, 0.8 * 0.66 + 0.2 * 0.78)
    # ColabFold's own embedded interface scores are preserved for comparison
    assert pred.extras["colabfold_ipsae"]["A-B"] == 0.91
    assert pred.extras["colabfold_pdockq2"]["B-A"] == 0.92

    df = evaluate(pred)
    row = df.iloc[0]
    assert row["n_chains"] == 2
    assert math.isclose(row["plddt_mean"], 80.0)
    assert row["pdockq"] > 0.0
    # constant PAE=5 across a 30+30 complex: n0res=30 -> d0 = calc_d0_array(30)
    d0 = 1.24 * (30 - 15) ** (1 / 3) - 1.8
    assert math.isclose(row["ipsae"], 1.0 / (1.0 + (5.0 / d0) ** 2), rel_tol=1e-9)


# ----------------------------------------------------------------- AlphaFold3
def test_af3_with_ligand(tmp_path):
    directory = write_af3_dir(tmp_path / "af3")
    units = discover(directory)
    assert [u.tool for u in units] == ["alphafold3"]

    pred = get_parser("alphafold3").load(units[0])
    assert pred.n_tokens == 20 + 15 + 3  # ligand contributes one token per atom
    assert pred.has_pae
    assert [t.kind for t in pred.tokens].count("ligand") == 3
    assert math.isclose(pred.get_pair_iptm("A", "B"), 0.7)
    assert math.isclose(pred.ranking_score, 0.75)

    summary_df, interfaces_df = evaluate_full(pred)
    assert summary_df.iloc[0]["n_chains"] == 3
    pairs = {
        (r["chain_a"], r["chain_b"]): r for _, r in interfaces_df.iterrows()
    }
    assert pairs[("A", "B")]["ipsae_mode"] == "residues"
    assert pairs[("A", "C")]["ipsae_mode"] == "tokens"
    assert pairs[("A", "B")]["ipsae"] > pairs[("A", "C")]["ipsae"]
    assert math.isclose(pairs[("A", "B")]["iptm_native"], 0.7)
    # ligand pair has no polymer-polymer contacts -> contact metrics undefined
    assert math.isnan(pairs[("A", "C")]["pdockq"])


# ---------------------------------------------------------- AlphaFold2 (JSON)
def test_af2_json_layout(tmp_path):
    directory = write_af2_json_dir(tmp_path / "af2")
    units = discover(directory)
    assert [u.tool for u in units] == ["alphafold2"]

    pred = get_parser("alphafold2").load(units[0])
    assert pred.n_tokens == 40
    assert pred.has_pae
    assert math.isclose(pred.iptm, 0.9)
    assert math.isclose(pred.ptm, 0.91)
    assert math.isclose(pred.ranking_score, 0.902)
    assert math.isclose(float(pred.plddt_arr.mean()), 77.0)  # from confidence JSON

    df = evaluate(pred)
    assert math.isclose(df.iloc[0]["lis"], (12.0 - 4.0) / 12.0)


# ---------------------------------------------------------------------- Boltz
def test_boltz_npz_and_scaling(tmp_path):
    directory = write_boltz_dir(tmp_path / "boltz")
    units = discover(directory)
    assert [u.tool for u in units] == ["boltz"]

    pred = get_parser("boltz").load(units[0])
    assert pred.n_tokens == 45
    assert pred.has_pae
    assert math.isclose(pred.ranking_score, 0.85)
    # plddt npz stores 0-1 values; they must be rescaled to 0-100
    assert math.isclose(float(pred.plddt_arr.mean()), 83.0, rel_tol=1e-9)
    assert math.isclose(pred.get_pair_iptm("A", "B"), 0.7)
    assert pred.extras["complex_iplddt"] == 0.8

    df = evaluate(pred)
    row = df.iloc[0]
    assert math.isclose(row["lis"], 0.5)  # constant PAE=6 -> (12-6)/12
    assert math.isclose(row["ipae_mean"], 6.0)


# --------------------------------------------------------------------- target
def test_infer_target_strips_run_components():
    from foldmetrics.models import infer_target

    assert infer_target("/x/af_server/barnase_barstar/seed318/model_0.cif") == "barnase_barstar"
    assert infer_target("/x/af2/barnase_barstar/pred0/unrelaxed.cif") == "barnase_barstar"
    assert infer_target("/x/af3/mpro/ranked/model.cif") == "mpro"
    assert infer_target("/x/boltz2/job1/sample-3/m.cif") == "job1"
    assert infer_target("demo/dimer_good/model.cif") == "dimer_good"


# ------------------------------------------------------------------ discovery
def test_discover_mixed_batch(tmp_path):
    write_colabfold_dir(tmp_path / "a" / "cf")
    write_boltz_dir(tmp_path / "b" / "boltz")
    units = discover(tmp_path)
    assert sorted(u.tool for u in units) == ["boltz", "colabfold"]

    df = evaluate(tmp_path)
    assert len(df) == 2
    assert set(df["tool"]) == {"boltz", "colabfold"}


def test_discover_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "does_not_exist")


def test_boltz_affinity_is_preserved(tmp_path):
    """Boltz-2 affinity mode writes one affinity JSON per job."""
    from conftest import write_boltz_dir

    directory = write_boltz_dir(tmp_path / "boltz", affinity=True)
    pred = get_parser("boltz").load(discover(directory)[0])
    assert math.isclose(pred.extras["boltz_affinity_pred_value"], -1.23)
    assert math.isclose(pred.extras["boltz_affinity_probability_binary"], 0.87)


def test_boltz_without_affinity_is_unaffected(tmp_path):
    from conftest import write_boltz_dir

    directory = write_boltz_dir(tmp_path / "boltz")
    pred = get_parser("boltz").load(discover(directory)[0])
    assert not any(k.startswith("boltz_affinity") for k in pred.extras)
    assert not pred.warnings


def test_af3_top_level_copy_is_not_counted_twice(tmp_path):
    """AlphaFold3 duplicates its best sample into the job directory."""
    from conftest import write_af3_dir

    job = tmp_path / "af3" / "P0__P1"
    write_af3_dir(job / "seed-2066_sample-0")
    write_af3_dir(job / "seed-2066_sample-1")
    write_af3_dir(job)  # the copy AlphaFold3 leaves beside the samples

    units = discover(tmp_path)
    assert len(units) == 2
    assert all(u.dir != job for u in units)


def test_af3_without_sample_directories_is_kept(tmp_path):
    """A job that only has the top-level model must still be scored."""
    from conftest import write_af3_dir

    write_af3_dir(tmp_path / "af3" / "P0__P1")
    assert len(discover(tmp_path)) == 1


def test_chai_stacked_pae_npz_is_indexed_per_model(tmp_path):
    """Recent Chai-1 writes one pae.npz holding every model's matrix."""
    import numpy as np
    from conftest import add_protein_chain, finish_structure, new_structure

    directory = tmp_path / "chai"
    directory.mkdir(parents=True)
    n_models, n = 3, 24
    for idx in range(n_models):
        st, model = new_structure()
        add_protein_chain(model, "A", 12, origin=(0.0, 0.0, 0.0))
        add_protein_chain(model, "B", 12, origin=(0.0, 5.0, 0.0))
        finish_structure(st, model, directory / f"pred.model_idx_{idx}.cif")
        np.savez(
            directory / f"scores.model_idx_{idx}.npz",
            ptm=np.array([0.9]), iptm=np.array([0.8]),
            aggregate_score=np.array([0.85]),
        )
    # a distinct constant per model so the slice can be identified
    stacked = np.stack([np.full((n, n), 2.0 + idx) for idx in range(n_models)])
    np.savez(directory / "pae.npz", pae=stacked)

    for unit in discover(directory):
        pred = get_parser("chai").load(unit)
        idx = int(unit.name.rsplit("_", 1)[1])
        assert pred.has_pae, unit.name
        assert not pred.warnings
        assert math.isclose(float(pred.pae[0, 0]), 2.0 + idx)


def test_af2_native_layout_counts_each_model_once(tmp_path):
    """AlphaFold2 writes every model four times; only one may be scored."""
    from conftest import write_af2_native_dir

    directory = write_af2_native_dir(tmp_path / "af2_multimer" / "P0__P1")
    structures = list(directory.glob("*.pdb")) + list(directory.glob("*.cif"))
    assert len(structures) == 20  # 5 models x (unrelaxed, ranked) x (pdb, cif)

    units = discover(tmp_path)
    assert len(units) == 5
    assert {u.name for u in units} == {
        f"model_{i}_multimer_v3_pred_0" for i in range(1, 6)
    }
    # the ranked copies must never be the scored structure
    assert not any("ranked_" in u.files["structure"].name for u in units)

    df = evaluate(tmp_path)
    assert len(df) == 5
    assert set(df["target"]) == {"P0__P1"}  # weights and samples are one target
    assert df["has_pae"].all()
    assert df["warnings"].fillna("").eq("").all()
