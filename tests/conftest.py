from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path

import gemmi
import numpy as np

from foldmetrics.models import Prediction, Token

# ------------------------------------------------------------ direct builders


def make_two_chain_pred(
    n_a: int = 40,
    n_b: int = 40,
    plddt: float = 80.0,
    pae: float | np.ndarray | None = None,
    pae_ab: float | None = None,
    pae_ba: float | None = None,
    spacing: float = 4.0,
    gap: float = 5.0,
    kinds: tuple[str, str] = ("protein", "protein"),
) -> Prediction:
    """Two straight chains of contact atoms; A at y=0, B at y=gap.

    With spacing=4 and gap=5, residue i of A contacts residues i-1, i, i+1
    of B (distances 5.0 and 6.40 Å), giving 3n-2 contact pairs.
    """
    tokens = []
    res_name = {"protein": "ALA", "rna": "A", "dna": "DA", "ligand": "LIG"}
    for i in range(n_a):
        tokens.append(
            Token("A", i + 1, res_name[kinds[0]], kinds[0], "CB",
                  (i * spacing, 0.0, 0.0), plddt, plddt)
        )
    for j in range(n_b):
        tokens.append(
            Token("B", j + 1, res_name[kinds[1]], kinds[1], "CB",
                  (j * spacing, gap, 0.0), plddt, plddt)
        )
    n = n_a + n_b
    matrix = None
    if pae is not None:
        matrix = np.asarray(pae, dtype=float)
        if matrix.ndim == 0:
            matrix = np.full((n, n), float(pae))
    elif pae_ab is not None or pae_ba is not None:
        matrix = np.full((n, n), 3.0)
        if pae_ab is not None:
            matrix[:n_a, n_a:] = pae_ab
        if pae_ba is not None:
            matrix[n_a:, :n_a] = pae_ba
    return Prediction(
        name="synthetic", tool="test", source=Path("synthetic"),
        tokens=tokens, pae=matrix,
    )


# --------------------------------------------------------- structure builders


def _new_model() -> gemmi.Model:
    try:
        return gemmi.Model("1")
    except TypeError:
        return gemmi.Model(1)


def _seqid(num: int) -> gemmi.SeqId:
    try:
        return gemmi.SeqId(num, " ")
    except TypeError:
        return gemmi.SeqId(str(num))


def _atom(name: str, element: str, pos: tuple[float, float, float], b: float) -> gemmi.Atom:
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*pos)
    atom.occ = 1.0
    atom.b_iso = b
    return atom


def add_protein_chain(
    model: gemmi.Model,
    name: str,
    n_res: int,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    plddt: float = 80.0,
    spacing: float = 4.0,
    cb_plddt: float | None = None,
) -> None:
    """``cb_plddt`` gives CB a different pLDDT from the backbone.

    Engines newer than AlphaFold2 emit per-atom pLDDT; the metrics that
    read a specific atom can only be pinned down with such a model.
    """
    chain = gemmi.Chain(name)
    x0, y0, z0 = origin
    for i in range(n_res):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = _seqid(i + 1)
        res.het_flag = "A"
        x = x0 + i * spacing
        # full backbone so external tools (e.g. DockQ) can superpose
        res.add_atom(_atom("N", "N", (x - 1.2, y0 + 1.9, z0), plddt))
        res.add_atom(_atom("CA", "C", (x, y0 + 1.5, z0), plddt))
        res.add_atom(_atom("C", "C", (x + 1.3, y0 + 2.0, z0), plddt))
        res.add_atom(_atom("O", "O", (x + 1.4, y0 + 3.2, z0), plddt))
        res.add_atom(_atom("CB", "C", (x, y0, z0), plddt if cb_plddt is None else cb_plddt))
        chain.add_residue(res)
    model.add_chain(chain)


def add_ligand_chain(
    model: gemmi.Model,
    name: str,
    n_atoms: int,
    origin: tuple[float, float, float],
    plddt: float = 70.0,
) -> None:
    chain = gemmi.Chain(name)
    res = gemmi.Residue()
    res.name = "LIG"
    res.seqid = _seqid(1)
    res.het_flag = "H"
    x0, y0, z0 = origin
    for k in range(n_atoms):
        res.add_atom(_atom(f"C{k + 1}", "C", (x0 + 1.4 * k, y0, z0), plddt))
    chain.add_residue(res)
    model.add_chain(chain)


def new_structure() -> tuple[gemmi.Structure, gemmi.Model]:
    st = gemmi.Structure()
    st.name = "synthetic"
    model = _new_model()
    return st, model


def finish_structure(st: gemmi.Structure, model: gemmi.Model, path: Path) -> None:
    st.add_model(model)
    st.setup_entities()
    if path.suffix == ".pdb":
        st.write_pdb(str(path))
    else:
        st.make_mmcif_document().write_file(str(path))


# ------------------------------------------------------------- tool fixtures


def write_colabfold_dir(
    directory: Path, n_a: int = 30, n_b: int = 30, job: str = "job"
) -> Path:
    """Write one ColabFold job; several jobs may share one directory."""
    directory.mkdir(parents=True, exist_ok=True)
    tag = "rank_001_alphafold2_multimer_v3_model_1_seed_000"
    st, model = new_structure()
    add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0))
    add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0))
    finish_structure(st, model, directory / f"{job}_unrelaxed_{tag}.pdb")

    n = n_a + n_b
    scores = {
        "plddt": [80.0] * n,
        "pae": np.full((n, n), 5.0).tolist(),
        "ptm": 0.78,
        "iptm": 0.66,
        "max_pae": 31.75,
        # recent ColabFold embeds its own interface scores (PAE cutoff 15)
        "ipsae": {"A-B": 0.91, "B-A": 0.90},
        "pdockq": {"A-B": 0.52},
        "pdockq2": {"A-B": 0.93, "B-A": 0.92},
    }
    (directory / f"{job}_scores_{tag}.json").write_text(json.dumps(scores))
    return directory


def write_af3_dir(
    directory: Path,
    n_a: int = 20,
    n_b: int = 15,
    n_lig: int = 3,
    cb_plddt: float | None = None,
) -> Path:
    """AlphaFold3 layout. ``cb_plddt`` makes the pLDDT vary within a residue,
    which only this parser preserves (it reads B-factors, not an npz)."""
    directory.mkdir(parents=True, exist_ok=True)
    st, model = new_structure()
    add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0), cb_plddt=cb_plddt)
    add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0), cb_plddt=cb_plddt)
    add_ligand_chain(model, "C", n_lig, origin=(0.0, 60.0, 0.0))
    finish_structure(st, model, directory / "model.cif")

    n = n_a + n_b + n_lig
    pae = np.full((n, n), 25.0)
    pae[: n_a + n_b, : n_a + n_b] = 5.0
    np.fill_diagonal(pae, 0.8)

    summary = {
        "ptm": 0.8,
        "iptm": 0.7,
        "ranking_score": 0.75,
        "chain_pair_iptm": [
            [0.0, 0.7, 0.2],
            [0.7, 0.0, 0.3],
            [0.2, 0.3, 0.0],
        ],
        "fraction_disordered": 0.05,
        "has_clash": 0,
    }
    (directory / "summary_confidences.json").write_text(json.dumps(summary))

    confidences = {
        "pae": pae.tolist(),
        "token_chain_ids": ["A"] * n_a + ["B"] * n_b + ["C"] * n_lig,
        "token_res_ids": list(range(1, n_a + 1))
        + list(range(1, n_b + 1))
        + [1] * n_lig,
    }
    (directory / "confidences.json").write_text(json.dumps(confidences))
    return directory


def write_af2_json_dir(directory: Path, n_a: int = 20, n_b: int = 20) -> Path:
    """AF2-Multimer JSON layout: iptm_ptm.json + confidence/pae JSONs + cif."""
    directory.mkdir(parents=True, exist_ok=True)
    tag = "model_1_multimer_v3_pred_0"
    st, model = new_structure()
    add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0))
    add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0))
    finish_structure(st, model, directory / f"unrelaxed_{tag}.cif")

    n = n_a + n_b
    (directory / f"confidence_{tag}.json").write_text(
        json.dumps(
            {
                "residueNumber": list(range(1, n + 1)),
                "confidenceScore": [77.0] * n,
                "confidenceCategory": ["H"] * n,
            }
        )
    )
    (directory / f"pae_{tag}.json").write_text(
        json.dumps(
            [
                {
                    "predicted_aligned_error": np.full((n, n), 4.0).tolist(),
                    "max_predicted_aligned_error": 31.75,
                }
            ]
        )
    )
    (directory / "iptm_ptm.json").write_text(
        json.dumps({tag: {"iptm": 0.9, "ptm": 0.91, "ranking_confidence": 0.902}})
    )
    (directory / "ranking_debug.json").write_text(
        json.dumps({"iptm+ptm": {tag: 0.902}, "order": [tag]})
    )
    return directory


def write_boltz_dir(
    directory: Path, n_a: int = 25, n_b: int = 20, affinity: bool = False
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    st, model = new_structure()
    add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0), plddt=0.0)
    add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0), plddt=0.0)
    finish_structure(st, model, directory / "job_model_0.cif")

    n = n_a + n_b
    confidence = {
        "confidence_score": 0.85,
        "ptm": 0.8,
        "iptm": 0.7,
        "complex_plddt": 0.83,
        "complex_iplddt": 0.8,
        "pair_chains_iptm": {"0": {"0": 0.8, "1": 0.7}, "1": {"0": 0.7, "1": 0.82}},
    }
    (directory / "confidence_job_model_0.json").write_text(json.dumps(confidence))
    np.savez(directory / "pae_job_model_0.npz", pae=np.full((n, n), 6.0))
    np.savez(directory / "plddt_job_model_0.npz", plddt=np.full(n, 0.83))
    if affinity:  # Boltz-2 affinity mode: one file per job
        (directory / "affinity_job.json").write_text(
            json.dumps(
                {
                    "affinity_pred_value": -1.23,
                    "affinity_probability_binary": 0.87,
                }
            )
        )
    return directory


def write_af2_native_dir(directory: Path, n_a: int = 15, n_b: int = 15) -> Path:
    """AlphaFold2-Multimer's real output directory.

    Five network weights (``model_1..5``) times the sampling index
    (``pred_N``) -- there is no seed dimension. Every model is written four
    times: ``unrelaxed_<tag>`` and the ranked copy ``ranked_<i>``, each as
    both ``.cif`` and ``.pdb``, so a naive scan counts four times too many.
    """
    import pickle

    directory.mkdir(parents=True, exist_ok=True)
    tags = [f"model_{i}_multimer_v3_pred_0" for i in range(1, 6)]
    n = n_a + n_b
    for rank, tag in enumerate(tags):
        st, model = new_structure()
        add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0))
        add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0))
        for stem in (f"unrelaxed_{tag}", f"ranked_{rank}"):
            for suffix in (".pdb", ".cif"):
                st2, model2 = new_structure()
                add_protein_chain(model2, "A", n_a, origin=(0.0, 0.0, 0.0))
                add_protein_chain(model2, "B", n_b, origin=(0.0, 5.0, 0.0))
                finish_structure(st2, model2, directory / f"{stem}{suffix}")
        with (directory / f"result_{tag}.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "plddt": np.full(n, 82.0),
                    "predicted_aligned_error": np.full((n, n), 4.0),
                    "ptm": 0.71,
                    "iptm": 0.63,
                    "ranking_confidence": 0.646,
                },
                fh,
            )
        (directory / f"confidence_{tag}.json").write_text(
            json.dumps({"confidenceScore": [82.0] * n})
        )
        (directory / f"pae_{tag}.json").write_text(
            json.dumps([{"predicted_aligned_error": np.full((n, n), 4.0).tolist()}])
        )
    (directory / "ranking_debug.json").write_text(
        json.dumps({"iptm+ptm": {tag: 0.65 for tag in tags}, "order": tags})
    )
    return directory
