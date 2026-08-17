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
) -> None:
    chain = gemmi.Chain(name)
    x0, y0, z0 = origin
    for i in range(n_res):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = _seqid(i + 1)
        res.het_flag = "A"
        res.add_atom(_atom("CA", "C", (x0 + i * spacing, y0 + 1.5, z0), plddt))
        res.add_atom(_atom("CB", "C", (x0 + i * spacing, y0, z0), plddt))
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


def write_colabfold_dir(directory: Path, n_a: int = 30, n_b: int = 30) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    tag = "rank_001_alphafold2_multimer_v3_model_1_seed_000"
    st, model = new_structure()
    add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0))
    add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0))
    finish_structure(st, model, directory / f"job_unrelaxed_{tag}.pdb")

    n = n_a + n_b
    scores = {
        "plddt": [80.0] * n,
        "pae": np.full((n, n), 5.0).tolist(),
        "ptm": 0.78,
        "iptm": 0.66,
        "max_pae": 31.75,
    }
    (directory / f"job_scores_{tag}.json").write_text(json.dumps(scores))
    return directory


def write_af3_dir(directory: Path, n_a: int = 20, n_b: int = 15, n_lig: int = 3) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    st, model = new_structure()
    add_protein_chain(model, "A", n_a, origin=(0.0, 0.0, 0.0))
    add_protein_chain(model, "B", n_b, origin=(0.0, 5.0, 0.0))
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


def write_boltz_dir(directory: Path, n_a: int = 25, n_b: int = 20) -> Path:
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
    return directory
