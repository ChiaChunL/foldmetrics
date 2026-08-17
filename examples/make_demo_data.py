"""Generate synthetic AlphaFold3-style prediction folders to try foldmetrics.

No prediction tool needed: this writes three fake but realistically shaped
jobs (structures, pLDDT profiles, PAE matrices, summary confidences) under
the output directory, ready for::

    python examples/make_demo_data.py demo_predictions
    foldmetrics score demo_predictions -o metrics.tsv --interfaces interfaces.tsv --plot plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import numpy as np


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


def _atom(name: str, element: str, pos, b: float) -> gemmi.Atom:
    atom = gemmi.Atom()
    atom.name = name
    atom.element = gemmi.Element(element)
    atom.pos = gemmi.Position(*pos)
    atom.occ = 1.0
    atom.b_iso = float(b)
    return atom


def smooth_noise(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    raw = rng.normal(0.0, 1.0, n + 8)
    kernel = np.hanning(9)
    kernel /= kernel.sum()
    return np.convolve(raw, kernel, mode="valid")[:n] * scale


def plddt_profile(n: int, base: float, rng: np.random.Generator) -> np.ndarray:
    edge = np.minimum(np.arange(n), np.arange(n)[::-1])
    falloff = 18.0 * np.exp(-edge / 6.0)  # floppy termini
    profile = base - falloff + smooth_noise(n, 4.0, rng)
    return np.clip(profile, 30.0, 98.0)


def add_protein_chain(model, name, plddts, origin, spacing=3.8):
    chain = gemmi.Chain(name)
    x0, y0, z0 = origin
    for i, b in enumerate(plddts):
        res = gemmi.Residue()
        res.name = "ALA"
        res.seqid = _seqid(i + 1)
        res.het_flag = "A"
        x = x0 + i * spacing
        y = y0 + 1.5 * np.sin(i / 3.0)  # mild wiggle, keeps chains in contact
        res.add_atom(_atom("CA", "C", (x, y + 1.5, z0), b))
        res.add_atom(_atom("CB", "C", (x, y, z0), b))
        chain.add_residue(res)
    model.add_chain(chain)


def add_ligand_chain(model, name, n_atoms, origin, plddt):
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


def build_pae(sizes: list[int], inter_levels: dict[tuple[int, int], float],
              rng: np.random.Generator) -> np.ndarray:
    """Banded intra-chain PAE plus block-constant inter-chain PAE with noise."""
    n = sum(sizes)
    offsets = np.cumsum([0, *sizes])
    pae = np.zeros((n, n))
    for ci, si in enumerate(sizes):
        a0, a1 = offsets[ci], offsets[ci + 1]
        idx = np.arange(si)
        band = 1.0 + 0.06 * np.abs(idx[:, None] - idx[None, :])
        pae[a0:a1, a0:a1] = np.minimum(band, 7.0)
        for cj in range(ci + 1, len(sizes)):
            b0, b1 = offsets[cj], offsets[cj + 1]
            level = inter_levels.get((ci, cj), 20.0)
            block = level + rng.normal(0.0, level * 0.15, (si, sizes[cj]))
            pae[a0:a1, b0:b1] = block
            pae[b0:b1, a0:a1] = block.T + rng.normal(0.0, 0.8, (sizes[cj], si))
    pae += np.abs(rng.normal(0.0, 0.4, (n, n)))
    np.fill_diagonal(pae, 0.2)
    return np.clip(pae, 0.1, 31.75)


def write_job(root: Path, name: str, protein_sizes: list[int], plddt_bases: list[float],
              n_ligand_atoms: int, inter_levels: dict[tuple[int, int], float],
              summary: dict, seed: int) -> None:
    rng = np.random.default_rng(seed)
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)

    st = gemmi.Structure()
    st.name = name
    model = _new_model()

    n_chains = len(protein_sizes) + (1 if n_ligand_atoms else 0)
    chain_names = [chr(ord("A") + i) for i in range(n_chains)]
    profiles = [plddt_profile(s, b, rng) for s, b in zip(protein_sizes, plddt_bases, strict=True)]
    for i, profile in enumerate(profiles):
        add_protein_chain(model, chain_names[i], profile, origin=(0.0, 5.0 * i, 0.0))
    sizes = list(protein_sizes)
    token_plddts = list(np.concatenate(profiles))
    if n_ligand_atoms:
        lig_chain = chain_names[len(protein_sizes)]
        lig_plddt = 72.0
        add_ligand_chain(model, lig_chain, n_ligand_atoms,
                         origin=(4.0, -6.0, 0.0), plddt=lig_plddt)
        sizes.append(n_ligand_atoms)
        token_plddts.extend([lig_plddt] * n_ligand_atoms)

    st.add_model(model)
    st.setup_entities()
    st.make_mmcif_document().write_file(str(directory / "model.cif"))

    pae = build_pae(sizes, inter_levels, rng)
    token_chain_ids: list[str] = []
    token_res_ids: list[int] = []
    for chain_name, size, is_ligand in zip(
        chain_names, sizes, [False] * len(protein_sizes) + [bool(n_ligand_atoms)], strict=False
    ):
        token_chain_ids.extend([chain_name] * size)
        token_res_ids.extend([1] * size if is_ligand else list(range(1, size + 1)))

    (directory / "summary_confidences.json").write_text(json.dumps(summary, indent=1))
    (directory / "confidences.json").write_text(
        json.dumps(
            {
                "pae": np.round(pae, 2).tolist(),
                "token_chain_ids": token_chain_ids,
                "token_res_ids": token_res_ids,
            }
        )
    )
    print(f"wrote {directory} ({sum(sizes)} tokens, {len(sizes)} chains)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", nargs="?", default="demo_predictions", type=Path)
    args = parser.parse_args()

    write_job(
        args.out, "dimer_good",
        protein_sizes=[110, 85], plddt_bases=[92.0, 90.0], n_ligand_atoms=0,
        inter_levels={(0, 1): 4.0},
        summary={
            "ptm": 0.87, "iptm": 0.82, "ranking_score": 0.84,
            "chain_pair_iptm": [[0.0, 0.82], [0.82, 0.0]],
            "fraction_disordered": 0.03, "has_clash": 0,
        },
        seed=1,
    )
    write_job(
        args.out, "receptor_peptide_medium",
        protein_sizes=[140, 30], plddt_bases=[88.0, 68.0], n_ligand_atoms=0,
        inter_levels={(0, 1): 11.0},
        summary={
            "ptm": 0.72, "iptm": 0.55, "ranking_score": 0.58,
            "chain_pair_iptm": [[0.0, 0.55], [0.55, 0.0]],
            "fraction_disordered": 0.12, "has_clash": 0,
        },
        seed=2,
    )
    write_job(
        args.out, "complex_with_ligand_poor",
        protein_sizes=[95, 75], plddt_bases=[80.0, 62.0], n_ligand_atoms=10,
        inter_levels={(0, 1): 21.0, (0, 2): 6.0, (1, 2): 24.0},
        summary={
            "ptm": 0.68, "iptm": 0.38, "ranking_score": 0.45,
            "chain_pair_iptm": [
                [0.0, 0.34, 0.71],
                [0.34, 0.0, 0.22],
                [0.71, 0.22, 0.0],
            ],
            "fraction_disordered": 0.2, "has_clash": 0,
        },
        seed=3,
    )


if __name__ == "__main__":
    main()
