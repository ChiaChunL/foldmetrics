"""Structure-file tokenization shared by all parsers.

Follows AF3-style tokenization so that token lists line up with token-level
PAE matrices: one token per standard polymer residue, one token per heavy
atom for ligands and non-standard residues.

Representative atoms mirror ipsae.py (Dunbrack Lab): the anchor atom is CA
(protein) or C1' (nucleic); the contact atom used for distance-based metrics
is CB (CA for GLY) or C3'.
"""

from __future__ import annotations

from pathlib import Path

import gemmi

from foldmetrics.models import Token

PROTEIN_RES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
DNA_RES = frozenset({"DA", "DC", "DG", "DT"})
RNA_RES = frozenset({"A", "C", "G", "U"})
WATER_RES = frozenset({"HOH", "WAT", "DOD"})


def _find_atom(atoms: list[gemmi.Atom], name: str) -> gemmi.Atom | None:
    for atom in atoms:
        if atom.name == name:
            return atom
    return None


def _xyz(atom: gemmi.Atom) -> tuple[float, float, float]:
    return (atom.pos.x, atom.pos.y, atom.pos.z)


def tokenize_structure(path: str | Path) -> list[Token]:
    """Read a PDB/mmCIF file and produce the normalized token list.

    B-factors are taken as pLDDT values (the convention of every supported
    tool); callers should pass the result through :func:`autoscale_plddt`.
    """
    structure = gemmi.read_structure(str(path))
    if len(structure) == 0:
        raise ValueError(f"no models found in structure file: {path}")
    model = structure[0]

    tokens: list[Token] = []
    for chain in model:
        for res in chain:
            res_name = res.name.strip().upper()
            if res_name in WATER_RES:
                continue
            heavy = [
                a for a in res if not a.is_hydrogen() and a.altloc in ("", "A", "\x00")
            ]
            if not heavy:
                continue

            if res_name in PROTEIN_RES:
                ca = _find_atom(heavy, "CA") or heavy[0]
                cb = ca if res_name == "GLY" else (_find_atom(heavy, "CB") or ca)
                tokens.append(
                    Token(
                        chain=chain.name,
                        res_id=res.seqid.num,
                        res_name=res_name,
                        kind="protein",
                        atom_name=cb.name,
                        xyz=_xyz(cb),
                        anchor_xyz=_xyz(ca),
                        plddt=ca.b_iso,
                        cb_plddt=cb.b_iso,
                    )
                )
            elif res_name in DNA_RES or res_name in RNA_RES:
                kind = "dna" if res_name in DNA_RES else "rna"
                anchor = _find_atom(heavy, "C1'") or heavy[0]
                contact = _find_atom(heavy, "C3'") or anchor
                tokens.append(
                    Token(
                        chain=chain.name,
                        res_id=res.seqid.num,
                        res_name=res_name,
                        kind=kind,
                        atom_name=contact.name,
                        xyz=_xyz(contact),
                        anchor_xyz=_xyz(anchor),
                        plddt=anchor.b_iso,
                        cb_plddt=contact.b_iso,
                    )
                )
            else:
                for atom in heavy:
                    tokens.append(
                        Token(
                            chain=chain.name,
                            res_id=res.seqid.num,
                            res_name=res_name,
                            kind="ligand",
                            atom_name=atom.name,
                            xyz=_xyz(atom),
                            anchor_xyz=_xyz(atom),
                            plddt=atom.b_iso,
                            cb_plddt=atom.b_iso,
                        )
                    )
    if not tokens:
        raise ValueError(f"no usable residues/atoms found in structure file: {path}")
    return tokens


def autoscale_plddt(tokens: list[Token]) -> float:
    """Convert 0-1 scaled pLDDT to 0-100 in place; return the factor applied."""
    max_val = 0.0
    for t in tokens:
        max_val = max(max_val, t.plddt, t.cb_plddt)
    if 0.0 < max_val <= 1.05:
        for t in tokens:
            t.plddt *= 100.0
            t.cb_plddt *= 100.0
        return 100.0
    return 1.0
