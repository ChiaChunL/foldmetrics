"""Normalized data model shared by all parsers and metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

_STRIP_DIR_RE = re.compile(r"^(seed|sample|pred|rank|ranked|model)[-_]?\d*$", re.IGNORECASE)


def infer_target(source: str | Path) -> str:
    """Best-effort target (job) name from a prediction's directory path.

    Walks up from the structure file's directory, skipping generic run
    components like ``seed318``, ``pred0``, ``sample-1`` or ``ranked``, so
    that the same complex predicted by different tools (usually organized
    as ``<tool>/<target>/<seed...>``) maps to one target label.
    """
    parts = Path(source).parent.parts
    for part in reversed(parts):
        if not _STRIP_DIR_RE.match(part):
            return part
    return Path(source).parent.name or "unknown"

POLYMER_KINDS = frozenset({"protein", "dna", "rna"})
NUCLEIC_KINDS = frozenset({"dna", "rna"})


@dataclass
class Token:
    """One scoring unit, following AF3-style tokenization.

    Standard polymer residues contribute one token each; ligands and
    non-standard residues contribute one token per heavy atom.
    """

    chain: str
    res_id: int
    res_name: str
    kind: str  # "protein" | "dna" | "rna" | "ligand"
    atom_name: str  # contact atom the coordinates refer to (CB / C3' / the atom itself)
    xyz: tuple[float, float, float]
    plddt: float  # anchor-atom pLDDT (CA / C1' / the atom itself), 0-100 scale
    cb_plddt: float  # contact-atom pLDDT, used by pDockQ/pDockQ2/ipLDDT


@dataclass
class Prediction:
    """A single predicted model, normalized across prediction tools.

    ``pae[i, j]`` is the expected position error (Å) of token *j* when the
    predicted and true structures are aligned on token *i*.
    """

    name: str
    tool: str
    source: Path
    tokens: list[Token]
    pae: np.ndarray | None = None
    ptm: float | None = None
    iptm: float | None = None
    ranking_score: float | None = None
    chain_pair_iptm: dict[tuple[str, str], float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pae is not None:
            pae = np.asarray(self.pae, dtype=float)
            n = len(self.tokens)
            if pae.ndim != 2 or pae.shape != (n, n):
                self.warnings.append(
                    f"PAE shape {pae.shape} does not match {n} tokens; "
                    "PAE-based metrics disabled"
                )
                self.pae = None
            else:
                self.pae = pae

    # ------------------------------------------------------------------ views
    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    @property
    def has_pae(self) -> bool:
        return self.pae is not None

    @cached_property
    def chain_arr(self) -> np.ndarray:
        return np.array([t.chain for t in self.tokens], dtype=object)

    @cached_property
    def chains(self) -> list[str]:
        seen: dict[str, None] = {}
        for t in self.tokens:
            seen.setdefault(t.chain)
        return list(seen)

    @cached_property
    def coords(self) -> np.ndarray:
        return np.array([t.xyz for t in self.tokens], dtype=float).reshape(self.n_tokens, 3)

    @cached_property
    def plddt_arr(self) -> np.ndarray:
        return np.array([t.plddt for t in self.tokens], dtype=float)

    @cached_property
    def cb_plddt_arr(self) -> np.ndarray:
        return np.array([t.cb_plddt for t in self.tokens], dtype=float)

    @cached_property
    def kind_arr(self) -> np.ndarray:
        return np.array([t.kind for t in self.tokens], dtype=object)

    @cached_property
    def polymer_mask(self) -> np.ndarray:
        return np.array([t.kind in POLYMER_KINDS for t in self.tokens], dtype=bool)

    @cached_property
    def chain_kinds(self) -> dict[str, str]:
        """Chain classification mirroring ipsae.py: nucleic wins over protein."""
        out: dict[str, str] = {}
        for c in self.chains:
            kinds = {t.kind for t in self.tokens if t.chain == c}
            if kinds & NUCLEIC_KINDS:
                out[c] = "nucleic"
            elif "protein" in kinds:
                out[c] = "protein"
            else:
                out[c] = "ligand"
        return out

    # ------------------------------------------------------------- selections
    def token_idx(self, chain: str, polymer_only: bool = False) -> np.ndarray:
        mask = self.chain_arr == chain
        if polymer_only:
            mask = mask & self.polymer_mask
        return np.where(mask)[0]

    def is_nucleic_pair(self, chain_a: str, chain_b: str) -> bool:
        kinds = self.chain_kinds
        return kinds[chain_a] == "nucleic" or kinds[chain_b] == "nucleic"

    def get_pair_iptm(self, chain_a: str, chain_b: str) -> float | None:
        for key in ((chain_a, chain_b), (chain_b, chain_a)):
            value = self.chain_pair_iptm.get(key)
            if value is not None:
                return float(value)
        return None
