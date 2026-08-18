"""Normalized data model shared by all parsers and metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

# Path components that only describe *which run* a model came from, never
# which complex it is: engine output trees interleave them freely
# (``seed_2066/predictions/``, ``seed-1_sample-0/``, ``pred0/``, ``ranked/``,
# ``trunk_0/`` from Chai-1 with several trunk samples).
_RUN_WORDS = frozenset(
    {
        "seed", "seeds", "sample", "samples", "pred", "preds",
        "prediction", "predictions", "output", "outputs", "outs", "out",
        "result", "results", "rank", "ranked", "ranking",
        "model", "models", "idx", "run", "runs", "fold", "folds",
        "trunk", "trunks", "diffn", "recycle",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")

# AlphaFold3 does not overwrite a non-empty output directory: it appends a
# timestamp (run_alphafold.py, ``<name>_%Y%m%d_%H%M%S``), so re-running a job
# would otherwise register as a separate target.
_TIMESTAMP_SUFFIX_RE = re.compile(r"_\d{8}_\d{6}$")


def _is_run_component(name: str) -> bool:
    """True when every word in ``name`` is a run descriptor or a number.

    Handles the flat (``seed318``, ``predictions``) and the compound
    (``seed-1_sample-0``, ``model_idx_0``) spellings alike, while leaving
    real job names such as ``P0__P1`` or ``boltz_results_boltz2`` alone.
    """
    tokens = _TOKEN_RE.findall(name)
    if not tokens:
        return False
    saw_word = False
    for token in tokens:
        if token.isdigit():
            continue
        if token.lower() in _RUN_WORDS:
            saw_word = True
            continue
        return False
    return saw_word


def _strip_run_suffixes(stem: str) -> str:
    """Drop trailing run descriptors from a file stem: ``job_sample_0`` -> ``job``.

    A doubled separator marks a field boundary inside job names
    (``P53__MDM2``), so stripping never crosses one: ``model__sample_0``
    keeps both halves even though each is spelled like a run word.
    """
    while True:
        match = re.search(r"([._-]+)([A-Za-z]*)(\d*)$", stem)
        if match is None:
            return stem
        separator, word, number = match.groups()
        if len(separator) > 1:
            return stem
        if not word and not number:
            return stem
        if word and word.lower() not in _RUN_WORDS:
            return stem
        stem = stem[: match.start()]


def infer_target(source: str | Path) -> str:
    """Best-effort target (job) name for a prediction file.

    Walks up from the structure file, skipping directories that only
    describe the run (``seed_2066``, ``predictions``, ``sample-1``,
    ``ranked``), so that the same complex predicted by different tools —
    and stored in each engine's own output tree — maps to one label.

    When every directory is a run descriptor (e.g. scoring straight from a
    ``predictions/`` folder), the file name is used instead:
    ``P0__P1_sample_0.cif`` -> ``P0__P1``.
    """
    path = Path(source)
    for part in reversed(path.parent.parts):
        if not _is_run_component(part):
            return _TIMESTAMP_SUFFIX_RE.sub("", part) or part

    stem = _strip_run_suffixes(path.stem)
    return stem or path.parent.name or "unknown"


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
    cb_plddt: float  # contact-atom pLDDT, used by pDockQ/ipLDDT
    # anchor atom (CA / C1' / the atom itself); None means "same as xyz"
    anchor_xyz: tuple[float, float, float] | None = None


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
    # set by parsers whose file naming states the job explicitly; takes
    # precedence over the path heuristic in :func:`infer_target`
    target: str | None = None
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
    def anchor_coords(self) -> np.ndarray:
        """CA / C1' positions, falling back to the contact atom."""
        return np.array(
            [t.anchor_xyz if t.anchor_xyz is not None else t.xyz for t in self.tokens],
            dtype=float,
        ).reshape(self.n_tokens, 3)

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
