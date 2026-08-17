"""Metric implementations.

Formulas and constants follow the published references and were checked
against the ipsae.py reference implementation (Dunbrack Lab):

- pDockQ:  Bryant, Pozzati & Elofsson, Nat. Commun. 13, 1265 (2022)
- pDockQ2: Zhu, Shenoy, Kundrotas & Elofsson, Bioinformatics 39, btad424 (2023)
- ipSAE:   Dunbrack, bioRxiv 10.1101/2025.02.10.637595 (2025)
- LIS:     Kim et al., bioRxiv 10.1101/2024.02.19.580970 (2024)

Do not change any constant here without a test and an updated citation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from foldmetrics.models import Prediction, infer_target

DEFAULT_PAE_CUTOFF = 10.0  # Å, ipSAE inclusion cutoff
DEFAULT_DIST_CUTOFF = 8.0  # Å, contact cutoff for pDockQ/pDockQ2/ipLDDT


# --------------------------------------------------------------------- pieces
def ptm_transform(pae: np.ndarray | float, d0: np.ndarray | float) -> np.ndarray | float:
    """TM-score kernel 1 / (1 + (PAE/d0)^2)."""
    return 1.0 / (1.0 + (np.asarray(pae, dtype=float) / d0) ** 2)


def calc_d0(n: float, nucleic: bool = False) -> float:
    """Scalar d0 (Yang & Skolnick 2004) as used by ipsae.py for chain-level n.

    Note the deliberate asymmetry with :func:`calc_d0_array`: here the TM
    formula applies only for n > 27, mirroring the reference implementation.
    """
    n = float(n)
    min_value = 2.0 if nucleic else 1.0
    d0 = 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8 if n > 27 else 1.0
    return max(min_value, d0)


def calc_d0_array(n: np.ndarray | float, nucleic: bool = False) -> np.ndarray:
    """Vectorized d0 with n clamped to >= 26, as used by ipsae.py per residue."""
    n = np.maximum(26.0, np.asarray(n, dtype=float))
    min_value = 2.0 if nucleic else 1.0
    return np.maximum(min_value, 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8)


def _contact_pairs(
    pred: Prediction, chain_a: str, chain_b: str, dist_cutoff: float
) -> tuple[np.ndarray, np.ndarray]:
    """Indices (into pred.tokens) of contact pairs between two chains.

    Contacts are measured between contact atoms (CB / GLY-CA / C3'),
    polymer tokens only, matching the pDockQ definition.
    """
    idx_a = pred.token_idx(chain_a, polymer_only=True)
    idx_b = pred.token_idx(chain_b, polymer_only=True)
    if idx_a.size == 0 or idx_b.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    diff = pred.coords[idx_a][:, None, :] - pred.coords[idx_b][None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    ii, jj = np.where(dist <= dist_cutoff)
    return idx_a[ii], idx_b[jj]


# -------------------------------------------------------------------- pDockQ
@dataclass
class PdockqResult:
    value: float
    n_pairs: int
    n_if_res: int
    mean_plddt: float


def pdockq(
    pred: Prediction,
    chain_a: str,
    chain_b: str,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> PdockqResult:
    """pDockQ (Bryant 2022). Symmetric in the two chains; 0.0 when no contacts."""
    ia, ib = _contact_pairs(pred, chain_a, chain_b, dist_cutoff)
    n_pairs = int(ia.size)
    if n_pairs == 0:
        return PdockqResult(0.0, 0, 0, float("nan"))
    unique = np.unique(np.concatenate([ia, ib]))
    mean_plddt = float(pred.cb_plddt_arr[unique].mean())
    x = mean_plddt * math.log10(n_pairs)
    value = 0.724 / (1.0 + math.exp(-0.052 * (x - 152.611))) + 0.018
    return PdockqResult(value, n_pairs, int(unique.size), mean_plddt)


def pdockq2_asym(
    pred: Prediction,
    chain_aligned: str,
    chain_scored: str,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> float:
    """pDockQ2 (Zhu 2023) for one direction (PAE rows from ``chain_aligned``).

    Returns NaN without a PAE matrix, 0.0 when the chains have no contacts.
    """
    if pred.pae is None:
        return float("nan")
    ia, ib = _contact_pairs(pred, chain_aligned, chain_scored, dist_cutoff)
    if ia.size == 0:
        return 0.0
    mean_ptm = float(np.mean(ptm_transform(pred.pae[ia, ib], 10.0)))
    unique = np.unique(np.concatenate([ia, ib]))
    mean_plddt = float(pred.cb_plddt_arr[unique].mean())
    x = mean_plddt * mean_ptm
    return 1.31 / (1.0 + math.exp(-0.075 * (x - 84.733))) + 0.005


# ----------------------------------------------------------------------- LIS
def lis_asym(pred: Prediction, chain_a: str, chain_b: str) -> float:
    """Local Interaction Score (Kim 2024) for PAE rows of ``chain_a``."""
    if pred.pae is None:
        return float("nan")
    idx_a = pred.token_idx(chain_a)
    idx_b = pred.token_idx(chain_b)
    if idx_a.size == 0 or idx_b.size == 0:
        return float("nan")
    block = pred.pae[np.ix_(idx_a, idx_b)]
    good = block[block < 12.0]
    if good.size == 0:
        return 0.0
    return float(np.mean((12.0 - good) / 12.0))


# --------------------------------------------------------------------- ipSAE
@dataclass
class IpsaeAsym:
    """One direction of ipSAE: ``chain_aligned`` rows scoring ``chain_scored``."""

    value: float  # max over aligned residues of the by-residue score
    best_token: int | None  # token index (into pred.tokens) achieving the max
    n0res: int  # n0res at the best residue
    d0res: float  # d0 at the best residue
    d0chn_value: float  # ipSAE_d0chn (fixed d0 from both chain lengths)
    iptm_d0chn: float  # PAE-derived ipTM (no PAE cutoff, d0 from chain lengths)
    byres: np.ndarray  # by-residue scores, aligned with idx_aligned
    idx_aligned: np.ndarray  # token indices of the aligned chain


def ipsae_asym(
    pred: Prediction,
    chain_aligned: str,
    chain_scored: str,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    polymer_only: bool = True,
) -> IpsaeAsym | None:
    """ipSAE (Dunbrack 2025) for one direction; None if a chain has no tokens.

    With ``polymer_only`` (the default, matching ipsae.py) ligand tokens are
    excluded; pass False to score ligand chains at token level (experimental).
    """
    if pred.pae is None:
        return None
    idx_a = pred.token_idx(chain_aligned, polymer_only=polymer_only)
    idx_b = pred.token_idx(chain_scored, polymer_only=polymer_only)
    if idx_a.size == 0 or idx_b.size == 0:
        return None
    nucleic = pred.is_nucleic_pair(chain_aligned, chain_scored)

    block = pred.pae[np.ix_(idx_a, idx_b)]
    valid = block < pae_cutoff
    n0res_byres = valid.sum(axis=1)
    d0res_byres = calc_d0_array(n0res_byres, nucleic)

    ptm_res = ptm_transform(block, d0res_byres[:, None])
    with np.errstate(invalid="ignore"):
        byres = np.where(
            n0res_byres > 0,
            (ptm_res * valid).sum(axis=1) / np.maximum(n0res_byres, 1),
            0.0,
        )

    d0chn = calc_d0(idx_a.size + idx_b.size, nucleic)
    ptm_chn = ptm_transform(block, d0chn)
    byres_chn = np.where(
        n0res_byres > 0,
        (ptm_chn * valid).sum(axis=1) / np.maximum(n0res_byres, 1),
        0.0,
    )
    iptm_byres = ptm_chn.mean(axis=1)

    best = int(np.argmax(byres))
    return IpsaeAsym(
        value=float(byres[best]),
        best_token=int(idx_a[best]),
        n0res=int(n0res_byres[best]),
        d0res=float(d0res_byres[best]),
        d0chn_value=float(np.max(byres_chn)),
        iptm_d0chn=float(np.max(iptm_byres)),
        byres=byres,
        idx_aligned=idx_a,
    )


# ---------------------------------------------------------------- aggregation
def _nan_stat(values: list[float], fn) -> float:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(fn(vals)) if vals else float("nan")


def pair_metrics(
    pred: Prediction,
    chain_a: str,
    chain_b: str,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> dict[str, Any]:
    """All interface metrics for one unordered chain pair."""
    kinds = pred.chain_kinds
    poly_a = pred.token_idx(chain_a, polymer_only=True).size
    poly_b = pred.token_idx(chain_b, polymer_only=True).size
    polymer_pair = poly_a > 0 and poly_b > 0

    row: dict[str, Any] = {
        "chain_a": chain_a,
        "chain_b": chain_b,
        "kind_a": kinds[chain_a],
        "kind_b": kinds[chain_b],
        "n_a": int(pred.token_idx(chain_a).size),
        "n_b": int(pred.token_idx(chain_b).size),
    }

    # Contact-based metrics: defined for polymer-polymer interfaces only.
    if polymer_pair:
        pq = pdockq(pred, chain_a, chain_b, dist_cutoff)
        row["n_contacts"] = pq.n_pairs
        row["n_if_res"] = pq.n_if_res
        row["iplddt"] = pq.mean_plddt
        row["pdockq"] = pq.value
        ab = pdockq2_asym(pred, chain_a, chain_b, dist_cutoff)
        ba = pdockq2_asym(pred, chain_b, chain_a, dist_cutoff)
        row["pdockq2_ab"], row["pdockq2_ba"] = ab, ba
        row["pdockq2"] = _nan_stat([ab, ba], max)
    else:
        row.update(
            n_contacts=0,
            n_if_res=0,
            iplddt=float("nan"),
            pdockq=float("nan"),
            pdockq2=float("nan"),
            pdockq2_ab=float("nan"),
            pdockq2_ba=float("nan"),
        )

    # PAE-based metrics. For pairs involving a pure-ligand chain, fall back
    # to token-level scoring (experimental; flagged in ipsae_mode).
    ipsae_mode = "residues" if polymer_pair else "tokens"
    res_ab = ipsae_asym(pred, chain_a, chain_b, pae_cutoff, polymer_only=polymer_pair)
    res_ba = ipsae_asym(pred, chain_b, chain_a, pae_cutoff, polymer_only=polymer_pair)
    row["ipsae_mode"] = ipsae_mode
    row["ipsae_ab"] = res_ab.value if res_ab else float("nan")
    row["ipsae_ba"] = res_ba.value if res_ba else float("nan")
    row["ipsae"] = _nan_stat([row["ipsae_ab"], row["ipsae_ba"]], max)
    row["ipsae_d0chn"] = _nan_stat(
        [
            res_ab.d0chn_value if res_ab else float("nan"),
            res_ba.d0chn_value if res_ba else float("nan"),
        ],
        max,
    )
    row["iptm_pae"] = _nan_stat(
        [
            res_ab.iptm_d0chn if res_ab else float("nan"),
            res_ba.iptm_d0chn if res_ba else float("nan"),
        ],
        max,
    )

    lis_ab = lis_asym(pred, chain_a, chain_b)
    lis_ba = lis_asym(pred, chain_b, chain_a)
    row["lis"] = _nan_stat([lis_ab, lis_ba], lambda v: sum(v) / len(v))

    if pred.pae is not None:
        idx_a = pred.token_idx(chain_a)
        idx_b = pred.token_idx(chain_b)
        blocks = np.concatenate(
            [
                pred.pae[np.ix_(idx_a, idx_b)].ravel(),
                pred.pae[np.ix_(idx_b, idx_a)].ravel(),
            ]
        )
        row["ipae_mean"] = float(blocks.mean())
        row["ipae_min"] = float(blocks.min())
    else:
        row["ipae_mean"] = float("nan")
        row["ipae_min"] = float("nan")

    iptm_native = pred.get_pair_iptm(chain_a, chain_b)
    row["iptm_native"] = iptm_native if iptm_native is not None else float("nan")
    return row


def compute_interfaces(
    pred: Prediction,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> list[dict[str, Any]]:
    """Interface metrics for every unordered chain pair."""
    return [
        pair_metrics(pred, a, b, pae_cutoff, dist_cutoff)
        for a, b in combinations(pred.chains, 2)
    ]


def compute_summary(
    pred: Prediction,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    interfaces: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """One summary row per model.

    For complexes with more than two chains, interface metrics (ipsae,
    pdockq, pdockq2, lis) report the best (maximum) interface; use
    :func:`compute_interfaces` for the per-pair breakdown.
    """
    if interfaces is None:
        interfaces = compute_interfaces(pred, pae_cutoff, dist_cutoff)

    summary: dict[str, Any] = {
        "model": pred.name,
        "tool": pred.tool,
        "target": infer_target(pred.source),
        "chains": ",".join(pred.chains),
        "n_chains": len(pred.chains),
        "n_tokens": pred.n_tokens,
        "n_res": int(pred.polymer_mask.sum()),
        "ptm": pred.ptm if pred.ptm is not None else float("nan"),
        "iptm": pred.iptm if pred.iptm is not None else float("nan"),
        "ranking_score": (
            pred.ranking_score if pred.ranking_score is not None else float("nan")
        ),
        "plddt_mean": float(pred.plddt_arr.mean()) if pred.n_tokens else float("nan"),
    }

    # Pooled interface pLDDT over every polymer-polymer contact residue.
    if_res: set[int] = set()
    for a, b in combinations(pred.chains, 2):
        ia, ib = _contact_pairs(pred, a, b, dist_cutoff)
        if_res.update(ia.tolist())
        if_res.update(ib.tolist())
    summary["iplddt"] = (
        float(pred.cb_plddt_arr[sorted(if_res)].mean()) if if_res else float("nan")
    )

    if pred.pae is not None and pred.n_tokens > 1:
        off_diag = ~np.eye(pred.n_tokens, dtype=bool)
        summary["pae_mean"] = float(pred.pae[off_diag].mean())
        inter = pred.chain_arr[:, None] != pred.chain_arr[None, :]
        summary["ipae_mean"] = (
            float(pred.pae[inter].mean()) if inter.any() else float("nan")
        )
    else:
        summary["pae_mean"] = float("nan")
        summary["ipae_mean"] = float("nan")

    for metric in ("ipsae", "pdockq", "pdockq2", "lis"):
        summary[metric] = _nan_stat([row[metric] for row in interfaces], max)
    summary["n_interfaces"] = sum(1 for row in interfaces if row["n_contacts"] > 0)
    summary["has_pae"] = pred.has_pae
    summary["source"] = str(pred.source)
    summary["warnings"] = "; ".join(pred.warnings)
    return summary


def compute_all(
    pred: Prediction,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summary row plus per-interface rows for one prediction."""
    interfaces = compute_interfaces(pred, pae_cutoff, dist_cutoff)
    summary = compute_summary(pred, pae_cutoff, dist_cutoff, interfaces=interfaces)
    for row in interfaces:
        row["model"] = pred.name
        row["tool"] = pred.tool
        row["target"] = summary["target"]
    return summary, interfaces
