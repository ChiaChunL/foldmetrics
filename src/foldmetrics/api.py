"""High-level DataFrame API."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from foldmetrics.metrics import (
    DEFAULT_DIST_CUTOFF,
    DEFAULT_PAE_CUTOFF,
    compute_all,
)
from foldmetrics.models import Prediction
from foldmetrics.parsers import load_predictions

Source = str | Path | Prediction | Iterable[str | Path | Prediction]

SUMMARY_COLUMNS = [
    "model", "tool", "target", "chains", "n_chains", "n_tokens", "n_res",
    "ptm", "iptm", "ranking_score", "plddt_mean", "iplddt",
    "pae_mean", "ipae_mean", "ipsae", "pdockq", "pdockq2", "lis",
    "n_interfaces", "has_pae", "source", "warnings",
]

INTERFACE_COLUMNS = [
    "model", "tool", "target", "chain_a", "chain_b", "kind_a", "kind_b", "n_a", "n_b",
    "n_contacts", "n_if_res", "iplddt", "pdockq",
    "pdockq2", "pdockq2_ab", "pdockq2_ba",
    "ipsae", "ipsae_ab", "ipsae_ba", "ipsae_d0chn", "iptm_pae",
    "lis", "iptm_native", "ipae_mean", "ipae_min", "ipae_max", "ipsae_mode",
]


def _as_predictions(source: Source, tool: str, on_error: str) -> list[Prediction]:
    if isinstance(source, Prediction):
        return [source]
    if isinstance(source, (str, Path)):
        return load_predictions(source, tool=tool, on_error=on_error)

    predictions: list[Prediction] = []
    paths: list[str | Path] = []
    for item in source:
        if isinstance(item, Prediction):
            predictions.append(item)
        else:
            paths.append(item)
    if paths:
        predictions.extend(load_predictions(paths, tool=tool, on_error=on_error))
    return predictions


def evaluate_full(
    source: Source,
    tool: str = "auto",
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    on_error: str = "raise",
    pdockq2_variant: str = "consensus",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summary and per-interface DataFrames for the given predictions.

    ``source`` may be paths (files or directories, scanned recursively),
    already-loaded :class:`Prediction` objects, or a mix.
    """
    summaries = []
    interface_rows: list[dict] = []
    for pred in _as_predictions(source, tool, on_error):
        summary, interfaces = compute_all(
            pred, pae_cutoff, dist_cutoff, pdockq2_variant
        )
        summaries.append(summary)
        interface_rows.extend(interfaces)

    df_summary = pd.DataFrame(summaries).reindex(columns=SUMMARY_COLUMNS)
    df_interfaces = pd.DataFrame(interface_rows).reindex(columns=INTERFACE_COLUMNS)
    return df_summary, df_interfaces


def evaluate(
    source: Source,
    tool: str = "auto",
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    on_error: str = "raise",
    pdockq2_variant: str = "consensus",
) -> pd.DataFrame:
    """One row of metrics per model."""
    return evaluate_full(
        source, tool, pae_cutoff, dist_cutoff, on_error, pdockq2_variant
    )[0]


def evaluate_interfaces(
    source: Source,
    tool: str = "auto",
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    on_error: str = "raise",
    pdockq2_variant: str = "consensus",
) -> pd.DataFrame:
    """One row of metrics per chain pair per model."""
    return evaluate_full(
        source, tool, pae_cutoff, dist_cutoff, on_error, pdockq2_variant
    )[1]


AGGREGATE_COLUMNS = [
    "target", "tool", "n_models", "best_model",
    "iptm_mean", "iptm_std", "iptm_max",
    "ipsae_mean", "ipsae_std", "ipsae_max",
    "pdockq2_max", "plddt_mean",
]


def aggregate_by_target(df: pd.DataFrame) -> pd.DataFrame:
    """Campaign/ensemble view: one row per (target, tool) over all its models.

    Collapses a summary DataFrame (multiple seeds/samples of the same
    complex) into mean/std/max statistics plus the best model per group,
    ranked by ranking_score (falling back to ipSAE, then ipTM).
    """
    rows = []
    for (target, tool), group in df.groupby(["target", "tool"], dropna=False):
        order = (
            group[["ranking_score", "ipsae", "iptm"]]
            .bfill(axis=1).iloc[:, 0].fillna(0.0)
        )
        rows.append(
            {
                "target": target,
                "tool": tool,
                "n_models": len(group),
                "best_model": group.loc[order.idxmax(), "model"],
                "iptm_mean": group["iptm"].mean(),
                "iptm_std": group["iptm"].std(),
                "iptm_max": group["iptm"].max(),
                "ipsae_mean": group["ipsae"].mean(),
                "ipsae_std": group["ipsae"].std(),
                "ipsae_max": group["ipsae"].max(),
                "pdockq2_max": group["pdockq2"].max(),
                "plddt_mean": group["plddt_mean"].mean(),
            }
        )
    return (
        pd.DataFrame(rows, columns=AGGREGATE_COLUMNS)
        .sort_values(["target", "tool"])
        .reset_index(drop=True)
    )
