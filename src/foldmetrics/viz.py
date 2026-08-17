"""Matplotlib figures: structure panel, pLDDT track, PAE heatmap, batch views.

Figure inventory (all saved by the CLI according to the shape of the input):

- :func:`plot_summary` — one model: pLDDT-colored 3D structure (PyMOL render
  when available, backbone trace otherwise), pLDDT track, PAE heatmap and a
  metrics panel.
- :func:`plot_batch` — many models: ranked dot plot of 0-1 scores plus mean
  pLDDT bars.
- :func:`plot_comparison` — targets x methods: one panel per metric, grouped
  by target with one color per prediction tool.

Chain colors use a colorblind-safe categorical palette assigned in fixed
order; chain identity is always also carried by direct text labels. PAE uses
the community-standard single-hue green scale; pLDDT uses the AlphaFold
confidence colors.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from foldmetrics.metrics import (
    DEFAULT_DIST_CUTOFF,
    DEFAULT_PAE_CUTOFF,
    compute_all,
)
from foldmetrics.models import Prediction

CHAIN_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"]
FALLBACK_CHAIN_COLOR = "#808080"
INK = "#333333"
MUTED = "#777777"

# AlphaFold confidence bands (status colors; always shown with their labels).
PLDDT_BANDS = [
    (90.0, 100.0, "#0053D6", "very high"),
    (70.0, 90.0, "#65CBF3", "confident"),
    (50.0, 70.0, "#FFDB13", "low"),
    (0.0, 50.0, "#FF7D45", "very low"),
]

PAE_CMAP = "Greens_r"
PAE_VMAX = 31.75  # Å, AlphaFold display convention

METRIC_LABELS = {
    "ptm": "pTM",
    "iptm": "ipTM",
    "ranking_score": "ranking",
    "ipsae": "ipSAE",
    "pdockq": "pDockQ",
    "pdockq2": "pDockQ2",
    "lis": "LIS",
    "plddt_mean": "mean pLDDT",
    "iplddt": "ipLDDT",
}


def chain_color(i: int) -> str:
    return CHAIN_COLORS[i] if i < len(CHAIN_COLORS) else FALLBACK_CHAIN_COLOR


def _plddt_color(value: float) -> str:
    for low, _, color, _ in PLDDT_BANDS:
        if value >= low:
            return color
    return PLDDT_BANDS[-1][2]


def _chain_segments(pred: Prediction) -> list[tuple[str, int, int]]:
    """Consecutive same-chain runs as (chain, start, stop) token ranges."""
    segments: list[tuple[str, int, int]] = []
    start = 0
    for i in range(1, pred.n_tokens + 1):
        if i == pred.n_tokens or pred.tokens[i].chain != pred.tokens[start].chain:
            segments.append((pred.tokens[start].chain, start, i))
            start = i
    return segments


# ------------------------------------------------------------- single panels
def plot_plddt(pred: Prediction, ax: plt.Axes) -> None:
    """Per-token pLDDT, one colored segment per chain, with confidence bands."""
    for low, high, color, label in PLDDT_BANDS:
        ax.axhspan(low, high, color=color, alpha=0.12, lw=0, zorder=0)
        ax.text(
            1.01,
            (low + high) / 2.0,
            label,
            transform=ax.get_yaxis_transform(),
            fontsize=7,
            color=MUTED,
            va="center",
        )

    color_of = {c: chain_color(i) for i, c in enumerate(pred.chains)}
    plddt = pred.plddt_arr
    for chain, start, stop in _chain_segments(pred):
        xs = np.arange(start, stop)
        ys = plddt[start:stop]
        # AlphaFold-style: the curve itself is colored by the confidence band.
        if len(xs) >= 2:
            points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            seg_colors = [
                _plddt_color((ys[k] + ys[k + 1]) / 2.0) for k in range(len(xs) - 1)
            ]
            ax.add_collection(
                LineCollection(
                    segments, colors=seg_colors, linewidths=1.7,
                    zorder=2, capstyle="round",
                )
            )
        else:
            ax.scatter(xs, ys, s=12, color=_plddt_color(float(ys[0])), zorder=2)
        ax.text(
            (start + stop - 1) / 2.0,
            104.0,
            chain,
            ha="center",
            va="bottom",
            fontsize=8,
            color=color_of[chain],
            clip_on=False,
        )
        if start > 0:
            ax.axvline(start - 0.5, color="#cccccc", lw=0.8, ls="--", zorder=1)

    ax.set_xlim(-0.5, pred.n_tokens - 0.5)
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 50, 70, 90, 100])
    ax.set_xlabel("Token", color=INK, fontsize=9)
    ax.set_ylabel("pLDDT", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_pae(pred: Prediction, ax: plt.Axes, colorbar: bool = True) -> Any:
    """PAE heatmap: numeric token axes, chain borders, chain labels on top."""
    if pred.pae is None:
        raise ValueError(f"prediction {pred.name!r} has no PAE matrix")
    im = ax.imshow(
        pred.pae,
        cmap=PAE_CMAP,
        vmin=0.0,
        vmax=max(PAE_VMAX, float(np.nanmax(pred.pae))),
        origin="upper",
        interpolation="nearest",
    )
    segments = _chain_segments(pred)
    color_of = {c: chain_color(i) for i, c in enumerate(pred.chains)}
    for _, start, _ in segments[1:]:
        ax.axvline(start - 0.5, color="#555555", lw=0.7)
        ax.axhline(start - 0.5, color="#555555", lw=0.7)
    for chain, start, stop in segments:
        ax.text(
            (start + stop - 1) / 2.0,
            1.015,
            chain,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=color_of[chain],
            clip_on=False,
        )
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.set_xlabel("Scored token", color=INK, fontsize=9)
    ax.set_ylabel("Aligned token", color=INK, fontsize=9)
    ax.tick_params(colors=INK, labelsize=8, length=2)
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Expected position error (Å)", color=INK, fontsize=8)
        cbar.ax.tick_params(colors=INK, labelsize=7)
    return im


def plot_structure_trace(pred: Prediction, ax: Any) -> None:
    """Backbone/token trace colored by pLDDT on a 3D axis (PyMOL fallback)."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    coords = pred.coords
    color_of = {c: chain_color(i) for i, c in enumerate(pred.chains)}
    for chain, start, stop in _chain_segments(pred):
        xyz = coords[start:stop]
        if len(xyz) >= 2:
            segs = np.stack([xyz[:-1], xyz[1:]], axis=1)
            colors = [_plddt_color(v) for v in pred.plddt_arr[start : stop - 1]]
            ax.add_collection3d(Line3DCollection(segs, colors=colors, linewidths=2.0))
        else:
            ax.scatter(*xyz.T, color=_plddt_color(pred.plddt_arr[start]), s=14)
        center = xyz.mean(axis=0)
        ax.text(*center, chain, fontsize=9, color=color_of[chain])

    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    half = float((maxs - mins).max()) / 2.0 or 1.0
    mid = (maxs + mins) / 2.0
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _structure_panel(pred: Prediction, fig: plt.Figure, slot: Any, renderer: str) -> None:
    """Fill a gridspec slot with the best available structure depiction."""
    image = None
    if renderer in ("auto", "pymol"):
        from foldmetrics.render import render_structure

        tmp = Path(tempfile.mkstemp(suffix=".png")[1])
        try:
            if render_structure(pred.source, tmp) is not None:
                image = plt.imread(str(tmp))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if image is not None:
        ax = fig.add_subplot(slot)
        ax.imshow(image)
        ax.set_axis_off()
    else:
        ax = fig.add_subplot(slot, projection="3d")
        plot_structure_trace(pred, ax)
    ax.set_title("Structure (pLDDT colors)", fontsize=8, color=MUTED, pad=2)


# --------------------------------------------------------------- summary fig
def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(v):
        return "—"
    return f"{v:.{digits}f}"


def _metrics_text(summary: dict, interfaces: list[dict]) -> str:
    lines = [
        f"tool         {summary['tool']}",
        f"chains       {summary['chains']}",
        "",
        f"pTM          {_fmt(summary['ptm'])}",
        f"ipTM         {_fmt(summary['iptm'])}",
        f"ranking      {_fmt(summary['ranking_score'])}",
        f"mean pLDDT   {_fmt(summary['plddt_mean'], 1)}",
        f"ipLDDT       {_fmt(summary['iplddt'], 1)}",
        f"mean PAE     {_fmt(summary['pae_mean'], 2)}",
        f"mean iPAE    {_fmt(summary['ipae_mean'], 2)}",
        "",
        f"ipSAE        {_fmt(summary['ipsae'])}",
        f"pDockQ       {_fmt(summary['pdockq'])}",
        f"pDockQ2      {_fmt(summary['pdockq2'])}",
        f"LIS          {_fmt(summary['lis'])}",
    ]
    if len(interfaces) > 1:
        lines += ["", "interface    ipSAE   pDockQ"]
        ranked = sorted(
            interfaces,
            key=lambda r: -1.0 if np.isnan(r["ipsae"]) else r["ipsae"],
            reverse=True,
        )
        for row in ranked[:8]:
            pair = f"{row['chain_a']}-{row['chain_b']}"
            lines.append(
                f"{pair:<12} {_fmt(row['ipsae']):>6}  {_fmt(row['pdockq']):>6}"
            )
    return "\n".join(lines)


def plot_summary(
    pred: Prediction,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    renderer: str = "auto",
) -> plt.Figure:
    """One-page summary: structure, pLDDT track, PAE heatmap, metrics panel.

    ``renderer``: "auto"/"pymol" try a headless PyMOL cartoon first and fall
    back to a matplotlib backbone trace; "trace" always uses the trace.
    """
    summary, interfaces = compute_all(pred, pae_cutoff, dist_cutoff)

    if pred.has_pae:
        fig = plt.figure(figsize=(13.2, 8.2))
        gs = fig.add_gridspec(
            2, 3, height_ratios=[1.0, 3.0], width_ratios=[1.3, 1.3, 0.75],
            hspace=0.34, wspace=0.22,
        )
        ax_plddt = fig.add_subplot(gs[0, :2])
        ax_text = fig.add_subplot(gs[:, 2])
        ax_pae = fig.add_subplot(gs[1, 1])
        plot_pae(pred, ax_pae)
        structure_slot = gs[1, 0]
    else:
        fig = plt.figure(figsize=(11.5, 7.0))
        gs = fig.add_gridspec(
            2, 2, height_ratios=[1.0, 2.5], width_ratios=[1.5, 1.0],
            hspace=0.38, wspace=0.15,
        )
        ax_plddt = fig.add_subplot(gs[0, :])
        ax_text = fig.add_subplot(gs[1, 1])
        structure_slot = gs[1, 0]

    _structure_panel(pred, fig, structure_slot, renderer)
    plot_plddt(pred, ax_plddt)

    ax_text.axis("off")
    ax_text.text(
        0.0,
        1.0,
        _metrics_text(summary, interfaces),
        transform=ax_text.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
        color=INK,
    )
    fig.suptitle(pred.name, color=INK, fontsize=11)
    return fig


def save_summary_plot(
    pred: Prediction,
    path: str | Path,
    dpi: int = 300,
    pae_cutoff: float = DEFAULT_PAE_CUTOFF,
    dist_cutoff: float = DEFAULT_DIST_CUTOFF,
    renderer: str = "auto",
) -> Path:
    fig = plot_summary(pred, pae_cutoff, dist_cutoff, renderer=renderer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- contact map
def contact_highlight(contacts: list[dict]) -> dict[str, list[int]]:
    """Interface residues per chain from a contact list."""
    highlight: dict[str, set[int]] = {}
    for row in contacts:
        highlight.setdefault(row["chain_a"], set()).add(row["res_a"])
        highlight.setdefault(row["chain_b"], set()).add(row["res_b"])
    return {chain: sorted(res) for chain, res in highlight.items()}


def _contacts_text(pred: Prediction, contacts: list[dict],
                   dist_cutoff: float, pae_cutoff: float | None) -> str:
    pae_txt = f"PAE < {pae_cutoff:g} Å (both directions)" if pae_cutoff else "no PAE filter"
    lines = [
        f"criteria      d ≤ {dist_cutoff:g} Å",
        f"              {pae_txt}",
        "",
        f"contact pairs {len(contacts)}",
    ]
    pairs: dict[tuple[str, str], list[dict]] = {}
    for row in contacts:
        pairs.setdefault((row["chain_a"], row["chain_b"]), []).append(row)
    if pairs:
        lines += ["", "pair    pairs  res_a  res_b"]
        for (a, b), rows in sorted(pairs.items()):
            res_a = len({r["res_a"] for r in rows})
            res_b = len({r["res_b"] for r in rows})
            lines.append(f"{a}-{b:<6} {len(rows):>4}  {res_a:>5}  {res_b:>5}")
    closest = sorted(contacts, key=lambda r: r["distance"])[:8]
    if closest:
        lines += ["", "closest contacts (Å):"]
        for row in closest:
            lines.append(
                f"{row['chain_a']}/{row['resname_a']}{row['res_a']:<4} — "
                f"{row['chain_b']}/{row['resname_b']}{row['res_b']:<4} {row['distance']:.1f}"
            )
    return "\n".join(lines)


def _contact_structure_panel(
    pred: Prediction, fig: plt.Figure, slot: Any, renderer: str,
    highlight: dict[str, list[int]],
) -> None:
    color_of = {c: chain_color(i) for i, c in enumerate(pred.chains)}
    image = None
    if renderer in ("auto", "pymol"):
        from foldmetrics.render import render_contacts

        tmp = Path(tempfile.mkstemp(suffix=".png")[1])
        try:
            if render_contacts(pred.source, tmp, highlight, color_of) is not None:
                image = plt.imread(str(tmp))
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    if image is not None:
        ax = fig.add_subplot(slot)
        ax.imshow(image)
        ax.set_axis_off()
    else:
        ax = fig.add_subplot(slot, projection="3d")
        plot_structure_trace(pred, ax)
        highlight_tokens = [
            i for i, t in enumerate(pred.tokens)
            if t.res_id in set(highlight.get(t.chain, []))
        ]
        if highlight_tokens:
            xyz = pred.coords[highlight_tokens]
            colors = [color_of[pred.tokens[i].chain] for i in highlight_tokens]
            ax.scatter(*xyz.T, s=20, c=colors, depthshade=False, zorder=4)
    ax.set_title("Interface residues highlighted", fontsize=8, color=MUTED, pad=2)


def plot_contact_map(
    pred: Prediction,
    contacts: list[dict],
    dist_cutoff: float = 8.0,
    pae_cutoff: float | None = 12.0,
    renderer: str = "auto",
) -> plt.Figure:
    """Confident-contact figure: highlighted structure + PAE with contact overlay."""
    highlight = contact_highlight(contacts)

    if pred.has_pae:
        fig = plt.figure(figsize=(13.5, 5.6))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.5, 0.75], wspace=0.25)
        ax_pae = fig.add_subplot(gs[0, 1])
        ax_text = fig.add_subplot(gs[0, 2])
        structure_slot = gs[0, 0]
        plot_pae(pred, ax_pae)
        if contacts:
            rows = [r["token_a"] for r in contacts] + [r["token_b"] for r in contacts]
            cols = [r["token_b"] for r in contacts] + [r["token_a"] for r in contacts]
            ax_pae.scatter(
                cols, rows, s=9, facecolor="#D55E00", edgecolor="white",
                linewidths=0.3, alpha=0.9, zorder=3, label="contact",
            )
            ax_pae.legend(
                loc="lower left", bbox_to_anchor=(0.0, 1.03), frameon=False, fontsize=7
            )
    else:
        fig = plt.figure(figsize=(9.5, 5.2))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.2)
        ax_text = fig.add_subplot(gs[0, 1])
        structure_slot = gs[0, 0]

    _contact_structure_panel(pred, fig, structure_slot, renderer, highlight)
    ax_text.axis("off")
    ax_text.text(
        0.0, 1.0, _contacts_text(pred, contacts, dist_cutoff, pae_cutoff),
        transform=ax_text.transAxes, va="top", ha="left",
        family="monospace", fontsize=7.5, color=INK,
    )
    fig.suptitle(f"{pred.name} — confident interface contacts", color=INK, fontsize=11)
    return fig


def save_contact_plot(
    pred: Prediction,
    contacts: list[dict],
    path: str | Path,
    dist_cutoff: float = 8.0,
    pae_cutoff: float | None = 12.0,
    renderer: str = "auto",
    dpi: int = 300,
) -> Path:
    fig = plot_contact_map(pred, contacts, dist_cutoff, pae_cutoff, renderer)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------------------- batch overview
BATCH_METRICS = [
    ("ptm", "#0072B2"),
    ("iptm", "#E69F00"),
    ("ipsae", "#009E73"),
    ("pdockq", "#D55E00"),
]


def plot_batch(df: Any, max_models: int = 40) -> plt.Figure:
    """Batch overview from a summary DataFrame (see :func:`foldmetrics.evaluate`).

    Left: 0-1 confidence scores (pTM, ipTM, ipSAE, pDockQ) per model as a dot
    plot, best models on top. Right: mean pLDDT bars with the 50/70/90
    confidence thresholds. Models are ranked by ranking_score (falling back
    to ipSAE, then ipTM).
    """
    d = df.copy()
    order = d[["ranking_score", "ipsae", "iptm"]].bfill(axis=1).iloc[:, 0].fillna(0.0)
    d = d.assign(_order=order).sort_values("_order", ascending=False)
    truncated = len(d) > max_models
    d = d.head(max_models)
    n = len(d)

    multi_tool = d["tool"].nunique() > 1
    labels = [
        f"{str(m)[:34]} · {t}" if multi_tool else str(m)[:36]
        for m, t in zip(d["model"], d["tool"], strict=True)
    ]
    y = np.arange(n)[::-1]  # best model on top
    fig_height = max(3.2, 0.38 * n + 1.8)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.5, fig_height), width_ratios=[2.2, 1.0]
    )

    for metric, color in BATCH_METRICS:
        ax1.scatter(
            d[metric], y, s=48, color=color, label=METRIC_LABELS.get(metric, metric),
            zorder=3, edgecolors="white", linewidths=0.8,
        )
    ax1.set_yticks(y, labels=labels)
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.6, n - 0.4)
    ax1.set_xlabel("score (0–1)", color=INK, fontsize=9)
    ax1.grid(axis="x", color="#e6e6e6", lw=0.7)
    ax1.set_axisbelow(True)
    ax1.tick_params(colors=INK, labelsize=8)
    ax1.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.005), ncols=4,
        frameon=False, fontsize=8,
    )
    for spine in ("top", "right"):
        ax1.spines[spine].set_visible(False)

    ax2.barh(y, d["plddt_mean"], height=0.55, color="#0072B2", alpha=0.8)
    for threshold in (50.0, 70.0, 90.0):
        ax2.axvline(threshold, color="#bbbbbb", lw=0.8, ls="--", zorder=0)
    ax2.set_yticks(y, labels=[""] * n)
    ax2.set_xlim(0.0, 100.0)
    ax2.set_ylim(-0.6, n - 0.4)
    ax2.set_xlabel("mean pLDDT", color=INK, fontsize=9)
    ax2.tick_params(colors=INK, labelsize=8)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    title = "Batch overview"
    if truncated:
        title += f" (top {n} of {len(df)} models by ranking)"
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save_batch_plot(df: Any, path: str | Path, dpi: int = 300) -> Path:
    fig = plot_batch(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ------------------------------------------------- targets x methods overview
def plot_comparison(
    df: Any,
    metrics: tuple[str, ...] = ("iptm", "ipsae", "pdockq2", "plddt_mean"),
    max_targets: int = 25,
) -> plt.Figure:
    """Targets x methods comparison from a summary DataFrame.

    One panel per metric; x-axis groups by target (complex), one color per
    tool. Small faint dots are individual models; large dots are the
    per-target/tool mean. Works for every batch shape: one target with
    several methods, many targets with one method, or the full grid.
    """
    d = df.copy()
    if "target" not in d.columns or d["target"].isna().all():
        d["target"] = d["model"]

    rank_metric = next((m for m in metrics if d[m].notna().any()), metrics[0])
    target_order = (
        d.groupby("target")[rank_metric].mean().sort_values(ascending=False).index
    )
    truncated = len(target_order) > max_targets
    targets = list(target_order[:max_targets])
    d = d[d["target"].isin(targets)]

    tools = list(dict.fromkeys(d["tool"]))
    color_of = {t: chain_color(i) for i, t in enumerate(tools)}
    n_targets, n_tools = len(targets), len(tools)
    width = 0.8 / max(n_tools, 1)

    fig_width = max(7.0, 1.6 + 0.42 * n_targets * max(n_tools, 2) * 0.55)
    fig, axes = plt.subplots(
        len(metrics), 1, sharex=True,
        figsize=(fig_width, 1.95 * len(metrics) + 1.1),
    )
    axes = np.atleast_1d(axes)

    for ax, metric in zip(axes, metrics, strict=True):
        for i, tool in enumerate(tools):
            offset = (i - (n_tools - 1) / 2.0) * width
            sub = d[d["tool"] == tool]
            for j, target in enumerate(targets):
                vals = sub.loc[sub["target"] == target, metric].dropna()
                if vals.empty:
                    continue
                x = j + offset
                ax.scatter(
                    np.full(len(vals), x), vals, s=13, color=color_of[tool],
                    alpha=0.35, linewidths=0, zorder=2,
                )
                ax.scatter(
                    [x], [vals.mean()], s=52, color=color_of[tool],
                    edgecolors="white", linewidths=0.8, zorder=3,
                )
        ax.set_ylabel(METRIC_LABELS.get(metric, metric), color=INK, fontsize=9)
        ax.set_ylim((0.0, 102.0) if "plddt" in metric else (0.0, 1.04))
        ax.grid(axis="y", color="#ececec", lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[-1].set_xticks(range(n_targets), labels=targets, rotation=30, ha="right")
    if n_tools > 1:
        handles = [
            Line2D([], [], marker="o", ls="", color=color_of[t], label=t, markersize=7)
            for t in tools
        ]
        axes[0].legend(
            handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.02),
            ncols=min(n_tools, 6), frameon=False, fontsize=8,
        )

    title = "Per-target comparison"
    if truncated:
        title += f" (top {n_targets} targets by {METRIC_LABELS.get(rank_metric, rank_metric)})"
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def save_comparison_plot(df: Any, path: str | Path, dpi: int = 300) -> Path:
    fig = plot_comparison(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
