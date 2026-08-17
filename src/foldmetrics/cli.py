"""Command-line interface: ``foldmetrics`` (alias ``fmx``)."""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

from foldmetrics import __version__
from foldmetrics.metrics import DEFAULT_DIST_CUTOFF, DEFAULT_PAE_CUTOFF
from foldmetrics.parsers import discover, load_predictions, parser_names

# metric name -> (summary columns, interface columns); used by --metrics and
# the per-metric subcommands (``fmx ipsae preds/`` == ``fmx score preds/ --metrics ipsae``)
METRIC_GROUPS: dict[str, tuple[list[str], list[str]]] = {
    "ptm": (["ptm"], []),
    "iptm": (["iptm"], ["iptm_native", "iptm_pae"]),
    "ranking": (["ranking_score"], []),
    "plddt": (["plddt_mean", "iplddt"], ["iplddt"]),
    "pae": (["pae_mean", "ipae_mean"], ["ipae_mean", "ipae_min"]),
    "ipsae": (["ipsae"], ["ipsae", "ipsae_ab", "ipsae_ba", "ipsae_d0chn", "ipsae_mode"]),
    "pdockq": (["pdockq"], ["n_contacts", "n_if_res", "pdockq"]),
    "pdockq2": (["pdockq2"], ["pdockq2", "pdockq2_ab", "pdockq2_ba"]),
    "lis": (["lis"], ["lis"]),
}
_SUMMARY_ID_COLS = ["model", "tool", "chains", "n_chains", "n_tokens"]
_INTERFACE_ID_COLS = ["model", "tool", "chain_a", "chain_b", "kind_a", "kind_b"]


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "paths", nargs="+", help="prediction files or directories (scanned recursively)"
    )
    sub.add_argument(
        "--tool",
        default="auto",
        choices=["auto", *parser_names()],
        help="restrict to one prediction tool (default: auto-detect)",
    )


def _add_cutoffs(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--pae-cutoff", type=float, default=DEFAULT_PAE_CUTOFF,
        help=f"PAE cutoff (Å) for ipSAE (default: {DEFAULT_PAE_CUTOFF:g})",
    )
    sub.add_argument(
        "--dist-cutoff", type=float, default=DEFAULT_DIST_CUTOFF,
        help=f"contact distance cutoff (Å) for pDockQ/ipLDDT (default: {DEFAULT_DIST_CUTOFF:g})",
    )


def _add_score_args(sub: argparse.ArgumentParser) -> None:
    _add_common(sub)
    _add_cutoffs(sub)
    sub.add_argument("-o", "--out", type=Path, help="write the summary table (.tsv/.csv/.json)")
    sub.add_argument(
        "--interfaces", type=Path,
        help="also write the per-chain-pair table to this file",
    )
    sub.add_argument(
        "--plot", type=Path, metavar="DIR",
        help="write a summary figure (PNG) per model into DIR",
    )
    sub.add_argument(
        "--renderer", default="auto", choices=["auto", "pymol", "trace"],
        help="structure panel renderer: auto/pymol use headless PyMOL when "
             "available (adds a few seconds per model), trace is a fast "
             "matplotlib backbone trace (default: auto)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foldmetrics",
        description=(
            "Confidence metrics (pTM, ipTM, pLDDT, ipLDDT, PAE, ipSAE, pDockQ, "
            "pDockQ2, LIS) for AlphaFold2/3, ColabFold, Boltz, Chai-1 and Protenix outputs."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="compute metrics for one or many predictions")
    _add_score_args(p_score)
    p_score.add_argument(
        "--metrics",
        help="report only these metrics (comma-separated): " + ", ".join(METRIC_GROUPS),
    )

    for metric in METRIC_GROUPS:
        p_metric = sub.add_parser(
            metric, help=f"compute only {metric} (shorthand for: score --metrics {metric})"
        )
        _add_score_args(p_metric)

    p_plot = sub.add_parser(
        "plot", help="render figures (structure + pLDDT + PAE + metrics, batch views)"
    )
    _add_common(p_plot)
    _add_cutoffs(p_plot)
    p_plot.add_argument(
        "-o", "--out", type=Path, default=Path("foldmetrics_plots"), metavar="DIR",
        help="output directory (default: foldmetrics_plots)",
    )
    p_plot.add_argument("--dpi", type=int, default=300)
    p_plot.add_argument(
        "--format", default="png", choices=["png", "pdf", "svg"],
        help="figure file format (default: png)",
    )
    p_plot.add_argument(
        "--renderer", default="auto", choices=["auto", "pymol", "trace"],
        help="structure panel renderer: auto/pymol use headless PyMOL when "
             "available (adds a few seconds per model), trace is a fast "
             "matplotlib backbone trace (default: auto)",
    )

    p_detect = sub.add_parser("detect", help="list recognized predictions without scoring")
    _add_common(p_detect)

    return parser


def _write_table(df, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False, na_rep="NA")
    elif suffix == ".json":
        df.to_json(path, orient="records", indent=2)
    else:
        df.to_csv(path, sep="\t", index=False, na_rep="NA")


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("_") or "model"


def _load(args: argparse.Namespace) -> list:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predictions = load_predictions(args.paths, tool=args.tool, on_error="warn")
    for w in caught:
        print(f"warning: {w.message}", file=sys.stderr)
    return predictions


def cmd_score(args: argparse.Namespace) -> int:
    from foldmetrics.api import evaluate_full

    predictions = _load(args)
    if not predictions:
        print("error: no predictions recognized under the given paths", file=sys.stderr)
        return 1

    df, df_interfaces = evaluate_full(
        predictions, pae_cutoff=args.pae_cutoff, dist_cutoff=args.dist_cutoff
    )
    df_full = df  # figures always use every metric, regardless of --metrics

    metrics = getattr(args, "metrics", None)
    if metrics:
        names = [m.strip() for m in metrics.split(",") if m.strip()]
        unknown = sorted(set(names) - set(METRIC_GROUPS))
        if unknown:
            print(
                f"error: unknown metric(s) {', '.join(unknown)}; "
                f"choose from: {', '.join(METRIC_GROUPS)}",
                file=sys.stderr,
            )
            return 2
        summary_cols = list(dict.fromkeys(
            _SUMMARY_ID_COLS + [c for m in names for c in METRIC_GROUPS[m][0]]
        ))
        interface_cols = list(dict.fromkeys(
            _INTERFACE_ID_COLS + [c for m in names for c in METRIC_GROUPS[m][1]]
        ))
        df = df[summary_cols + ["source", "warnings"]]
        df_interfaces = df_interfaces[interface_cols]

    display = df.drop(columns=["source", "warnings"], errors="ignore")
    print(display.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    for pred in predictions:
        for note in pred.warnings:
            print(f"note [{pred.name}]: {note}", file=sys.stderr)

    if args.out:
        _write_table(df, args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    if args.interfaces:
        _write_table(df_interfaces, args.interfaces)
        print(f"wrote {args.interfaces}", file=sys.stderr)
    if args.plot:
        _render_plots(
            predictions, df_full, args.plot, 300,
            args.pae_cutoff, args.dist_cutoff,
            renderer=getattr(args, "renderer", "auto"),
        )
    return 0


def _render_plots(
    predictions, df, out_dir: Path, dpi: int, pae_cutoff: float, dist_cutoff: float,
    renderer: str = "auto", fmt: str = "png",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from foldmetrics.viz import save_batch_plot, save_comparison_plot, save_summary_plot

    out_dir.mkdir(parents=True, exist_ok=True)
    for pred in predictions:
        target = out_dir / f"{_safe_name(pred.name)}.{fmt}"
        save_summary_plot(
            pred, target, dpi=dpi, pae_cutoff=pae_cutoff, dist_cutoff=dist_cutoff,
            renderer=renderer,
        )
        print(f"wrote {target}", file=sys.stderr)
    if df is not None and len(df) > 1:
        target = out_dir / f"batch_overview.{fmt}"
        save_batch_plot(df, target, dpi=dpi)
        print(f"wrote {target}", file=sys.stderr)
        if df["tool"].nunique() > 1 or df["target"].nunique() > 1:
            target = out_dir / f"comparison.{fmt}"
            save_comparison_plot(df, target, dpi=dpi)
            print(f"wrote {target}", file=sys.stderr)


def cmd_plot(args: argparse.Namespace) -> int:
    from foldmetrics.api import evaluate_full

    predictions = _load(args)
    if not predictions:
        print("error: no predictions recognized under the given paths", file=sys.stderr)
        return 1
    df, _ = evaluate_full(
        predictions, pae_cutoff=args.pae_cutoff, dist_cutoff=args.dist_cutoff
    )
    _render_plots(
        predictions, df, args.out, args.dpi, args.pae_cutoff, args.dist_cutoff,
        renderer=args.renderer, fmt=args.format,
    )
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    units = discover(args.paths, tool=args.tool)
    if not units:
        print("no predictions recognized under the given paths", file=sys.stderr)
        return 1
    for unit in units:
        files = ", ".join(f"{role}={path.name}" for role, path in sorted(unit.files.items()))
        print(f"{unit.tool:<12} {unit.name:<40} {unit.dir}  [{files}]")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in METRIC_GROUPS:
        args.metrics = args.command
        args.command = "score"
    handlers = {"score": cmd_score, "plot": cmd_plot, "detect": cmd_detect}
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
