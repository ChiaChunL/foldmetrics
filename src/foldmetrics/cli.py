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

    p_contacts = sub.add_parser(
        "contacts",
        help="extract confident inter-chain contacts (distance + PAE filtered)",
    )
    _add_common(p_contacts)
    p_contacts.add_argument(
        "--dist-cutoff", type=float, default=8.0,
        help="contact-atom distance cutoff in Å (default: 8)",
    )
    p_contacts.add_argument(
        "--pae-cutoff", type=float, default=12.0,
        help="keep a contact only when both PAE directions are below this (Å); "
             "negative disables the PAE filter (default: 12)",
    )
    p_contacts.add_argument(
        "-o", "--out", type=Path, help="write the contact table (.tsv/.csv/.json)"
    )
    p_contacts.add_argument(
        "--plot", type=Path, metavar="DIR",
        help="write per-model contact figure (PNG) plus a PyMOL .pml script into DIR",
    )
    p_contacts.add_argument(
        "--renderer", default="auto", choices=["auto", "pymol", "trace"],
        help="structure panel renderer (default: auto)",
    )

    p_dockq = sub.add_parser(
        "dockq",
        help="DockQ vs a reference structure (needs: pip install 'foldmetrics[dockq]')",
    )
    _add_common(p_dockq)
    p_dockq.add_argument(
        "--ref", type=Path, required=True,
        help="reference (experimental/trusted) structure, PDB or mmCIF",
    )
    p_dockq.add_argument(
        "--mapping",
        help="model:reference chain mapping such as 'A:A,B:D' "
             "(default: by name when both structures share chain names, else by order)",
    )
    p_dockq.add_argument(
        "--small-molecule", action="store_true",
        help="also score small-molecule ligand poses (DockQ small_molecule mode)",
    )
    p_dockq.add_argument(
        "--capri-peptide", action="store_true",
        help="use CAPRI peptide criteria (for protein-peptide interfaces)",
    )
    p_dockq.add_argument(
        "-o", "--out", type=Path, help="write the DockQ table (.tsv/.csv/.json)"
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


CONTACT_COLUMNS = [
    "model", "tool", "chain_a", "res_a", "resname_a", "atom_a", "kind_a",
    "chain_b", "res_b", "resname_b", "atom_b", "kind_b",
    "distance", "pae_ab", "pae_ba", "plddt_a", "plddt_b",
]

DOCKQ_COLUMNS = [
    "model", "tool", "interface", "native_pair", "dockq", "dockq_class",
    "fnat", "fnonnat", "f1", "irmsd", "lrmsd", "clashes",
    "len_a", "len_b", "total_dockq",
]


def cmd_contacts(args: argparse.Namespace) -> int:
    import pandas as pd

    from foldmetrics.metrics import find_contacts

    predictions = _load(args)
    if not predictions:
        print("error: no predictions recognized under the given paths", file=sys.stderr)
        return 1
    pae_cutoff = None if args.pae_cutoff < 0 else args.pae_cutoff

    all_rows: list[dict] = []
    for pred in predictions:
        rows = find_contacts(pred, args.dist_cutoff, pae_cutoff)
        if pred.pae is None:
            print(
                f"note [{pred.name}]: no PAE available — distance-only contacts",
                file=sys.stderr,
            )
        for row in rows:
            row["model"] = pred.name
            row["tool"] = pred.tool
        all_rows.extend(rows)
        n_res = len({(r["chain_a"], r["res_a"]) for r in rows}) + len(
            {(r["chain_b"], r["res_b"]) for r in rows}
        )
        print(
            f"{pred.name}: {len(rows)} contact pairs, {n_res} interface residues",
            file=sys.stderr,
        )

        if args.plot:
            import matplotlib

            matplotlib.use("Agg")
            from foldmetrics.render import write_contacts_pml
            from foldmetrics.viz import chain_color, contact_highlight, save_contact_plot

            args.plot.mkdir(parents=True, exist_ok=True)
            base = _safe_name(pred.name)
            target = args.plot / f"{base}_contacts.png"
            save_contact_plot(
                pred, rows, target,
                dist_cutoff=args.dist_cutoff, pae_cutoff=pae_cutoff,
                renderer=args.renderer,
            )
            print(f"wrote {target}", file=sys.stderr)
            colors = {c: chain_color(i) for i, c in enumerate(pred.chains)}
            pml = write_contacts_pml(
                args.plot / f"{base}_contacts.pml",
                pred.source,
                contact_highlight(rows),
                colors,
                header=(
                    f"foldmetrics contacts: {pred.name} "
                    f"(d <= {args.dist_cutoff:g} A, PAE < {pae_cutoff or 'off'})"
                ),
            )
            print(f"wrote {pml}", file=sys.stderr)

    df = pd.DataFrame(all_rows).reindex(columns=CONTACT_COLUMNS)
    if len(df) == 0:
        print("no contacts passed the cutoffs")
    elif len(df) <= 60:
        print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    else:
        print(df.head(20).to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"... {len(df)} rows total — use -o FILE for the full table")
    if args.out:
        _write_table(df, args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def cmd_dockq(args: argparse.Namespace) -> int:
    import pandas as pd

    from foldmetrics.dockq import _require_dockq, compute_dockq, parse_mapping

    try:
        _require_dockq()
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not args.ref.exists():
        print(f"error: reference structure not found: {args.ref}", file=sys.stderr)
        return 1
    mapping = None
    if args.mapping:
        try:
            mapping = parse_mapping(args.mapping)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    units = discover(args.paths, tool=args.tool)
    if not units:
        print("error: no predictions recognized under the given paths", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    failures = 0
    for unit in units:
        try:
            rows, total = compute_dockq(
                unit.files["structure"], args.ref, mapping,
                small_molecule=args.small_molecule,
                capri_peptide=args.capri_peptide,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            print(f"warning: DockQ failed for {unit.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        for row in rows:
            row["model"] = unit.name
            row["tool"] = unit.tool
            row["total_dockq"] = total
        all_rows.extend(rows)

    if not all_rows:
        print("error: DockQ produced no results", file=sys.stderr)
        return 2 if failures else 1
    df = pd.DataFrame(all_rows).reindex(columns=DOCKQ_COLUMNS)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if args.out:
        _write_table(df, args.out)
        print(f"wrote {args.out}", file=sys.stderr)
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
    handlers = {
        "score": cmd_score,
        "plot": cmd_plot,
        "detect": cmd_detect,
        "contacts": cmd_contacts,
        "dockq": cmd_dockq,
    }
    try:
        return handlers[args.command](args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
