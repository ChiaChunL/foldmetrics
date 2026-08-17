"""Python API walkthrough, using the real predictions in examples/data.

Usage (from the repository root)::

    python examples/python_api.py examples/data
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import foldmetrics as fmx
from foldmetrics.metrics import ipsae_asym
from foldmetrics.viz import plot_pae, plot_plddt, save_batch_plot, save_summary_plot


def main(path: str) -> None:
    out = Path("example_output")
    out.mkdir(exist_ok=True)

    # 1. Batch scoring: one row per model ------------------------------------
    df = fmx.evaluate(path)  # accepts a directory, a file, or a list of either
    print(df[["model", "tool", "ptm", "iptm", "ipsae", "pdockq", "plddt_mean"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    df.to_csv(out / "metrics.tsv", sep="\t", index=False)

    # 2. Per chain-pair breakdown -------------------------------------------
    dfi = fmx.evaluate_interfaces(path)
    print(dfi[["model", "chain_a", "chain_b", "ipsae", "pdockq2", "iptm_native"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # 3. Working with a single prediction object ----------------------------
    pred = fmx.load_predictions(path)[0]
    summary, interfaces = fmx.compute_all(pred)
    print(f"\n{pred.name}: {pred.n_tokens} tokens, chains {pred.chains}")
    print(f"  ipSAE={summary['ipsae']:.3f}  pDockQ={summary['pdockq']:.3f}")

    # Per-residue ipSAE profile for one direction (chain A aligned, B scored):
    a, b = pred.chains[:2]
    result = ipsae_asym(pred, a, b)
    if result is not None:
        print(f"  best aligned residue for {a}->{b}: token {result.best_token} "
              f"(byres max {result.value:.3f}, n0res={result.n0res})")

    # 4. Confident interface contacts ---------------------------------------
    contacts = fmx.find_contacts(pred, dist_cutoff=8.0, pae_cutoff=12.0)
    print(f"  {len(contacts)} confident contacts; closest:")
    for row in sorted(contacts, key=lambda r: r["distance"])[:3]:
        print(f"    {row['chain_a']}/{row['resname_a']}{row['res_a']} - "
              f"{row['chain_b']}/{row['resname_b']}{row['res_b']}  "
              f"{row['distance']:.1f} A (PAE {row['pae_ab']:.1f}/{row['pae_ba']:.1f})")

    # 5. Figures -------------------------------------------------------------
    # Ready-made one-page summary (structure + pLDDT track + PAE + metrics):
    save_summary_plot(pred, out / f"{pred.name}_summary.png")

    # Batch overview across every model scored above:
    save_batch_plot(df, out / "batch_overview.png")

    # Or compose your own figure from the building blocks:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), height_ratios=[1, 2.6], constrained_layout=True
    )
    plot_plddt(pred, ax1)
    plot_pae(pred, ax2)
    fig.savefig(out / f"{pred.name}_custom.png", dpi=150)
    plt.close(fig)

    print(f"\nfigures and tables written to {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/data")
