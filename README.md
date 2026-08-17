# foldmetrics

Unified confidence metrics for structure-prediction models.

`foldmetrics` ingests the raw output folders of the mainstream structure
predictors — **AlphaFold2 / AlphaFold-Multimer, ColabFold, AlphaFold3, Boltz,
Chai-1, Protenix** — and computes a consistent set of quality metrics for
monomers and complexes (protein, nucleic acid and small-molecule ligands),
for a single model or whole batches:

| Metric | What it tells you | Source |
|---|---|---|
| `ptm`, `iptm` | global / interface predicted TM-score | read from tool output |
| `ranking_score` | tool-native model ranking (AF3 `ranking_score`, Boltz `confidence_score`, Chai `aggregate_score`, AF2 `ranking_confidence`) | read from tool output |
| `plddt_mean` | mean per-token pLDDT | computed |
| `iplddt` | mean pLDDT over interface residues (contact atoms within 8 Å across chains) | computed |
| `pae_mean`, `ipae_mean` | mean PAE (all off-diagonal / inter-chain blocks) | computed |
| `ipsae` | interface score from PAE with per-residue d0 (Dunbrack 2025) | computed |
| `pdockq` | interface score from contacts + pLDDT (Bryant 2022) | computed |
| `pdockq2` | interface score from contacts + pLDDT + PAE (Zhu 2023) | computed |
| `lis` | Local Interaction Score from PAE (Kim 2024) | computed |

Computed metrics were validated against the `ipsae.py` reference
implementation (Dunbrack Lab), including its exact `d0` conventions.

## Install

```bash
pip install foldmetrics
```

(conda-forge packaging is planned; until then use pip inside a conda env.)

Development install:

```bash
git clone https://github.com/ChiaChunL/foldmetrics.git && cd foldmetrics
pip install -e ".[dev]"
```

## Quickstart

CLI (`foldmetrics`, short alias `fmx`):

```bash
# score every prediction found under a directory (any mix of tools)
foldmetrics score path/to/predictions/ -o metrics.tsv

# per chain-pair breakdown + summary figures (pLDDT track, PAE heatmap, metrics)
fmx score preds/ -o metrics.tsv --interfaces interfaces.tsv --plot plots/

# a single model: point at any of its files
fmx score run1/fold_job_full_data_0.json

# one metric only (every metric name is also a subcommand)
fmx ipsae preds/ --interfaces ipsae_per_pair.tsv
fmx score preds/ --metrics ipsae,pdockq2,lis

# what would be scored?
fmx detect preds/

# figures only
fmx plot preds/ -o plots/
```

Python:

```python
import foldmetrics as fm

df = fm.evaluate("path/to/predictions/")          # one row per model
dfi = fm.evaluate_interfaces("path/to/predictions/")  # one row per chain pair

# lower-level access
preds = fm.load_predictions("path/to/predictions/")
summary, interfaces = fm.compute_all(preds[0])
```

More recipes (including runnable demo data that needs no prediction tool) live
in [examples/](examples/).

## Visualization

`--plot DIR` (on `score` and every metric subcommand) or the `plot`
subcommand renders publication-oriented figures, adapting automatically to
the shape of the batch:

| Input shape | Figures written into DIR |
|---|---|
| every model | `<model>.png` — pLDDT-colored structure + pLDDT track + PAE heatmap + metrics panel |
| more than one model | plus `batch_overview.png` — ranked confidence dot plot + mean pLDDT bars |
| more than one target and/or tool | plus `comparison.png` — one panel per metric, grouped by target, one color per tool |

The structure panel uses **headless PyMOL** when available (auto-detected
from `FOLDMETRICS_PYMOL`, PATH, or common conda locations; ~2–3 s per
model) and falls back to a fast matplotlib backbone trace otherwise —
select explicitly with `--renderer pymol|trace`. The `plot` subcommand also
takes `--format png|pdf|svg` and `--dpi` (default 300).

Single model (AlphaFold3, SARS-CoV-2 Mpro + nirmatrelvir):

![per-model summary](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_summary.png)

Targets × methods comparison (real batch: 10 complexes × 4 tools):

![per-target comparison](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_comparison.png)

Batch overview (one target, AlphaFold2 + AlphaFold3 models):

![batch overview](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_batch.png)

## Supported tools and files

| Tool | Detected files | pTM/ipTM | pLDDT | PAE |
|---|---|---|---|---|
| ColabFold | `*_scores_rank_*.json` + `*_(un)relaxed_rank_*.pdb` | yes | yes | yes |
| AlphaFold2 (pickle layout) | `result_model_*.pkl` + `unrelaxed_*.pdb` / `ranked_*.pdb` | yes | yes | yes |
| AlphaFold2 (JSON layout) | `iptm_ptm.json` + `confidence_*.json` / `pae_*.json` + `unrelaxed_*.cif/.pdb` | yes | yes | yes |
| AlphaFold3 (server/local) | `*model*.cif` + `*summary_confidences*.json` + `*confidences*/full_data*.json` | yes | yes | yes |
| Boltz-1/2 | `confidence_*_model_*.json` + `*_model_*.cif` + `pae_*.npz` / `plddt_*.npz` | yes | yes | yes |
| Chai-1 | `scores.model_idx_*.npz` + `pred.model_idx_*.cif` | yes | yes | if exported |
| Protenix | `*summary_confidence*.json` + matching `.cif` (+ `*full_data*.json` with `token_pair_pae`) | yes | yes | yes |
| HelixFold3 | planned | — | — | — |

## Validation

- Numerical parity with the `ipsae.py` reference implementation (Dunbrack
  Lab) verified digit-for-digit on real AlphaFold3 server output: ipSAE
  (both directions and d0chn variant), pDockQ, pDockQ2 and LIS all match to
  6 decimal places at the default cutoffs (10/10).
- Batch-tested on 720 real predictions across AlphaFold2-Multimer,
  AlphaFold3 (server + local), Boltz-2 and Protenix — including
  protein–small-molecule complexes, homodimers, monomers and negative
  controls — with zero parse errors; known binders score ipSAE 0.9+, decoy
  pairs < 0.1, monomers report NA.
- ColabFold and Chai-1 parsers are currently validated on synthetic
  fixtures only; real-output samples welcome.

Native per-tool extras (e.g. Boltz `complex_iplddt`/`ligand_iptm`, AF3
`chain_pair_pae_min`, Chai clash flags) are preserved on
`Prediction.extras` and chain-pair ipTM is surfaced as `iptm_native` in the
interface table.

## What each metric needs

The structure file is always required (it defines chains and tokens); the
table shows which additional inputs each metric consumes. When an input is
missing the metric is `NA` and a note lands in the `warnings` column —
nothing crashes.

| Metric (= subcommand) | pLDDT | Coordinates | PAE | Source |
|---|---|---|---|---|
| `ptm`, `iptm`, `ranking` | – | – | – | read from the tool's confidence file |
| `plddt` (mean pLDDT, ipLDDT) | yes | ipLDDT only | – | B-factors, or the tool's pLDDT file |
| `pae` (mean PAE, inter-chain PAE) | – | – | yes | tool's PAE matrix |
| `pdockq` | yes | yes | – | contacts at 8 Å between CB/C3' atoms |
| `pdockq2` | yes | yes | yes | |
| `ipsae`, `lis` | – | – | yes | chain mapping from the structure |

## Outputs and paths

- Summary table → stdout; `-o FILE` writes it. The extension picks the
  format: `.tsv` (default), `.csv`, `.json`; missing values are `NA`.
- `--interfaces FILE` → the per chain-pair table (same formats).
- `--plot DIR` → figures as described under Visualization; model names are
  sanitized (`[^\w.-]` → `_`) for use as filenames.
- `plot -o DIR` defaults to `./foldmetrics_plots/`.
- Exit codes: `0` success, `1` nothing recognized/found, `2` bad arguments.

## Conventions worth knowing

- **Tokens.** Standard residues are one token; ligands and modified residues
  are one token per heavy atom (AF3-style), so token-level PAE matrices line
  up across tools. pLDDT is stored on the 0–100 scale everywhere (Boltz 0–1
  values are rescaled).
- **Complex-level interface metrics are the best interface.** For >2 chains,
  `ipsae`/`pdockq`/`pdockq2`/`lis` in the summary table are the maximum over
  chain pairs; use `--interfaces` for the full breakdown.
- **Ligand interfaces.** `pdockq`/`pdockq2`/`iplddt` are defined for
  polymer–polymer interfaces only. For chain pairs involving a ligand chain,
  `ipsae`/`lis` are computed over ligand atom tokens (experimental) and marked
  `ipsae_mode = "tokens"` in the interface table.
- **Missing data degrades gracefully.** No PAE → PAE-based metrics are NaN
  and a note lands in the `warnings` column; nothing crashes.
- **Directionality.** PAE is asymmetric, so `pdockq2`/`ipsae`/`lis` have two
  directional values; the interface table reports both (`*_ab`, `*_ba`) plus
  the aggregate used everywhere else (max for ipSAE/pDockQ2, mean for LIS,
  matching the reference implementations).

## References

- Bryant P, Pozzati G, Elofsson A. *Improved prediction of protein-protein
  interactions using AlphaFold2.* Nat Commun 13, 1265 (2022). — pDockQ
- Zhu W, Shenoy A, Kundrotas P, Elofsson A. *Evaluation of AlphaFold-Multimer
  prediction on multi-chain protein complexes.* Bioinformatics 39, btad424
  (2023). — pDockQ2
- Dunbrack RL. *ipSAE: scoring pairwise interactions in AlphaFold models.*
  bioRxiv 10.1101/2025.02.10.637595 (2025). — ipSAE
- Kim AR et al. *Enhanced protein-protein interaction discovery via
  AlphaFold-Multimer.* bioRxiv 10.1101/2024.02.19.580970 (2024). — LIS

## Roadmap

- HelixFold3 parser (sample outputs welcome — please open an issue)
- Real-output samples for ColabFold and Chai-1
- Per-interface PAE/ipSAE figures; interactive HTML report
- conda-forge feedstock; CI (lint + tests) on GitHub Actions

## License

MIT
