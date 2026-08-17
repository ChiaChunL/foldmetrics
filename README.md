<div align="center">
<img src="https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/foldmetrics_banner.svg" alt="foldmetrics" width="78%">
</div>

-----------------

# foldmetrics: unified confidence metrics for structure-prediction models

| Testing | [![CI](https://github.com/ChiaChunL/foldmetrics/actions/workflows/ci.yml/badge.svg)](https://github.com/ChiaChunL/foldmetrics/actions/workflows/ci.yml) |
|---|---|
| Package | [![PyPI Latest Release](https://img.shields.io/pypi/v/foldmetrics.svg)](https://pypi.org/project/foldmetrics/) [![Python versions](https://img.shields.io/pypi/pyversions/foldmetrics.svg)](https://pypi.org/project/foldmetrics/) [![PyPI Downloads](https://img.shields.io/pypi/dm/foldmetrics.svg)](https://pypi.org/project/foldmetrics/) |
| Meta | [![License - MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |

## 🧬 What is it?

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
| `dockq` | true interface accuracy vs a reference structure (Basu & Wallner 2016) | computed, optional |

Beyond scores it extracts **confident interface contacts** (distance + PAE
filtered residue pairs, with PyMOL scripts) and renders
**publication-oriented figures**.

## 📦 Installation

```bash
pip install foldmetrics
```

With DockQ support (reference-based scoring):

```bash
pip install "foldmetrics[dockq]"
```

Development install:

```bash
git clone https://github.com/ChiaChunL/foldmetrics.git && cd foldmetrics
pip install -e ".[dev]"
```

## ⚡ Quickstart

CLI (`foldmetrics`, short alias `fmx`):

```bash
# score every prediction found under a directory (any mix of tools)
foldmetrics score path/to/predictions/ -o metrics.tsv

# per chain-pair breakdown + summary figures (structure, pLDDT, PAE, metrics)
fmx score preds/ -o metrics.tsv --interfaces interfaces.tsv --plot plots/

# a single model: point at any of its files
fmx score run1/fold_job_full_data_0.json

# one metric only (every metric name is also a subcommand)
fmx ipsae preds/ --interfaces ipsae_per_pair.tsv
fmx score preds/ --metrics ipsae,pdockq2,lis

# confident interface contacts (+ figure + PyMOL script)
fmx contacts preds/ -o contacts.tsv --plot plots/

# DockQ against an experimental / reference structure
fmx dockq preds/ --ref native.pdb -o dockq.tsv

# what would be scored?
fmx detect preds/
```

Python:

```python
import foldmetrics as fm

df = fm.evaluate("path/to/predictions/")              # one row per model
dfi = fm.evaluate_interfaces("path/to/predictions/")  # one row per chain pair

preds = fm.load_predictions("path/to/predictions/")
summary, interfaces = fm.compute_all(preds[0])
contacts = fm.find_contacts(preds[0], dist_cutoff=8.0, pae_cutoff=12.0)
```

More recipes (including runnable demo data that needs no prediction tool)
live in [examples/](examples/).

## 🛠️ Command-line reference

In every example above, `preds/` is a **placeholder for wherever your
prediction outputs live**: a directory (scanned recursively; different
tools can be mixed freely), one specific model file, or any number of
paths at once.

| Option | Commands | Meaning |
|---|---|---|
| `paths` | all | prediction files and/or directories to process |
| `--tool NAME` | all | restrict to one tool (`colabfold`, `alphafold2`, `alphafold3`, `boltz`, `chai`, `protenix`); default auto-detects |
| `-o, --out FILE` | score, metric subcommands, contacts, dockq | write the result table; format follows the extension (`.tsv`/`.csv`/`.json`) |
| `--interfaces FILE` | score, metric subcommands | also write the per chain-pair table |
| `--metrics LIST` | score | report only these metrics, e.g. `--metrics ipsae,pdockq2` |
| `--plot DIR` | score, metric subcommands, contacts | write figures into DIR (contacts also writes a `.pml`) |
| `--pae-cutoff Å` | score family (default 10, for ipSAE) · contacts (default 12; negative disables) | PAE confidence threshold |
| `--dist-cutoff Å` | score family, contacts (default 8) | contact-atom distance threshold |
| `--renderer {auto,pymol,trace}` | score family, plot, contacts | structure panel renderer (PyMOL vs fast trace) |
| `--format {png,pdf,svg}` / `--dpi N` | plot | figure file format and resolution |
| `--ref FILE` | dockq | reference structure to compare against (required) |
| `--mapping A:A,B:D` | dockq | model:reference chain pairing |
| `--small-molecule` | dockq | also score small-molecule ligand poses |

`fmx <command> --help` prints the complete option list for any command.

## 🖼️ Visualization

`--plot DIR` (on `score` and every metric subcommand) or the `plot`
subcommand renders figures that adapt automatically to the shape of the
batch:

| Input shape | Figures written into DIR |
|---|---|
| every model | `<model>.png` — pLDDT-colored structure + pLDDT track + PAE heatmap + metrics panel |
| more than one model | plus `batch_overview.png` — ranked confidence dot plot + mean pLDDT bars |
| more than one target and/or tool | plus `comparison.png` — one panel per metric, grouped by target, one color per tool |

The structure panel uses **headless PyMOL** when available (auto-detected
from `FOLDMETRICS_PYMOL`, PATH, or common conda locations; ~2–3 s per
model) and falls back to a fast matplotlib backbone trace otherwise —
select explicitly with `--renderer pymol|trace`. The `plot` subcommand also
takes `--format png|pdf|svg` and `--dpi` (default 300). The pLDDT track is
colored by the AlphaFold confidence bands (blue / cyan / yellow / orange).

Single model (AlphaFold3, SARS-CoV-2 Mpro + nirmatrelvir):

![per-model summary](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_summary.png)

Targets × methods comparison (real batch: 10 complexes × 4 tools):

![per-target comparison](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_comparison.png)

Batch overview (one target, AlphaFold2 + AlphaFold3 models):

![batch overview](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_batch.png)

## 🤝 Confident interface contacts

`fmx contacts` extracts the inter-chain residue (and ligand-atom) pairs the
model is *confident* about, following the interface-contact idea of Zhang
et al. 2022: a pair is kept when the contact atoms are within
`--dist-cutoff` (default 8 Å) **and** both PAE directions are below
`--pae-cutoff` (default 12 Å; negative disables). Ligand atoms participate,
so binding-site contacts are reported too.

```bash
fmx contacts preds/ -o contacts.tsv --plot plots/
```

Outputs per model: the contact table (residue pair, distance, both PAE
directions, pLDDTs), a figure with the interface residues highlighted on
the structure and overlaid on the PAE heatmap, and a ready-to-run
`<model>_contacts.pml` with named PyMOL selections (`if_A`, `if_B`,
`interface`) for interactive inspection.

![contact map](https://raw.githubusercontent.com/ChiaChunL/foldmetrics/main/docs/assets/demo_contacts.png)

## 🎯 DockQ against a reference

When an experimental (or otherwise trusted) structure exists, `fmx dockq`
computes the *actual* interface accuracy via the official DockQ
implementation (install with `pip install "foldmetrics[dockq]"`):

```bash
fmx dockq preds/ --ref 1brs.pdb -o dockq.tsv
fmx dockq preds/ --ref native.cif --mapping A:A,B:D   # explicit chain pairing
fmx dockq preds/ --ref complex.cif --small-molecule   # score ligand poses too
```

Reports DockQ, fnat, iRMSD, LRMSD and the CAPRI-style class per interface.
Chains are matched by name when both structures share names, otherwise by
order — check the reported pairing (mmCIF label vs auth chain ids differ
between tools) and override with `--mapping MODEL:REF,...`.

## 📊 How to read the scores

| Score | Guidance | Basis |
|---|---|---|
| pLDDT | > 90 very high (side chains reliable); 70–90 backbone confident; 50–70 low; < 50 likely disordered | AlphaFold confidence bands (Jumper 2021) |
| pTM / ipTM | > 0.8 confident; 0.6–0.8 gray zone, inspect; < 0.6 likely wrong (interface) | AlphaFold-Multimer / AF3 guidance |
| PAE | < 5 Å: relative placement of the two positions is reliable; > ~15 Å: unreliable | AlphaFold documentation |
| pDockQ | > 0.23 acceptable or better; > 0.5 confident | Bryant 2022 |
| pDockQ2 | estimates DockQ, so DockQ classes apply: < 0.23 incorrect; ≥ 0.23 acceptable; ≥ 0.49 medium; ≥ 0.80 high | Zhu 2023; Basu & Wallner 2016 |
| ipSAE | no published universal cutoff; in our 720-model validation known binders scored ≥ 0.88 and decoys ≤ 0.10 — values above ≈ 0.5 indicate a confidently predicted interface | Dunbrack 2025 + our validation |
| LIS | higher is better; the authors propose ≈ 0.2 as the interaction cutoff | Kim 2024 |
| DockQ | < 0.23 incorrect; 0.23–0.49 acceptable; 0.49–0.80 medium; ≥ 0.80 high | Basu & Wallner 2016 (CAPRI classes) |

Single scores can mislead — pDockQ ignores PAE and can stay deceptively
high on confidently-folded but wrongly-docked chains, which pDockQ2/ipSAE
expose. Read them together (that is rather the point of this package).

## 🧰 Supported tools and files

| Tool | Detected files | pTM/ipTM | pLDDT | PAE |
|---|---|---|---|---|
| ColabFold | `*_scores_rank_*.json` + `*_(un)relaxed_rank_*.pdb` | yes | yes | yes |
| AlphaFold2 (pickle layout) | `result_model_*.pkl` + `unrelaxed_*.pdb` / `ranked_*.pdb` | yes | yes | yes |
| AlphaFold2 (JSON layout) | `iptm_ptm.json` + `confidence_*.json` / `pae_*.json` + `unrelaxed_*.cif/.pdb` | yes | yes | yes |
| AlphaFold3 (server/local) | `*model*.cif` + `*summary_confidences*.json` + `*confidences*/full_data*.json` | yes | yes | yes |
| Boltz-1/2 | `confidence_*_model_*.json` + `*_model_*.cif` + `pae_*.npz` / `plddt_*.npz` | yes | yes | yes |
| Chai-1 | `scores.model_idx_*.npz` + `pred.model_idx_*.cif` | yes | yes | if exported |
| Protenix | `*summary_confidence*.json` + matching `.cif` (+ `*full_data*.json` with `token_pair_pae`) | yes | yes | yes |

Native per-tool extras (e.g. Boltz `complex_iplddt`/`ligand_iptm`, AF3
`chain_pair_pae_min`, Chai clash flags) are preserved on
`Prediction.extras` and chain-pair ipTM is surfaced as `iptm_native` in the
interface table.

## ✅ Validation

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
  fixtures only; real-output samples welcome (please open an issue).

## 📋 What each metric needs

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
| `contacts` | reported | yes | recommended | distance always; PAE filter when present |
| `dockq` | – | yes | – | plus a reference structure (`--ref`) |

## 📁 Outputs and paths

- Summary table → stdout; `-o FILE` writes it. The extension picks the
  format: `.tsv` (default), `.csv`, `.json`; missing values are `NA`.
- `--interfaces FILE` → the per chain-pair table (same formats).
- `--plot DIR` → figures as described under Visualization; `contacts
  --plot` adds `<model>_contacts.png` + `<model>_contacts.pml`. Model names
  are sanitized (`[^\w.-]` → `_`) for use as filenames.
- `plot -o DIR` defaults to `./foldmetrics_plots/`.
- Exit codes: `0` success, `1` nothing recognized/found, `2` bad arguments
  or missing optional dependency.

## 💡 Conventions worth knowing

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

## 📚 References

- Jumper J et al. *Highly accurate protein structure prediction with
  AlphaFold.* Nature 596, 583–589 (2021).
  [doi:10.1038/s41586-021-03819-2](https://doi.org/10.1038/s41586-021-03819-2)
  — pLDDT / PAE and their confidence bands
- Bryant P, Pozzati G, Elofsson A. *Improved prediction of protein-protein
  interactions using AlphaFold2.* Nat Commun 13, 1265 (2022).
  [doi:10.1038/s41467-022-28865-w](https://doi.org/10.1038/s41467-022-28865-w)
  — pDockQ
- Zhu W, Shenoy A, Kundrotas P, Elofsson A. *Evaluation of AlphaFold-Multimer
  prediction on multi-chain protein complexes.* Bioinformatics 39, btad424
  (2023).
  [doi:10.1093/bioinformatics/btad424](https://doi.org/10.1093/bioinformatics/btad424)
  — pDockQ2
- Dunbrack RL. *ipSAE: scoring pairwise interactions in AlphaFold models.*
  bioRxiv (2025).
  [doi:10.1101/2025.02.10.637595](https://doi.org/10.1101/2025.02.10.637595)
  — ipSAE
- Kim AR et al. *Enhanced protein-protein interaction discovery via
  AlphaFold-Multimer.* bioRxiv (2024).
  [doi:10.1101/2024.02.19.580970](https://doi.org/10.1101/2024.02.19.580970)
  — LIS
- Basu S, Wallner B. *DockQ: A quality measure for protein-protein docking
  models.* PLoS ONE 11, e0161879 (2016).
  [doi:10.1371/journal.pone.0161879](https://doi.org/10.1371/journal.pone.0161879);
  Mirabello C, Wallner B. *DockQ v2.* Bioinformatics (2024).
  [github.com/bjornwallner/DockQ](https://github.com/bjornwallner/DockQ)
  — DockQ
- Zhang J, Pei J, Durham J, Bos T, Cong Q. *Computed cancer interactome
  explains the effects of somatic mutations in cancers.* Protein Sci 31,
  e4479 (2022).
  [doi:10.1002/pro.4479](https://doi.org/10.1002/pro.4479)
  — confident-contact criteria

## 📄 License

[MIT](LICENSE)
