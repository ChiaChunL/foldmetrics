# Examples

Everything here runs without any prediction tool installed: `make_demo_data.py`
generates realistic synthetic AlphaFold3-style outputs to play with. Substitute
your real prediction folders anywhere a `demo_predictions` path appears.

```bash
python examples/make_demo_data.py demo_predictions
```

## CLI recipes

`fmx` is a short alias for `foldmetrics`; they are identical.

### Score a batch (any mix of supported tools, scanned recursively)

```bash
fmx score demo_predictions -o metrics.tsv --interfaces interfaces.tsv --plot plots
```

stdout (also written to `metrics.tsv`; missing values are written as `NA`):

```
                   model       tool chains  n_chains  n_tokens  n_res   ptm  iptm  ranking_score  plddt_mean  iplddt  pae_mean  ipae_mean  ipsae  pdockq  pdockq2   lis  n_interfaces  has_pae
complex_with_ligand_poor alphafold3  A,B,C         3       180    170 0.680 0.380          0.450      69.624  68.924    12.315     19.943  0.266   0.467    0.011 0.473             1     True
              dimer_good alphafold3    A,B         2       195    195 0.870 0.820          0.840      88.462  88.645     3.824      4.310  0.488   0.712    0.436 0.641             1     True
 receptor_peptide_medium alphafold3    A,B         2       170    170 0.720 0.550          0.580      81.796  72.580     6.130     11.326  0.052   0.278    0.029 0.142             1     True
```

Reading `complex_with_ligand_poor` is instructive: `pdockq` is deceptively
decent (0.47 — it only sees contacts and pLDDT) while `pdockq2` (0.011) and the
per-pair ipSAE expose the bad protein–protein interface; the summary `ipsae`
(0.266) comes from the *best* interface, which is the protein–ligand pair.
The `--interfaces` table shows exactly this breakdown per chain pair, including
the two directional values (`*_ab`, `*_ba`) and the tool's own chain-pair ipTM
(`iptm_native`).

`--plot plots/` writes one summary figure per model (pLDDT-colored structure
+ pLDDT track + PAE heatmap + metrics panel), plus `batch_overview.png` when
there are several models, plus `comparison.png` (targets x methods, one panel
per metric) when the batch spans several targets or tools. The structure
panel uses headless PyMOL when installed (auto-detected; ~2-3 s per model) and
a fast matplotlib backbone trace otherwise:

```bash
fmx plot demo_predictions -o plots --renderer trace   # fast, no PyMOL
fmx plot demo_predictions -o plots --format pdf       # vector output
```

### Single metrics

Every metric name is also a subcommand, so computing just one metric over a
batch is:

```bash
fmx ipsae demo_predictions                      # only ipSAE columns
fmx pdockq2 demo_predictions -o pdockq2.tsv     # same options as 'score'
fmx score demo_predictions --metrics ipsae,lis  # any subset
```

Available names: `ptm`, `iptm`, `ranking`, `plddt`, `pae`, `ipsae`,
`pdockq`, `pdockq2`, `lis`.

### Confident interface contacts

```bash
fmx contacts demo_predictions -o contacts.tsv --plot plots
fmx contacts preds/ --dist-cutoff 6 --pae-cutoff 10     # stricter
fmx contacts preds/ --pae-cutoff -1                     # distance-only
```

Writes the residue-pair table, a contact figure (structure highlight + PAE
overlay), and a `<model>_contacts.pml` PyMOL script with `if_<chain>` /
`interface` selections.

### DockQ vs a reference structure

Requires `pip install "foldmetrics[dockq]"`:

```bash
fmx dockq preds/ --ref native.pdb -o dockq.tsv
fmx dockq preds/ --ref native.cif --mapping A:A,B:D    # model:reference chains
```

### Other commands

```bash
fmx detect demo_predictions            # list what would be scored, per tool
fmx plot demo_predictions -o plots     # figures only
fmx score run1/fold_x_full_data_0.json # a single model, via any of its files
fmx score preds/ --tool boltz          # restrict auto-detection to one tool
fmx score preds/ --pae-cutoff 15       # loosen the ipSAE PAE cutoff
fmx score run1/ run2/ model_dir/       # any number of paths
```

Output formats follow the file extension: `.tsv` (default), `.csv`, `.json`.

## Python API

See [python_api.py](python_api.py) for a complete walkthrough:

```bash
python examples/python_api.py demo_predictions
```

It covers:

1. `fm.evaluate(path)` — summary DataFrame (one row per model)
2. `fm.evaluate_interfaces(path)` — per chain-pair DataFrame
3. `fm.load_predictions` + `fm.compute_all` — full access to one model,
   including per-residue ipSAE profiles via `foldmetrics.metrics.ipsae_asym`
4. Figures: `save_summary_plot`, `save_batch_plot`, and composing your own
   figure from `plot_plddt` / `plot_pae`
