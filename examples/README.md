# Examples

[`data/`](data/) contains **real prediction outputs** (barnase–barstar by
three tools, plus SARS-CoV-2 Mpro + nirmatrelvir by AlphaFold3) so every
command below runs as-is from the repository root — no prediction tool
needed:

```
examples/data/
├── af3_server/       AlphaFold Server download   (cif + summary + full_data)
├── af2_multimer/     AlphaFold2-Multimer, JSON layout
├── boltz2/           Boltz-2                     (cif + json + npz)
└── af3_mpro_ligand/  AlphaFold3 local, protein + small-molecule ligand
```

`fmx` is a short alias for `foldmetrics`; they are identical.

## Score everything (any mix of tools, auto-detected)

```bash
fmx score examples/data -o metrics.tsv --interfaces interfaces.tsv --plot plots
```

Real output (note the chain-naming diversity across tools — handled
transparently):

```
                            model       tool   ptm  iptm  ranking_score  plddt_mean  ipsae  pdockq  pdockq2   lis
       model_1_multimer_v3_pred_0 alphafold2 0.945 0.937          0.939      98.094  0.897   0.535    0.952 0.787
                mpro_nirmatrelvir alphafold3 0.950 0.970          0.970      97.527  0.836      NA       NA 0.690
fold_barnase_barstar_s318_model_0 alphafold3 0.940 0.930          0.930      97.487  0.890   0.528    0.944 0.777
          barnase_barstar_model_0      boltz 0.965 0.959          0.965      96.702  0.935   0.498    0.939 0.780
```

(`mpro_nirmatrelvir` shows `NA` for pDockQ/pDockQ2 because its only
interface is protein–ligand; its ipSAE/LIS come from ligand-token scoring.)

## Single metrics

Every metric name is also a subcommand:

```bash
fmx ipsae examples/data                     # only ipSAE columns
fmx pdockq2 examples/data -o pdockq2.tsv    # same options as 'score'
fmx score examples/data --metrics ipsae,lis # any subset
```

Available names: `ptm`, `iptm`, `ranking`, `plddt`, `pae`, `ipsae`,
`pdockq`, `pdockq2`, `lis`.

## Confident interface contacts

```bash
fmx contacts examples/data/af3_server -o contacts.tsv --plot plots
fmx contacts examples/data/af3_mpro_ligand --plot plots   # drug binding site
fmx contacts preds/ --dist-cutoff 6 --pae-cutoff 10       # stricter
```

Writes the residue-pair table, a figure (structure highlight + PAE
overlay), and ready-to-open viewer files with a publication preset
applied: `_contacts.pse` (PyMOL session) and `_contacts.cxs` (ChimeraX
session) when those programs are installed, plus the `_contacts.pml` /
`_contacts.cxc` scripts always. Double-click a session and the styled
scene (pastel cartoons, interface sticks, labeled hotspot residues) is
there. On the barnase–barstar example the closest extracted contacts are
the literature hotspots (R59/R83/R87/H102 against D35/D39).

## DockQ vs a reference structure

Requires `pip install "foldmetrics[dockq]"`. Here the AlphaFold Server
model serves as the reference for the AlphaFold2 model of the same complex:

```bash
fmx dockq examples/data/af2_multimer --ref examples/data/af3_server/fold_barnase_barstar_s318_model_0.cif
```

```
                     model       tool interface native_pair  dockq dockq_class  fnat  irmsd  lrmsd  total_dockq
model_1_multimer_v3_pred_0 alphafold2       B-C          AB  0.973        high 0.964  0.309  0.537        0.973
```

Chains were paired automatically by order (AF2 names them B,C; the
reference uses A,B) — override with `--mapping MODEL:REF,...`, or use
`--best-mapping` to search every assignment (recommended for
homo-multimers). More switches: `--small-molecule`, `--no-align`,
`--low-memory`, `--capri-peptide`.

## Other commands

```bash
fmx detect examples/data               # list what would be scored, per tool
fmx plot examples/data -o plots        # figures only (batch + comparison views)
fmx plot examples/data --renderer trace --format pdf   # fast, vector output
fmx score run1/fold_x_full_data_0.json # a single model, via any of its files
fmx score preds/ --tool boltz          # restrict auto-detection to one tool
```

Output formats follow the file extension: `.tsv` (default), `.csv`, `.json`.

## Python API

See [python_api.py](python_api.py) for a complete walkthrough:

```bash
python examples/python_api.py examples/data
```

It covers:

1. `fm.evaluate(path)` — summary DataFrame (one row per model)
2. `fm.evaluate_interfaces(path)` — per chain-pair DataFrame
3. `fm.load_predictions` + `fm.compute_all` — full access to one model,
   including per-residue ipSAE profiles via `foldmetrics.metrics.ipsae_asym`
4. `fm.find_contacts` — confident-contact extraction
5. Figures: `save_summary_plot`, `save_batch_plot`, and composing your own
   figure from `plot_plddt` / `plot_pae`
