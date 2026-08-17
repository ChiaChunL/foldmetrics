"""DockQ against a reference structure (optional integration).

Wraps the official DockQ implementation (Basu & Wallner, PLoS ONE 2016;
DockQ v2: Mirabello & Wallner, Bioinformatics 2024). Install with::

    pip install "foldmetrics[dockq]"

Unlike every other foldmetrics score, DockQ is a *true accuracy* measure:
it needs a reference (experimental or trusted) structure to compare
against, and reports per-interface DockQ / fnat / iRMSD / LRMSD plus the
standard CAPRI-style quality class.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from typing import Any

# CAPRI-style DockQ quality classes (Basu & Wallner 2016).
DOCKQ_CLASSES = [(0.80, "high"), (0.49, "medium"), (0.23, "acceptable"), (0.0, "incorrect")]


def dockq_class(value: float) -> str:
    for threshold, name in DOCKQ_CLASSES:
        if value >= threshold:
            return name
    return "incorrect"


def _require_dockq():
    try:
        from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
    except ImportError as exc:  # pragma: no cover - depends on extras
        raise ImportError(
            "the DockQ package is required for this feature; "
            'install it with: pip install "foldmetrics[dockq]"'
        ) from exc
    return load_PDB, run_on_all_native_interfaces


def _rows_from_result(result: dict, chain_map: dict[str, str]) -> list[dict[str, Any]]:
    mapping_str = ",".join(f"{m}:{n}" for n, m in chain_map.items())
    rows: list[dict[str, Any]] = []
    for key, r in result.items():
        native_pair = "-".join(key) if isinstance(key, (tuple, list)) else str(key)
        value = float(r["DockQ"])
        rows.append(
            {
                "interface": f"{r.get('chain1', '?')}-{r.get('chain2', '?')}",
                "native_pair": native_pair,
                "dockq": value,
                "dockq_class": dockq_class(value),
                "fnat": float(r.get("fnat", float("nan"))),
                "fnonnat": float(r.get("fnonnat", float("nan"))),
                "f1": float(r.get("F1", float("nan"))),
                "irmsd": float(r.get("iRMSD", float("nan"))),
                "lrmsd": float(r.get("LRMSD", float("nan"))),
                "clashes": int(r.get("clashes", 0)),
                "len_a": int(r.get("len1", 0)),
                "len_b": int(r.get("len2", 0)),
                "mapping": mapping_str,
            }
        )
    rows.sort(key=lambda x: x["interface"])
    return rows


def compute_dockq(
    model_path: str | Path,
    native_path: str | Path,
    mapping: dict[str, str] | None = None,
    small_molecule: bool = False,
    capri_peptide: bool = False,
    no_align: bool = False,
    low_memory: bool = False,
    best_mapping: bool = False,
    max_search_chains: int = 5,
) -> tuple[list[dict[str, Any]], float]:
    """DockQ of one model against a reference structure.

    ``mapping`` maps **model chains to reference chains** (e.g.
    ``{"A": "A", "B": "D"}``). Without it, chains are matched by name when
    the two structures use the same names, otherwise by order of
    appearance. Note that mmCIF chain naming can differ between tools
    (label vs auth asym ids), so check the ``mapping`` column of the result.

    ``best_mapping=True`` instead tries every chain assignment and keeps
    the one maximizing total DockQ — important for homo-multimers where
    name/order pairing is ambiguous. The search is factorial in the chain
    count and therefore refused above ``max_search_chains`` chains (use an
    explicit ``mapping`` there).

    ``no_align`` skips the sequence alignment (only when model and
    reference share residue numbering); ``low_memory`` reduces memory use
    on very large complexes; ``capri_peptide`` applies CAPRI peptide
    criteria; ``small_molecule`` scores ligand poses too.

    Returns (per-interface rows, total DockQ).
    """
    load_pdb, run_all = _require_dockq()
    model = load_pdb(str(model_path), small_molecule=small_molecule)
    native = load_pdb(str(native_path), small_molecule=small_molecule)
    model_chains = [c.id for c in model]
    native_chains = [c.id for c in native]

    def run(chain_map: dict[str, str]):
        return run_all(
            model, native, chain_map=chain_map,
            no_align=no_align, capri_peptide=capri_peptide, low_memory=low_memory,
        )

    if mapping is not None:
        # user gives model->native; the DockQ API wants native->model
        chain_maps = [{native: model for model, native in mapping.items()}]
    elif best_mapping:
        n_assign = min(len(model_chains), len(native_chains))
        if max(len(model_chains), len(native_chains)) > max_search_chains:
            raise ValueError(
                f"best-mapping search over {len(model_chains)}x{len(native_chains)} "
                f"chains is too large; provide an explicit mapping instead"
            )
        natives = native_chains[:n_assign] if len(native_chains) <= len(model_chains) \
            else native_chains
        chain_maps = [
            dict(zip(natives, perm, strict=False))
            for perm in permutations(model_chains, len(natives))
        ]
    elif set(model_chains) == set(native_chains):
        chain_maps = [{c: c for c in native_chains}]
    else:
        chain_maps = [
            {
                native: model
                for native, model in zip(native_chains, model_chains, strict=False)
            }
        ]

    best: tuple[list[dict[str, Any]], float] | None = None
    last_error: Exception | None = None
    for chain_map in chain_maps:
        try:
            result, total = run(chain_map)
        except Exception as exc:  # noqa: BLE001 - other permutations may work
            last_error = exc
            continue
        if best is None or total > best[1]:
            best = (_rows_from_result(result, chain_map), float(total))
    if best is None:
        raise RuntimeError(f"DockQ failed for every chain mapping: {last_error}")
    return best


def parse_mapping(text: str) -> dict[str, str]:
    """Parse a CLI mapping string like ``A:A,B:D`` (model:reference)."""
    mapping: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"bad mapping segment {part!r}; expected model:reference pairs like A:A,B:D"
            )
        model_chain, native_chain = part.split(":", 1)
        mapping[model_chain.strip()] = native_chain.strip()
    if not mapping:
        raise ValueError("empty chain mapping")
    return mapping
