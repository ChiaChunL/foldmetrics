"""3D structure rendering and viewer-session generation.

Static renders use headless PyMOL (auto-detected) with a matplotlib
backbone-trace fallback handled in :mod:`foldmetrics.viz`.

For interface contacts a shared "publication preset" is emitted in four
forms so users can double-click straight into a styled scene:

- ``.pml`` — PyMOL command script (text, always written)
- ``.pse`` — PyMOL session (written when PyMOL is installed)
- ``.cxc`` — ChimeraX command script (text, always written)
- ``.cxs`` — ChimeraX session (written when ChimeraX is installed)

The preset: pastel cartoons per chain, interface residues as sticks in the
chain's accent color (heteroatoms by element), the closest contact
residues highlighted in orange and labeled.

Executable discovery: ``FOLDMETRICS_PYMOL`` / ``FOLDMETRICS_CHIMERAX``
environment variables, then PATH, then common install locations.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

_PYMOL_GLOBS = (
    "/Applications/miniconda*/envs/*/bin/pymol",
    "/Applications/PyMOL.app/Contents/MacOS/PyMOL",
    "~/miniconda*/envs/*/bin/pymol",
    "~/miniforge*/envs/*/bin/pymol",
    "~/mambaforge*/envs/*/bin/pymol",
    "~/anaconda*/envs/*/bin/pymol",
    "/opt/conda/envs/*/bin/pymol",
    "/opt/homebrew/bin/pymol",
    "/usr/local/bin/pymol",
)

_CHIMERAX_GLOBS = (
    "/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX",
    "/Applications/UCSF ChimeraX*.app/Contents/MacOS/ChimeraX",
    "/usr/bin/chimerax",
    "/usr/local/bin/chimerax",
    "/opt/UCSF/ChimeraX*/bin/ChimeraX",
)

# Cartoon tints (pastel) and accent colors per chain, fixed order.
PASTEL_CHAIN_COLORS = ["#F2F2F2", "#CBBFE8", "#C4E3D2", "#F6DDBA", "#C2DAEF", "#EFCADD"]
HOTSPOT_COLOR = "#D55E00"


@lru_cache(maxsize=1)
def find_pymol() -> str | None:
    """Locate a PyMOL executable, or None if not installed."""
    env = os.environ.get("FOLDMETRICS_PYMOL")
    if env:
        return env if Path(env).exists() else None
    exe = shutil.which("pymol")
    if exe:
        return exe
    for pattern in _PYMOL_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[0]
    return None


@lru_cache(maxsize=1)
def find_chimerax() -> str | None:
    """Locate a ChimeraX executable, or None if not installed."""
    env = os.environ.get("FOLDMETRICS_CHIMERAX")
    if env:
        return env if Path(env).exists() else None
    for name in ("chimerax", "ChimeraX"):
        exe = shutil.which(name)
        if exe:
            return exe
    for pattern in _CHIMERAX_GLOBS:
        hits = sorted(glob.glob(os.path.expanduser(pattern)))
        if hits:
            return hits[-1]  # newest version
    return None


def pastel_color(i: int) -> str:
    return PASTEL_CHAIN_COLORS[i % len(PASTEL_CHAIN_COLORS)]


def _resi_ranges(res_ids: list[int], sep: str = "+") -> str:
    """Compress residue ids: PyMOL ``3+7-10`` (sep '+') or ChimeraX ``3,7-10`` (sep ',')."""
    ids = sorted(set(res_ids))
    parts: list[str] = []
    start = prev = ids[0]
    for r in ids[1:] + [None]:  # type: ignore[list-item]
        if r is not None and r == prev + 1:
            prev = r
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if r is not None:
            start = prev = r
    return sep.join(parts)


def _hex_rgb(color: str) -> list[float]:
    color = color.lstrip("#")
    return [round(int(color[i : i + 2], 16) / 255.0, 3) for i in (0, 2, 4)]


# ------------------------------------------------------------- PyMOL preset
def contacts_pymol_commands(
    structure_path: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
    labels: bool = True,
) -> list[str]:
    """Publication-preset PyMOL commands highlighting interface contacts.

    Flat, outlined look (``ray_trace_mode 1``); cartoon colors are pinned
    per chain so residue highlighting never bleeds onto the cartoon;
    ``pairs`` draws dashed contact lines between specific atoms
    ((chain_a, res_a, atom_a, chain_b, res_b, atom_b) tuples).
    """
    lines = [
        "bg_color white",
        f"load {structure_path}, fm_model",
        "hide everything",
        "show cartoon",
        "set cartoon_fancy_helices, 1",
        "set cartoon_side_chain_helper, 1",
        "set ray_shadows, 0",
        "set specular, 0",
        "set ambient, 0.45",
        "set ray_trace_mode, 1",
        "set ray_trace_color, gray30",
        "set stick_radius, 0.22",
        "set dash_color, gray50",
        "set dash_gap, 0.45",
        "set dash_width, 2.0",
        "show sticks, hetatm and not solvent",
        "util.cnc hetatm and not solvent",
    ]
    for i, chain in enumerate(chains):
        lines.append(f"set_color fm_pastel_{chain}, {_hex_rgb(pastel_color(i))}")
        lines.append(f"color fm_pastel_{chain}, chain {chain}")
        # pin the cartoon color: residue recoloring must not bleed onto it
        lines.append(f"set cartoon_color, fm_pastel_{chain}, chain {chain}")

    selections = []
    for chain, res_ids in highlight.items():
        if not res_ids:
            continue
        name = f"if_{chain}"
        lines.append(f"set_color fm_accent_{chain}, {_hex_rgb(chain_colors[chain])}")
        lines.append(f"select {name}, chain {chain} and resi {_resi_ranges(res_ids)}")
        lines.append(f"show sticks, {name}")
        lines.append(f"color fm_accent_{chain}, {name}")
        lines.append(f"util.cnc {name}")
        selections.append(name)

    hot_parts = [
        f"(chain {chain} and resi {_resi_ranges(res_ids)})"
        for chain, res_ids in hotspots.items()
        if res_ids
    ]
    if hot_parts:
        lines.append(f"select hotspots, {' or '.join(hot_parts)}")
        lines.append(f"set_color fm_hotspot, {_hex_rgb(HOTSPOT_COLOR)}")
        lines.append("color fm_hotspot, hotspots and elem C")
    if labels and selections:
        # open-source PyMOL builds lack `one_letter` in the label namespace;
        # inject our own via `stored`, which label expressions can always see
        one_letter = (
            "{'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q',"
            "'GLU':'E','GLY':'G','HIS':'H','ILE':'I','LEU':'L','LYS':'K',"
            "'MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',"
            "'TYR':'Y','VAL':'V'}"
        )
        lines.append(f"/from pymol import stored; stored.fm_one = {one_letter}")
        lines.append(
            f"label ({' or '.join(selections)}) and name CA and polymer, "
            "stored.fm_one.get(resn, resn)+resi"
        )
        lines.append("set label_size, 13")
        lines.append("set label_color, gray40")

    for k, (ca, ra, aa, cb, rb, ab) in enumerate(pairs or []):
        lines.append(
            f"distance fm_contact_{k}, "
            f"fm_model and chain {ca} and resi {ra} and name {aa}, "
            f"fm_model and chain {cb} and resi {rb} and name {ab}"
        )
        lines.append(f"hide labels, fm_contact_{k}")

    if selections:
        lines.append(f"select interface, {' or '.join(selections)}")
        lines.append("orient interface")
        lines.append("zoom interface, 8")
    else:
        lines.append("orient")
    lines.append("deselect")
    lines.append("set ray_opaque_background, 1")
    lines.append("set antialias, 2")
    return lines


def write_contacts_pml(
    pml_path: str | Path,
    structure_path: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
    header: str = "",
) -> Path:
    """Write an interactive PyMOL script (double-click / ``@script.pml``)."""
    pml_path = Path(pml_path)
    pml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {header}"] if header else []
    lines += contacts_pymol_commands(
        Path(structure_path).resolve(), chains, highlight, hotspots, chain_colors, pairs
    )
    pml_path.write_text("\n".join(lines) + "\n")
    return pml_path


def _run_pymol_script(script: str, out_file: Path, timeout: int) -> Path | None:
    exe = find_pymol()
    if exe is None:
        return None
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        script_path = fh.name
    try:
        result = subprocess.run(
            [exe, "-cq", script_path], capture_output=True, timeout=timeout, check=False
        )
        if result.returncode != 0 or not out_file.exists():
            return None
        return out_file
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


def save_contacts_pse(
    pse_path: str | Path,
    structure_path: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
    timeout: int = 300,
) -> Path | None:
    """PyMOL session with the contact preset applied; None if no PyMOL."""
    structure_path = Path(structure_path)
    if not structure_path.exists():
        return None
    pse_path = Path(pse_path)
    commands = contacts_pymol_commands(
        structure_path.resolve(), chains, highlight, hotspots, chain_colors, pairs
    )
    script = "from pymol import cmd, util\n" + "\n".join(
        f"cmd.do({line!r})" for line in commands
    )
    script += f"\ncmd.save({str(pse_path)!r})\n"
    return _run_pymol_script(script, pse_path, timeout)


def render_contacts(
    structure_path: str | Path,
    out_png: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
    width: int = 1400,
    height: int = 1150,
    dpi: int = 300,
    timeout: int = 300,
) -> Path | None:
    """Ray-traced PNG of the contact preset; None if no PyMOL."""
    structure_path = Path(structure_path)
    if not structure_path.exists():
        return None
    out_png = Path(out_png)
    commands = contacts_pymol_commands(
        structure_path.resolve(), chains, highlight, hotspots, chain_colors, pairs
    )
    script = "from pymol import cmd, util\n" + "\n".join(
        f"cmd.do({line!r})" for line in commands
    )
    script += f"\ncmd.ray({width}, {height})\ncmd.png({str(out_png)!r}, dpi={dpi})\n"
    return _run_pymol_script(script, out_png, timeout)


# ---------------------------------------------------------- ChimeraX preset
def contacts_cxc_commands(
    structure_path: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
) -> list[str]:
    """Publication-preset ChimeraX commands (silhouette style).

    ``pairs`` draws dashed contact lines between specific atoms
    ((chain_a, res_a, atom_a, chain_b, res_b, atom_b) tuples).
    """
    lines = [
        f"open {structure_path}",
        "set bgColor white",
        "hide atoms",
        "show cartoons",
        "lighting soft",
        "graphics silhouettes true",
        "show ligand atoms",
        "style ligand stick",
        "color ligand byhetero",
    ]
    for i, chain in enumerate(chains):
        lines.append(f"color /{chain} {pastel_color(i)} target c")

    interface_specs = []
    for chain, res_ids in highlight.items():
        if not res_ids:
            continue
        spec = f"/{chain}:{_resi_ranges(res_ids, sep=',')}"
        lines.append(f"show {spec} atoms")
        lines.append(f"style {spec} stick")
        lines.append(f"color {spec} {chain_colors[chain]} target a")
        lines.append(f"color {spec} byhetero")
        interface_specs.append(spec)

    # dashed contact lines first, so the later residue labels survive
    if pairs:
        for ca, ra, aa, cb, rb, ab in pairs:
            lines.append(f"distance /{ca}:{ra}@{aa} /{cb}:{rb}@{ab}")
        lines.append("distance style color #666666 radius 0.05 dashes 6")
        lines.append("label delete pseudobonds")

    hot_specs = [
        f"/{chain}:{_resi_ranges(res_ids, sep=',')}"
        for chain, res_ids in hotspots.items()
        if res_ids
    ]
    if hot_specs:
        hot = "|".join(hot_specs)
        lines.append(f"color {hot} {HOTSPOT_COLOR} target a")
        lines.append(f"color {hot} byhetero")
        lines.append(
            f'label {hot} residues text "{{0.label_one_letter_code}}{{0.number}}"'
        )
    if interface_specs:
        lines.append(f"view {'|'.join(interface_specs)}")
    else:
        lines.append("view")
    return lines


def write_contacts_cxc(
    cxc_path: str | Path,
    structure_path: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
    header: str = "",
) -> Path:
    """Write an interactive ChimeraX script (open it in ChimeraX to run)."""
    cxc_path = Path(cxc_path)
    cxc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {header}"] if header else []
    lines += contacts_cxc_commands(
        Path(structure_path).resolve(), chains, highlight, hotspots, chain_colors, pairs
    )
    cxc_path.write_text("\n".join(lines) + "\n")
    return cxc_path


def save_contacts_cxs(
    cxs_path: str | Path,
    structure_path: str | Path,
    chains: list[str],
    highlight: dict[str, list[int]],
    hotspots: dict[str, list[int]],
    chain_colors: dict[str, str],
    pairs: list[tuple] | None = None,
    timeout: int = 300,
) -> Path | None:
    """ChimeraX session with the contact preset applied; None if no ChimeraX."""
    exe = find_chimerax()
    structure_path = Path(structure_path)
    if exe is None or not structure_path.exists():
        return None
    cxs_path = Path(cxs_path)
    cxs_path.parent.mkdir(parents=True, exist_ok=True)

    commands = contacts_cxc_commands(
        structure_path.resolve(), chains, highlight, hotspots, chain_colors, pairs
    )
    commands += [f"save {cxs_path.resolve()} format session", "exit"]
    with tempfile.NamedTemporaryFile("w", suffix=".cxc", delete=False) as fh:
        fh.write("\n".join(commands) + "\n")
        script_path = fh.name
    try:
        result = subprocess.run(
            [exe, "--nogui", "--silent", script_path],
            capture_output=True, timeout=timeout, check=False,
        )
        if result.returncode != 0 or not cxs_path.exists():
            return None
        return cxs_path
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


# ------------------------------------------------------- pLDDT still render
_PYMOL_SCRIPT = """\
from pymol import cmd

cmd.bg_color("white")
cmd.load({structure!r}, "m")
cmd.hide("everything")
cmd.show("cartoon")
cmd.show("sticks", "hetatm and not solvent")
cmd.set("stick_radius", 0.18)
cmd.set_color("af_vhigh", [0.000, 0.325, 0.839])
cmd.set_color("af_high", [0.396, 0.796, 0.953])
cmd.set_color("af_low", [1.000, 0.859, 0.075])
cmd.set_color("af_vlow", [1.000, 0.490, 0.271])
cmd.color("af_vlow", "all")
cmd.color("af_low", "b > 50")
cmd.color("af_high", "b > 70")
cmd.color("af_vhigh", "b > 90")
cmd.orient()
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("cartoon_fancy_helices", 1)
cmd.set("specular", 0.2)
cmd.ray({width}, {height})
cmd.png({out!r}, dpi={dpi})
"""


def render_structure(
    structure_path: str | Path,
    out_png: str | Path,
    width: int = 1400,
    height: int = 1150,
    dpi: int = 300,
    timeout: int = 300,
) -> Path | None:
    """Render a pLDDT-colored cartoon PNG with PyMOL; None if unavailable/failed."""
    structure_path = Path(structure_path)
    if not structure_path.exists():
        return None
    out_png = Path(out_png)
    script = _PYMOL_SCRIPT.format(
        structure=str(structure_path), out=str(out_png),
        width=width, height=height, dpi=dpi,
    )
    return _run_pymol_script(script, out_png, timeout)
