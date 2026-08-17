"""3D structure rendering for figures.

Primary renderer: headless PyMOL (auto-detected), producing a ray-traced
cartoon colored by pLDDT with the AlphaFold confidence colors. When PyMOL
is unavailable, figures fall back to a matplotlib backbone trace drawn by
:func:`foldmetrics.viz.plot_structure_trace`.

PyMOL discovery order: the ``FOLDMETRICS_PYMOL`` environment variable, the
``pymol`` executable on PATH, then common conda/app install locations.
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

# AlphaFold pLDDT confidence colors as PyMOL RGB triples.
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


def render_structure(
    structure_path: str | Path,
    out_png: str | Path,
    width: int = 1400,
    height: int = 1150,
    dpi: int = 300,
    timeout: int = 300,
) -> Path | None:
    """Render a pLDDT-colored cartoon PNG with PyMOL; None if unavailable/failed."""
    exe = find_pymol()
    structure_path = Path(structure_path)
    if exe is None or not structure_path.exists():
        return None
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    script = _PYMOL_SCRIPT.format(
        structure=str(structure_path), out=str(out_png),
        width=width, height=height, dpi=dpi,
    )
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        script_path = fh.name
    try:
        result = subprocess.run(
            [exe, "-cq", script_path],
            capture_output=True, timeout=timeout, check=False,
        )
        if result.returncode != 0 or not out_png.exists():
            return None
        return out_png
    except (OSError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
