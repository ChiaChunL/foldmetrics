from __future__ import annotations

import math

import pytest
from conftest import write_colabfold_dir

from foldmetrics.cli import main
from foldmetrics.dockq import dockq_class, parse_mapping

pytest.importorskip("DockQ")


def test_dockq_class_boundaries():
    assert dockq_class(0.95) == "high"
    assert dockq_class(0.60) == "medium"
    assert dockq_class(0.30) == "acceptable"
    assert dockq_class(0.10) == "incorrect"


def test_parse_mapping():
    assert parse_mapping("A:A,B:D") == {"A": "A", "B": "D"}
    with pytest.raises(ValueError):
        parse_mapping("AB")


def test_dockq_self_comparison_is_perfect(tmp_path):
    from foldmetrics.dockq import compute_dockq

    directory = write_colabfold_dir(tmp_path / "cf")
    structure = next(directory.glob("*_unrelaxed_*.pdb"))
    rows, total = compute_dockq(structure, structure)
    assert rows, "expected at least one interface"
    assert math.isclose(total, 1.0, abs_tol=1e-6)
    assert rows[0]["dockq_class"] == "high"
    assert math.isclose(rows[0]["fnat"], 1.0, abs_tol=1e-6)
    assert rows[0]["irmsd"] < 1e-6


def test_dockq_cli(tmp_path, capsys):
    directory = write_colabfold_dir(tmp_path / "cf")
    structure = next(directory.glob("*_unrelaxed_*.pdb"))
    out = tmp_path / "dockq.tsv"
    code = main(["dockq", str(directory), "--ref", str(structure), "-o", str(out)])
    assert code == 0
    assert "dockq" in capsys.readouterr().out.lower()
    assert out.exists()
