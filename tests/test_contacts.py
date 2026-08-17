from __future__ import annotations

import math

from conftest import make_two_chain_pred, write_colabfold_dir

from foldmetrics.cli import main
from foldmetrics.metrics import find_contacts


def test_contacts_distance_and_pae_filters():
    # gap=5, spacing=4 -> 3n-2 geometric contacts between the chains
    pred = make_two_chain_pred(n_a=20, n_b=20, pae=5.0)
    rows = find_contacts(pred, dist_cutoff=8.0, pae_cutoff=12.0)
    assert len(rows) == 3 * 20 - 2
    assert all(r["distance"] <= 8.0 for r in rows)
    assert all(max(r["pae_ab"], r["pae_ba"]) < 12.0 for r in rows)
    # tighter distance excludes the diagonal neighbours (6.40 Å)
    assert len(find_contacts(pred, dist_cutoff=5.5, pae_cutoff=12.0)) == 20


def test_contacts_pae_direction_is_strict():
    # one direction confident, the other not -> max(PAE) filter rejects all
    pred = make_two_chain_pred(pae_ab=4.0, pae_ba=20.0)
    assert find_contacts(pred, pae_cutoff=12.0) == []
    # disabling the PAE filter keeps the geometric contacts
    assert len(find_contacts(pred, pae_cutoff=None)) == 3 * 40 - 2


def test_contacts_without_pae_fall_back_to_distance():
    pred = make_two_chain_pred(n_a=10, n_b=10, pae=None)
    rows = find_contacts(pred)
    assert len(rows) == 3 * 10 - 2
    assert all(math.isnan(r["pae_ab"]) for r in rows)


def test_contacts_cli(tmp_path, capsys):
    directory = write_colabfold_dir(tmp_path / "cf")
    out = tmp_path / "contacts.tsv"
    plots = tmp_path / "plots"
    code = main(
        ["contacts", str(directory), "-o", str(out),
         "--plot", str(plots), "--renderer", "trace", "--no-sessions"]
    )
    assert code == 0
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1 + (3 * 30 - 2)  # header + contact rows (PAE=5 passes)
    assert "distance" in lines[0]
    assert len(list(plots.glob("*_contacts.png"))) == 1

    pml = list(plots.glob("*_contacts.pml"))
    assert len(pml) == 1
    pml_text = pml[0].read_text()
    assert "select if_A" in pml_text
    assert "select hotspots" in pml_text  # labeled closest contacts

    cxc = list(plots.glob("*_contacts.cxc"))
    assert len(cxc) == 1
    cxc_text = cxc[0].read_text()
    assert cxc_text.startswith("#")
    assert "show cartoons" in cxc_text
    assert "byhetero" in cxc_text
