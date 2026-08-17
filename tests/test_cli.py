from __future__ import annotations

from conftest import write_colabfold_dir

from foldmetrics.cli import main


def test_score_stdout_and_tsv(tmp_path, capsys):
    directory = write_colabfold_dir(tmp_path / "cf")
    out = tmp_path / "metrics.tsv"
    interfaces = tmp_path / "interfaces.tsv"

    code = main(["score", str(directory), "-o", str(out), "--interfaces", str(interfaces)])
    assert code == 0

    captured = capsys.readouterr()
    assert "colabfold" in captured.out
    assert "ipsae" in captured.out.lower() or "ipsae" in out.read_text().lower()

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2  # header + one model
    assert "pdockq" in lines[0]
    assert interfaces.exists()


def test_detect(tmp_path, capsys):
    directory = write_colabfold_dir(tmp_path / "cf")
    assert main(["detect", str(directory)]) == 0
    assert "colabfold" in capsys.readouterr().out


def test_plot_writes_png(tmp_path):
    directory = write_colabfold_dir(tmp_path / "cf")
    plots = tmp_path / "plots"
    assert main(["plot", str(directory), "-o", str(plots), "--renderer", "trace"]) == 0
    pngs = list(plots.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].stat().st_size > 10_000


def test_plot_batch_and_comparison(tmp_path):
    from conftest import write_boltz_dir

    write_colabfold_dir(tmp_path / "t1" / "cf")
    write_boltz_dir(tmp_path / "t2" / "boltz")
    plots = tmp_path / "plots"
    code = main(
        ["plot", str(tmp_path / "t1"), str(tmp_path / "t2"),
         "-o", str(plots), "--renderer", "trace", "--dpi", "80"]
    )
    assert code == 0
    names = {p.name for p in plots.glob("*.png")}
    assert "batch_overview.png" in names
    assert "comparison.png" in names  # two tools and two targets present
    assert len(names) == 4  # two per-model figures + the two batch views


def test_metric_subcommand(tmp_path, capsys):
    directory = write_colabfold_dir(tmp_path / "cf")
    assert main(["ipsae", str(directory)]) == 0
    out = capsys.readouterr().out
    assert "ipsae" in out
    assert "pdockq" not in out  # filtered to the requested metric only


def test_score_metrics_filter(tmp_path, capsys):
    directory = write_colabfold_dir(tmp_path / "cf")
    assert main(["score", str(directory), "--metrics", "pdockq,lis"]) == 0
    out = capsys.readouterr().out
    assert "pdockq" in out and "lis" in out
    assert "ipsae" not in out
    assert main(["score", str(directory), "--metrics", "nope"]) == 2


def test_score_empty_dir_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["score", str(empty)]) == 1
