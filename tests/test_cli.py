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


def test_score_by_target(tmp_path, capsys):
    from conftest import write_boltz_dir

    write_colabfold_dir(tmp_path / "complex1" / "cf")
    write_boltz_dir(tmp_path / "complex2" / "boltz")
    out = tmp_path / "agg.tsv"
    code = main(["score", str(tmp_path), "--by-target", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "n_models" in text and "best_model" in text
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 3  # header + two (target, tool) groups
    assert "ipsae_mean" in lines[0]


def test_aggregate_by_target_api(tmp_path):
    from foldmetrics.api import aggregate_by_target, evaluate

    write_colabfold_dir(tmp_path / "t1" / "cf")
    df = evaluate(tmp_path)
    agg = aggregate_by_target(df)
    assert len(agg) == 1
    row = agg.iloc[0]
    assert row["n_models"] == 1
    assert row["best_model"] == df.iloc[0]["model"]
    assert row["iptm_max"] == df.iloc[0]["iptm"]


def test_score_empty_dir_errors(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["score", str(empty)]) == 1


def test_score_by_target_aggregation(tmp_path, capsys):
    from conftest import write_boltz_dir

    write_colabfold_dir(tmp_path / "targetA" / "cf")
    write_boltz_dir(tmp_path / "targetA" / "boltz")
    out = tmp_path / "by_target.tsv"
    assert main(["score", str(tmp_path), "--by-target", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "n_models" in printed and "best_model" in printed
    header, *rows = out.read_text().strip().splitlines()
    assert "ipsae_mean" in header and "ipsae_std" in header
    assert len(rows) == 2  # one row per (target, tool)
