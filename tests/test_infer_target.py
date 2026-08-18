"""Path-only tests for target inference.

These use bare paths on purpose: the tidied fixtures under examples/ are
already ``<target>/<seed>/`` shaped, so they cannot catch mistakes in how
each engine's *native* output tree is interpreted.
"""

from __future__ import annotations

import pytest

from foldmetrics.models import infer_target

# (path, expected target)
NATIVE_LAYOUTS = [
    # Protenix: the job directory sits above seed/, with predictions/ below
    ("results/protenix/panel/P0__P1/seed_2066/predictions/P0__P1_sample_0.cif", "P0__P1"),
    # Boltz: the job directory sits *inside* predictions/
    (
        "results/boltz2/panel/boltz_results_boltz2/predictions/P0__P0/P0__P0_model_0.cif",
        "P0__P0",
    ),
    # ColabFold: everything flat inside the job directory
    ("cf/P0__P1/P0__P1_unrelaxed_rank_001.pdb", "P0__P1"),
    # AlphaFold2: ranked_N.pdb directly in the job directory
    ("af2/P0__P1/ranked_0.pdb", "P0__P1"),
    # AlphaFold3 local: per-sample directories are compound run names
    ("af3/P0__P1/seed-1_sample-0/model.cif", "P0__P1"),
    ("af3/P0__P1/P0__P1_model.cif", "P0__P1"),
    # Chai-1: user-chosen output directory per job
    ("chai/outputs/P0__P1/pred.model_idx_0.cif", "P0__P1"),
    # tidied layouts (as bundled in examples/) must keep working
    ("af_server/barnase_barstar/seed318/fold_x_model_0.cif", "barnase_barstar"),
    ("boltz2/job1/sample-3/m.cif", "job1"),
    ("af3/mpro/ranked/model.cif", "mpro"),
    ("demo/dimer_good/model.cif", "dimer_good"),
]


@pytest.mark.parametrize(("path", "expected"), NATIVE_LAYOUTS)
def test_infer_target_on_native_layouts(path, expected):
    assert infer_target(path) == expected
    assert infer_target("/abs/prefix/" + path) == expected


def test_run_directories_do_not_collapse_distinct_targets():
    """The regression: distinct complexes must not share one target label."""
    paths = [
        f"results/protenix/panel/P{i}__P{j}/seed_2066/predictions/P{i}__P{j}_sample_{s}.cif"
        for i, j in ((0, 1), (0, 2), (1, 2))
        for s in range(5)
    ]
    targets = {infer_target(p) for p in paths}
    assert targets == {"P0__P1", "P0__P2", "P1__P2"}


def test_falls_back_to_the_file_name_when_every_directory_is_generic():
    assert infer_target("predictions/P0__P1_sample_0.cif") == "P0__P1"
    assert infer_target("seed_2066/predictions/job7_model_3.cif") == "job7"


def test_job_names_containing_run_words_are_kept():
    # a real job directory is kept even when a run word appears inside it
    assert infer_target("x/boltz_results_boltz2/model_0.cif") == "boltz_results_boltz2"
    assert infer_target("x/sample_prep_A/seed1/m.cif") == "sample_prep_A"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # `__` separates the two halves of a job name; a half that happens to
        # be spelled like a run word must survive the file-name fallback
        ("predictions/model__sample_0.cif", "model__sample"),
        ("predictions/fold__run_model_0.cif", "fold__run"),
        ("predictions/RANK1__MODEL2_sample_0.cif", "RANK1__MODEL2"),
        # single separators are still stripped
        ("predictions/P53__MDM2_sample_0.cif", "P53__MDM2"),
        ("predictions/job7_model_3.cif", "job7"),
    ],
)
def test_file_name_fallback_does_not_cross_field_boundaries(path, expected):
    assert infer_target(path) == expected


def test_normal_layout_is_unaffected_by_the_fallback():
    assert (
        infer_target("out/P53__MDM2/seed_1/predictions/P53__MDM2_sample_0.cif")
        == "P53__MDM2"
    )


# --------------------------------------------------------------- ColabFold
# colabfold_batch writes an entire panel flat into one output directory, so
# no directory can carry the job name; the file name is authoritative.
COLABFOLD_NAMES = [
    ("A__B_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb", "A__B"),
    ("A__B_relaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb", "A__B"),
    (
        "P53__MDM2_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json",
        "P53__MDM2",
    ),
    ("A__B_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.cif", "A__B"),
    ("not_a_colabfold_file.pdb", None),
]


@pytest.mark.parametrize(("filename", "expected"), COLABFOLD_NAMES)
def test_colabfold_target_from_filename(filename, expected):
    from foldmetrics.parsers.colabfold import target_from_filename

    assert target_from_filename(filename) == expected


def test_colabfold_flat_panel_keeps_targets_apart(tmp_path):
    """A flat panel must not collapse onto the output directory name."""
    from conftest import write_colabfold_dir

    from foldmetrics.api import aggregate_by_target, evaluate

    panel = tmp_path / "results" / "colabfold" / "panel"
    for job in ("A__B", "A__C", "P53__MDM2"):
        write_colabfold_dir(panel, n_a=12, n_b=12, job=job)

    df = evaluate(panel)
    assert set(df["target"]) == {"A__B", "A__C", "P53__MDM2"}
    assert len(aggregate_by_target(df)) == 3


def test_colabfold_per_job_directories_still_work(tmp_path):
    from conftest import write_colabfold_dir

    from foldmetrics.api import evaluate

    for job in ("A__B", "A__C"):
        write_colabfold_dir(tmp_path / "cf" / job, n_a=12, n_b=12, job=job)
    assert set(evaluate(tmp_path / "cf")["target"]) == {"A__B", "A__C"}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # AlphaFold3 re-runs land in "<job>_<date>_<time>" instead of overwriting
        ("af3/P0__P1/P0__P1_20260817_143022/seed-1_sample-0/model.cif", "P0__P1"),
        ("af3/P0__P1/P0__P1_20260817_143022/P0__P1_model.cif", "P0__P1"),
        # a first run has no suffix and must be unchanged
        ("af3/P0__P1/P0__P1/seed-1_sample-0/model.cif", "P0__P1"),
        # digits that are not a timestamp are part of the name
        ("af3/P0__P1_2/model.cif", "P0__P1_2"),
    ],
)
def test_rerun_timestamps_do_not_split_a_target(path, expected):
    assert infer_target(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Chai-1 adds trunk_<N>/ when several trunk samples are requested
        ("chai/P0__P1/trunk_0/pred.model_idx_0.cif", "P0__P1"),
        ("chai/P0__P2/trunk_1/pred.model_idx_0.cif", "P0__P2"),
        # a single trunk sample writes straight into the job directory
        ("chai/P0__P1/pred.model_idx_0.cif", "P0__P1"),
    ],
)
def test_chai_trunk_directories_are_run_components(path, expected):
    assert infer_target(path) == expected
