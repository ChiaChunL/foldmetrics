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
