"""Independent-review probes for the high-risk remediation batch.

Added by the evaluator (not the implementer) as extra evidence around gaps the
shipped tests leave open.  Nothing here asserts product behaviour that the
implementer weakened; each test documents a contract the batch claims and
strengthens the coverage where the shipped suite is thin:

* D1 (``370adf5``) is pinned with ``n_slabs > 1`` so the slab expansion of the
  forwarded blueprint masks is actually exercised (the shipped XY/ZW tests all
  use a single somatic label).
* D2 (``45d3f5d``) is pinned on the ``eggs_per_female == 0`` trigger named in
  the CHANGELOG but not directly tested there.
* D4 (``562eabb``) alias-safety of ``_coerce_adjacency_dense`` is checked
  against caller-owned, CSR-tuple and scipy buffers.
* B1 (``7962ba6``) collapsed-age history decoding is checked with more than one
  observation group, which the shipped test does not do.

The D4 section also pins the accepted ``adjust_migration_on_edge`` deviation at
<= 1 ulp and records the precedence of the ambiguous 2-D ``migration_rate``
shape (``(n_sexes, n_ages)`` wins over ``(n_demes, n_ages)`` when the two
coincide).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

import natal as nt
from natal.frontend.hooks import Op
from natal.frontend.patterns import IndividualSelector
from natal.frontend.spatial.migration import normalize_migration_rate_column
from natal.frontend.spatial.population import _coerce_adjacency_dense


# ── D1: structure-derived sex masks survive slab expansion ───────────────────


def _xy_species(name: str, somatic_labels: list[str] | None) -> nt.Species:
    """Return a minimal XY species with an optional somatic-slab axis."""
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrA": {"loci": {"A": ["A", "a"]}},
            "chrX": {"sex_type": "X", "loci": {"sx": ["X1", "X2"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
        },
        unordered=False,
        somatic_labels=somatic_labels,
    )


def _build_xy(
    species: nt.Species, name: str, *, use_age_structure: bool
) -> nt.AgeStructuredPopulation:
    """Build the two-parent XY cohort, optionally through ``age_structure()``."""
    builder = nt.AgeStructuredPopulation.setup(
        species=species, name=name, stochastic=False
    )
    if use_age_structure:
        builder = builder.age_structure(n_ages=2, new_adult_age=1)
    female = species.get_genotype_from_str("A|A;X1|X2")
    male = species.get_genotype_from_str("A|A;X1|Y1")
    return (
        builder.initial_state(
            individual_count={"female": {female: {1: 600}}, "male": {male: {1: 600}}}
        )
        .survival(
            female_age_based_survival=[1.0, 0.0],
            male_age_based_survival=[1.0, 0.0],
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0],
            eggs_per_female=2,
            sex_ratio=0.5,
        )
        .competition(
            carrying_capacity=1e12, low_density_growth_rate=2.0, growth_mode="fixed"
        )
        .build()
    )


@pytest.mark.parametrize("use_age_structure", [False, True], ids=["from_species", "age_structure"])
def test_sex_masks_survive_slab_expansion(use_age_structure: bool) -> None:
    """With ``n_slabs > 1`` the forwarded masks must still expand correctly.

    ``PopulationBuilder.age_structure()`` forwards the *unexpanded* blueprint
    masks and lets ``build_population_config`` expand them over the somatic
    slab axis.  The shipped regression suite only covers ``n_slabs == 1``, so a
    wrong (expanded) mask would have gone unnoticed there.
    """
    species = _xy_species(f"eval_xy_slabs_{use_age_structure}", ["normal", "infected"])
    blueprint = species.get_config_blueprint()
    assert blueprint["n_slabs"] == 2
    assert blueprint["n_ztypes"] == 2 * blueprint["n_genotypes"]
    # The forwarded masks are on the *unexpanded* genotype axis.
    assert blueprint["female_only_by_sex_chrom"].shape == (
        blueprint["n_genotypes"],
    )

    pop = _build_xy(species, f"eval_xy_slabs_{use_age_structure}", use_age_structure=use_age_structure)
    pop.run(1)

    counts = pop.state.individual_count
    assert float(counts.sum()) == 1200.0
    assert float(counts[0].sum()) == 600.0
    assert float(counts[1].sum()) == 600.0


# ── D2: the eggs_per_female == 0 trigger named in the CHANGELOG ──────────────


def _hook_injected_discrete(name: str, mode: str | None) -> nt.DiscreteGenerationPopulation:
    """Discrete population whose only recruits come from a per-tick hook.

    ``eggs_per_female == 0`` makes the derived equilibrium competition strength
    zero, while the hook installs age-0 individuals each tick.  With the
    default curve those recruits are cleared; ``no_competition`` preserves
    them.
    """
    species = nt.Species.from_dict(
        name=f"eval_eggs0_{name}",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )
    kwargs = {} if mode is None else {"growth_mode": mode}
    op = Op.add(genotypes="WT|WT", ages=0, sex="male", delta=200.0)
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": [10.0, 0.0]},
                "male": {"WT|WT": [10.0, 0.0]},
            }
        )
        .reproduction(eggs_per_female=0.0, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .competition(
            carrying_capacity=1e9, low_density_growth_rate=2.0, **kwargs
        )
        .hooks(op)
        .build()
    )


def test_eggs_per_female_zero_default_extinguishes_hook_recruits() -> None:
    """The default curve clears hook-injected age-0 individuals at zero eggs.

    The CHANGELOG names ``eggs_per_female == 0`` as a trigger for the
    zero-equilibrium rule but ships no test that exercises that trigger
    directly; the shipped hook fixtures only pin the ``no_competition``
    workaround.
    """
    default = _hook_injected_discrete("eval_eggs0_default", None)
    preserved = _hook_injected_discrete("eval_eggs0_none", "no_competition")
    assert default.params.growth_mode == nt.BEVERTON_HOLT
    assert preserved.params.growth_mode == nt.NO_COMPETITION

    for _ in range(3):
        default.run(1)
        preserved.run(1)

    assert float(default.state.individual_count.sum()) == 0.0
    assert float(preserved.state.individual_count.sum()) > 0.0


# ── D4: adjacency coercion is alias-safe and keeps all-zero rows ─────────────


def test_adjacency_coercion_never_writes_caller_buffers() -> None:
    """Row normalization must return a fresh array, not touch the input.

    The contract says the builder must not write through a caller-owned dense
    array, a CSR tuple's ``data`` buffer, or a scipy matrix's buffer.
    """
    dense = np.array(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 7.0], [0.0, 0.0, 0.0]], dtype=np.float64
    )
    dense_before = dense.copy()
    out = _coerce_adjacency_dense(dense, 3)
    assert not np.shares_memory(out, dense)
    np.testing.assert_array_equal(dense, dense_before)
    # The all-zero (isolated) row is preserved verbatim.
    np.testing.assert_array_equal(out[2], np.zeros(3))
    np.testing.assert_allclose(out[:2].sum(axis=1), [1.0, 1.0])

    indptr = np.array([0, 1, 2, 3], dtype=np.int64)
    indices = np.array([1, 0, 1], dtype=np.int64)
    data = np.array([2.0, 3.0, 5.0], dtype=np.float64)
    data_before = data.copy()
    csr_out = _coerce_adjacency_dense((indptr, indices, data), 3)
    np.testing.assert_array_equal(data, data_before)
    out_after = csr_out.copy()
    csr_out[0, 0] = 123.0
    np.testing.assert_array_equal(data, data_before)
    del out_after

    scipy = pytest.importorskip("scipy.sparse")
    matrix = scipy.csr_matrix(
        np.array([[0.0, 2.0, 0.0], [3.0, 0.0, 7.0], [0.0, 0.0, 0.0]])
    )
    scipy_before = matrix.toarray().copy()
    scipy_out = _coerce_adjacency_dense(matrix, 3)
    np.testing.assert_array_equal(matrix.toarray(), scipy_before)
    np.testing.assert_allclose(scipy_out[:2].sum(axis=1), [1.0, 1.0])


def test_per_deme_age_table_when_demes_equal_sexes_actual_semantics() -> None:
    """Pin the *actual* semantics of the ambiguous ``(D, A)`` shape.

    With ``n_demes == n_sexes`` the shape test treats the 2-D array as the
    shared ``(S, A)`` table and tiles it over demes, so the per-deme reading is
    unavailable for that shape; ``batch_setting`` remains the unambiguous way
    to give each deme its own age vector.
    """
    per_deme = np.array([[0.1, 0.0], [0.4, 0.0]], dtype=np.float64)
    column = normalize_migration_rate_column(
        per_deme, n_demes=2, n_sexes=2, n_ages=2, adult_start_age=1
    )
    # Row d is a sex row, identical for every deme.
    np.testing.assert_allclose(column, np.tile(per_deme, (2, 1, 1)))

    # The canonical (D, S, A) column has no such ambiguity.
    canonical = np.zeros((2, 2, 2), dtype=np.float64)
    canonical[:, :, 1] = np.array([[0.1], [0.4]])
    direct = normalize_migration_rate_column(
        canonical, n_demes=2, n_sexes=2, n_ages=2, adult_start_age=1
    )
    np.testing.assert_allclose(direct, canonical)


# ── D4: adjust_migration_on_edge is a no-op only up to ~1 ulp ────────────────


def test_adjust_on_edge_deviation_is_at_most_one_ulp() -> None:
    """The flag must not change the destination distribution beyond rounding.

    The docs now state "up to ~1 ulp"; this pins that bound exactly on a
    non-uniform kernel over a 3x3 grid, where boundary rows exercise the
    differing scale factors.
    """
    from natal.frontend.spatial.migration import fold_migration_csr
    from natal.frontend.spatial.topology import SquareGrid

    topology = SquareGrid(3, 3)
    n_demes = 9
    kernel = np.array(
        [[0.0, 2.0, 0.0], [3.0, 0.0, 7.0], [0.0, 5.0, 0.0]], dtype=np.float64
    )
    adjacency = np.zeros((n_demes, n_demes), dtype=np.float64)

    def fold(adjust: bool):
        return fold_migration_csr(
            n_demes=n_demes,
            topology=topology,
            adjacency_dense=adjacency,
            migration_kernel=kernel,
            kernel_bank=None,
            deme_kernel_ids=None,
            kernel_include_center=True,
            adjust_on_edge=adjust,
            mode="kernel",
        )

    adjusted = fold(True)
    plain = fold(False)
    # Structure is identical; only the last-bit weights may differ.
    np.testing.assert_array_equal(adjusted.indptr, plain.indptr)
    np.testing.assert_array_equal(adjusted.dest_idx, plain.dest_idx)
    difference = np.abs(adjusted.weights - plain.weights)
    ulps = difference / np.spacing(np.maximum(np.abs(plain.weights), np.finfo(np.float64).tiny))
    assert float(ulps.max()) <= 1.0, (difference.max(), float(ulps.max()))


# ── B1: collapsed-age history decoding with more than one group ──────────────


def _history_population(name: str, groups, *, collapse_age: bool) -> nt.AgeStructuredPopulation:
    """Four-age population recording an observation-mode history."""
    species = nt.Species.from_dict(
        name=f"{name}_species", structure={"chr1": {"loc": ["WT", "Dr"]}}
    )
    return (
        nt.AgeStructuredPopulation.setup(
            species=species, name=name, stochastic=False, continuous_sampling=False
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [1.0, 2.0, 3.0, 4.0], "Dr|Dr": [1.0, 1.0, 1.0, 1.0]},
                "male": {"WT|WT": [5.0, 6.0, 7.0, 8.0], "Dr|Dr": [1.0, 1.0, 1.0, 1.0]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 1.0],
            eggs_per_female=10.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8],
            male_age_based_survival=[1.0, 0.9, 0.8],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            old_juvenile_carrying_capacity=500,
            expected_num_new_adult_females=10,
        )
        .with_observation(groups=groups, collapse_age=collapse_age)
        .record_history(mode="observation")
        .build()
    )


@pytest.mark.parametrize("collapse_age", [False, True], ids=["full", "collapsed"])
def test_multi_group_history_translation(collapse_age: bool) -> None:
    """Two observation groups must decode in both full and collapsed layouts.

    The shipped regression only covers a single group, so a layout mistake in
    the group axis would not have been caught.
    """
    groups = OrderedDict(
        (
            ("wild", IndividualSelector(ztype="WT|WT")),
            ("dr", IndividualSelector(ztype="Dr|Dr")),
        )
    )
    pop = _history_population(f"eval_hist_{collapse_age}", groups, collapse_age=collapse_age)
    pop.run(2)

    readable = nt.population_observation_history_to_readable_dict(pop)
    assert readable["collapse_age"] is collapse_age
    snapshot = readable["snapshots"][-1]["observed"]
    assert set(snapshot) == {"wild", "dr"}
    for group_payload in snapshot.values():
        # collapse_age stores one value per sex; the full layout stores an
        # age vector per sex.
        assert set(group_payload) == {"female", "male"}
        for value in group_payload.values():
            if collapse_age:
                assert isinstance(value, float)
            else:
                assert isinstance(value, dict)
