"""Frozen user-surface contract samples: the chained configuration API.

RUST_ONLY_REFACTOR_PLAN.md section 2.1 freezes the user-facing chained
syntax: configuration method names, parameter names, chaining, and
build expressions must stay available while the execution backend
becomes Rust-only.  Each test here is an executable sample of that
surface, built only from the public ``natal`` namespace and pinned with
numerical assertions on deterministic dynamics.

Deliberately excluded from the frozen surface: the ``backend=``
selection argument (an explicitly accepted behavior change per plan
section 2.2) and any internal symbol.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import natal as nt

# ── Shared species definitions (singleton-scoped names) ──────────────────────


def _two_allele_species() -> nt.Species:
    """Return the shared two-allele species for chained-API samples."""
    return nt.Species.from_dict(
        name="FrozenChainSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Age-structured chain
# ══════════════════════════════════════════════════════════════════════════════


def test_age_structured_full_chain_builds_and_runs() -> None:
    """A full age-structured chain (demos/mosquito.py shape) builds and runs."""
    species = _two_allele_species()
    pop = (
        nt.AgeStructuredPopulation.setup(
            species=species,
            name="FrozenAgeChain",
            stochastic=False,
            continuous_sampling=False,
        )
        .age_structure(n_ages=4, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [0.0, 50.0, 0.0, 0.0]},
                "male": {"WT|WT": [0.0, 50.0, 0.0, 0.0]},
            }
        )
        .reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            eggs_per_female=8.0,
        )
        .survival(
            female_age_based_survival=[1.0, 0.8, 0.6, 0.0],
            male_age_based_survival=[1.0, 0.8, 0.6, 0.0],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            carrying_capacity=500.0,
        )
        .with_observation(
            groups={
                "adults": nt.IndividualSelector(age=range(1, 4)),
                "juveniles": nt.IndividualSelector(age=0),
            },
            collapse_age=False,
        )
        .build()
    )

    pop.run(5, finish=True)

    assert pop.tick == 5
    observed = pop.observe()
    values = observed.values
    # Groups in declaration order, sexes both present, finite population.
    assert observed.labels["group"] == ("adults", "juveniles")
    assert np.isfinite(values).all()
    # Female and male vital rates and the initial census are identical,
    # so the deterministic trajectory must keep both sexes exactly equal
    # in every group and age cell.
    np.testing.assert_array_equal(values[:, 0], values[:, 1])
    # The two groups partition the age axis, so their union equals the
    # whole census regardless of how the population split across ages.
    np.testing.assert_allclose(values.sum(), pop.state.individual_count.sum())
    # The partition invariant plus an active adult cohort.
    assert values[0].sum() > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Discrete-generation chain
# ══════════════════════════════════════════════════════════════════════════════


def test_discrete_generation_chain_reproduces_exactly() -> None:
    """A discrete chain (demos/discrete.py shape) holds a fixed point exactly.

    With 10 WT|WT females each laying 2 eggs at sex ratio 0.5, full
    age-0 survival, and a non-binding carrying capacity, every
    generation is exactly 10 females + 10 males of WT|WT.
    """
    species = _two_allele_species()
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name="FrozenDiscreteChain",
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )

    pop.run(3)

    counts = pop.state.individual_count
    assert counts.shape == (2, 2, 3)  # (sex, canonical age pair, genotype)
    # Genotype order is WT|WT, WT|Dr, Dr|Dr (unordered species).
    np.testing.assert_allclose(counts[:, 1, 0], [10.0, 10.0])
    np.testing.assert_allclose(counts[:, :, 1:], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Spatial chain
# ══════════════════════════════════════════════════════════════════════════════


def _ring_adjacency(n: int) -> NDArray[np.float64]:
    """Return the dense adjacency of an open chain 0-1-...-(n-1)."""
    m = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        m[i, i + 1] = 1.0
        m[i + 1, i] = 1.0
    return m


def _deme_initial(wt: float, dr: float) -> dict[str, dict[str, float]]:
    """Return a mirrored female/male initial census for one deme."""
    return {
        "female": {"WT|WT": wt, "WT|Dr": dr},
        "male": {"WT|WT": wt, "WT|Dr": dr},
    }


def test_spatial_chain_with_batch_setting_and_migration() -> None:
    """A spatial chain (demos/spatial.py shape) runs and stays symmetric.

    The initial census is mirror-symmetric around the chain center and
    the topology is symmetric, so mirrored demes must stay equal while
    migration mixes them.
    """
    species = _two_allele_species()
    spatial = (
        nt.SpatialPopulation.builder(
            species, n_demes=4, pop_type="discrete_generation"
        )
        .setup(name="FrozenSpatialChain", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [
                    _deme_initial(100.0, 0.0),
                    _deme_initial(60.0, 40.0),
                    _deme_initial(20.0, 80.0),
                    _deme_initial(0.0, 100.0),
                ]
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=10000.0, low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(4), migration_rate=0.1)
        .build()
    )

    spatial.run(2)

    assert spatial.tick == 2
    states = [d.state.individual_count for d in spatial.demes]
    totals = [s.sum() for s in states]
    # Female and male vital rates and each deme's census are identical
    # across sexes, so every deme stays exactly sex-symmetric.
    for s in states:
        np.testing.assert_array_equal(s[0], s[1])
    # Mirror symmetry of the topology and of the census totals: deme 0 ↔ 3
    # and deme 1 ↔ 2 stay equal because per-deme total dynamics are
    # genotype-blind under neutral fitness.
    np.testing.assert_allclose(totals[0], totals[3])
    np.testing.assert_allclose(totals[1], totals[2])
    # Every deme stays positive and finite while migration mixes them.
    assert all(np.isfinite(t) and t > 0.0 for t in totals)


def test_spatial_batch_setting_matrix_form() -> None:
    """``batch_setting`` accepts a 2-D (row, col) ndarray, row-major.

    Matrix values land on demes in row-major order: deme 0 ← grid[0, 0],
    deme 1 ← grid[0, 1], then the next row.
    """
    species = _two_allele_species()
    grid = np.array(
        [[100.0, 200.0], [300.0, 400.0]],
        dtype=np.float64,
    )
    spatial = (
        nt.SpatialPopulation.builder(
            species, n_demes=4, pop_type="discrete_generation"
        )
        .setup(name="FrozenSpatialMatrix", stochastic=False)
        .initial_state(
            individual_count=nt.batch_setting(
                [_deme_initial(10.0, 0.0)] * 4
            )
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=nt.batch_setting(grid), low_density_growth_rate=2.0)
        .migration(adjacency=_ring_adjacency(4), migration_rate=0.0)
        .build()
    )

    per_deme_k = [d.params.carrying_capacity for d in spatial.demes]
    np.testing.assert_allclose(per_deme_k, grid.ravel())


# ══════════════════════════════════════════════════════════════════════════════
# Runtime update chain
# ══════════════════════════════════════════════════════════════════════════════


def test_runtime_update_chain_changes_params_and_custom() -> None:
    """``pop.update()`` re-enters the chained surface at runtime."""
    species = _two_allele_species()
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name="FrozenUpdateChain",
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .custom(temperature=25.0, debug=False)
        .build()
    )

    assert pop.params.carrying_capacity == 100000.0
    assert pop.config.custom["temperature"] == 25.0

    pop.update().competition(carrying_capacity=4321.0)
    pop.update().custom(temperature=35.0, debug=True)

    assert pop.params.carrying_capacity == 4321.0
    assert pop.config.custom["temperature"] == 35.0
    assert pop.config.custom["debug"] is True


def test_custom_slots_keep_scalar_types_and_array_shapes() -> None:
    """``custom`` preserves bool/int/float values and 3-D array shapes."""
    species = _two_allele_species()
    grid = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name="FrozenCustomSlots",
            stochastic=False,
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=1000.0, low_density_growth_rate=2.0)
        .custom(cohort_id=7, temperature=2.5, debug=True, grid=grid)
        .build()
    )

    custom = pop.config.custom
    assert isinstance(custom["cohort_id"], int)
    assert custom["cohort_id"] == 7
    assert isinstance(custom["temperature"], float)
    assert custom["temperature"] == 2.5
    assert isinstance(custom["debug"], bool)
    assert custom["debug"] is True
    stored = custom["grid"]
    assert isinstance(stored, np.ndarray)
    assert stored.shape == (2, 2, 3)
    np.testing.assert_array_equal(stored, grid)


# ══════════════════════════════════════════════════════════════════════════════
# Build-time fitness and modifier chains
# ══════════════════════════════════════════════════════════════════════════════


def _fixed_point_builder(
    name: str,
) -> nt.PopulationBuilder:
    """Return a 10+10 WT|WT discrete builder at its exact replacement point."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_two_allele_species(),
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )


def test_build_time_fitness_chain_writes_tensors_and_shapes_dynamics() -> None:
    """``cfg.fitness(viability=...)`` lands per sex and damps juveniles.

    With 10 females laying 2 eggs at sex ratio 0.5, both sexes start with
    10 age-0 juveniles; sex-specific viability then must leave exactly
    ``10 * viability`` recruits per sex.
    """
    pop = _fixed_point_builder("FrozenBuildFitness").fitness(
        viability={"female": {"WT|WT": 0.5}, "male": {"WT|WT": 0.25}}
    ).build()

    # Public tensor read: female age-0 WT|WT is 0.5, male is 0.25, all
    # other cells stay at the neutral 1.0.
    tensor = pop.params.viability_fitness.array
    np.testing.assert_allclose(tensor[0, 0, 0], 0.5)
    np.testing.assert_allclose(tensor[1, 0, 0], 0.25)
    np.testing.assert_allclose(tensor[:, :, 1:], 1.0)
    np.testing.assert_allclose(tensor[:, 1, :], 1.0)

    pop.run(1)
    counts = pop.state.individual_count
    # 20 eggs -> 10 female + 10 male juveniles -> viability-scaled adults.
    np.testing.assert_allclose(counts[:, 1, 0], [10.0 * 0.5, 10.0 * 0.25])
    np.testing.assert_allclose(counts[:, :, 1:], 0.0)


def test_build_time_modifiers_chain_full_conversion() -> None:
    """``cfg.modifiers(gamete_modifiers=[...])`` rewrites meiosis at build.

    A full-conversion gamete modifier sends every WT|Dr heterozygote
    gamete to ``Dr``, so a WT|Dr population must breed to pure Dr|Dr in
    one generation.
    """

    def full_conversion() -> dict[tuple[int, int], dict[str, float]]:
        """Return the meiosis override: WT|Dr (ztype 1) yields only Dr."""
        return {(sex, 1): {"Dr": 1.0} for sex in (0, 1)}

    species = _two_allele_species()
    pop = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            name="FrozenBuildModifier",
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|Dr": 10},
                "male": {"WT|Dr": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .modifiers(gamete_modifiers=[full_conversion])
        .build()
    )

    # Diagnostic read of the public ``config`` snapshot: the heterozygote
    # meiosis row is a point mass on the Dr gamete for both sexes
    # (Mendelian would be 0.5/0.5), while the homozygote rows are
    # unaffected (WT|WT passes only WT, Dr|Dr only Dr).
    names = [str(gt) for gt in pop.index_registry.index_to_genotype]
    meiosis = pop.config.zygotes_to_gametes_map
    np.testing.assert_allclose(meiosis[:, names.index("WT|Dr"), :], [[0.0, 1.0]] * 2)
    np.testing.assert_allclose(meiosis[:, 0, :], [[1.0, 0.0]] * 2)
    np.testing.assert_allclose(meiosis[:, 2, :], [[0.0, 1.0]] * 2)

    pop.run(1)
    counts = pop.state.individual_count
    np.testing.assert_allclose(counts[:, 1, 2], [10.0, 10.0])
    np.testing.assert_allclose(counts[:, :, :2], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# Observation-mode history
# ══════════════════════════════════════════════════════════════════════════════


def test_record_history_observation_mode_matches_direct_observe() -> None:
    """``record_history(mode="observation")`` stores what ``observe()`` returns.

    The two genotype groups partition the ztype axis, so their sum must
    equal the census per sex, the last recorded row must equal a direct
    observation taken at the same tick, and the stored array must be
    read-only (mutation of the return value must not corrupt history).
    """
    pop = _fixed_point_builder("FrozenObsHistory").with_observation(
        groups={
            "wild": nt.IndividualSelector(ztype="WT|WT"),
            "drive_carriers": nt.IndividualSelector(ztype="WT|Dr, Dr|Dr"),
        },
        collapse_age=True,
    ).record_history(mode="observation").build()

    pop.run(2, record_every=1)

    assert pop.history.ticks == (0, 1, 2)
    values = pop.history.values
    assert values.shape == (3, 2, 2)
    # Partition: wild + drive carriers equal the whole census per sex.
    np.testing.assert_allclose(values[-1].sum(axis=0), 10.0)
    # The recorded row at tick 2 equals a direct observation at tick 2.
    np.testing.assert_array_equal(values[-1], pop.observe().values)
    # Ownership: history hands out a read-only buffer.
    assert not values.flags.writeable

