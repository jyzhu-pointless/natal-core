"""End-to-end dynamics tests for the PointMutation preset.

These tests run real populations through the Rust lifecycle and compare the
resulting allele-frequency trajectories with expectations derived from the
documented model rather than from the implementation:

- **Gametic mutation only**: the wild-type fraction of the gamete pool is
  multiplied by ``1 - mu`` each generation, so ``q_B(t) = 1 - (1 - q_B(0)) *
  (1 - mu)^t`` exactly (no selection, no drift in the deterministic engine).
- **Competing targets**: the wild-type pool decays by ``1 - sum(r_k)`` per
  generation and every target draws the fixed share ``r_k / sum(r)`` of the
  converted mass, so the ratio ``B : C`` stays ``r_1 : r_2`` at *every*
  generation.  A sequential (uncompensated) cascade would drift this ratio
  towards the first-declared target, so this trajectory is the end-to-end
  signature of the compensation.
- **Selection**: one discrete generation maps the post-selection parental
  frequency ``q`` to the gamete pool ``q_g = 1 - (1 - mu)(1 - q)``, then
  random union, then viability selection with the preset's ``viability_mode``
  weights.  ``_reference_trajectory`` implements exactly that textbook
  lifecycle independently of the engine.
- **Recessive lethal fixed point**: with ``w_BB = 0`` the equilibrium solves
  ``q = q_g / (1 + q_g)``, ``q_g = mu + q(1 - mu)``, i.e.
  ``q^2 (1 - mu) + 2 mu q - mu = 0`` ⇒ ``q_hat = sqrt(mu) / (1 + sqrt(mu))``.
  (The textbook ``sqrt(mu)`` assumes selection *before* reproduction; the
  engine selects the offspring, which is why the denominator appears.)
- **Sex-specific rates**: only the female half of the gamete pool mutates, so
  the wild-type fraction decays by ``1 - mu_f / 2`` per generation.
- **X-linked locus**: every X of the next generation comes from a mutated
  maternal or paternal pool, so the X-pool frequency is ``1 - (1 - mu)^t`` in
  both sexes (males are hemizygous and are counted as one X copy).
- **Spatial**: migration moves individuals, not alleles, so the *global*
  allele frequency keeps the panmictic recursion while demes mix towards it.
- **Overlapping generations**: with a stable age structure the total
  wild-type fraction decays geometrically at a constant per-generation factor
  ``lambda``.  ``lambda`` is *not* ``1 - mu``: the standing age structure keeps
  older cohorts, which carry fewer mutations, so the population-level decay is
  slower.  ``lambda`` is the dominant root of the renewal equation, so two
  different initial age profiles converge to the same value.  Under the engine
  default (5 % sperm displacement) that equation also includes the
  storage/remating dynamics — stored, older sperm carries fewer mutations and
  raises the root further; with storage disabled the gamete pool is exactly the
  current adults' count-weighted average, and the root computed from the
  measured age profile pins ``lambda`` exactly.
- **Stochastic**: identical builds share the engine's default seed 0, so two
  stochastic populations are bit-identical; the realized one-generation
  mutation frequency is binomially distributed around ``mu``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.builder import batch_setting


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _species(name: str) -> nt.Species:
    """Return a single-locus species with two competing target alleles."""
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B", "C"]}},
        gamete_labels=["default"],
    )


def _discrete_population(
    name: str,
    species: nt.Species,
    preset: nt.PointMutation,
    initial: dict[str, dict[str, float]],
    *,
    stochastic: bool = False,
    eggs_per_female: float = 4.0,
    carrying_capacity: float = 50_000.0,
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic (or stochastic) discrete-generation population."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=stochastic
        )
        .initial_state(individual_count=initial)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=eggs_per_female, sex_ratio=0.5)
        .competition(
            carrying_capacity=carrying_capacity, low_density_growth_rate=2.0
        )
        .presets(preset)
        .build()
    )


def _frequency(
    registry: nt.IndexRegistry,
    counts: np.ndarray,
    species: nt.Species,
    allele: str,
) -> float:
    """Return the autosomal allele frequency of *allele* over *counts*.

    ``counts`` is a ``(sex, age, ztype)`` array or a ``(deme, sex, age,
    ztype)`` stack; only the last axis is addressed, so both layouts work.
    """
    array = np.asarray(counts)
    gene = species.get_gene(allele)
    copies = total = 0.0
    for ztype_index, (genotype, _slab) in enumerate(registry.index_to_ztype):
        count = float(array[..., ztype_index].sum())
        if count:
            copies += count * nt.count_allele_copies(genotype, gene)
            total += 2.0 * count
    return copies / total if total else 0.0


def _population_frequency(pop: nt.DiscreteGenerationPopulation, allele: str) -> float:
    """Return the allele frequency of one panmictic population."""
    return _frequency(
        pop.index_registry, pop.state.individual_count, pop.species, allele
    )


def _reference_trajectory(
    generations: int,
    *,
    mutation_rate: float,
    q0: float = 0.0,
    w_ab: float = 1.0,
    w_bb: float = 1.0,
) -> list[float]:
    """Iterate the documented discrete lifecycle independently of the engine.

    One generation is: gametic mutation (``A`` survives with ``1 - mu``) →
    random union of gametes → viability selection with ``w_AA = 1``,
    ``w_AB = w_ab``, ``w_BB = w_bb``.  Returns the post-selection frequency
    after each generation (index 0 is the initial frequency).
    """
    trajectory = [q0]
    q = q0
    for _ in range(generations):
        q_gamete = 1.0 - (1.0 - mutation_rate) * (1.0 - q)
        aa = (1.0 - q_gamete) ** 2
        ab = 2.0 * q_gamete * (1.0 - q_gamete)
        bb = q_gamete**2
        mean_fitness = aa + ab * w_ab + bb * w_bb
        q = (ab * w_ab / 2.0 + bb * w_bb) / mean_fitness
        trajectory.append(q)
    return trajectory


def _fixed_point(*, mutation_rate: float, w_ab: float, w_bb: float) -> float:
    """Return the fixed point of ``_reference_trajectory`` by iteration."""
    q = 0.0
    for _ in range(200_000):
        q_next = _reference_trajectory(
            1, mutation_rate=mutation_rate, q0=q, w_ab=w_ab, w_bb=w_bb
        )[-1]
        if abs(q_next - q) < 1e-16:
            return q_next
        q = q_next
    raise AssertionError("reference trajectory did not converge")


# ══════════════════════════════════════════════════════════════════════════════
# Panmictic dynamics
# ══════════════════════════════════════════════════════════════════════════════


def test_neutral_mutation_accumulates_geometrically() -> None:
    """Without selection, ``q_B(t) = 1 - (1 - mu)^t`` at every generation."""
    species = _species("_point_mutation_dyn_neutral")
    mutation_rate = 1e-3
    preset = nt.PointMutation(
        "NeutralMut", source_allele="A", target_allele="B", mutation_rate=mutation_rate
    )
    pop = _discrete_population(
        "DynNeutral",
        species,
        preset,
        {"female": {"A|A": 500}, "male": {"A|A": 500}},
    )

    assert _population_frequency(pop, "B") == 0.0
    for generation in range(1, 11):
        pop.run(1)
        expected = 1.0 - (1.0 - mutation_rate) ** generation
        assert _population_frequency(pop, "B") == pytest.approx(expected, rel=1e-9)


def test_competing_targets_keep_their_declared_ratio_over_time() -> None:
    """The B:C ratio stays ``r_B : r_C`` as the wild-type pool decays.

    This is the end-to-end signature of the cascade compensation: an
    uncompensated sequential cascade would let the first-declared target take
    its mass first, so the ratio would drift generation after generation.
    """
    species = _species("_point_mutation_dyn_competing")
    rate_b, rate_c = 3e-3, 5e-3
    preset = nt.PointMutation(
        "CompetingMut",
        source_allele="A",
        target_alleles=["B", "C"],
        mutation_rates=[rate_b, rate_c],
    )
    pop = _discrete_population(
        "DynCompeting",
        species,
        preset,
        {"female": {"A|A": 500}, "male": {"A|A": 500}},
    )

    for generation in range(1, 8):
        pop.run(1)
        freq_b = _population_frequency(pop, "B")
        freq_c = _population_frequency(pop, "C")
        wild_type = _population_frequency(pop, "A")
        assert freq_b / freq_c == pytest.approx(rate_b / rate_c, rel=1e-9)
        assert wild_type == pytest.approx((1.0 - rate_b - rate_c) ** generation, rel=1e-9)
        assert freq_b + freq_c + wild_type == pytest.approx(1.0, rel=1e-12)


def test_female_specific_rate_halves_the_accumulation() -> None:
    """A female-only rate mutates half the gamete pool, so ``A`` decays by ``1 - mu/2``."""
    species = _species("_point_mutation_dyn_sex_specific")
    mutation_rate = 2e-3
    preset = nt.PointMutation(
        "FemaleOnlyMut",
        source_allele="A",
        target_allele="B",
        mutation_rate=(mutation_rate, 0.0),
    )
    pop = _discrete_population(
        "DynFemaleOnly",
        species,
        preset,
        {"female": {"A|A": 500}, "male": {"A|A": 500}},
    )

    for generation in range(1, 6):
        pop.run(1)
        expected = 1.0 - (1.0 - mutation_rate / 2.0) ** generation
        assert _population_frequency(pop, "B") == pytest.approx(expected, rel=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Mutation-selection balance
# ══════════════════════════════════════════════════════════════════════════════


def test_recessive_lethal_reaches_the_mutation_selection_fixed_point() -> None:
    """A recessive lethal balances at ``sqrt(mu) / (1 + sqrt(mu))``.

    The trajectory is also compared generation by generation with the
    independently written ``_reference_trajectory`` (gametic mutation →
    Mendelian union → viability selection), so a lifecycle-ordering mistake
    cannot hide behind the equilibrium value.
    """
    species = _species("_point_mutation_dyn_lethal")
    mutation_rate = 1e-4
    preset = nt.PointMutation(
        "LethalMut",
        source_allele="A",
        target_allele="B",
        mutation_rate=mutation_rate,
        viability_scaling=0.0,
        viability_mode="recessive",
    )
    pop = _discrete_population(
        "DynLethal",
        species,
        preset,
        {"female": {"A|A": 2000}, "male": {"A|A": 2000}},
    )

    observed: list[float] = []
    for _ in range(50):
        pop.run(1)
        observed.append(_population_frequency(pop, "B"))
    reference = _reference_trajectory(
        50, mutation_rate=mutation_rate, w_bb=0.0
    )[1:]
    np.testing.assert_allclose(observed, reference, rtol=1e-9, atol=1e-12)

    # Run to the fixed point: the analytic root of q^2(1-mu) + 2 mu q - mu = 0.
    pop.run(750)
    equilibrium = mutation_rate**0.5 / (1.0 + mutation_rate**0.5)
    assert _population_frequency(pop, "B") == pytest.approx(equilibrium, rel=1e-5)


def test_multiplicative_deleterious_mutation_reaches_its_fixed_point() -> None:
    """A multiplicative fitness cost balances where the reference recursion does."""
    species = _species("_point_mutation_dyn_multiplicative")
    mutation_rate = 1e-3
    scaling = 0.9
    preset = nt.PointMutation(
        "MultiplicativeMut",
        source_allele="A",
        target_allele="B",
        mutation_rate=mutation_rate,
        viability_scaling=scaling,
        viability_mode="multiplicative",
    )
    pop = _discrete_population(
        "DynMultiplicative",
        species,
        preset,
        {"female": {"A|A": 1000}, "male": {"A|A": 1000}},
    )

    observed: list[float] = []
    for _ in range(30):
        pop.run(1)
        observed.append(_population_frequency(pop, "B"))
    reference = _reference_trajectory(
        30, mutation_rate=mutation_rate, w_ab=scaling, w_bb=scaling**2
    )[1:]
    np.testing.assert_allclose(observed, reference, rtol=1e-9, atol=1e-12)

    pop.run(370)
    equilibrium = _fixed_point(
        mutation_rate=mutation_rate, w_ab=scaling, w_bb=scaling**2
    )
    assert _population_frequency(pop, "B") == pytest.approx(equilibrium, rel=1e-4)


# ══════════════════════════════════════════════════════════════════════════════
# Sex chromosomes
# ══════════════════════════════════════════════════════════════════════════════


def _x_linked_species(name: str) -> nt.Species:
    """Return a species whose mutation locus sits on the X chromosome."""
    return nt.Species.from_dict(
        name=name,
        structure={
            "chrX": {"sex_type": "X", "loci": {"sx": ["A", "B"]}},
            "chrY": {"sex_type": "Y", "loci": {"sy": ["Y1"]}},
            "chrA": {"loci": {"a": ["C"]}},
        },
        gamete_labels=["default"],
    )


def _x_pool_frequency(pop: nt.DiscreteGenerationPopulation, allele: str, sex: int) -> float:
    """Return the X-chromosome allele frequency of one sex (males are hemizygous)."""
    species = pop.species
    x_chromosome = species.get_chromosome("chrX")
    gene = species.get_gene(allele)
    registry = pop.index_registry
    counts = pop.state.individual_count
    copies = x_copies = 0.0
    for ztype_index, (genotype, _slab) in enumerate(registry.index_to_ztype):
        count = float(counts[sex, :, ztype_index].sum())
        if not count:
            continue
        copies += count * nt.count_allele_copies(genotype, gene)
        for haplotype in (genotype.maternal, genotype.paternal):
            try:
                haplotype.get_haplotype_for_chromosome(x_chromosome)
            except ValueError:
                continue  # Y-bearing haplotype carries no X copy
            x_copies += count
    return copies / x_copies if x_copies else 0.0


def test_x_linked_mutation_accumulates_in_both_sexes() -> None:
    """Every X copy is drawn from a mutated pool, so both sexes follow ``1-(1-mu)^t``."""
    species = _x_linked_species("_point_mutation_dyn_x_linked")
    mutation_rate = 5e-3
    preset = nt.PointMutation(
        "XLinkedMut", source_allele="A", target_allele="B", mutation_rate=mutation_rate
    )
    pop = _discrete_population(
        "DynXLinked",
        species,
        preset,
        {"female": {"A|A; C|C": 500}, "male": {"A|Y1; C|C": 500}},
    )

    for generation in range(1, 6):
        pop.run(1)
        expected = 1.0 - (1.0 - mutation_rate) ** generation
        assert _x_pool_frequency(pop, "B", nt.Sex.FEMALE) == pytest.approx(
            expected, rel=1e-9
        )
        assert _x_pool_frequency(pop, "B", nt.Sex.MALE) == pytest.approx(
            expected, rel=1e-9
        )


# ══════════════════════════════════════════════════════════════════════════════
# Spatial demes
# ══════════════════════════════════════════════════════════════════════════════


def _spatial_population(
    name: str,
    species: nt.Species,
    preset: nt.PointMutation,
    initial: Sequence[dict[str, dict[str, float]]],
    *,
    migration_rate: float,
) -> nt.SpatialPopulation:
    """Build a three-deme discrete-generation spatial population."""
    adjacency = np.array(
        [[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=np.float64
    )
    return (
        nt.SpatialPopulation.builder(species, n_demes=3, pop_type="discrete_generation")
        .setup(stochastic=False)
        .initial_state(individual_count=batch_setting(list(initial)))
        .reproduction(eggs_per_female=4.0, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .competition(
            juvenile_growth_mode=nt.NO_COMPETITION, carrying_capacity=100_000
        )
        .migration(adjacency=adjacency, migration_rate=migration_rate)
        .presets(preset)
        .build()
    )


def _deme_frequencies(pop: nt.SpatialPopulation, allele: str) -> list[float]:
    """Return the allele frequency of every deme."""
    return [
        _frequency(deme.index_registry, deme.state.individual_count, pop.species, allele)
        for deme in pop.demes
    ]


def _global_frequency(pop: nt.SpatialPopulation, allele: str) -> float:
    """Return the allele frequency pooled over every deme."""
    stacked = np.stack([deme.state.individual_count for deme in pop.demes])
    return _frequency(
        pop.demes[0].index_registry, stacked, pop.species, allele
    )


def test_spatial_mutation_keeps_the_global_recursion() -> None:
    """Migration redistributes the mutant but does not create or destroy alleles."""
    species = _species("_point_mutation_dyn_spatial")
    mutation_rate = 1e-3
    preset = nt.PointMutation(
        "SpatialMut", source_allele="A", target_allele="B", mutation_rate=mutation_rate
    )
    deme_state = {"female": {"A|A": 300.0}, "male": {"A|A": 300.0}}
    pop = _spatial_population(
        "DynSpatial",
        species,
        preset,
        [deme_state, deme_state, deme_state],
        migration_rate=0.2,
    )

    for tick in range(1, 6):
        pop.run_tick()
        expected = 1.0 - (1.0 - mutation_rate) ** tick
        assert _global_frequency(pop, "B") == pytest.approx(expected, rel=1e-9)
        # Identical symmetric demes stay identical under symmetric migration.
        np.testing.assert_allclose(
            _deme_frequencies(pop, "B"), [expected] * 3, rtol=1e-9
        )


def test_spatial_migration_transports_a_local_mutant_across_demes() -> None:
    """A mutant seeded in one deme spreads under migration; without it, it stays local.

    The global frequency is pinned by the panmictic recursion started from the
    seeded frequency ``q0 = 1/12`` (one deme carries ``A|B`` males); the
    per-deme spread is checked by the shrinking gap between the seeded deme
    and its neighbours, with a no-migration control that keeps the gap.
    """
    species = _species("_point_mutation_dyn_spatial_transport")
    mutation_rate = 1e-3
    preset = nt.PointMutation(
        "TransportMut", source_allele="A", target_allele="B", mutation_rate=mutation_rate
    )
    seeded = {"female": {"A|A": 400.0}, "male": {"A|B": 400.0}}
    unseeded = {"female": {"A|A": 400.0}, "male": {"A|A": 400.0}}

    def build(name: str, migration_rate: float) -> nt.SpatialPopulation:
        return _spatial_population(
            name, species, preset, [seeded, unseeded, unseeded],
            migration_rate=migration_rate,
        )

    migrating = build("DynTransportMix", 0.3)
    isolated = build("DynTransportIsolated", 0.0)
    # Global B frequency before any tick: 400 B copies of 4800 alleles.
    initial_global = 400.0 / 4800.0
    assert _global_frequency(migrating, "B") == pytest.approx(initial_global)

    for tick in range(1, 9):
        migrating.run_tick()
        isolated.run_tick()
        expected = 1.0 - (1.0 - initial_global) * (1.0 - mutation_rate) ** tick
        assert _global_frequency(migrating, "B") == pytest.approx(expected, rel=1e-9)

    def deme_gap(pop: nt.SpatialPopulation) -> float:
        frequencies = _deme_frequencies(pop, "B")
        return frequencies[0] - frequencies[1]

    # Mixing pulls the seeded deme towards its neighbours; isolation does not.
    assert deme_gap(migrating) < 0.25 * deme_gap(isolated)
    assert deme_gap(isolated) > 0.1


# ══════════════════════════════════════════════════════════════════════════════
# Overlapping generations
# ══════════════════════════════════════════════════════════════════════════════


_AGE_MATING_RATES = [0.0, 0.0, 1.0, 1.0, 1.0]


def _age_structured_population(
    name: str,
    species: nt.Species,
    preset: nt.PointMutation,
    initial: dict[str, dict[str, list[float]]],
    *,
    sperm_displacement_rate: float | None = None,
) -> nt.AgeStructuredPopulation:
    """Build a viable five-age deterministic population (overlapping generations).

    ``sperm_displacement_rate=None`` keeps the engine default (5 % of stored
    sperm displaced by a remating); ``1.0`` disables storage, so the gamete
    pool is exactly the current adults' count-weighted average.
    """
    return (
        nt.AgeStructuredPopulation.setup(species=species, name=name, stochastic=False)
        .age_structure(n_ages=5, new_adult_age=2)
        .initial_state(individual_count=initial)
        .reproduction(
            female_age_based_mating_rate=_AGE_MATING_RATES,
            male_age_based_mating_rate=_AGE_MATING_RATES,
            eggs_per_female=4.0,
            sperm_displacement_rate=sperm_displacement_rate,
        )
        .survival(
            female_age_based_survival=[1.0, 0.9, 0.8, 0.7],
            male_age_based_survival=[1.0, 0.9, 0.8, 0.7],
        )
        .competition(
            juvenile_growth_mode="beverton_holt",
            old_juvenile_carrying_capacity=500,
            expected_num_new_adult_females=400,
        )
        .presets(preset)
        .build()
    )


def _renewal_root(age_weights: Sequence[float], mutation_rate: float) -> float:
    """Solve the renewal equation ``sum_a w_a [1 - (1 - mu) lam^-a] = 0``.

    With sperm storage disabled the gamete pool is the count-weighted average
    over the mating ages, so the wild-type fraction obeys
    ``p_{t+1} = (1 - mu) sum_a w_a p_{t-a}`` and decays as ``lam^t``, where
    ``lam`` is the dominant root of that equation.  The root is bracketed by
    ``(1 - mu, 1)`` and found by bisection.
    """
    ages = range(len(age_weights))

    def residual(lam: float) -> float:
        return sum(
            weight * (1.0 - (1.0 - mutation_rate) * lam ** (-age))
            for age, weight in zip(ages, age_weights)
        )

    low, high = 1.0 - mutation_rate, 1.0
    assert residual(low) < 0.0 < residual(high)
    for _ in range(200):
        mid = (low + high) / 2.0
        if residual(mid) < 0.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _late_decay_ratios(
    pop: nt.AgeStructuredPopulation,
    *,
    warmup: int = 60,
    window: int = 20,
) -> list[float]:
    """Return exactly *window* per-generation wild-type decay factors.

    The population is advanced ``warmup`` generations first, then one ratio is
    recorded per further generation, each against the previous generation's
    state (including the state the call started from).
    """

    def wild_type() -> float:
        return 1.0 - _frequency(
            pop.index_registry, pop.state.individual_count, pop.species, "B"
        )

    previous = wild_type()
    ratios: list[float] = []
    for generation in range(1, warmup + window + 1):
        pop.run(1)
        current = wild_type()
        if generation > warmup:
            ratios.append(current / previous)
        previous = current
    return ratios


def test_age_structured_decay_rate_is_schedule_determined() -> None:
    """Overlapping generations decay at a constant rate set by the age schedule.

    Three independent claims are pinned: (1) the late-generation decay factor
    is constant, i.e. the wild-type fraction is geometric once the age profile
    is stable; (2) it is strictly slower than the gametic rate ``1 - mu``
    because older cohorts carry fewer mutations; (3) two different initial age
    profiles converge to the *same* factor, so it is a property of the
    survival/fecundity schedule rather than of the starting state.
    """
    species = _species("_point_mutation_dyn_age_structured")
    mutation_rate = 5e-3

    def mutation_preset(name: str) -> nt.PointMutation:
        """One preset instance per population (registration is per population)."""
        return nt.PointMutation(
            name, source_allele="A", target_allele="B", mutation_rate=mutation_rate
        )

    all_adults = _age_structured_population(
        "DynAgeAdults",
        species,
        mutation_preset("AgeAdultsMut"),
        {
            "female": {"A|A": [0, 0, 400, 0, 0]},
            "male": {"A|A": [0, 0, 400, 0, 0]},
        },
    )
    spread = _age_structured_population(
        "DynAgeSpread",
        species,
        mutation_preset("AgeSpreadMut"),
        {
            "female": {"A|A": [0, 50, 150, 150, 150]},
            "male": {"A|A": [0, 50, 150, 150, 150]},
        },
    )

    adult_ratios = _late_decay_ratios(all_adults)
    spread_ratios = _late_decay_ratios(spread)
    for ratios in (adult_ratios, spread_ratios):
        # (1) geometric decay with a stable age profile.
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-9)

    adult_rate = float(np.mean(adult_ratios))
    spread_rate = float(np.mean(spread_ratios))
    # (2) the age structure stores less-mutated older cohorts.
    assert 1.0 - mutation_rate < adult_rate < 1.0
    # (3) both starting profiles converge to the same renewal root.
    assert adult_rate == pytest.approx(spread_rate, rel=1e-9)


def test_age_structured_decay_matches_the_renewal_root_without_sperm_storage() -> None:
    """With sperm storage disabled, decay equals the renewal root of the age profile.

    The gamete pool is then exactly the count-weighted average over the mating
    ages, so the wild-type fraction obeys ``p_{t+1} = (1 - mu) sum_a w_a
    p_{t-a}`` and decays at the dominant root of that renewal equation.
    Building the root from the *measured* stable age profile turns the
    qualitative "schedule-determined" claim into an exact quantitative
    contract, which also pins the mutation-rate magnitude in this lifecycle.
    """
    species = _species("_point_mutation_dyn_renewal_root")
    mutation_rate = 5e-3
    preset = nt.PointMutation(
        "RenewalMut", source_allele="A", target_allele="B", mutation_rate=mutation_rate
    )
    pop = _age_structured_population(
        "DynRenewal",
        species,
        preset,
        {
            "female": {"A|A": [0, 0, 400, 0, 0]},
            "male": {"A|A": [0, 0, 400, 0, 0]},
        },
        sperm_displacement_rate=1.0,
    )
    pop.run(100)  # reach the stable age profile
    counts = pop.state.individual_count
    weights = [
        _AGE_MATING_RATES[age] * float(counts[:, age, :].sum()) for age in range(5)
    ]
    measured_rate = float(np.mean(_late_decay_ratios(pop, warmup=0, window=20)))

    assert measured_rate == pytest.approx(
        _renewal_root(weights, mutation_rate), rel=1e-9
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stochastic runs
# ══════════════════════════════════════════════════════════════════════════════


def test_stochastic_runs_are_reproducible_and_binomially_consistent() -> None:
    """Two identical stochastic builds are bit-identical and mutate at rate ``mu``.

    Quantity tested: the B allele frequency after exactly one generation.  All
    parents are ``A|A`` and no fitness scaling is configured, so every B allele
    is a mutation event and the count is binomial with mean ``mu`` and variance
    ``mu(1-mu)/n_alleles``.  Tolerance: four standard errors (a two-sided
    99.99% bound under that model), with ``n_alleles`` taken conservatively
    from the *post-competition* count.  The engine seeds every population from
    the same default stream (seed 0), so both builds follow the identical
    trajectory and the assertion is deterministic, not flaky.
    """
    species = _species("_point_mutation_dyn_stochastic")
    mutation_rate = 5e-3
    preset = nt.PointMutation(
        "StochasticMut",
        source_allele="A",
        target_allele="B",
        mutation_rate=mutation_rate,
    )
    initial = {"female": {"A|A": 2000}, "male": {"A|A": 2000}}
    first = _discrete_population(
        "DynStochasticA", species, preset, initial, stochastic=True, eggs_per_female=5.0
    )
    second = _discrete_population(
        "DynStochasticB", species, preset, initial, stochastic=True, eggs_per_female=5.0
    )

    assert first.config.stochastic and second.config.stochastic

    first.run(1)
    second.run(1)
    np.testing.assert_array_equal(
        first.state.individual_count, second.state.individual_count
    )

    observed = _population_frequency(first, "B")
    n_alleles = 2.0 * float(first.state.individual_count.sum())
    standard_error = (mutation_rate * (1.0 - mutation_rate) / n_alleles) ** 0.5
    assert abs(observed - mutation_rate) < 4.0 * standard_error
