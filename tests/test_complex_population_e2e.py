"""Public spatial lifecycle checks with independently derived cohort trajectories."""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.genetics.compile import RecipeHost
from natal.frontend.modifiers import GameteConversionRuleSet
from natal.frontend.modifiers.module import GameteModifier
from natal.frontend.presets import PresetFitnessPatch
from natal.frontend.spatial.builder import batch_setting


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("migration_rate", [0.0, 0.2, 1.0])
def test_directed_migration_preserves_labeled_cohort_survival(
    compress: bool, migration_rate: float
) -> None:
    """Track sex, age, genotype, and label through four spatial ticks.

    Reproduction is disabled. Each cohort therefore follows the directed
    migration matrix raised to the tick number, multiplied by its sex- and
    age-specific survival product. Distinct starting deme vectors expose
    swapped migration directions and mixed genotype/label coordinates;
    compressed and full registries must give the same public outcome.
    """
    species = nt.Species.from_dict(
        f"directed_cohorts_{compress}_{migration_rate}",
        {"autosome": {"marker": ["A", "B"]}},
        somatic_labels=["default", "infected"],
    )
    # (sex, initial age, exact ZType, counts by source deme).
    cohorts = [
        ("female", 1, "A|A@infected", [120.0, 30.0, 0.0]),
        ("female", 2, "A|B@default", [0.0, 80.0, 20.0]),
        ("male", 1, "B|B@default", [10.0, 0.0, 90.0]),
        ("male", 2, "A|A@infected", [40.0, 60.0, 10.0]),
    ]
    initial_states: list[dict[str, dict[str, list[float]]]] = []
    for deme in range(3):
        state: dict[str, dict[str, list[float]]] = {"female": {}, "male": {}}
        for sex, age, ztype, counts in cohorts:
            ages = [0.0] * 7
            ages[age] = counts[deme]
            state[sex][ztype] = ages
        initial_states.append(state)
    female_survival = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0]
    male_survival = [1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.0]
    adjacency = np.array([[0.0, 0.75, 0.25], [0.5, 0.0, 0.5], [1.0, 0.0, 0.0]])
    population = (
        nt.SpatialPopulation.builder(species, n_demes=3)
        .setup(stochastic=False, compress=compress)
        .age_structure(n_ages=7, new_adult_age=1)
        .initial_state(individual_count=batch_setting(initial_states))
        .survival(
            female_age_based_survival=female_survival,
            male_age_based_survival=male_survival,
        )
        .reproduction(
            eggs_per_female=0.0,
            female_age_based_mating_rate=[0.0] * 7,
            male_age_based_mating_rate=[0.0] * 7,
        )
        .competition(expected_num_new_adult_females=100)
        .migration(adjacency=adjacency, migration_rate=migration_rate)
        .build()
    )
    transition = (1.0 - migration_rate) * np.eye(3) + migration_rate * adjacency
    for tick in range(1, 5):
        population.run_tick()
        assert population.tick == tick
        expected = [
            np.zeros_like(deme.state.individual_count) for deme in population.demes
        ]
        for sex, age, ztype, counts in cohorts:
            sex_index = 0 if sex == "female" else 1
            survival = female_survival if sex == "female" else male_survival
            destinations = np.asarray(counts) @ np.linalg.matrix_power(transition, tick)
            destinations *= np.prod(survival[age : age + tick])
            genotype_text, label = ztype.split("@")
            genotype = species.get_genotype_from_str(genotype_text)
            for deme_index, deme in enumerate(population.demes):
                ztype_index = deme.index_registry.ztype_index(genotype, label)
                expected[deme_index][sex_index, age + tick, ztype_index] += (
                    destinations[deme_index]
                )
        for deme, expected_counts in zip(population.demes, expected, strict=True):
            # Four small matrix/survival products accumulate only rounding error.
            np.testing.assert_allclose(
                deme.state.individual_count, expected_counts, rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("compress", [False, True])
def test_labeled_stored_sperm_survives_batched_compression_and_migration(
    compress: bool,
) -> None:
    """Stored sire coordinates travel and survive with their female cohort."""
    species = nt.Species.from_dict(
        f"stored_sperm_cohorts_{compress}",
        {"autosome": {"marker": ["A", "B"]}},
        somatic_labels=["default", "infected"],
    )
    population = (
        nt.SpatialPopulation.builder(species, n_demes=2)
        .setup(stochastic=False, compress=compress)
        .age_structure(n_ages=5, new_adult_age=1)
        .initial_state(
            individual_count=batch_setting(
                [
                    {
                        "female": {"A|A@infected": {1: 100.0}},
                        "male": {"B|B@infected": {1: 40.0}},
                    },
                    {
                        "female": {"A|A@infected": {1: 60.0}},
                        "male": {"B|B@infected": {1: 20.0}},
                    },
                ]
            ),
            sperm_storage=batch_setting(
                [
                    {"A|A@infected": {"B|B@infected": {1: 80.0}}},
                    {"A|A@infected": {"B|B@infected": {1: 20.0}}},
                ]
            ),
        )
        .survival(
            female_age_based_survival=[1.0, 0.8, 0.75, 0.5, 0.0],
            male_age_based_survival=[1.0, 0.6, 0.5, 0.25, 0.0],
        )
        .reproduction(
            eggs_per_female=0.0,
            female_age_based_mating_rate=[0.0] * 5,
            male_age_based_mating_rate=[0.0] * 5,
        )
        .competition(expected_num_new_adult_females=100)
        .migration(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), migration_rate=0.25)
        .build()
    )
    transition = np.array([[0.75, 0.25], [0.25, 0.75]])
    for tick in range(3):
        if tick:
            population.run_tick()
        female_survival = [1.0, 0.8, 0.6][tick]
        male_survival = [1.0, 0.6, 0.3][tick]
        movement = np.linalg.matrix_power(transition, tick)
        females = np.array([100.0, 60.0]) @ movement * female_survival
        males = np.array([40.0, 20.0]) @ movement * male_survival
        sperm = np.array([80.0, 20.0]) @ movement * female_survival
        for index, deme in enumerate(population.demes):
            registry = deme.index_registry
            female = registry.ztype_index(
                species.get_genotype_from_str("A|A"), "infected"
            )
            male = registry.ztype_index(
                species.get_genotype_from_str("B|B"), "infected"
            )
            counts_expected = np.zeros_like(deme.state.individual_count)
            counts_expected[0, 1 + tick, female] = females[index]
            counts_expected[1, 1 + tick, male] = males[index]
            sperm_expected = np.zeros_like(deme.state.sperm_storage)
            sperm_expected[1 + tick, female, male] = sperm[index]
            np.testing.assert_allclose(
                deme.state.individual_count, counts_expected, rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                deme.state.sperm_storage, sperm_expected, rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("compress", [False, True])
def test_discrete_spatial_fitness_variants_reproduce_before_migration(
    compress: bool,
) -> None:
    """Distinct genetics variants obey local fecundity/viability then movement."""
    species = nt.Species.from_dict(
        f"discrete_fitness_cohorts_{compress}",
        {"autosome": {"marker": ["A", "B"]}},
        somatic_labels=["default", "infected"],
    )
    population = (
        nt.SpatialPopulation.builder(species, n_demes=2, pop_type="discrete_generation")
        .setup(stochastic=False, compress=compress)
        .initial_state(
            individual_count=batch_setting(
                [
                    {"female": {"A|A@infected": 100.0}, "male": {"A|A@infected": 50.0}},
                    {"female": {"A|A@infected": 40.0}, "male": {"A|A@infected": 30.0}},
                ]
            )
        )
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .fitness(viability=batch_setting([{"A|A": 0.8}, {"A|A": 0.5}]), mode="multiply")
        .competition(juvenile_growth_mode=nt.NO_COMPETITION, carrying_capacity=1000)
        .migration(adjacency=np.array([[0.0, 1.0], [1.0, 0.0]]), migration_rate=0.3)
        .build()
    )
    transition = np.array([[0.7, 0.3], [0.3, 0.7]])
    # Two eggs per female, equal sex probability, local viability and no
    # density regulation give F_next = (F * viability) @ transition. All
    # descendants carry the default label without a transmission rule.
    expected_females = np.array([100.0, 40.0])
    for _tick in range(3):
        population.run_tick()
        expected_females = (expected_females * [0.8, 0.5]) @ transition
        for index, deme in enumerate(population.demes):
            ztype = deme.index_registry.ztype_index(
                species.get_genotype_from_str("A|A"), "default"
            )
            expected = np.zeros_like(deme.state.individual_count)
            expected[:, 1, ztype] = expected_females[index]
            np.testing.assert_allclose(
                deme.state.individual_count, expected, rtol=1e-12, atol=1e-12
            )


class _ZeroRateSlabPreset(nt.GeneticPreset):
    """Combine a pruned zero-rate target and one slab-level fitness patch."""

    def __init__(self, patch_key: str, slab: str = "default") -> None:
        super().__init__(name=f"zero_rate_{patch_key}")
        self.patch_key = patch_key
        self.slab = slab

    def gamete_modifier(self, host: RecipeHost) -> GameteModifier:
        """A valid zero-rate target need not remain in a compressed registry."""
        return (
            GameteConversionRuleSet()
            .add_gtype_convert(to="B@unused", rate=0.0)
            .to_gamete_modifier(host)
        )

    def fitness_patch(self) -> PresetFitnessPatch:
        """Return a public preset patch with an explicit label name."""
        return {self.patch_key: {self.slab: 0.5}}

    def zygote_modifier(self, host: RecipeHost) -> None:
        """Leave Mendelian zygote formation unchanged."""
        return None


@pytest.mark.parametrize("patch_key", ["sexual_selection_per_slab", "zygote_per_slab"])
@pytest.mark.parametrize("speed_mode", [0, 3])
def test_zero_rate_pruned_gamete_target_and_slab_fitness_survive_refresh(
    patch_key: str,
    speed_mode: int,
) -> None:
    """Refresh keeps valid absent targets inert and reapplies slab fitness once."""
    species = nt.Species.from_dict(
        f"zero_rate_refresh_{patch_key}",
        {"autosome": {"marker": ["A", "B"]}},
        gamete_labels=["default", "unused"],
        somatic_labels=["default", "infected"],
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            stochastic=False,
            compress=True,
            extreme_speed_mode=speed_mode,
        )
        .initial_state(
            individual_count={"female": {"A|A": 100.0}, "male": {"A|A": 100.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .competition(growth_mode="no_competition")
        .presets(_ZeroRateSlabPreset(patch_key))
        .build()
    )
    assert population.registry.n_ztypes == 1
    assert population.registry.n_gtypes == 1
    for tick in range(1, 4):
        population.refresh_modifiers()
        population.refresh_modifiers()
        population.run(1)
        # A common nonzero mate preference cancels in conditional choice;
        # offspring survival of one half instead reduces abundance each tick.
        survivors = 100.0 * (0.5**tick if patch_key == "zygote_per_slab" else 1.0)
        expected = np.zeros_like(population.state.individual_count)
        expected[:, 1, 0] = survivors
        np.testing.assert_allclose(
            population.state.individual_count, expected, rtol=1e-12, atol=1e-12
        )
        config = population.export_config()
        fitness = (
            config.zygote_viability_fitness
            if patch_key == "zygote_per_slab"
            else config.sexual_selection_fitness
        )
        np.testing.assert_array_equal(fitness, np.full_like(fitness, 0.5))


@pytest.mark.parametrize(
    "patch_key",
    [
        "viability_per_slab",
        "fecundity_per_slab",
        "sexual_selection_per_slab",
        "zygote_per_slab",
    ],
)
def test_unknown_slab_fitness_still_fails_before_compressed_run(patch_key: str) -> None:
    """Compression may omit known labels, but never excuses a misspelled label."""
    species = nt.Species.from_dict(
        f"invalid_slab_{patch_key}",
        {"autosome": {"marker": ["A", "B"]}},
        gamete_labels=["default", "unused"],
    )
    with pytest.raises(KeyError, match="Unknown somatic label 'typo'"):
        (
            nt.DiscreteGenerationPopulation.setup(
                species=species, stochastic=False, compress=True
            )
            .initial_state(
                individual_count={"female": {"A|A": 100.0}, "male": {"A|A": 100.0}}
            )
            .presets(_ZeroRateSlabPreset(patch_key, slab="typo"))
            .build()
        )


@pytest.mark.parametrize("speed_mode", [0, 2])
@pytest.mark.parametrize("continuous", [False, True])
def test_stochastic_zygote_survival_matches_sex_specific_poisson_thinning(
    speed_mode: int,
    continuous: bool,
) -> None:
    """Check offspring means, variances, and support across independent draws.

    One hundred mothers each contribute Poisson(2) eggs, split equally by
    sex, then survive with female probability 1/4 and male probability 3/4.
    Thus offspring counts have means/variances (25, 75). Continuous sampling
    uses moment-matched gamma/beta draws: the same moments follow from total
    expectation and variance (the small-count fallback has negligible mass
    at these abundances). The fused Poisson mode samples these means directly.

    Reimporting abundance preserves the live RNG stream, giving 128 successive
    nonoverlapping trials. Six standard-error bounds on the means and sample
    variances control false failures across the eight sex/mode comparisons;
    they reject missing survival, swapped sex probabilities, and zero variance.
    """
    species = nt.Species.from_dict(
        f"stochastic_zygote_{speed_mode}_{continuous}",
        {"autosome": {"marker": ["A"]}},
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            stochastic=True,
            continuous_sampling=continuous,
            extreme_speed_mode=speed_mode,
        )
        .initial_state(
            individual_count={"female": {"A|A": 100.0}, "male": {"A|A": 100.0}}
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .fitness(zygote_viability={"A|A": {"female": 0.25, "male": 0.75}})
        .competition(growth_mode="no_competition")
        .build()
    )
    initial = population.export_state()
    trials = 128
    samples = np.zeros((trials, 2))
    for trial in range(trials):
        population.import_state(initial)
        population.run(1)
        samples[trial] = population.state.individual_count[:, 1, 0]
    means = np.array([25.0, 75.0])
    assert np.isfinite(samples).all()
    assert (samples >= 0).all()
    if not continuous or speed_mode == 2:
        np.testing.assert_array_equal(samples, np.round(samples))
    np.testing.assert_array_less(
        np.abs(samples.mean(axis=0) - means),
        6.0 * np.sqrt(means / trials),
    )
    # For Poisson samples, Var(S²) = mu/n + 2*mu²/(n-1). Gamma/beta
    # moment matching changes higher moments slightly; at means >=25 the
    # broad six-SE bound also accommodates that continuous approximation.
    np.testing.assert_array_less(
        np.abs(samples.var(axis=0, ddof=1) - means),
        6.0 * np.sqrt(means / trials + 2.0 * means**2 / (trials - 1)),
    )


@pytest.mark.parametrize("speed_mode", [0, 3])
def test_zygote_and_juvenile_survival_multiply_on_the_correct_sex_axis(
    speed_mode: int,
) -> None:
    """Early and later survival apply once each, before the adult census."""
    species = nt.Species.from_dict(
        f"two_survival_stages_{speed_mode}",
        {"autosome": {"marker": ["A"]}},
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(
            species=species,
            stochastic=False,
            extreme_speed_mode=speed_mode,
        )
        .initial_state(
            individual_count={"female": {"A|A": 100.0}, "male": {"A|A": 100.0}}
        )
        .survival(female_age0_survival=0.8, male_age0_survival=0.4)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
        .fitness(zygote_viability={"A|A": {"female": 0.25, "male": 0.75}})
        .competition(growth_mode="no_competition")
        .build()
    )
    population.run(1)
    expected = np.zeros_like(population.state.individual_count)
    expected[0, 1, 0] = 100.0 * 0.25 * 0.8
    expected[1, 1, 0] = 100.0 * 0.75 * 0.4
    np.testing.assert_allclose(
        population.state.individual_count, expected, rtol=1e-12, atol=1e-12
    )
