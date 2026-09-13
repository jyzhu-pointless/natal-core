"""Complex public model trajectories checked without using compiled genetic maps."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import TypeAlias

import numpy as np
import pytest

import natal as nt
from natal.frontend.genetics.compile import RecipeHost
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier

GeneticState: TypeAlias = tuple[int, tuple[str, str], bool]
Census: TypeAlias = dict[GeneticState, float]


class _CascadePreset(nt.GeneticPreset):
    """Submit rules through the public preset/build boundary, not tensor wrappers."""

    def __init__(
        self, name: str, *, gamete: GameteConversionRuleSet | None = None,
        zygote: ZygoteConversionRuleSet | None = None,
    ) -> None:
        super().__init__(name=name)
        self.gamete_rules = gamete
        self.zygote_rules = zygote

    def gamete_modifier(self, host: RecipeHost) -> GameteModifier | None:
        """Compile gamete declarations against the build's active registry."""
        return self.gamete_rules.to_gamete_modifier(host) if self.gamete_rules else None

    def zygote_modifier(self, host: RecipeHost) -> ZygoteModifier | None:
        """Compile zygote declarations against the build's active registry."""
        return self.zygote_rules.to_zygote_modifier(host) if self.zygote_rules else None


def _maternal_drive_step(census: Census, transmission: float) -> Census:
    """Independent parental-cross oracle for tagging, drive, resistance and fitness.

    Each mother lays two eggs, scaled by 3/4 if infected. Fathers are
    sampled proportional to their adult counts; infected fathers also
    multiply pair fecundity by 3/4. Each gamete chooses a
    parental copy with probability 1/2; infected D-carrying mothers convert
    W to D with probability 2/5. Maternal infection is transmitted with
    probability q. In infected embryos each W copy independently converts
    to R with probability 1/4, and embryo viability is 4/5. Sex is an
    independent fair coin. There is no density regulation or adult carryover.
    """
    result: defaultdict[GeneticState, float] = defaultdict(float)
    male_total = sum(count for (sex, _, _), count in census.items() if sex == 1)
    order = {allele: index for index, allele in enumerate(("W", "D", "R"))}
    for (sex, mother, mother_infected), females in census.items():
        if sex != 0:
            continue
        eggs = females * 2.0 * (0.75 if mother_infected else 1.0)
        for (father_sex, father, father_infected), males in census.items():
            if father_sex != 1:
                continue
            for egg_copy, sperm_copy in product(mother, father):
                homing = 0.4 if mother_infected and "D" in mother and egg_copy == "W" else 0.0
                for maternal_gene, gamete_p in ((egg_copy, 1.0 - homing), ("D", homing)):
                    infection_p = transmission if mother_infected else 0.0
                    for infected, label_p in ((False, 1.0 - infection_p), (True, infection_p)):
                        weight = eggs * males / male_total * (0.75 if father_infected else 1.0) * 0.25 * gamete_p * label_p
                        if not weight:
                            continue
                        options = [
                            [(allele, 0.75), ("R", 0.25)] if infected and allele == "W" else [(allele, 1.0)]
                            for allele in (maternal_gene, sperm_copy)
                        ]
                        for (first, p_first), (second, p_second) in product(*options):
                            pair = tuple(sorted((first, second), key=order.__getitem__))
                            mass = weight * p_first * p_second * (0.8 if infected else 1.0) * 0.5
                            result[0, pair, infected] += mass
                            result[1, pair, infected] += mass
    return dict(result)


def _assert_census(population: nt.DiscreteGenerationPopulation, census: Census) -> None:
    """Compare every sex/age/ZType cell, including all expected-zero outcomes."""
    expected = np.zeros_like(population.state.individual_count)
    observation = np.zeros((3, 2))
    for (sex, pair, infected), count in census.items():
        genotype = population.species.get_genotype_from_str("|".join(pair))
        label = "infected" if infected else "default"
        index = population.registry.ztype_index(genotype, label)
        expected[sex, 1, index] += count
        observation[0 if infected else 2, sex] += count
        if infected and "R" in pair:
            observation[1, sex] += count
    # Five generations of small probability products: tolerance is rounding,
    # not a biological allowance (well below one individual at this census).
    np.testing.assert_allclose(population.state.individual_count, expected, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(population.observe().values, observation, rtol=2e-12, atol=1e-10)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("transmission", [0.0, 0.6, 1.0])
def test_four_rule_families_joint_inheritance_fitness_and_resume(
    compress: bool, transmission: float,
) -> None:
    """Check five generations, two refreshes and restored continuation against crosses."""
    species = nt.Species.from_dict(
        name=f"joint_inheritance_{compress}_{transmission}",
        structure={"autosome": {"drive": ["W", "D", "R", "unused"]}},
        gamete_labels=["default", "maternal", "unused"],
        somatic_labels=["default", "infected", "unused"],
    )
    tagging = GameteConversionRuleSet().add_gtype_convert(
        to="*@maternal", rate=1.0,
        filters={"parent_sex": "female", "parent": "*@infected", "current": "*@default"},
    )
    homing = GameteConversionRuleSet().add_allele_convert(
        from_allele="W", to_allele="D", rate=0.4,
        filters={"parent_sex": "female", "parent": "*::D@infected", "current": "W@maternal"},
    )
    infection = ZygoteConversionRuleSet().add_ztype_convert(
        to="*@infected", rate=transmission,
        filters={"maternal": "*@maternal", "paternal": "*@default"},
    )
    embryo = ZygoteConversionRuleSet().add_allele_convert(
        from_allele="W", to_allele="R", rate=0.25, side="both",
        filters={"current": "*@infected", "maternal": "*@maternal", "paternal": "*@default"},
    )
    population = (
        nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False, compress=compress)
        .initial_state(individual_count={
            "female": {"W|D@infected": 600, "W|W@default": 400},
            "male": {"W|R@infected": 200, "W|W@default": 800},
        })
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.5, fixed_egg_count=True)
        .competition(growth_mode="no_competition")
        .presets(
            _CascadePreset("tagging", gamete=tagging),
            _CascadePreset("homing", gamete=homing),
            _CascadePreset("infection", zygote=infection),
            _CascadePreset("embryo", zygote=embryo),
            nt.TransgenicBackground(
                name="infection_cost", tg_slab="infected", wt_slab="default",
                fecundity_scaling=0.75, viability_scaling=0.8,
            ),
        )
        .with_observation(groups={
            "infected": nt.IndividualSelector(ztype="*@infected"),
            "resistant_infected": nt.IndividualSelector(ztype="*::R@infected"),
            "uninfected": nt.IndividualSelector(ztype="*@default"),
        }, collapse_age=True)
        .build()
    )
    if compress:
        assert population.registry.n_ztypes < len(species.get_all_genotypes()) * 3
    census: Census = {
        (0, ("W", "D"), True): 600.0, (0, ("W", "W"), False): 400.0,
        (1, ("W", "R"), True): 200.0, (1, ("W", "W"), False): 800.0,
    }
    _assert_census(population, census)
    saved = None
    for tick in range(1, 6):
        if tick == 3:
            population.refresh_modifiers()
            population.refresh_modifiers()
        census = _maternal_drive_step(census, transmission)
        population.run(1)
        assert population.tick == tick
        _assert_census(population, census)
        if tick == 2:
            saved = population.export_state()
    assert saved is not None
    population.import_state(saved)
    assert population.tick == 2
    population.refresh_modifiers()
    population.run(3)
    assert population.tick == 5
    _assert_census(population, census)


def _linked_population(system: str, rate: float, compress: bool, mode: int) -> nt.DiscreteGenerationPopulation:
    """Build a phase-preserving two-locus model with unequal sex-chromosome lengths."""
    primary, partner = system
    species = nt.Species.from_dict(
        name=f"linked_{system}_{rate}_{compress}", unordered=False,
        # Deliberately put sex chromosomes before the autosome in declaration order.
        structure={
            "partner": {"sex_type": partner, "loci": {"partner_marker": [partner]}},
            "primary": {"sex_type": primary, "loci": {"primary_marker": [primary], "extra": ["E"]}},
            "autosome": {"locus_a": ["A", "a"], "locus_b": ["B", "b"]},
        },
    )
    chromosome = species.get_chromosome("autosome")
    chromosome.set_recombination(chromosome.loci[0], chromosome.loci[1], rate)
    female_sex, male_sex = (("X/E|X/E", "X/E|Y") if system == "XY" else ("W|Z/E", "Z/E|Z/E"))
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, stochastic=False, compress=compress, extreme_speed_mode=mode,
        )
        .initial_state(individual_count={
            "female": {f"A/B|a/b;{female_sex}": 1000},
            "male": {f"A/B|a/b;{male_sex}": 1000},
        })
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.9, fixed_egg_count=True)
        .competition(growth_mode="no_competition")
        .build()
    )


def _assert_linkage_generation(
    population: nt.DiscreteGenerationPopulation, system: str, rate: float, tick: int,
) -> None:
    """After the first cross, linkage disequilibrium decays by (1-r) each generation."""
    disequilibrium = (1.0 - 2.0 * rate) * (1.0 - rate) ** (tick - 1) / 4.0
    gametes = {"A/B": 0.25 + disequilibrium, "a/b": 0.25 + disequilibrium,
               "A/b": 0.25 - disequilibrium, "a/B": 0.25 - disequilibrium}
    expected = np.zeros_like(population.state.individual_count)
    sex_pairs = ("X/E|X/E", "X/E|Y") if system == "XY" else ("W|Z/E", "Z/E|Z/E")
    for sex, sex_pair in enumerate(sex_pairs):
        for mother, father in product(gametes, repeat=2):
            count = 1000.0 * gametes[mother] * gametes[father]
            if count == 0:
                continue  # Correctly pruned under r=0 and compression.
            genotype = population.species.get_genotype_from_str(f"{mother}|{father};{sex_pair}")
            index = population.registry.ztype_index(genotype, "default")
            expected[sex, 1, index] += count
    np.testing.assert_allclose(population.state.individual_count, expected, rtol=2e-12, atol=1e-10)


@pytest.mark.parametrize("compress", [False, True])
@pytest.mark.parametrize("system", ["XY", "ZW"])
@pytest.mark.parametrize("rate", [0.0, 0.2, 0.5])
@pytest.mark.parametrize("mode", [0, 3])
def test_multigeneration_linkage_with_sex_chromosomes(
    compress: bool, system: str, rate: float, mode: int,
) -> None:
    """Verify every ordered autosomal pair and genetic sex across four generations."""
    population = _linked_population(system, rate, compress, mode)
    for tick in range(1, 5):
        if tick == 3:
            population.refresh_modifiers()
        population.run(1)
        assert population.tick == tick
        _assert_linkage_generation(population, system, rate, tick)


@pytest.mark.parametrize("system", ["XY", "ZW"])
@pytest.mark.parametrize("compress", [False, True])
def test_shared_recombination_edit_changes_new_build_not_running_population(
    system: str, compress: bool,
) -> None:
    """A shared rate-array write affects a new model, while an existing engine stays fixed."""
    original = _linked_population(system, 0.1, compress, 0)
    species = original.species
    rates = np.asarray(species.get_chromosome("autosome").recombination_map)
    rates[:] = 0.4
    female_sex, male_sex = (("X/E|X/E", "X/E|Y") if system == "XY" else ("W|Z/E", "Z/E|Z/E"))
    rebuilt = (
        nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False, compress=compress)
        .initial_state(individual_count={
            "female": {f"A/B|a/b;{female_sex}": 1000},
            "male": {f"A/B|a/b;{male_sex}": 1000},
        })
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2.0, sex_ratio=0.9, fixed_egg_count=True)
        .competition(growth_mode="no_competition")
        .build()
    )
    for tick in range(1, 4):
        original.run(1)
        rebuilt.run(1)
        _assert_linkage_generation(original, system, 0.1, tick)
        _assert_linkage_generation(rebuilt, system, 0.4, tick)
