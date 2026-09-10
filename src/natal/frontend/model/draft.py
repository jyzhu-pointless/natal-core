"""ModelDraft: the single build-time draft configuration.

``ModelDraft`` replaces the former ``PopulationConfig`` /
``DiscretePopulationConfig`` pair.  One NamedTuple serves both
granularities: discrete-generation models are normalized at construction
(their per-generation scalars become the unified ``(2, n_ages)``
vectors), so no ``isinstance`` fork survives downstream.

Discipline:
    - Scalar fields are plain Python floats/ints (the 0-d ndarray idiom
      retired with the Numba removal).
    - Array contents may be mutated in place; scalar metadata requires
      ``_replace``.
    - The draft is a build-time object. ``build()`` materializes it into
      the ``Blueprint`` and ``Params`` contracts; runtime sessions consume
      those owned contracts while configuration snapshots remain drafts.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, TypeAlias

import numpy as np
from numpy.typing import NDArray

from natal.frontend.utils.types import Sex

# Mirrors ``natal.contracts.params.CustomValue``.  Defined locally because
# importing the contracts package here would cycle (contracts.materialize
# imports this module to materialize ModelDraft).
CustomValue: TypeAlias = bool | int | float | NDArray[np.float64]

__all__ = ["CustomValue", "ModelDraft"]


class ModelDraft(NamedTuple):
    """Unified build-time draft consumed by ``materialize`` and engines.

    Attributes:
        stochastic: Whether demographic events are stochastic.
        continuous_sampling: Dirichlet sampling for gamete proportions
            when True, multinomial when False.
        n_sexes: Number of sexes (always 2 in practice).
        n_ages: Number of age classes (discrete models normalize to 2).
        n_ztypes: Engine-visible zygote-type axis size.
        n_gtypes: Total gamete types (haploid genotypes x gamete
            labels).
        n_glabs: Gamete-label variants per haplotype.
        n_slabs: Somatic-label variants per genotype (pre-compression).
        new_adult_age: First adult age class.
        adult_ages: (A_adult,) int64 adult age indices.
        extreme_speed_mode: 0 off, 1 multinomial, 2 poisson, 3
            deterministic Wright-Fisher fused tick.
        ztype_names: Canonical ``"<genotype>:<slab>"`` name per ztype
            index.
        gtype_names: Canonical ``"<haplotype>:<glab>"`` name per gtype
            index.
        age_based_survival_rates: (2, n_ages) survival probabilities.
        age_based_mating_rates: (2, n_ages) mating probabilities.
        age_based_reproduction_rates: (n_ages,) female reproduction
            participation.
        female_age_based_fertility: (n_ages,) relative female fertility.
        age_based_relative_competition_strength: (n_ages,) competition
            weights.
        carrying_capacity: float, carrying capacity K.
        eggs_per_female: float, expected eggs per female.
        sex_ratio: float, newborn fraction female.
        sperm_displacement_rate: float, sperm displacement
            probability.
        low_density_growth_rate: float, low-density growth rate r.
        juvenile_growth_mode: int, density-regulation selector.
        viability_fitness: (2, n_ages, n_ztypes) viability coefficients.
        fecundity_fitness: (2, n_ztypes) fecundity coefficients.
        sexual_selection_fitness: (n_ztypes, n_ztypes) mating weights.
        zygote_viability_fitness: (2, n_ztypes) zygote survival
            coefficients.
        zygotes_to_gametes_map: (2, n_ztypes, n_gtypes) meiosis
            probabilities.
        gametes_to_zygotes_map: (n_gtypes, n_gtypes, n_ztypes) gamete
            pair to zygote mapping.
        offspring_tensor: (n_ztypes, n_ztypes, n_ztypes) offspring
            ztype probabilities per (mother, father) ztype pair.
        female_ztype_compatibility: (n_ztypes,) female mating weights.
        male_ztype_compatibility: (n_ztypes,) male mating weights.
        female_only_by_sex_chrom: (n_ztypes,) female-only mask.
        male_only_by_sex_chrom: (n_ztypes,) male-only mask.
        initial_individual_count: (2, n_ages, n_ztypes) initial
            population.
        initial_sperm_storage: (n_ages, n_ztypes, n_ztypes) initial
            sperm storage.
        equilibrium_individual_distribution: Optional (2, n_ages)
            declared equilibrium; None selects derivation mode.
        custom: User custom slots as a plain ``{name: value}`` dict
            (empty when none registered).  Values are validated and
            normalized by ``build_custom_slots``; scalars reach the Rust
            session via ``Params.custom_slots``.
        fixed_egg_count: Deterministic expected egg count when True.
        has_sex_chromosomes: Sex-chromosome constraints active.
        external_expected_eggs: Optional Champer-model egg override;
            None means unused.
        discrete_generation: True when the draft is in the
            discrete-generation normalization (built by
            ``build_discrete_engine_config``).  Granularity is a data
            flag on the unified draft, not a type.
    """

    # -- sampling flags --
    stochastic: bool
    continuous_sampling: bool
    # -- dimensions --
    n_sexes: int
    n_ages: int
    n_ztypes: int
    n_gtypes: int
    n_glabs: int
    n_slabs: int
    new_adult_age: int
    adult_ages: NDArray[np.int64]
    extreme_speed_mode: int
    # -- symbolic name directory --
    ztype_names: tuple[str, ...]
    gtype_names: tuple[str, ...]
    # -- demographic vectors --
    age_based_survival_rates: NDArray[np.float64]
    age_based_mating_rates: NDArray[np.float64]
    # Optional at runtime: None means "not declared", and consumers fall
    # back to the female mating-rate row.
    age_based_reproduction_rates: NDArray[np.float64] | None
    female_age_based_fertility: NDArray[np.float64]
    age_based_relative_competition_strength: NDArray[np.float64]
    # -- ecological scalars (immutable NamedTuple fields) --
    carrying_capacity: float
    eggs_per_female: float
    sex_ratio: float
    sperm_displacement_rate: float
    low_density_growth_rate: float
    juvenile_growth_mode: int
    # generation_time is a declared static descriptor (age_structure
    # kwarg / build-time derivation); the equilibrium metrics are NOT
    # stored — read them via the always-fresh derive surface
    # (pop.params.expected_*).
    generation_time: float
    # -- fitness tensors --
    viability_fitness: NDArray[np.float64]
    fecundity_fitness: NDArray[np.float64]
    sexual_selection_fitness: NDArray[np.float64]
    zygote_viability_fitness: NDArray[np.float64]
    # -- inheritance maps --
    zygotes_to_gametes_map: NDArray[np.float64]
    gametes_to_zygotes_map: NDArray[np.float64]
    offspring_tensor: NDArray[np.float64]
    # -- compatibility and sex-chromosome masks --
    female_ztype_compatibility: NDArray[np.float64]
    male_ztype_compatibility: NDArray[np.float64]
    female_only_by_sex_chrom: NDArray[np.bool_]
    male_only_by_sex_chrom: NDArray[np.bool_]
    # -- initial state --
    initial_individual_count: NDArray[np.float64]
    initial_sperm_storage: NDArray[np.float64]
    # -- declarations and plumbing --
    equilibrium_individual_distribution: Optional[NDArray[np.float64]]
    custom: dict[str, CustomValue]
    fixed_egg_count: bool
    has_sex_chromosomes: bool
    external_expected_eggs: Optional[float] = None
    discrete_generation: bool = False

    def set_viability_fitness(
        self, sex: int, ztype_idx: int, value: float, age: int = -1
    ) -> None:
        """Write one viability fitness coefficient in place.

        Args:
            sex: Sex index.
            ztype_idx: Zygote-type index.
            value: Fitness value.
            age: Age class; negative selects the last juvenile age
                (``new_adult_age - 1``).
        """
        if age < 0:
            age = self.new_adult_age - 1
        self.viability_fitness[sex, age, ztype_idx] = value

    def set_fecundity_fitness(self, sex: int, ztype_idx: int, value: float) -> None:
        """Write one fecundity fitness coefficient in place.

        Args:
            sex: Sex index.
            ztype_idx: Zygote-type index.
            value: Fitness value.
        """
        self.fecundity_fitness[sex, ztype_idx] = value

    def set_sexual_selection_fitness(
        self, female_ztype_idx: int, male_ztype_idx: int, value: float
    ) -> None:
        """Write one sexual-selection weight in place.

        Args:
            female_ztype_idx: Female zygote-type index.
            male_ztype_idx: Male zygote-type index.
            value: Mating weight.
        """
        self.sexual_selection_fitness[female_ztype_idx, male_ztype_idx] = value

    def set_zygote_viability_fitness(
        self, sex: int, ztype_idx: int, value: float
    ) -> None:
        """Write one zygote-viability coefficient in place.

        Args:
            sex: Sex index.
            ztype_idx: Zygote-type index.
            value: Survival probability in [0, 1].
        """
        self.zygote_viability_fitness[sex, ztype_idx] = value

    def compute_generation_time(self) -> float:
        """Compute the mean generation time from current demographics.

        Returns:
            Mean generation time averaged over sexes.
        """
        gen_times = np.zeros(self.n_sexes, dtype=np.float64)
        for sex in range(self.n_sexes):
            cumulative_survival = np.ones(self.n_ages, dtype=np.float64)
            for age in range(1, self.n_ages):
                cumulative_survival[age] = (
                    cumulative_survival[age - 1]
                    * self.age_based_survival_rates[sex, age - 1]
                )

            numerator = 0.0
            denominator = 0.0
            for age in range(self.n_ages):
                cumulative_mating_value = self.age_based_mating_rates[sex, age]
                if sex == Sex.FEMALE:
                    cumulative_mating_value *= self.female_age_based_fertility[age]
                if cumulative_mating_value > 0:
                    numerator += age * cumulative_survival[age] * cumulative_mating_value
                    denominator += cumulative_survival[age] * cumulative_mating_value

            if denominator > 0:
                gen_times[sex] = numerator / denominator

        return float(np.mean(gen_times))
