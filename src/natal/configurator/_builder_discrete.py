"""DiscreteGenerationPopulationBuilder extracted from _factory.py."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
    overload,
)

if TYPE_CHECKING:
    from natal.population.discrete_generation import DiscreteGenerationPopulation

import numpy as np

from natal.configurator._builder_base import (
    ArrayF64,
    FecundityMap,
    FitnessOperation,
    GenotypeSelector,
    HookFn,
    HookMap,
    InitialAgeCountValue,
    InitialIndividualCountInput,
    ModifierSpec,
    PopulationBuilderBase,
    SexualSelectionMap,
    ViabilityMap,
    ZygoteViabilityMap,
)
from natal.data import LOGISTIC
from natal.genetics import Genotype, Species
from natal.utils.helpers import resolve_sex_label


class DiscreteGenerationPopulationBuilder(PopulationBuilderBase):
    """Builder for DiscreteGenerationPopulation.

    For populations with discrete, non-overlapping generations.

    Note:
        This builder fixes ``n_ages=2`` and ``new_adult_age=1``.
        In discrete engine, juvenile competition strength is computed from
        total age-0 abundance directly.
    """

    def __init__(self, species: Species):
        super().__init__(species)

        self.name: str = "DiscreteGenerationPop"
        self.stochastic: bool = True
        self.continuous_sampling: bool = False
        self.fixed_egg_count: bool = False

        self.initial_individual_count: Optional[InitialIndividualCountInput] = None

        self.eggs_per_female: float = 50.0
        self.sex_ratio: float = 0.5

        self.female_age0_survival: float = 1.0
        self.male_age0_survival: float = 1.0

        self.female_adult_mating_rate: float = 1.0
        self.male_adult_mating_rate: float = 1.0

        self.juvenile_growth_mode: Union[int, str] = LOGISTIC
        self.low_density_growth_rate: float = 1.0
        self.carrying_capacity: Optional[float] = None
        self.equilibrium_individual_distribution: Optional[ArrayF64] = None

        self.gamete_modifiers: Optional[List[ModifierSpec]] = None
        self.zygote_modifiers: Optional[List[ModifierSpec]] = None
        self._fitness_operations: List[FitnessOperation] = []
        self._hooks: HookMap = {}

    def presets(self, *preset_list: Any) -> DiscreteGenerationPopulationBuilder:
        if preset_list:
            self._presets = list(preset_list)
        return self

    def fitness(
        self,
        viability: Optional[ViabilityMap] = None,
        fecundity: Optional[FecundityMap] = None,
        sexual_selection: Optional[SexualSelectionMap] = None,
        zygote_viability: Optional[ViabilityMap] = None,
        mode: str = "replace",
    ) -> DiscreteGenerationPopulationBuilder:
        if viability is not None:
            self._fitness_operations.append(("viability", (viability,), {'mode': mode}))
        if fecundity is not None:
            self._fitness_operations.append(("fecundity", (fecundity,), {'mode': mode}))
        if sexual_selection is not None:
            self._fitness_operations.append(("sexual_selection", (sexual_selection,), {'mode': mode}))
        if zygote_viability is not None:
            self._fitness_operations.append(("zygote_viability", (zygote_viability,), {'mode': mode}))
        return self

    @staticmethod
    def _iter_sexual_selection_entries(
        sexual_selection: Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
    ) -> Iterable[Tuple[GenotypeSelector, GenotypeSelector, float]]:
        if not sexual_selection:
            return []
        has_nested = any(isinstance(v, dict) for v in sexual_selection.values())
        entries: List[Tuple[GenotypeSelector, GenotypeSelector, float]] = []
        if has_nested:
            for female_selector, male_map in sexual_selection.items():
                if not isinstance(male_map, dict):
                    raise TypeError(
                        "When using nested sexual_selection, each female key must map to a dict of male->value"
                    )
                for male_selector, value in male_map.items():
                    entries.append((female_selector, male_selector, float(value)))
            return entries
        for male_selector, value in sexual_selection.items():
            assert isinstance(value, float), "In flat sexual_selection form, values must be floats"
            entries.append(("*", male_selector, value))
        return entries

    def modifiers(
        self,
        gamete_modifiers: Optional[List[ModifierSpec]] = None,
        zygote_modifiers: Optional[List[ModifierSpec]] = None,
    ) -> DiscreteGenerationPopulationBuilder:
        if gamete_modifiers is not None:
            self.gamete_modifiers = gamete_modifiers
        if zygote_modifiers is not None:
            self.zygote_modifiers = zygote_modifiers
        return self

    def setup(
        self,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        **kwargs: object,
    ) -> DiscreteGenerationPopulationBuilder:
        self.name = name
        self.stochastic = stochastic
        self.continuous_sampling = continuous_sampling
        self.fixed_egg_count = fixed_egg_count
        self._set_param("setup.stochastic", stochastic)
        self._set_param("setup.continuous_sampling", continuous_sampling)
        self._set_param("setup.fixed_egg_count", fixed_egg_count)
        self._set_params(domain="setup", **kwargs)
        return self

    @overload
    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[str, InitialAgeCountValue]],
    ) -> DiscreteGenerationPopulationBuilder:
        ...

    @overload
    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[Genotype, InitialAgeCountValue]],
    ) -> DiscreteGenerationPopulationBuilder:
        ...

    @overload
    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[Genotype | str, InitialAgeCountValue]],
    ) -> DiscreteGenerationPopulationBuilder:
        ...

    def initial_state(
        self,
        individual_count: InitialIndividualCountInput,
    ) -> DiscreteGenerationPopulationBuilder:
        self.initial_individual_count = individual_count
        return self

    def reproduction(
        self,
        eggs_per_female: float = 50.0,
        sex_ratio: float = 0.5,
        female_adult_mating_rate: float = 1.0,
        male_adult_mating_rate: float = 1.0,
        **kwargs: object,
    ) -> DiscreteGenerationPopulationBuilder:
        self.eggs_per_female = eggs_per_female
        self.sex_ratio = sex_ratio
        self.female_adult_mating_rate = female_adult_mating_rate
        self.male_adult_mating_rate = male_adult_mating_rate
        self._set_params(domain="reproduction", **kwargs)
        return self

    def survival(
        self,
        female_age0_survival: float = 1.0,
        male_age0_survival: float = 1.0,
        **kwargs: object,
    ) -> DiscreteGenerationPopulationBuilder:
        self.female_age0_survival = female_age0_survival
        self.male_age0_survival = male_age0_survival
        self._set_params(domain="survival", **kwargs)
        return self

    def competition(
        self,
        juvenile_growth_mode: Union[int, str] = "logistic",
        low_density_growth_rate: float = 1.0,
        carrying_capacity: Optional[float] = None,
        **kwargs: object,
    ) -> DiscreteGenerationPopulationBuilder:
        self.juvenile_growth_mode = juvenile_growth_mode
        self.low_density_growth_rate = low_density_growth_rate
        if carrying_capacity is not None:
            self.carrying_capacity = carrying_capacity
        self._set_params(domain="competition", **kwargs)
        return self

    def hooks(
        self,
        *hook_items: Union[HookFn, HookMap]
    ) -> DiscreteGenerationPopulationBuilder:
        for item in hook_items:
            if isinstance(item, dict):
                hook_map = cast(HookMap, item)
                for event, registrations in hook_map.items():
                    if event not in self._hooks:
                        self._hooks[event] = []
                    self._hooks[event].extend(registrations)
            elif callable(item):
                meta = getattr(item, 'meta', {})
                event = meta.get('event') or getattr(item, 'event', None)
                if not event:
                    raise ValueError(
                        f"Hook '{getattr(item, '__name__', str(item))}' missing event. "
                        "Please specify with @hook(event='...')"
                    )
                priority = meta.get('priority', getattr(item, 'priority', 0))
                name = getattr(item, '__name__', None)
                if event not in self._hooks:
                    self._hooks[event] = []
                self._hooks[event].append((item, name, priority))
            else:
                if item is not None:
                    raise TypeError(f"Unsupported hook type: {type(item)}")
        return self

    def build(self) -> DiscreteGenerationPopulation:
        from natal.configurator._factory import PopulationConfigBuilder
        from natal.population.discrete_generation import DiscreteGenerationPopulation

        if self.initial_individual_count is None:
            raise ValueError(
                "initial_individual_count is required. "
                "Use .initial_state() before .build()"
            )

        initial_individual_count = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
            species=self.species,
            distribution=self.initial_individual_count,
        )

        female_survival = [self.female_age0_survival, 0.0]
        male_survival = [self.male_age0_survival, 0.0]

        female_mating = np.array([0.0, self.female_adult_mating_rate], dtype=np.float64)
        male_mating = np.array([0.0, self.male_adult_mating_rate], dtype=np.float64)

        female_relative_fertility = np.array([0.0, 1.0], dtype=np.float64)

        pop_config = PopulationConfigBuilder.build(
            species=self.species,
            n_ages=2,
            new_adult_age=1,
            stochastic=self.stochastic,
            continuous_sampling=self.continuous_sampling,
            female_age_based_survival=female_survival,
            male_age_based_survival=male_survival,
            female_age_based_mating_rate=female_mating,
            male_age_based_mating_rate=male_mating,
            age_based_reproduction_rate=female_mating,
            female_age_based_fertility=female_relative_fertility,
            eggs_per_female=self.eggs_per_female,
            fixed_egg_count=self.fixed_egg_count,
            sex_ratio=self.sex_ratio,
            use_sperm_storage=False,
            sperm_displacement_rate=0.0,
            relative_competition_factor=1.0,
            juvenile_growth_mode=self.juvenile_growth_mode,
            low_density_growth_rate=self.low_density_growth_rate,
            age_1_carrying_capacity=self.carrying_capacity,
            old_juvenile_carrying_capacity=None,
            expected_num_adult_females=(
                self.carrying_capacity * self.sex_ratio
                if self.carrying_capacity is not None
                else None
            ),
            equilibrium_individual_distribution=self.equilibrium_individual_distribution,
            gamete_modifiers=self.gamete_modifiers,
            zygote_modifiers=self.zygote_modifiers,
            generation_time=1,
            initial_individual_count=initial_individual_count,
            custom_specs=self._custom_specs or None,
        )

        pop = DiscreteGenerationPopulation(
            species=self.species,
            population_config=pop_config,
            name=self.name,
            hooks=self._hooks,
        )

        for preset in self._presets:
            pop.apply_preset(preset)

        for operation in self._fitness_operations:
            method_name, args, kwargs = operation
            mode = kwargs.get('mode', 'replace')
            is_multiply = (mode == 'multiply')

            if method_name == 'viability':
                viability_map = cast(ViabilityMap, args[0])
                for genotype_selector, values in viability_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='viability',
                    )
                    for genotype in matched_genotypes:
                        new_adult_age = 1
                        target_age = new_adult_age - 1
                        viability_updates = self._iter_viability_updates(
                            values=values,
                            n_ages=2,
                            default_age=target_age,
                        )
                        for z_idx in pop.index_registry.ztype_indices_for(genotype):
                            for sex_idx, age_idx, raw_val in viability_updates:
                                val = raw_val
                                if is_multiply:
                                    current = pop.config.viability_fitness[sex_idx, age_idx, z_idx]
                                    val *= current
                                pop.config.set_viability_fitness(sex_idx, z_idx, val, age=age_idx)

            elif method_name == 'fecundity':
                fecundity_map = cast(FecundityMap, args[0])
                for genotype_selector, values in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )
                    for genotype in matched_genotypes:
                        for z_idx in pop.index_registry.ztype_indices_for(genotype):
                            if isinstance(values, dict):
                                for sex_label, value in values.items():
                                    sex_idx = resolve_sex_label(sex_label)
                                    val = float(value)
                                    if is_multiply:
                                        current = pop.config.fecundity_fitness[sex_idx, z_idx]
                                        val *= current
                                    pop.config.set_fecundity_fitness(sex_idx, z_idx, val)
                            else:
                                for sex_idx in (0, 1):
                                    val = float(values)
                                    if is_multiply:
                                        current = pop.config.fecundity_fitness[sex_idx, z_idx]
                                        val *= current
                                    pop.config.set_fecundity_fitness(sex_idx, z_idx, val)

            elif method_name == 'sexual_selection':
                preferences = cast(SexualSelectionMap, args[0])
                for f_selector, m_selector, preference in self._iter_sexual_selection_entries(preferences):
                    matched_f_genotypes = pop.species.resolve_genotype_selectors(
                        selector=f_selector,
                        context='sexual_selection (female)',
                    )
                    matched_m_genotypes = pop.species.resolve_genotype_selectors(
                        selector=m_selector,
                        context='sexual_selection (male)',
                    )
                    for f_genotype in matched_f_genotypes:
                        for m_genotype in matched_m_genotypes:
                            for f_z in pop.index_registry.ztype_indices_for(f_genotype):
                                for m_z in pop.index_registry.ztype_indices_for(m_genotype):
                                    val = float(preference)
                                    if is_multiply:
                                        current = pop.config.sexual_selection_fitness[f_z, m_z]
                                        val *= current
                                    pop.config.set_sexual_selection_fitness(f_z, m_z, val)

            elif method_name == 'zygote_viability':
                zygote_viability_map = cast(ZygoteViabilityMap, args[0])
                for genotype_selector, values in zygote_viability_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='zygote',
                    )
                    for genotype in matched_genotypes:
                        for z_idx in pop.index_registry.ztype_indices_for(genotype):
                            if isinstance(values, dict):
                                for sex_label, value in values.items():
                                    sex_idx = resolve_sex_label(sex_label)
                                    if isinstance(value, dict):
                                        raise TypeError("Zygote fitness does not support age-specific values. Use a float value instead.")
                                    val = float(value)
                                    if is_multiply:
                                        current = pop.config.zygote_viability_fitness[sex_idx, z_idx]
                                        val *= current
                                    pop.config.set_zygote_viability_fitness(sex_idx, z_idx, val)
                            else:
                                for sex_idx in (0, 1):
                                    val = float(values)
                                    if is_multiply:
                                        current = pop.config.zygote_viability_fitness[sex_idx, z_idx]
                                        val *= current
                                    pop.config.set_zygote_viability_fitness(sex_idx, z_idx, val)

        if self._observation_groups is not None:
            pop.set_observations(
                self._observation_groups,
                collapse_age=self._observation_collapse_age,
            )

        return pop
