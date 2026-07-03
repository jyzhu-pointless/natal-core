"""AgeStructuredPopulationBuilder extracted from _factory.py."""

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
    from natal.population.age_structured import AgeStructuredPopulation

import numpy as np
from numpy.typing import NDArray

from natal.configurator._builder_base import (
    ArrayF64,
    FecundityMap,
    FitnessOperation,
    GenotypeSelector,
    HookFn,
    HookMap,
    InitialAgeCountValue,
    InitialIndividualCountInput,
    InitialSpermStorageInput,
    ModifierSpec,
    PopulationBuilderBase,
    SexualSelectionMap,
    ViabilityMap,
    ZygoteViabilityMap,
)
from natal.data import (
    LOGISTIC,
)
from natal.genetics import Genotype, Species
from natal.utils.helpers import resolve_sex_label


class AgeStructuredPopulationBuilder(PopulationBuilderBase):
    """Builder for AgeStructuredPopulation with organized group methods.

    Note:
        Fitness and modifiers are applied AFTER presets during build().
        This allows presets to set base values, which can then be overridden.
    """

    def __init__(self, species: Species):
        super().__init__(species)
        # Store builder parameters directly
        self.name: str = "AgeStructuredPop"
        self.stochastic: bool = True
        self.continuous_sampling: bool = False
        self.fixed_egg_count: bool = False

        # Age structure
        self.n_ages: int = 8
        self.new_adult_age: int = 2
        self.generation_time: Optional[float] = None
        self.equilibrium_individual_distribution: Optional[ArrayF64] = None

        # Initial state (required)
        self.initial_individual_count: Optional[InitialIndividualCountInput] = None
        self.initial_sperm_storage: Optional[InitialSpermStorageInput] = None

        # Survival and mating
        self.female_age_based_survival: Optional[Any] = None
        self.male_age_based_survival: Optional[Any] = None
        self.female_age_based_mating_rate: Optional[ArrayF64] = None
        self.male_age_based_mating_rate: Optional[ArrayF64] = None
        self.age_based_reproduction_rate: Optional[ArrayF64] = None
        self.female_age_based_fertility: Optional[ArrayF64] = None

        # Reproduction
        self.eggs_per_female: float = 50.0
        self.sex_ratio: float = 0.5
        self.use_sperm_storage: bool = False
        self.sperm_displacement_rate: float = 0.0

        # Competition
        self.relative_competition_factor: float = 1.0
        self.juvenile_growth_mode: Union[int, str] = LOGISTIC
        self.low_density_growth_rate: float = 1.0
        self.age_1_carrying_capacity: Optional[float] = None
        self.old_juvenile_carrying_capacity: Optional[float] = None
        self.expected_num_adult_females: Optional[float] = None

        # Fitness and modifiers (delayed until build)
        self._fitness_operations: List[FitnessOperation] = []
        self.gamete_modifiers: Optional[List[ModifierSpec]] = None
        self.zygote_modifiers: Optional[List[ModifierSpec]] = None

        # Hooks
        self._hooks: HookMap = {}

    def setup(
        self,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        **kwargs: object,
    ) -> AgeStructuredPopulationBuilder:
        self.name = name
        self.stochastic = stochastic
        self.continuous_sampling = continuous_sampling
        self.fixed_egg_count = fixed_egg_count
        self._set_param("setup.stochastic", stochastic)
        self._set_param("setup.continuous_sampling", continuous_sampling)
        self._set_param("setup.fixed_egg_count", fixed_egg_count)
        self._set_params(domain="setup", **kwargs)
        return self

    def age_structure(
        self,
        n_ages: int = 8,
        new_adult_age: int = 2,
        generation_time: Optional[float] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None,
        **kwargs: object,
    ) -> AgeStructuredPopulationBuilder:
        self.n_ages = n_ages
        self.new_adult_age = new_adult_age
        self._set_param("age_structure.n_ages", n_ages)
        self._set_param("age_structure.new_adult_age", new_adult_age)
        if generation_time is not None:
            self.generation_time = generation_time
            self._set_param("age_structure.generation_time", generation_time)
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = np.array(equilibrium_distribution)
        self._set_params(domain="age_structure", **kwargs)
        return self

    @overload
    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[str, InitialAgeCountValue]],
        sperm_storage: Optional[Mapping[str, Mapping[str, InitialAgeCountValue]]] = None,
    ) -> AgeStructuredPopulationBuilder:
        ...

    @overload
    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[Genotype, InitialAgeCountValue]],
        sperm_storage: Optional[Mapping[Genotype, Mapping[Genotype, InitialAgeCountValue]]] = None,
    ) -> AgeStructuredPopulationBuilder:
        ...

    @overload
    def initial_state(
        self,
        individual_count: Mapping[str, Mapping[Genotype | str, InitialAgeCountValue]],
        sperm_storage: Optional[Mapping[Genotype | str, Mapping[Genotype | str, InitialAgeCountValue]]] = None,
    ) -> AgeStructuredPopulationBuilder:
        ...

    def initial_state(
        self,
        individual_count: InitialIndividualCountInput,
        sperm_storage: Optional[InitialSpermStorageInput] = None,
    ) -> AgeStructuredPopulationBuilder:
        self.initial_individual_count = individual_count
        if sperm_storage is not None:
            self.initial_sperm_storage = sperm_storage
        return self

    def survival(
        self,
        female_age_based_survival: Optional[Any] = None,
        male_age_based_survival: Optional[Any] = None,
        generation_time: Optional[float] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None,
        **kwargs: object,
    ) -> AgeStructuredPopulationBuilder:
        if female_age_based_survival is not None:
            self.female_age_based_survival = female_age_based_survival
        if male_age_based_survival is not None:
            self.male_age_based_survival = male_age_based_survival
        if generation_time is not None:
            self.generation_time = generation_time
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = np.array(equilibrium_distribution)
        self._set_params(domain="survival", **kwargs)
        return self

    def reproduction(
        self,
        female_age_based_mating_rate: Optional[Union[List[float], NDArray[np.float64]]] = None,
        male_age_based_mating_rate: Optional[Union[List[float], NDArray[np.float64]]] = None,
        age_based_reproduction_rate: Optional[Union[List[float], NDArray[np.float64]]] = None,
        female_age_based_fertility: Optional[Union[List[float], NDArray[np.float64]]] = None,
        eggs_per_female: float = 50.0,
        fixed_egg_count: bool = False,
        sex_ratio: float = 0.5,
        use_sperm_storage: bool = True,
        sperm_displacement_rate: float = 0.05,
        **kwargs: object,
    ) -> AgeStructuredPopulationBuilder:
        if female_age_based_mating_rate is not None:
            self.female_age_based_mating_rate = np.array(female_age_based_mating_rate)
        if male_age_based_mating_rate is not None:
            self.male_age_based_mating_rate = np.array(male_age_based_mating_rate)
        if age_based_reproduction_rate is not None:
            self.age_based_reproduction_rate = np.array(age_based_reproduction_rate)
        if female_age_based_fertility is not None:
            self.female_age_based_fertility = np.array(female_age_based_fertility)
        self.eggs_per_female = eggs_per_female
        self.fixed_egg_count = fixed_egg_count
        self.sex_ratio = sex_ratio
        self.use_sperm_storage = use_sperm_storage
        self.sperm_displacement_rate = sperm_displacement_rate
        self._set_params(domain="reproduction", **kwargs)
        return self

    def competition(
        self,
        competition_strength: float = 5.0,
        juvenile_growth_mode: Union[int, str] = "logistic",
        low_density_growth_rate: float = 6.0,
        age_1_carrying_capacity: Optional[float] = None,
        old_juvenile_carrying_capacity: Optional[float] = None,
        expected_num_adult_females: Optional[float] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None,
        **kwargs: object,
    ) -> AgeStructuredPopulationBuilder:
        self.relative_competition_factor = competition_strength
        self.juvenile_growth_mode = juvenile_growth_mode
        self.low_density_growth_rate = low_density_growth_rate
        if age_1_carrying_capacity is not None:
            self.age_1_carrying_capacity = age_1_carrying_capacity
        if old_juvenile_carrying_capacity is not None:
            self.old_juvenile_carrying_capacity = old_juvenile_carrying_capacity
        if expected_num_adult_females is not None:
            self.expected_num_adult_females = expected_num_adult_females
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = np.array(equilibrium_distribution)
        self._set_params(domain="competition", **kwargs)
        return self

    def presets(self, *preset_list: Any) -> AgeStructuredPopulationBuilder:
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
    ) -> AgeStructuredPopulationBuilder:
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
    ) -> AgeStructuredPopulationBuilder:
        if gamete_modifiers is not None:
            self.gamete_modifiers = gamete_modifiers
        if zygote_modifiers is not None:
            self.zygote_modifiers = zygote_modifiers
        return self

    def hooks(
        self,
        *hook_items: Union[HookFn, HookMap]
    ) -> AgeStructuredPopulationBuilder:
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
                raise TypeError(f"Unsupported hook type: {type(item)}")
        return self

    def build(self) -> AgeStructuredPopulation:
        from natal.configurator._factory import PopulationConfigBuilder
        from natal.population.age_structured import AgeStructuredPopulation

        if self.initial_individual_count is None:
            raise ValueError(
                "initial_individual_count is required. "
                "Use .initial_state() before .build()"
            )

        initial_individual_count = PopulationConfigBuilder.resolve_age_structured_initial_individual_count(
            species=self.species,
            distribution=self.initial_individual_count,
            n_ages=self.n_ages,
            new_adult_age=self.new_adult_age,
        )

        initial_sperm_storage = None
        if self.initial_sperm_storage is not None:
            initial_sperm_storage = PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage(
                species=self.species,
                sperm_storage=self.initial_sperm_storage,
                n_ages=self.n_ages,
                new_adult_age=self.new_adult_age,
            )

        pop_config = PopulationConfigBuilder.build(
            species=self.species,
            n_ages=self.n_ages,
            new_adult_age=self.new_adult_age,
            stochastic=self.stochastic,
            continuous_sampling=self.continuous_sampling,
            female_age_based_survival=self.female_age_based_survival,
            male_age_based_survival=self.male_age_based_survival,
            female_age_based_mating_rate=self.female_age_based_mating_rate,
            male_age_based_mating_rate=self.male_age_based_mating_rate,
            age_based_reproduction_rate=self.age_based_reproduction_rate,
            female_age_based_fertility=self.female_age_based_fertility,
            eggs_per_female=self.eggs_per_female,
            fixed_egg_count=self.fixed_egg_count,
            sex_ratio=self.sex_ratio,
            use_sperm_storage=self.use_sperm_storage,
            sperm_displacement_rate=self.sperm_displacement_rate,
            relative_competition_factor=self.relative_competition_factor,
            juvenile_growth_mode=self.juvenile_growth_mode,
            low_density_growth_rate=self.low_density_growth_rate,
            age_1_carrying_capacity=self.age_1_carrying_capacity,
            old_juvenile_carrying_capacity=None,
            expected_num_adult_females=self.expected_num_adult_females,
            equilibrium_individual_distribution=self.equilibrium_individual_distribution,
            gamete_modifiers=self.gamete_modifiers,
            zygote_modifiers=self.zygote_modifiers,
            generation_time=self.generation_time,
            initial_individual_count=initial_individual_count,
            initial_sperm_storage=initial_sperm_storage,
            custom_specs=self._custom_specs or None,
        )

        pop = AgeStructuredPopulation(
            species=self.species,
            population_config=pop_config,
            name=self.name,
            hooks=self._hooks
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
                        target_age = pop.new_adult_age - 1
                        viability_updates = self._iter_viability_updates(
                            values=values,
                            n_ages=self.n_ages,
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
                        context='zygote_viability',
                    )
                    for genotype in matched_genotypes:
                        for z_idx in pop.index_registry.ztype_indices_for(genotype):
                            if isinstance(values, dict):
                                for sex_label, value in values.items():
                                    sex_idx = resolve_sex_label(sex_label)
                                    if isinstance(value, dict):
                                        raise TypeError("Zygote viability fitness does not support age-specific values. Use a float value instead.")
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
