"""Legacy population builder classes.

.. deprecated::
    Use ``Configurator`` (``configurator/_base.py``) instead.
    The default ``setup()`` path now goes through ``DiscreteConfigurator`` /
    ``AgeStructuredConfigurator``.  These Builder classes remain available
    for backward compatibility only.
"""


from collections.abc import Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

import natal.data as _population_config
from natal.data import (
    PopulationConfig,
    build_custom_array,
    build_population_config,
)
from natal.genetics import Genotype, HaploidGenome, Species
from natal.registry.index import IndexRegistry
from natal.utils.helpers import resolve_sex_label
from natal.utils.types import Sex

if TYPE_CHECKING:
    pass

from natal.configurator._builder_age import (
    AgeStructuredPopulationBuilder,  # noqa: F401, pyright: ignore[reportUnusedImport]
)
from natal.configurator._builder_discrete import (
    DiscreteGenerationPopulationBuilder,  # noqa: F401, pyright: ignore[reportUnusedImport]
)
from natal.configurator._params import (
    build_equilibrium_distribution,
    compute_expected_eggs_from_females,
    resolve_age_param,
    resolve_carrying_capacity,
    resolve_growth_mode,
)

__all__ = [
    "AgeStructuredPopulationBuilder",     # legacy — use AgeStructuredConfigurator
    "DiscreteGenerationPopulationBuilder",  # legacy — use DiscreteConfigurator
    "PopulationConfigBuilder",
]

GenotypeSelectorAtom = Union[Genotype, str]
GenotypeSelector = Union[GenotypeSelectorAtom, Tuple[GenotypeSelectorAtom, ...]]
ArrayF64 = NDArray[np.float64]
InitialAgeCountValue: TypeAlias = (
    Sequence[float | int] | Mapping[int, float | int] | ArrayF64 | int | float
)
InitialIndividualCountInput: TypeAlias = Mapping[str, Mapping[Any, InitialAgeCountValue]]
InitialSpermStorageInput: TypeAlias = Mapping[Any, Mapping[Any, InitialAgeCountValue]]
HookFn = Callable[..., object]
ModifierSpec = Tuple[int, Optional[str], HookFn]
HookRegistration = Tuple[HookFn, Optional[str], Optional[int]]
HookMap = Dict[str, List[HookRegistration]]
SexScalarMap = Dict[str, float]
AgeScalarMap = Dict[int, float]
ViabilityNestedMap = Dict[Union[str, Sex, int], Union[float, AgeScalarMap]]
ViabilityMap = Dict[GenotypeSelector, Union[float, ViabilityNestedMap]]
FecundityMap = Dict[GenotypeSelector, Union[float, SexScalarMap]]
SexualSelectionMap = Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
ZygoteViabilityMap = Dict[GenotypeSelector, Union[float, SexScalarMap]]
FitnessOperationName = Literal["viability", "fecundity", "sexual_selection", "zygote_viability"]
FitnessOperation = Tuple[FitnessOperationName, Tuple[object], Dict[str, str]]

InitializeMapFn = Callable[..., NDArray[np.float64]]
initialize_gamete_map = cast(InitializeMapFn, _population_config.initialize_gamete_map)
initialize_zygote_map = cast(InitializeMapFn, _population_config.initialize_zygote_map)


class PopulationConfigBuilder:
    """Internal builder for constructing PopulationConfig.

    Handles all low-level configuration details and array initialization. It
    encapsulating the complexity of converting builder parameters.
    """

    @staticmethod
    def build(
        species: Species,
        # Basic settings
        n_ages: int,
        new_adult_age: int,
        stochastic: bool,
        continuous_sampling: bool,
        # Survival & Mating
        female_age_based_survival: Optional[Any],
        male_age_based_survival: Optional[Any],
        female_age_based_mating_rate: Optional[ArrayF64],
        male_age_based_mating_rate: Optional[ArrayF64],
        age_based_reproduction_rate: Optional[ArrayF64],
        female_age_based_fertility: Optional[ArrayF64],
        # Reproduction
        eggs_per_female: float,
        fixed_egg_count: bool,
        sex_ratio: float,
        use_sperm_storage: bool,  # TODO
        sperm_displacement_rate: float,
        # Competition
        relative_competition_factor: float,
        juvenile_growth_mode: Union[int, str],
        low_density_growth_rate: float,
        age_1_carrying_capacity: Optional[float],
        old_juvenile_carrying_capacity: Optional[float],
        expected_num_adult_females: Optional[float],
        equilibrium_individual_distribution: Optional[ArrayF64],
        # Modifiers
        gamete_modifiers: Optional[List[ModifierSpec]],
        zygote_modifiers: Optional[List[ModifierSpec]],
        # Generation time
        generation_time: Optional[float],
        # Initial state arrays (already parsed by builder)
        initial_individual_count: Optional[NDArray[np.float64]] = None,
        initial_sperm_storage: Optional[NDArray[np.float64]] = None,
        # Custom fields
        custom_specs: Optional[dict[str, float | int | bool | NDArray[np.float64]]] = None,
    ) -> PopulationConfig:
        """Construct a complete PopulationConfig from builder parameters.

        Args:
            species (Species): Genetic architecture.
            n_ages (int): Number of age classes.
            new_adult_age (int): Minimum age for adults.
            stochastic (bool): Whether to use stochastic sampling.
            continuous_sampling (bool): Whether to use Dirichlet sampling.
            female_age_based_survival (Any): Survival rates for females.
            male_age_based_survival (Any): Survival rates for males.
            female_age_based_mating_rate (NDArray): Mating rates for females.
            male_age_based_mating_rate (NDArray): Mating rates for males.
            age_based_reproduction_rate (NDArray): Reproduction participation
                rates for females.
            female_age_based_fertility (NDArray): Fertility weights for females.
            eggs_per_female (float): Average egg production.
            fixed_egg_count (bool): Whether egg count is deterministic.
            sex_ratio (float): Offspring sex ratio.
            use_sperm_storage (bool): Whether to enable sperm storage.
            sperm_displacement_rate (float): Rate of sperm displacement.
            relative_competition_factor (float): Competition intensity.
            juvenile_growth_mode (Union[int, str]): Growth model type.
            low_density_growth_rate (float): Intrinsic growth rate.
            age_1_carrying_capacity (Optional[float]): Population carrying capacity at age=1.
            old_juvenile_carrying_capacity (Optional[float]): Alias for age_1_carrying_capacity (deprecated).
            expected_num_adult_females (Optional[float]): Target adult female count.
            equilibrium_individual_distribution (Optional[NDArray]): Expected distribution.
            gamete_modifiers (List[Tuple]): Custom gamete modifiers.
            zygote_modifiers (List[Tuple]): Custom zygote modifiers.
            generation_time (Optional[int]): Calculated generation time.
            initial_individual_count (Optional[NDArray[np.float64]]): Initial counts array.
            initial_sperm_storage (Optional[NDArray[np.float64]]): Initial sperm storage array.

        Returns:
            PopulationConfig: A fully initialized PopulationConfig instance.

        Raises:
            ValueError: If n_ages, new_adult_age or other parameters are invalid.
            TypeError: If input types are incorrect.

        """
        # ===== Validation =====
        if n_ages <= 1:
            raise ValueError(f"n_ages must be at least 2, got {n_ages}")
        if new_adult_age < 0 or new_adult_age >= n_ages:
            raise ValueError(f"new_adult_age must be in [0, {n_ages}), got {new_adult_age}")

        # ===== Extract genotypes =====
        raw_gamete_labels = cast(Optional[List[str]], getattr(species, "gamete_labels", None))
        gamete_labels = raw_gamete_labels or ["default"]
        genotypes = species.get_all_genotypes(unordered=species.unordered)
        haploid_genotypes = species.get_all_haploid_genotypes()

        n_genotypes = len(genotypes)
        n_haplogenotypes = len(haploid_genotypes)
        n_glabs = len(gamete_labels)

        gamete_tensor_mods, zygote_tensor_mods = PopulationConfigBuilder._setup_modifiers(gamete_modifiers, zygote_modifiers)

        # ===== Build genotype/gamete maps =====
        gamete_map = initialize_gamete_map(
            diploid_genotypes=genotypes,
            haploid_genotypes=haploid_genotypes,
            n_glabs=n_glabs,
            gamete_modifiers=gamete_tensor_mods
        )

        zygote_map = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=genotypes,
            n_glabs=n_glabs,
            zygote_modifiers=zygote_tensor_mods
        )

        # ===== Resolve survival rates =====
        _default_female = np.ones(n_ages - 1, dtype=np.float64)
        _default_male = np.ones(n_ages - 1, dtype=np.float64)

        female_survival = resolve_age_param(
            female_age_based_survival, n_ages, _default_female
        )
        male_survival = resolve_age_param(
            male_age_based_survival, n_ages, _default_male
        )

        age_based_survival_rates = np.array([female_survival, male_survival], dtype=np.float64)

        # TODO: 所有的 age-based 参数都应当支持类似 survival_rates 的灵活输入格式
        # ===== Mating rates =====
        if female_age_based_mating_rate is not None:
            if len(female_age_based_mating_rate) != n_ages:
                raise ValueError(
                    f"female_age_based_mating_rate length {len(female_age_based_mating_rate)} != n_ages {n_ages}"
                )
            female_mating = np.array(female_age_based_mating_rate, dtype=np.float64)
        else:
            female_mating = np.zeros(n_ages, dtype=np.float64)
            female_mating[new_adult_age:] = 1.0

        if male_age_based_mating_rate is not None:
            if len(male_age_based_mating_rate) != n_ages:
                raise ValueError(
                    f"male_age_based_mating_rate length {len(male_age_based_mating_rate)} != n_ages {n_ages}"
                )
            male_mating = np.array(male_age_based_mating_rate, dtype=np.float64)
        else:
            male_mating = np.zeros(n_ages, dtype=np.float64)
            male_mating[new_adult_age:] = 1.0

        age_based_mating_rates = np.array([female_mating, male_mating], dtype=np.float64)

        # ===== Female reproduction participation rates =====
        if age_based_reproduction_rate is not None:
            if len(age_based_reproduction_rate) != n_ages:
                raise ValueError(
                    f"age_based_reproduction_rate length {len(age_based_reproduction_rate)} != n_ages {n_ages}"
                )
            female_reproduction = np.array(age_based_reproduction_rate, dtype=np.float64)
        else:
            # Backward compatible default: reuse female mating rates.
            female_reproduction = female_mating.copy()

        # ===== Female fertility =====
        if female_age_based_fertility is not None:
            if len(female_age_based_fertility) != n_ages:
                raise ValueError(
                    f"female_age_based_fertility length {len(female_age_based_fertility)} != n_ages {n_ages}"
                )
            female_fertility = np.array(female_age_based_fertility, dtype=np.float64)
        else:
            female_fertility = np.ones(n_ages, dtype=np.float64)

        # ===== Fitness tensors (default) =====
        viability_fitness = np.ones((2, n_ages, n_genotypes), dtype=np.float64)
        fecundity_fitness = np.ones((2, n_genotypes), dtype=np.float64)
        sexual_selection_fitness = np.ones((n_genotypes, n_genotypes), dtype=np.float64)
        zygote_viability_fitness = np.ones((2, n_genotypes), dtype=np.float64)

        # ===== Competition strength =====
        age_based_relative_competition_strength = np.ones(n_ages, dtype=np.float64)
        if relative_competition_factor > 0 and n_ages > 1:
            # Keep new larvae (age 0) at baseline competition strength 1.0,
            # and scale only old larvae (age 1), matching the SLiM formula:
            # new_larvae + old_larvae * OLD_LARVA_COMPETITION_FACTOR.
            age_based_relative_competition_strength[1] = relative_competition_factor

        # ===== Parse juvenile growth mode =====
        juvenile_growth_mode_int = resolve_growth_mode(juvenile_growth_mode)

        # ===== Resolve carrying capacity K (age-1 total at equilibrium) =====
        K = resolve_carrying_capacity(
            age_1_carrying_capacity=age_1_carrying_capacity,
            old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
            initial_individual_count=initial_individual_count,
        )

        # ===== Build equilibrium distribution =====
        if equilibrium_individual_distribution is not None:
            eq_dist = equilibrium_individual_distribution
        else:
            eq_dist = build_equilibrium_distribution(
                K=K,
                sex_ratio=sex_ratio,
                age_based_survival_rates=age_based_survival_rates,
                n_ages=n_ages,
            )

        # ===== Compute expected egg production =====
        # expected_num_adult_females independently determines expected eggs;
        # otherwise fall back to the equilibrium distribution's adult females.
        if expected_num_adult_females is not None:
            external_eggs = compute_expected_eggs_from_females(
                expected_num_adult_females=expected_num_adult_females,
                eggs_per_female=eggs_per_female,
                age_based_survival_rates=age_based_survival_rates,
                age_based_reproduction_rates=female_reproduction,
                female_age_based_fertility=female_fertility,
                sex_ratio=sex_ratio,
                new_adult_age=new_adult_age,
                n_ages=n_ages,
            )
        else:
            external_eggs = None

        # ===== Create and return PopulationConfig =====
        cfg = build_population_config(
            n_genotypes=n_genotypes,
            n_gtypes=n_haplogenotypes * n_glabs,
            n_sexes=2,
            n_ages=n_ages,
            n_glabs=n_glabs,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            age_based_survival_rates=age_based_survival_rates,
            age_based_mating_rates=age_based_mating_rates,
            age_based_reproduction_rates=female_reproduction,
            female_age_based_fertility=female_fertility,
            viability_fitness=viability_fitness,
            fecundity_fitness=fecundity_fitness,
            sexual_selection_fitness=sexual_selection_fitness,
            zygote_viability_fitness=zygote_viability_fitness,
            age_based_relative_competition_strength=age_based_relative_competition_strength,
            new_adult_age=new_adult_age,
            sperm_displacement_rate=sperm_displacement_rate,
            eggs_per_female=eggs_per_female,
            fixed_egg_count=fixed_egg_count,
            carrying_capacity=K,
            sex_ratio=sex_ratio,
            low_density_growth_rate=low_density_growth_rate,
            juvenile_growth_mode=juvenile_growth_mode_int,
            age_1_carrying_capacity=age_1_carrying_capacity or old_juvenile_carrying_capacity,
            old_juvenile_carrying_capacity=None,
            equilibrium_individual_distribution=eq_dist,
            external_expected_eggs=external_eggs,
            zygotes_to_gametes_map=gamete_map,
            gametes_to_zygotes_map=zygote_map,
            generation_time=generation_time,
            initial_individual_count=initial_individual_count,
        )

        if initial_sperm_storage is not None:
            cfg = cfg._replace(initial_sperm_storage=initial_sperm_storage.copy())

        # Build structured custom array from builder .custom() specs
        if custom_specs:
            cfg = cfg._replace(custom=build_custom_array(custom_specs))

        return cfg

    # ── Delegation stubs — forward to _params module for backward compatibility
    # These were originally static methods on this class; tests and external
    # code may still call them.

    @staticmethod
    def resolve_age_param(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """Delegate to :func:`natal.configurator._params.resolve_age_param`."""
        return resolve_age_param(*args, **kwargs)

    @staticmethod
    def _resolve_growth_mode(*args: Any, **kwargs: Any) -> int:
        """Delegate to :func:`natal.configurator._params.resolve_growth_mode`."""
        return resolve_growth_mode(*args, **kwargs)

    @staticmethod
    def _resolve_carrying_capacity(*args: Any, **kwargs: Any) -> float:
        """Delegate to :func:`natal.configurator._params.resolve_carrying_capacity`."""
        return resolve_carrying_capacity(*args, **kwargs)

    @staticmethod
    def _build_equilibrium_distribution(*args: Any, **kwargs: Any) -> NDArray[np.float64]:
        """Delegate to :func:`natal.configurator._params.build_equilibrium_distribution`."""
        return build_equilibrium_distribution(*args, **kwargs)

    @staticmethod
    def compute_expected_eggs_from_females(*args: Any, **kwargs: Any) -> float:
        """Delegate to :func:`natal.configurator._params.compute_expected_eggs_from_females`."""
        return compute_expected_eggs_from_females(*args, **kwargs)

    @staticmethod
    def _setup_modifiers(
        gamete_modifiers: Optional[List[ModifierSpec]],
        zygote_modifiers: Optional[List[ModifierSpec]],
    ) -> Tuple[List[HookFn], List[HookFn]]:
        """Helper to organize and build modifier tensors."""
        gamete_modifiers_list = list(gamete_modifiers) if gamete_modifiers else []
        zygote_modifiers_list = list(zygote_modifiers) if zygote_modifiers else []

        gamete_modifiers_list.sort(key=lambda x: float(x[0]))
        zygote_modifiers_list.sort(key=lambda x: float(x[0]))

        gamete_tensor_mods = PopulationConfigBuilder._build_modifier_tensors(gamete_modifiers_list, "gamete")
        zygote_tensor_mods = PopulationConfigBuilder._build_modifier_tensors(zygote_modifiers_list, "zygote")
        return gamete_tensor_mods, zygote_tensor_mods

    @staticmethod
    def _compute_expected_eggs_from_distribution(
        equilibrium_distribution: NDArray[np.float64],
        eggs_per_female: float,
        age_based_reproduction_rates: NDArray[np.float64],
        female_age_based_fertility: NDArray[np.float64],
        new_adult_age: int,
        n_ages: int,
    ) -> float:
        """Compute total expected egg production from an equilibrium distribution.

        Args:
            equilibrium_distribution: (2, n_ages) equilibrium distribution.
            eggs_per_female: Base eggs per female.
            age_based_reproduction_rates: Female reproduction participation by age.
            female_age_based_fertility: Relative fertility by age.
            new_adult_age: First adult age class.
            n_ages: Total age classes.

        Returns:
            float: Total expected egg production.
        """
        eggs = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = float(equilibrium_distribution[0, age])
            p_reproducing = min(1.0, max(0.0, float(age_based_reproduction_rates[age])))
            eggs += n_f * p_reproducing * female_age_based_fertility[age] * eggs_per_female
        return eggs

    @staticmethod
    def _get_all_haploid_genotypes(species: Species) -> List[HaploidGenome]:
        """Extract all haploid genomes from Species-level genotype iterators.

        Args:
            species (Species): The species instance to query.

        Returns:
            List[HaploidGenome]: A list of all haploid genotypes.
        """
        return list(species.iter_haploid_genotypes())

    @staticmethod
    def _resolve_sex_index(sex_key: Union[str, Sex]) -> int:
        """Resolve a sex key into an integer index (0 or 1).

        Args:
            sex_key (Union[str, Sex]): The sex label or enum.

        Returns:
            int: 0 for female, 1 for male.

        Raises:
            TypeError: If sex_key is neither str nor Sex.
        """
        if isinstance(sex_key, Sex):
            return int(sex_key.value)
        return resolve_sex_label(sex_key)

    @staticmethod
    def _resolve_age_counts_age_structured(
        age_data: InitialAgeCountValue,
        n_ages: int,
        new_adult_age: int,
    ) -> Dict[int, float]:
        """Normalize age-based distribution data into a sparse dictionary.

        Args:
            age_data (Union[List, Dict, float]): Raw age distribution data.
            n_ages (int): Total number of age classes.
            new_adult_age (int): Minimum age for adults.

        Returns:
            Dict[int, float]: Mapping of age to individual count.

        Raises:
            ValueError: If counts are negative or ages are out of range.
            TypeError: If data type is unsupported.
        """
        if isinstance(age_data, Mapping):
            age_map = age_data
            out: Dict[int, float] = {}
            for age, count in age_map.items():
                if age < 0 or age >= n_ages:
                    raise ValueError(f"Age {age} out of range [0, {n_ages})")
                fcount = float(count)
                if fcount < 0:
                    raise ValueError(f"Count must be non-negative, got {fcount}")
                if fcount > 0:
                    out[age] = fcount
            return out

        if isinstance(age_data, (Sequence, np.ndarray)) and not isinstance(
            age_data, (str, bytes, bytearray)
        ):
            arr = np.asarray(age_data, dtype=np.float64)
            out = {}
            for age, count in enumerate(arr):
                if age >= n_ages:
                    break
                if count < 0:
                    raise ValueError(f"Count must be non-negative, got {count}")
                if count > 0:
                    out[age] = float(count)
            return out

        fcount = float(age_data)
        if fcount < 0:
            raise ValueError(f"Count must be non-negative, got {fcount}")
        if fcount <= 0:
            return {}
        return dict.fromkeys(range(new_adult_age, n_ages), fcount)

    @staticmethod
    def resolve_age_structured_initial_individual_count(
        species: Species,
        distribution: InitialIndividualCountInput,
        n_ages: int,
        new_adult_age: int,
    ) -> NDArray[np.float64]:
        """Resolve initial individual counts for age-structured models.

        Args:
            species (Species): The bound Species object.
            distribution (Dict): User-provided distribution mapping.
            n_ages (int): Total number of age classes.
            new_adult_age (int): Minimum age for adults.

        Returns:
            NDArray[np.float64]: A 3D array [sex, age, genotype].
        """
        registry = IndexRegistry()
        slabs = species.somatic_labels or ["default"]
        for slab in slabs:
            registry.register_somatic_label(slab)
        genotypes = species.get_all_genotypes(unordered=species.unordered)
        for gt in genotypes:
            registry.register_genotype(gt)
        out = np.zeros((2, n_ages, registry.n_ztypes), dtype=np.float64)
        for sex_key, genotype_dist in distribution.items():
            sex_idx = PopulationConfigBuilder._resolve_sex_index(sex_key)
            for genotype_key, age_data in genotype_dist.items():
                from natal.patterns import (
                    GenotypePatternParser,
                    ZygoteTypePattern,
                )

                if isinstance(genotype_key, tuple):
                    _key, _slab = cast("tuple[object, str]", genotype_key)
                    if isinstance(_key, Genotype):
                        pattern = ZygoteTypePattern.from_pair(_key, _slab, species)
                    elif isinstance(_key, str):
                        _gt = species.get_genotype_from_str(_key)
                        _key = str(_gt)
                        pattern = ZygoteTypePattern.parse(f"{_key}@{_slab}", species)
                    else:
                        raise TypeError(
                            f"Tuple first element must be Genotype or str, got {type(_key)}"
                        )
                elif isinstance(genotype_key, str):
                    pattern = ZygoteTypePattern.from_slab_key(genotype_key, species)
                elif isinstance(genotype_key, Genotype):
                    parser = GenotypePatternParser(species)
                    pattern = ZygoteTypePattern(
                        parser.parse(str(genotype_key)), slab=None
                    )
                else:
                    raise TypeError(
                        f"genotype_key must be Genotype, str, or tuple, got {type(genotype_key)}"
                    )

                z_idx = registry.resolve_default_ztype_index(pattern)
                age_counts = PopulationConfigBuilder._resolve_age_counts_age_structured(
                    age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
                )
                for age, count in age_counts.items():
                    out[sex_idx, age, z_idx] += float(count)
        return out

    @staticmethod
    def resolve_age_structured_initial_sperm_storage(
        species: Species,
        sperm_storage: InitialSpermStorageInput,
        n_ages: int,
        new_adult_age: int,
    ) -> NDArray[np.float64]:
        """Resolve initial sperm storage for age-structured models.

        Args:
            species (Species): The bound Species object.
            sperm_storage (Dict): User-provided sperm storage mapping.
            n_ages (int): Total number of age classes.
            new_adult_age (int): Minimum age for adults.

        Returns:
            NDArray[np.float64]: A 3D array [age, female_genotype, male_genotype].

        Raises:
            TypeError: If storage value is not a dictionary.
        """
        registry = IndexRegistry()
        slabs = species.somatic_labels or ["default"]
        for slab in slabs:
            registry.register_somatic_label(slab)
        genotypes = species.get_all_genotypes(unordered=species.unordered)
        for gt in genotypes:
            registry.register_genotype(gt)
        out = np.zeros((n_ages, registry.n_ztypes, registry.n_ztypes), dtype=np.float64)

        for female_key, male_dict in sperm_storage.items():
            from natal.patterns import GenotypePatternParser, ZygoteTypePattern

            if isinstance(female_key, str):
                female_pattern = ZygoteTypePattern.from_slab_key(female_key, species)
            elif isinstance(female_key, Genotype):
                parser = GenotypePatternParser(species)
                female_pattern = ZygoteTypePattern(
                    parser.parse(str(female_key)), slab=None
                )
            else:
                raise TypeError(
                    f"female_key must be Genotype or str, got {type(female_key)}"
                )
            f_z = registry.resolve_default_ztype_index(female_pattern)

            for male_key, age_data in male_dict.items():
                if isinstance(male_key, str):
                    male_pattern = ZygoteTypePattern.from_slab_key(male_key, species)
                elif isinstance(male_key, Genotype):
                    parser = GenotypePatternParser(species)
                    male_pattern = ZygoteTypePattern(
                        parser.parse(str(male_key)), slab=None
                    )
                else:
                    raise TypeError(
                        f"male_key must be Genotype or str, got {type(male_key)}"
                    )
                m_z = registry.resolve_default_ztype_index(male_pattern)

                age_counts = PopulationConfigBuilder._resolve_age_counts_age_structured(
                    age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
                )
                for age, count in age_counts.items():
                    out[age, f_z, m_z] += float(count)
        return out

    @staticmethod
    def _resolve_discrete_age_distribution(
        age_data: InitialAgeCountValue,
    ) -> Tuple[float, float]:
        """Normalize discrete distribution data into (age0, age1) counts.

        Args:
            age_data: Raw distribution data.

        Returns:
            Tuple[float, float]: Count for age 0 and age 1.

        Raises:
            ValueError: If negative counts or invalid lengths are provided.
        """
        if isinstance(age_data, (int, float)) and not isinstance(age_data, bool):
            value = float(age_data)
            if value < 0:
                raise ValueError(f"Count must be non-negative, got {value}")
            return 0.0, value

        if isinstance(age_data, (Sequence, np.ndarray)) and not isinstance(
            age_data, (str, bytes, bytearray)
        ):
            arr = np.asarray(age_data, dtype=np.float64)
            if arr.size == 0:
                return 0.0, 0.0
            if arr.size == 1:
                if arr[0] < 0:
                    raise ValueError(f"Count must be non-negative, got {arr[0]}")
                return 0.0, float(arr[0])
            if arr.size == 2:
                if np.any(arr < 0):
                    raise ValueError(f"Count must be non-negative, got {arr}")
                return float(arr[0]), float(arr[1])
            raise ValueError(f"Discrete initial list/array must have length <= 2, got {arr.size}")

        if isinstance(age_data, Mapping):
            age_map = age_data
            unsupported_keys = [k for k in age_map.keys() if k not in (0, 1)]
            if unsupported_keys:
                raise ValueError(
                    f"Discrete initial dict supports only age keys 0 and 1, got {unsupported_keys}"
                )
            age0 = float(age_map.get(0, 0.0))
            age1 = float(age_map.get(1, 0.0))
            if age0 < 0 or age1 < 0:
                raise ValueError("Count must be non-negative")
            return age0, age1

        raise TypeError(f"Unsupported age_data type: {type(age_data)}")

    @staticmethod
    def resolve_discrete_initial_individual_count(
        species: Species,
        distribution: InitialIndividualCountInput,
    ) -> NDArray[np.float64]:
        """Resolve initial individual counts for discrete generation models.

        Args:
            species (Species): The bound Species object.
            distribution (Dict): User-provided distribution mapping.

        Returns:
            NDArray[np.float64]: A 3D array [sex, age, genotype] with age max 2.
        """
        registry = IndexRegistry()
        slabs = species.somatic_labels or ["default"]
        for slab in slabs:
            registry.register_somatic_label(slab)
        genotypes = species.get_all_genotypes(unordered=species.unordered)
        for gt in genotypes:
            registry.register_genotype(gt)
        out = np.zeros((2, 2, registry.n_ztypes), dtype=np.float64)

        for sex_key, genotype_dist in distribution.items():
            sex_idx = PopulationConfigBuilder._resolve_sex_index(sex_key)
            for genotype_key, age_data in genotype_dist.items():
                from natal.patterns import (
                    GenotypePatternParser,
                    ZygoteTypePattern,
                )

                if isinstance(genotype_key, tuple):
                    _key, _slab = cast("tuple[object, str]", genotype_key)
                    if isinstance(_key, Genotype):
                        pattern = ZygoteTypePattern.from_pair(_key, _slab, species)
                    elif isinstance(_key, str):
                        _gt = species.get_genotype_from_str(_key)
                        _key = str(_gt)
                        pattern = ZygoteTypePattern.parse(f"{_key}@{_slab}", species)
                    else:
                        raise TypeError(
                            f"Tuple first element must be Genotype or str, got {type(_key)}"
                        )
                elif isinstance(genotype_key, str):
                    pattern = ZygoteTypePattern.from_slab_key(genotype_key, species)
                elif isinstance(genotype_key, Genotype):
                    parser = GenotypePatternParser(species)
                    pattern = ZygoteTypePattern(
                        parser.parse(str(genotype_key)), slab=None
                    )
                else:
                    raise TypeError(
                        f"genotype_key must be Genotype, str, or tuple, got {type(genotype_key)}"
                    )

                z_idx = registry.resolve_default_ztype_index(pattern)
                age0, age1 = PopulationConfigBuilder._resolve_discrete_age_distribution(age_data)
                out[sex_idx, 0, z_idx] += age0
                out[sex_idx, 1, z_idx] += age1
        return out

    @staticmethod
    def _build_modifier_tensors(modifiers: List[ModifierSpec], modifier_type: str) -> List[HookFn]:
        """Convert modifier tuples to tensor modifier format (placeholder).

        Args:
            modifiers (List): List of modifier tuples.
            modifier_type (str): Type tag (gamete or zygote).

        Returns:
            List: The modifier list.
        """
        if not modifiers:
            return []
        return [fn for _, _, fn in modifiers]

