"""Builder for constructing population instances with fluent API.

This module provides PopulationBuilder classes for streamlined, chainable
population construction. It separates configuration management from object
instantiation, preventing parameter bloat and enabling clear, readable code.
"""

import inspect
from collections.abc import Iterable, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

import natal.population_config as _population_config
from natal.genetic_entities import Genotype, HaploidGenome
from natal.genetic_structures import Species
from natal.helpers import resolve_sex_label
from natal.population_config import (
    BEVERTON_HOLT,
    CONCAVE,
    FIXED,
    LINEAR,
    LOGISTIC,
    NO_COMPETITION,
    PopulationConfig,
    build_population_config,
)
from natal.type_def import Sex

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation
    from natal.discrete_generation_population import DiscreteGenerationPopulation

__all__ = ["AgeStructuredPopulationBuilder", "DiscreteGenerationPopulationBuilder"]

GenotypeSelectorAtom = Union[Genotype, str]
GenotypeSelector = Union[GenotypeSelectorAtom, Tuple[GenotypeSelectorAtom, ...]]
ArrayF64 = NDArray[np.float64]
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
ZygoteFitnessMap = Dict[GenotypeSelector, Union[float, SexScalarMap]]
FitnessOperationName = Literal["viability", "fecundity", "sexual_selection", "zygote"]
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
        is_stochastic: bool,
        use_continuous_sampling: bool,
        # Survival & Mating
        female_age_based_survival_rates: Optional[Any],
        male_age_based_survival_rates: Optional[Any],
        female_age_based_mating_rates: Optional[ArrayF64],
        male_age_based_mating_rates: Optional[ArrayF64],
        female_age_based_relative_fertility: Optional[ArrayF64],
        # Reproduction
        expected_eggs_per_female: float,
        use_fixed_egg_count: bool,
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
        generation_time: Optional[int],
        # Initial state arrays (already parsed by builder)
        initial_individual_count: Optional[NDArray[np.float64]] = None,
        initial_sperm_storage: Optional[NDArray[np.float64]] = None,
    ) -> PopulationConfig:
        """Construct a complete PopulationConfig from builder parameters.

        Args:
            species (Species): Genetic architecture.
            n_ages (int): Number of age classes.
            new_adult_age (int): Minimum age for adults.
            is_stochastic (bool): Whether to use stochastic sampling.
            use_continuous_sampling (bool): Whether to use Dirichlet sampling.
            female_age_based_survival_rates (Any): Survival rates for females.
            male_age_based_survival_rates (Any): Survival rates for males.
            female_age_based_mating_rates (NDArray): Mating rates for females.
            male_age_based_mating_rates (NDArray): Mating rates for males.
            female_age_based_relative_fertility (NDArray): Fertility weights for females.
            expected_eggs_per_female (float): Average egg production.
            use_fixed_egg_count (bool): Whether egg count is deterministic.
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
        # print("⏳ Building population config...")

        # ===== Validation =====
        if n_ages <= 1:
            raise ValueError(f"n_ages must be at least 2, got {n_ages}")
        if new_adult_age < 0 or new_adult_age >= n_ages:
            raise ValueError(f"new_adult_age must be in [0, {n_ages}), got {new_adult_age}")

        # ===== Extract genotypes =====
        raw_gamete_labels = cast(Optional[List[str]], getattr(species, "gamete_labels", None))
        gamete_labels = raw_gamete_labels or ["default"]
        genotypes = species.get_all_genotypes()
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
        _default_female = [1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0.0]
        _default_male = [1.0, 1.0, 2/3, 1/2, 0.0, 0.0, 0.0, 0.0]

        female_survival = PopulationConfigBuilder._resolve_survival_param(
            female_age_based_survival_rates, n_ages, _default_female
        )
        male_survival = PopulationConfigBuilder._resolve_survival_param(
            male_age_based_survival_rates, n_ages, _default_male
        )

        age_based_survival_rates = np.array([female_survival, male_survival], dtype=np.float64)

        # ===== Mating rates =====
        if female_age_based_mating_rates is not None:
            if len(female_age_based_mating_rates) != n_ages:
                raise ValueError(
                    f"female_age_based_mating_rates length {len(female_age_based_mating_rates)} != n_ages {n_ages}"
                )
            female_mating = np.array(female_age_based_mating_rates, dtype=np.float64)
        else:
            female_mating = np.zeros(n_ages, dtype=np.float64)
            female_mating[new_adult_age:] = 1.0

        if male_age_based_mating_rates is not None:
            if len(male_age_based_mating_rates) != n_ages:
                raise ValueError(
                    f"male_age_based_mating_rates length {len(male_age_based_mating_rates)} != n_ages {n_ages}"
                )
            male_mating = np.array(male_age_based_mating_rates, dtype=np.float64)
        else:
            male_mating = np.zeros(n_ages, dtype=np.float64)
            male_mating[new_adult_age:] = 1.0

        age_based_mating_rates = np.array([female_mating, male_mating], dtype=np.float64)

        # ===== Female fertility =====
        if female_age_based_relative_fertility is not None:
            if len(female_age_based_relative_fertility) != n_ages:
                raise ValueError(
                    f"female_age_based_relative_fertility length {len(female_age_based_relative_fertility)} != n_ages {n_ages}"
                )
            female_fertility = np.array(female_age_based_relative_fertility, dtype=np.float64)
        else:
            female_fertility = np.ones(n_ages, dtype=np.float64)

        # ===== Fitness tensors (default) =====
        viability_fitness = np.ones((2, n_ages, n_genotypes), dtype=np.float64)
        fecundity_fitness = np.ones((2, n_genotypes), dtype=np.float64)
        sexual_selection_fitness = np.ones((n_genotypes, n_genotypes), dtype=np.float64)

        # ===== Competition strength =====
        age_based_relative_competition_strength = np.ones(n_ages, dtype=np.float64)
        if relative_competition_factor > 0:
            age_based_relative_competition_strength[0] = relative_competition_factor
            if n_ages > 1:
                age_based_relative_competition_strength[1] = relative_competition_factor

        # ===== Parse juvenile growth mode =====
        juvenile_growth_mode_int = PopulationConfigBuilder._resolve_growth_mode(juvenile_growth_mode)

        # ===== Compute carrying capacity =====
        resolved_carrying_capacity = PopulationConfigBuilder._resolve_carrying_capacity(
            carrying_capacity=None,  # Not used in _resolve_carrying_capacity
            age_1_carrying_capacity=age_1_carrying_capacity,
            old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
            expected_num_adult_females=expected_num_adult_females,
            expected_eggs_per_female=expected_eggs_per_female,
            age_based_survival_rates=age_based_survival_rates,
            age_based_mating_rates=age_based_mating_rates,
            female_age_based_relative_fertility=female_fertility,
            sex_ratio=sex_ratio,
            new_adult_age=new_adult_age,
            n_ages=n_ages,
            age_based_relative_competition_strength=age_based_relative_competition_strength,
            initial_individual_count=initial_individual_count,
        )

        # print("🔧 Initializing population configuration...")

        # ===== Create and return PopulationConfig =====
        cfg = build_population_config(
            n_genotypes=n_genotypes,
            n_haploid_genotypes=n_haplogenotypes,
            n_sexes=2,
            n_ages=n_ages,
            n_glabs=n_glabs,
            is_stochastic=is_stochastic,
            use_continuous_sampling=use_continuous_sampling,
            age_based_survival_rates=age_based_survival_rates,
            age_based_mating_rates=age_based_mating_rates,
            female_age_based_relative_fertility=female_fertility,
            viability_fitness=viability_fitness,
            fecundity_fitness=fecundity_fitness,
            sexual_selection_fitness=sexual_selection_fitness,
            age_based_relative_competition_strength=age_based_relative_competition_strength,
            new_adult_age=new_adult_age,
            sperm_displacement_rate=sperm_displacement_rate,
            expected_eggs_per_female=expected_eggs_per_female,
            use_fixed_egg_count=use_fixed_egg_count,
            carrying_capacity=resolved_carrying_capacity,
            sex_ratio=sex_ratio,
            low_density_growth_rate=low_density_growth_rate,
            juvenile_growth_mode=juvenile_growth_mode_int,
            age_1_carrying_capacity=age_1_carrying_capacity or old_juvenile_carrying_capacity,
            old_juvenile_carrying_capacity=None,  # Not used, use age_1_carrying_capacity
            expected_num_adult_females=expected_num_adult_females,
            equilibrium_individual_distribution=equilibrium_individual_distribution,
            genotype_to_gametes_map=gamete_map,
            gametes_to_zygote_map=zygote_map,
            generation_time=generation_time,
            initial_individual_count=initial_individual_count,
        )

        if initial_sperm_storage is not None:
            cfg = cfg._replace(initial_sperm_storage=initial_sperm_storage.copy())

        # print("✅ Population configuration initialized")

        return cfg

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
    def _resolve_growth_mode(mode: Union[int, str]) -> int:
        """Resolve juvenile growth mode string or int to internal constant."""
        if isinstance(mode, int):
            if mode not in [NO_COMPETITION, FIXED, LOGISTIC, CONCAVE, BEVERTON_HOLT, LINEAR]:
                raise ValueError(f"Invalid growth mode constant: {mode}")
            return mode
        mode_map = {
            'NO_COMPETITION': NO_COMPETITION, 'FIXED': FIXED,
            'LOGISTIC': LOGISTIC, 'CONCAVE': CONCAVE,
            'BEVERTON_HOLT': BEVERTON_HOLT, 'LINEAR': LINEAR
        }
        upper_mode = mode.upper()
        if upper_mode not in mode_map:
            raise ValueError(f"Unknown growth mode string: {mode}")
        return mode_map[upper_mode]

    @staticmethod
    def _resolve_carrying_capacity(
        carrying_capacity: Optional[float],
        age_1_carrying_capacity: Optional[float],
        old_juvenile_carrying_capacity: Optional[float],
        expected_num_adult_females: Optional[float],
        expected_eggs_per_female: float,
        age_based_survival_rates: Optional[NDArray[np.float64]] = None,
        age_based_mating_rates: Optional[NDArray[np.float64]] = None,
        female_age_based_relative_fertility: Optional[NDArray[np.float64]] = None,
        sex_ratio: float = 0.5,
        new_adult_age: int = 1,
        n_ages: Optional[int] = None,
        age_based_relative_competition_strength: Optional[NDArray[np.float64]] = None,
        initial_individual_count: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """Logic to resolve carrying capacity from multiple possible sources.

        Args:
            carrying_capacity: Unused (kept for compatibility).
            age_1_carrying_capacity: Population carrying capacity at age=1.
            old_juvenile_carrying_capacity: Alias for age_1_carrying_capacity (deprecated).
            expected_num_adult_females: Target adult female count (can infer K from this).
            expected_eggs_per_female: Base egg production per female.
            age_based_survival_rates: Survival rates array with shape (2, n_ages).
            age_based_mating_rates: Mating rates array with shape (2, n_ages).
            female_age_based_relative_fertility: Female fertility by age (n_ages,).
            sex_ratio: Offspring sex ratio.
            new_adult_age: Minimum age for adults.
            n_ages: Total number of age classes.
            age_based_relative_competition_strength: Competition weights by age.
            initial_individual_count: Initial population distribution (2, n_ages).

        Returns:
            float: Resolved carrying capacity.

        Raises:
            ValueError: If no valid carrying capacity source found.

        Note:
            Priority order:
            1. age_1_carrying_capacity (preferred name)
            2. old_juvenile_carrying_capacity (alias, for backward compatibility)
            3. expected_num_adult_females (compute via equilibrium if rates available)
            4. initial_individual_count (fallback, use for inference)
        """
        # Priority 1: age_1_carrying_capacity (new preferred name)
        if age_1_carrying_capacity is not None:
            return float(age_1_carrying_capacity)

        # Priority 2: old_juvenile_carrying_capacity (legacy alias)
        if old_juvenile_carrying_capacity is not None:
            return float(old_juvenile_carrying_capacity)

        # Priority 3: expected_num_adult_females with equilibrium inference
        if expected_num_adult_females is not None:
            # Try equilibrium-based approach if all necessary parameters available
            if (age_based_survival_rates is not None and
                age_based_mating_rates is not None and
                female_age_based_relative_fertility is not None and
                n_ages is not None):
                # Build equilibrium distribution from expected_num_adult_females
                k_val = PopulationConfigBuilder._compute_equilibrium_carrying_capacity_from_female_count(
                    expected_num_adult_females=expected_num_adult_females,
                    expected_eggs_per_female=expected_eggs_per_female,
                    age_based_survival_rates=age_based_survival_rates,
                    age_based_mating_rates=age_based_mating_rates,
                    female_age_based_relative_fertility=female_age_based_relative_fertility,
                    sex_ratio=sex_ratio,
                    new_adult_age=new_adult_age,
                    n_ages=n_ages,
                )
                return k_val

            # Fallback without rates: just scale by egg production
            return float(expected_num_adult_females) * expected_eggs_per_female

        # Priority 4: initial_individual_count (fallback for no explicit sources)
        if initial_individual_count is not None:
            total_females = initial_individual_count[0].sum()
            if total_females > 0.1:
                total_males = initial_individual_count[1].sum()
                # Use initial population as proxy for carrying capacity
                # Calculate sex ratio from initial state
                total_both = total_females + total_males
                if total_both > 0:
                    observed_sex_ratio = total_females / total_both
                else:
                    observed_sex_ratio = sex_ratio

                # If rates are available, try equilibrium inference
                if (age_based_survival_rates is not None and
                    age_based_mating_rates is not None and
                    female_age_based_relative_fertility is not None and
                    n_ages is not None):
                    try:
                        k_val = PopulationConfigBuilder._compute_equilibrium_carrying_capacity_from_female_count(
                            expected_num_adult_females=float(total_females),
                            expected_eggs_per_female=expected_eggs_per_female,
                            age_based_survival_rates=age_based_survival_rates,
                            age_based_mating_rates=age_based_mating_rates,
                            female_age_based_relative_fertility=female_age_based_relative_fertility,
                            sex_ratio=observed_sex_ratio,
                            new_adult_age=new_adult_age,
                            n_ages=n_ages,
                        )
                        return k_val
                    except Exception:
                        # If equilibrium inference fails, fall through to simple estimate
                        pass

                # Simple estimate: use total population or scale from females
                if total_both > 0:
                    return total_both
                else:
                    return total_females * expected_eggs_per_female

        raise ValueError("No valid carrying capacity source found.")


    @staticmethod
    def _compute_equilibrium_carrying_capacity_from_female_count(
        expected_num_adult_females: float,
        expected_eggs_per_female: float,
        age_based_survival_rates: NDArray[np.float64],
        age_based_mating_rates: NDArray[np.float64],
        female_age_based_relative_fertility: NDArray[np.float64],
        sex_ratio: float,
        new_adult_age: int,
        n_ages: int,
    ) -> float:
        """Compute carrying capacity (K at age=1) from expected adult female count.

        Uses equilibrium distribution inference similar to compute_equilibrium_metrics:
        1. Distribute expected_num_adult_females across ages using survival rates
        2. Calculate expected age-0 egg production from adult females
        3. Infer K as the age-1 total at equilibrium

        Args:
            expected_num_adult_females: Target count of adult females.
            expected_eggs_per_female: Base egg production per female.
            age_based_survival_rates: Survival rates (2, n_ages).
            age_based_mating_rates: Mating rates (2, n_ages).
            female_age_based_relative_fertility: Female fertility by age (n_ages,).
            sex_ratio: Offspring sex ratio.
            new_adult_age: Minimum adult age.
            n_ages: Total number of age classes.

        Returns:
            float: Inferred carrying capacity at age=1.

        Note:
            This method mirrors the logic in compute_equilibrium_metrics but
            works in reverse: given adult female count, infer the age-1 total.
        """
        # Build equilibrium distribution centered at age=new_adult_age
        expected_distribution = np.zeros((2, n_ages), dtype=np.float64)

        # Distribute expected_num_adult_females across adult ages.
        # Assume proportional distribution based on relative survival to each age.
        # Age progression backward from a reference age (new_adult_age).
        # Forward: N(age) = N(age-1) * survival(age-1)
        # Backward: If total adults should be expected_num_adult_females,
        # allocate at new_adult_age as the reference point.
        expected_distribution[0, new_adult_age] = expected_num_adult_females

        # Backward propagation: age < new_adult_age (juveniles)
        for age in range(new_adult_age - 1, -1, -1):
            if age > 0:
                expected_distribution[0, age] = expected_distribution[0, age + 1] / (age_based_survival_rates[0, age] + 1e-10)
            else:
                # Age 0: from age 0 to age 1 via survival_rate[0]
                expected_distribution[0, age] = expected_distribution[0, 1] / (age_based_survival_rates[0, 0] + 1e-10)

        # Forward propagation: age > new_adult_age (older adults)
        for age in range(new_adult_age + 1, n_ages):
            expected_distribution[0, age] = expected_distribution[0, age - 1] * age_based_survival_rates[0, age - 1]

        # For males, assume proportional distribution
        for age in range(n_ages):
            if age >= new_adult_age:
                if age == new_adult_age:
                    # At reference age, allocate males proportionally
                    male_count = expected_num_adult_females * (1.0 - sex_ratio) / sex_ratio
                    expected_distribution[1, age] = male_count
                else:
                    # Older males: forward propagation
                    expected_distribution[1, age] = expected_distribution[1, age - 1] * age_based_survival_rates[1, age - 1]

        # Calculate cumulative mating rates for females (holding sperm)
        p_mated = np.zeros(n_ages, dtype=np.float64)
        p_unmated = 1.0
        for age in range(new_adult_age, n_ages):
            m_rate = age_based_mating_rates[0, age]
            p_unmated *= (1.0 - m_rate)
            p_mated[age] = 1.0 - p_unmated

        # Calculate produced age-0 eggs
        produced_age_0 = 0.0
        for age in range(new_adult_age, n_ages):
            n_f = expected_distribution[0, age]
            produced_age_0 += n_f * p_mated[age] * female_age_based_relative_fertility[age] * expected_eggs_per_female

        # Calculate sex-weighted age-0 survival rate
        s_0_avg = sex_ratio * age_based_survival_rates[0, 0] + (1.0 - sex_ratio) * age_based_survival_rates[1, 0]

        # Infer carrying capacity: at equilibrium, age-1 total = produced_age_0 * s_0_avg * expected_survival_rate
        # For simplicity, assume expected_survival_rate ≈ 1.0 at equilibrium
        # => K ≈ produced_age_0 * s_0_avg
        if produced_age_0 > 1e-10:
            carrying_capacity = produced_age_0 * s_0_avg
        else:
            carrying_capacity = expected_eggs_per_female * expected_num_adult_females

        return max(1.0, carrying_capacity)

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
    def _resolve_survival_param(
        param: Optional[Any],
        expected_length: int,
        default: List[float]
    ) -> NDArray[np.float64]:
        """Resolve flexible survival spec into a 1D float array.

        Note:
            Supports various input types:
            - None: uses default.
            - numeric scalar: fills all ages with this value.
            - sequence/ndarray: truncated or padded with 0.
            - dict[int, float]: sparse age map, unspecified ages default to 1.0.
            - callable(age): returns float for each age.

        Args:
            param (Optional[Any]): The flexible survival parameter to resolve.
            expected_length (int): Required length of the output array.
            default (List[float]): Default values to fallback to.

        Returns:
            NDArray[np.float64]: A 1D array of resolved survival rates.

        Raises:
            ValueError: If rates are negative or out of range.
            TypeError: If input type is unsupported.
        """
        if param is None:
            out = np.array(default[:expected_length], dtype=np.float64)
            if out.size < expected_length:
                out = np.pad(out, (0, expected_length - out.size), constant_values=0.0)
            return out

        if isinstance(param, (int, float)) and not isinstance(param, bool):
            val = float(param)
            if val < 0:
                raise ValueError("Survival rates must be non-negative")
            return np.full(expected_length, val, dtype=np.float64)

        if isinstance(param, dict):
            param_map = cast(Dict[int, float], param)
            out = np.ones(expected_length, dtype=np.float64)
            for age, value in param_map.items():
                if age < 0 or age >= expected_length:
                    raise ValueError(f"Age {age} out of range [0, {expected_length})")
                fval = float(value)
                if fval < 0:
                    raise ValueError("Survival rates must be non-negative")
                out[age] = fval
            return out

        if callable(param):
            sig = inspect.signature(param)
            required_positional = 0
            accepts_var_positional = False
            for p in sig.parameters.values():
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                    if p.default is inspect.Signature.empty:
                        required_positional += 1
                elif p.kind == inspect.Parameter.VAR_POSITIONAL:
                    accepts_var_positional = True
            if required_positional > 1 or (required_positional == 0 and not accepts_var_positional):
                raise TypeError("Survival callable must accept one int age argument")

            vals = np.empty(expected_length, dtype=np.float64)
            for age in range(expected_length):
                try:
                    value = param(age)
                    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, bool):
                        raise TypeError(
                            f"Survival callable must return a float-compatible number, got {type(value)}"
                        )
                    numeric_value = cast(int | float | np.integer[Any] | np.floating[Any], value)
                    vals[age] = float(numeric_value)
                except Exception as exc:
                    raise ValueError(f"Error calling survival rate function at age {age}: {exc}") from exc
            if np.any(vals < 0):
                raise ValueError("Survival rates must be non-negative")
            return vals

        if isinstance(param, (list, tuple, np.ndarray)):
            obj_arr = np.array(param, dtype=object)
            if obj_arr.size == 0:
                return np.zeros(expected_length, dtype=np.float64)

            if obj_arr[-1] is None:
                non_none = None
                for value in obj_arr[::-1]:
                    if value is not None:
                        non_none = float(value)
                        break
                if non_none is None:
                    out = np.array(default[:expected_length], dtype=np.float64)
                    if out.size < expected_length:
                        out = np.pad(out, (0, expected_length - out.size), constant_values=0.0)
                    return out
                prefix_vals: List[float] = []
                for value in obj_arr[:-1]:
                    if value is None:
                        raise TypeError("None only allowed as final sentinel in survival list")
                    prefix_vals.append(float(value))
                out = np.empty(expected_length, dtype=np.float64)
                prefix = min(len(prefix_vals), expected_length)
                if prefix > 0:
                    out[:prefix] = np.asarray(prefix_vals[:prefix], dtype=np.float64)
                if prefix < expected_length:
                    out[prefix:] = float(non_none)
                if np.any(out < 0):
                    raise ValueError("Survival rates must be non-negative")
                return out

            arr = np.asarray(obj_arr, dtype=np.float64)
            out = np.zeros(expected_length, dtype=np.float64)
            prefix = min(arr.size, expected_length)
            if prefix > 0:
                out[:prefix] = arr[:prefix]
            if np.any(out < 0):
                raise ValueError("Survival rates must be non-negative")
            return out

        raise TypeError(
            "survival rates must be None, sequence, dict, callable or numeric constant"
        )

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
    def _resolve_genotype_index(
        species: Species,
        genotype_key: Union[Genotype, str],
        genotype_to_index: Dict[Genotype, int],
    ) -> int:
        """Resolve a genotype key into its registered integer index.

        Args:
            species (Species): The species to resolve against.
            genotype_key (Union[Genotype, str]): The genotype instance or string label.
            genotype_to_index (Dict[Genotype, int]): Index mapping.

        Returns:
            int: The index of the genotype.

        Raises:
            TypeError: If genotype_key is invalid type.
            ValueError: If genotype does not belong to the species.
        """
        if isinstance(genotype_key, str):
            genotype = species.get_genotype_from_str(genotype_key)
        else:
            genotype = genotype_key
        if genotype.species is not species:
            raise ValueError("Genotype must belong to this species")
        return int(genotype_to_index[genotype])

    @staticmethod
    def _resolve_age_counts_age_structured(
        age_data: Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float],
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
        if isinstance(age_data, dict):
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

        if isinstance(age_data, (list, tuple, np.ndarray)):
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
        distribution: Mapping[str, Mapping[Union[Genotype, str], Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float]]],
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
        genotypes = species.get_all_genotypes()
        genotype_to_index = {gt: idx for idx, gt in enumerate(genotypes)}
        out = np.zeros((2, n_ages, len(genotypes)), dtype=np.float64)

        for sex_key, genotype_dist in distribution.items():
            sex_idx = PopulationConfigBuilder._resolve_sex_index(sex_key)
            for genotype_key, age_data in genotype_dist.items():
                genotype_idx = PopulationConfigBuilder._resolve_genotype_index(
                    species, genotype_key, genotype_to_index
                )
                age_counts = PopulationConfigBuilder._resolve_age_counts_age_structured(
                    age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
                )
                for age, count in age_counts.items():
                    out[sex_idx, age, genotype_idx] += float(count)
        return out

    @staticmethod
    def resolve_age_structured_initial_sperm_storage(
        species: Species,
        sperm_storage: Mapping[
            Union[Genotype, str],
            Mapping[Union[Genotype, str], Union[Dict[int, float], List[float], Tuple[float, ...], NDArray[np.float64], int, float]],
        ],
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
        genotypes = species.get_all_genotypes()
        genotype_to_index = {gt: idx for idx, gt in enumerate(genotypes)}
        out = np.zeros((n_ages, len(genotypes), len(genotypes)), dtype=np.float64)

        for female_key, male_dict in sperm_storage.items():
            female_idx = PopulationConfigBuilder._resolve_genotype_index(
                species, female_key, genotype_to_index
            )
            for male_key, age_data in male_dict.items():
                male_idx = PopulationConfigBuilder._resolve_genotype_index(
                    species, male_key, genotype_to_index
                )
                age_counts = PopulationConfigBuilder._resolve_age_counts_age_structured(
                    age_data=age_data, n_ages=n_ages, new_adult_age=new_adult_age
                )
                for age, count in age_counts.items():
                    out[age, female_idx, male_idx] += float(count)
        return out

    @staticmethod
    def _resolve_discrete_age_distribution(
        age_data: Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float],
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

        if isinstance(age_data, (list, tuple, np.ndarray)):
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

        if isinstance(age_data, dict):
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
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float]]],
    ) -> NDArray[np.float64]:
        """Resolve initial individual counts for discrete generation models.

        Args:
            species (Species): The bound Species object.
            distribution (Dict): User-provided distribution mapping.

        Returns:
            NDArray[np.float64]: A 3D array [sex, age, genotype] with age max 2.
        """
        genotypes = species.get_all_genotypes()
        genotype_to_index = {gt: idx for idx, gt in enumerate(genotypes)}
        out = np.zeros((2, 2, len(genotypes)), dtype=np.float64)

        for sex_key, genotype_dist in distribution.items():
            sex_idx = PopulationConfigBuilder._resolve_sex_index(sex_key)
            for genotype_key, age_data in genotype_dist.items():
                genotype_idx = PopulationConfigBuilder._resolve_genotype_index(
                    species, genotype_key, genotype_to_index
                )
                age0, age1 = PopulationConfigBuilder._resolve_discrete_age_distribution(age_data)
                out[sex_idx, 0, genotype_idx] += age0
                out[sex_idx, 1, genotype_idx] += age1
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



class PopulationBuilderBase:
    """Abstract base builder with common chainable methods.

    Attributes:
        species (Species): Genetic architecture for the population.
    """

    def __init__(self, species: Species):
        """Initialize builder with required species.

        Args:
            species (Species): Genetic architecture for the population.
        """
        self.species = species
        self._presets: List[Any] = []

    @staticmethod
    def _resolve_viability_age(age_key: object, n_ages: int) -> int:
        """Resolve and validate a viability age key.

        Args:
            age_key (object): Candidate age key.
            n_ages (int): Number of available age classes.

        Returns:
            int: Validated age index.

        Raises:
            TypeError: If age key is not an integer.
            ValueError: If age key is out of range.
        """
        if not isinstance(age_key, int) or isinstance(age_key, bool):
            raise TypeError(f"viability age key must be int, got {type(age_key)}")
        age = int(age_key)
        if age < 0 or age >= n_ages:
            raise ValueError(f"viability age {age} out of range [0, {n_ages})")
        return age

    @staticmethod
    def _iter_viability_updates(
        values: Union[float, ViabilityNestedMap],
        n_ages: int,
        default_age: int,
    ) -> List[Tuple[int, int, float]]:
        """Expand viability value specs into (sex_idx, age_idx, value) triples.

        Supported forms:
            - float: applies to both sexes at default_age.
            - {"female": 0.9, "male": 0.8}: per-sex at default_age.
            - {0: 0.95, 1: 0.85}: both sexes, age-specific.
            - {"female": {0: 0.95}, "male": {1: 0.9}}: sex+age specific.

        Args:
            values (Union[float, ViabilityNestedMap]): Viability value specification.
            n_ages (int): Number of age classes.
            default_age (int): Fallback age when age is not provided.

        Returns:
            List[Tuple[int, int, float]]: Expanded updates.

        Raises:
            TypeError: If input structure is unsupported.
            ValueError: If map is empty or age keys are invalid.
        """
        if not isinstance(values, dict):
            scalar = float(values)
            return [(0, default_age, scalar), (1, default_age, scalar)]

        if not values:
            raise ValueError("viability mapping cannot be empty")

        updates: List[Tuple[int, int, float]] = []
        for key, key_value in values.items():
            if isinstance(key, int) and not isinstance(key, bool):
                if isinstance(key_value, dict):
                    raise TypeError("age-based viability values must be numeric")
                age_idx = PopulationBuilderBase._resolve_viability_age(key, n_ages)
                val = float(key_value)
                updates.append((0, age_idx, val))
                updates.append((1, age_idx, val))
                continue

            if isinstance(key, (str, Sex)):
                sex_idx = int(key.value) if isinstance(key, Sex) else resolve_sex_label(key)
                if isinstance(key_value, dict):
                    for age_key, age_value in key_value.items():
                        age_idx = PopulationBuilderBase._resolve_viability_age(age_key, n_ages)
                        updates.append((sex_idx, age_idx, float(age_value)))
                else:
                    updates.append((sex_idx, default_age, float(key_value)))
                continue

            raise TypeError(
                "viability map keys must be sex labels (str/Sex) or age indices (int)"
            )

        return updates

    def add_preset(self, preset: Any) -> 'PopulationBuilderBase':
        """Add a gene drive preset to apply during build.

        Presets are applied in the order they are added.

        Args:
            preset (Any): A GeneDrivePreset or similar modification system.

        Returns:
            PopulationBuilderBase: Self for chaining.
        """
        self._presets.append(preset)
        return self

    def build(self) -> Any:
        """Build and return the configured Population.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError


class AgeStructuredPopulationBuilder(PopulationBuilderBase):
    """Builder for AgeStructuredPopulation with organized group methods.

    Note:
        Fitness and modifiers are applied AFTER presets during build().
        This allows presets to set base values, which can then be overridden.
    """

    def __init__(self, species: Species):
        """Initialize builder.

        Args:
            species (Species): Genetic architecture for the population.
        """
        super().__init__(species)
        # Store builder parameters directly
        self.name: str = "AgeStructuredPop"
        self.is_stochastic: bool = True
        self.use_continuous_sampling: bool = False
        self.use_fixed_egg_count: bool = False

        # Age structure
        self.n_ages: int = 8
        self.new_adult_age: int = 2
        self.generation_time: Optional[int] = None
        self.equilibrium_individual_distribution: Optional[ArrayF64] = None

        # Initial state (required)
        self.initial_individual_count: Optional[
            Mapping[
                str,
                Mapping[
                    Union[Genotype, str],
                    Union[List[float], Tuple[float, ...], ArrayF64, Dict[int, float], int, float],
                ],
            ]
        ] = None
        self.initial_sperm_storage: Optional[
            Mapping[
                Union[Genotype, str],
                Mapping[Union[Genotype, str], Union[Dict[int, float], List[float], Tuple[float, ...], ArrayF64, int, float]],
            ]
        ] = None

        # Survival and mating
        self.female_age_based_survival_rates: Optional[Any] = None
        self.male_age_based_survival_rates: Optional[Any] = None
        self.female_age_based_mating_rates: Optional[ArrayF64] = None
        self.male_age_based_mating_rates: Optional[ArrayF64] = None
        self.female_age_based_relative_fertility: Optional[ArrayF64] = None

        # Reproduction
        self.expected_eggs_per_female: float = 50.0
        self.sex_ratio: float = 0.5
        self.use_sperm_storage: bool = False
        self.sperm_displacement_rate: float = 0.0

        # Competition
        self.relative_competition_factor: float = 1.0
        self.juvenile_growth_mode: Union[int, str] = LOGISTIC
        self.low_density_growth_rate: float = 1.0
        self.age_1_carrying_capacity: Optional[int] = None
        self.old_juvenile_carrying_capacity: Optional[int] = None
        self.expected_num_adult_females: Optional[int] = None

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
        use_continuous_sampling: bool = False,
        use_fixed_egg_count: bool = False
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure basic population settings.

        Args:
            name (str): Human-readable population name.
            stochastic (bool): Whether to use stochastic sampling.
            use_continuous_sampling (bool): If True, use Dirichlet; else standard sampling.
            use_fixed_egg_count (bool): If True, egg count is fixed; if False, Poisson.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        self.name = name
        self.is_stochastic = stochastic
        self.use_continuous_sampling = use_continuous_sampling
        self.use_fixed_egg_count = use_fixed_egg_count
        return self

    def age_structure(
        self,
        n_ages: int = 8,
        new_adult_age: int = 2,
        generation_time: Optional[int] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure age structure and generation time.

        Args:
            n_ages (int): Number of age classes.
            new_adult_age (int): Age at which individuals become adults.
            generation_time (Optional[int]): Optional time for one generation.
            equilibrium_distribution (Optional[Union[List, NDArray]]): Scaling distribution.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        self.n_ages = n_ages
        self.new_adult_age = new_adult_age
        if generation_time is not None:
            self.generation_time = generation_time
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = np.array(equilibrium_distribution)
        return self

    def initial_state(
        self,
        individual_count: Mapping[
            str,
            Mapping[
                Union[Genotype, str],
                Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float],
            ],
        ],
        sperm_storage: Optional[
            Mapping[
                Union[Genotype, str],
                Mapping[Union[Genotype, str], Union[Dict[int, float], List[float], Tuple[float, ...], NDArray[np.float64], int, float]],
            ]
        ] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure initial population state and sperm storage.

        Args:
            individual_count (Dict): Initial population distribution (required).
                Format: {sex: {genotype: counts_by_age}}
            sperm_storage (Optional[Dict]): Optional initial sperm storage state.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        self.initial_individual_count = individual_count
        if sperm_storage is not None:
            self.initial_sperm_storage = sperm_storage
        return self

    def survival(
        self,
        female_age_based_survival_rates: Optional[Any] = None,
        male_age_based_survival_rates: Optional[Any] = None,
        generation_time: Optional[int] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure survival rates and related parameters.

        Args:
            female_age_based_survival_rates (Optional[Any]): Per-age female survival rates.
            male_age_based_survival_rates (Optional[Any]): Per-age male survival rates.
            generation_time (Optional[int]): Optional time scale for generation.
            equilibrium_distribution (Optional[Union[List, NDArray]]): Scaling distribution.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        if female_age_based_survival_rates is not None:
            self.female_age_based_survival_rates = female_age_based_survival_rates
        if male_age_based_survival_rates is not None:
            self.male_age_based_survival_rates = male_age_based_survival_rates
        if generation_time is not None:
            self.generation_time = generation_time
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = np.array(equilibrium_distribution)
        return self

    def reproduction(
        self,
        female_age_based_mating_rates: Optional[Union[List[float], NDArray[np.float64]]] = None,
        male_age_based_mating_rates: Optional[Union[List[float], NDArray[np.float64]]] = None,
        female_age_based_relative_fertility: Optional[Union[List[float], NDArray[np.float64]]] = None,
        eggs_per_female: float = 50.0,
        use_fixed_egg_count: bool = False,
        sex_ratio: float = 0.5,
        use_sperm_storage: bool = True,
        sperm_displacement_rate: float = 0.05
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure reproduction parameters including mating, fertility, and sperm storage.

        Args:
            female_age_based_mating_rates (Optional[Union[List, NDArray]]): Female mating rates.
            male_age_based_mating_rates (Optional[Union[List, NDArray]]): Male mating rates.
            female_age_based_relative_fertility (Optional[Union[List, NDArray]]): Fertility weights.
            eggs_per_female (float): Baseline eggs per adult female.
            use_fixed_egg_count (bool): If True, egg count is fixed; else Poisson.
            sex_ratio (float): Proportion of offspring that are female (0-1).
            use_sperm_storage (bool): Whether to model sperm storage.
            sperm_displacement_rate (float): Rate of sperm displacement (0-1).

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        if female_age_based_mating_rates is not None:
            self.female_age_based_mating_rates = np.array(female_age_based_mating_rates)
        if male_age_based_mating_rates is not None:
            self.male_age_based_mating_rates = np.array(male_age_based_mating_rates)
        if female_age_based_relative_fertility is not None:
            self.female_age_based_relative_fertility = np.array(female_age_based_relative_fertility)
        self.expected_eggs_per_female = eggs_per_female
        self.use_fixed_egg_count = use_fixed_egg_count
        self.sex_ratio = sex_ratio
        self.use_sperm_storage = use_sperm_storage
        self.sperm_displacement_rate = sperm_displacement_rate
        return self

    def competition(
        self,
        competition_strength: float = 5.0,
        juvenile_growth_mode: Union[int, str] = "logistic",
        low_density_growth_rate: float = 6.0,
        age_1_carrying_capacity: Optional[int] = None,
        old_juvenile_carrying_capacity: Optional[int] = None,
        expected_num_adult_females: Optional[int] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure competition, carrying capacity, and density-dependent parameters.

        Args:
            competition_strength (float): Intensity of competition factor.
            juvenile_growth_mode (Union[int, str]): Growth model ("logistic", etc.).
            low_density_growth_rate (float): Growth rate at low density.
            age_1_carrying_capacity (Optional[int]): Population capacity at age=1.
            old_juvenile_carrying_capacity (Optional[int]): Alias for age_1_carrying_capacity (deprecated).
            expected_num_adult_females (Optional[int]): Equilibrium number of adult females.
            equilibrium_distribution (Optional[Union[List, NDArray]]): Scaling distribution.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        self.relative_competition_factor = competition_strength
        self.juvenile_growth_mode = juvenile_growth_mode
        self.low_density_growth_rate = low_density_growth_rate
        if age_1_carrying_capacity is not None:
            self.age_1_carrying_capacity = age_1_carrying_capacity
        elif old_juvenile_carrying_capacity is not None:
            self.age_1_carrying_capacity = old_juvenile_carrying_capacity
        if expected_num_adult_females is not None:
            self.expected_num_adult_females = expected_num_adult_females
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = np.array(equilibrium_distribution)
        return self

    def presets(self, *preset_list: Any) -> 'AgeStructuredPopulationBuilder':
        """Add preset preset packages (applied during build).

        Presets are preset configurations that may include fitness tensors,
        modifiers, and other modifications. They are applied first, then
        overridden by explicit fitness(), modifiers(), and hooks() settings
        if provided.

        Args:
            *preset_list (Any): Variable number of preset objects to apply.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        if preset_list:
            self._presets = list(preset_list)
        return self

    def fitness(
        self,
        viability: Optional[ViabilityMap] = None,
        fecundity: Optional[FecundityMap] = None,
        sexual_selection: Optional[SexualSelectionMap] = None,
        zygote: Optional[ViabilityMap] = None,
        mode: str = "replace",
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure fitness via population methods (applied after presets).

        Fitness is set using the population's set_viability(), set_fecundity(),
        set_sexual_selection(), and set_zygote() methods AFTER presets are applied.
        This allows presets to set base fitness values which can then be overridden.

        Args:
            viability (Optional[Dict]): Mapping genotype -> {sex: value} or genotype -> value.
                If value is a dict with 'female'/'male' keys, applies per-sex.
            fecundity (Optional[Dict]): Mapping genotype -> fitness value (float or dict).
                - Float: Applies to both sexes.
                - Dict: {sex: value} applies per-sex.
            sexual_selection (Optional[Dict]): Nested mapping of female_selector -> {male_selector: value}.
            zygote (Optional[Dict]): Mapping genotype -> zygote fitness value (float or dict).
                Zygote fitness represents the probability that a zygote survives to become
                an individual, applied during reproduction stage before survival and competition.
                - female_selector can be omitted by passing flat form {male_selector: value},
                    which applies to all female genotypes.
            zygote (Optional[Dict]): Mapping genotype -> fitness value (float or dict).
                Applied during reproduction stage to newly formed offspring.
            mode (str): Scaling mode. 'replace' (default) overwrites existing values.
                'multiply' scales existing values by the provided factor.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        if viability is not None:
            self._fitness_operations.append(('viability', (viability,), {'mode': mode}))

        if fecundity is not None:
            self._fitness_operations.append(('fecundity', (fecundity,), {'mode': mode}))

        if sexual_selection is not None:
            self._fitness_operations.append(('sexual_selection', (sexual_selection,), {'mode': mode}))

        if zygote is not None:
            self._fitness_operations.append(('zygote', (zygote,), {'mode': mode}))

        return self

    @staticmethod
    def _iter_sexual_selection_entries(
        sexual_selection: Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
    ) -> Iterable[Tuple[GenotypeSelector, GenotypeSelector, float]]:
        """Normalize sexual selection mapping into (female_selector, male_selector, value).

        Args:
            sexual_selection (Dict): The sexual selection mapping to iterate over.

        Yields:
            Tuple[GenotypeSelector, GenotypeSelector, float]: Resolved selection entries.
        """
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

        # Flat form: {male_selector: value} => apply to all female genotypes.
        for male_selector, value in sexual_selection.items():
            assert isinstance(value, float), "In flat sexual_selection form, values must be floats"
            entries.append(("*", male_selector, value))
        return entries

    def modifiers(
        self,
        gamete_modifiers: Optional[List[ModifierSpec]] = None,
        zygote_modifiers: Optional[List[ModifierSpec]] = None,
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure custom modifier functions (applied after presets).

        Modifiers are registered AFTER presets are applied, allowing presets
        to establish base state which can then be modified.

        Args:
            gamete_modifiers (Optional[List]): (hook_id, name, modifier_func) for gametes.
            zygote_modifiers (Optional[List]): (hook_id, name, modifier_func) for zygotes.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
        if gamete_modifiers is not None:
            self.gamete_modifiers = gamete_modifiers
        if zygote_modifiers is not None:
            self.zygote_modifiers = zygote_modifiers
        return self

    def hooks(
        self,
        *hook_items: Union[HookFn, HookMap]
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure event hook registrations.

        Args:
            *hook_items (Union[Callable, Dict]): Functions decorated with @hook or mappings
                to hook registrations.

        Returns:
            AgeStructuredPopulationBuilder: Self for chaining.
        """
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

    def build(self) -> 'AgeStructuredPopulation':
        """Build and return the configured AgeStructuredPopulation.

        Note:
            PopulationConfig is immutable after population creation.
            Fitness must be set during build phase via this method.
            Required configuration like initial_individual_count must be set.

        Returns:
            AgeStructuredPopulation: Initialized AgeStructuredPopulation instance.

        Raises:
            ValueError: If required config like initial_individual_count is missing.
        """
        # Import here to avoid circular imports
        from natal.age_structured_population import AgeStructuredPopulation

        # Validate required config
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

        # 1️⃣ Build PopulationConfig via PopulationConfigBuilder
        pop_config = PopulationConfigBuilder.build(
            species=self.species,
            n_ages=self.n_ages,
            new_adult_age=self.new_adult_age,
            is_stochastic=self.is_stochastic,
            use_continuous_sampling=self.use_continuous_sampling,
            female_age_based_survival_rates=self.female_age_based_survival_rates,
            male_age_based_survival_rates=self.male_age_based_survival_rates,
            female_age_based_mating_rates=self.female_age_based_mating_rates,
            male_age_based_mating_rates=self.male_age_based_mating_rates,
            female_age_based_relative_fertility=self.female_age_based_relative_fertility,
            expected_eggs_per_female=self.expected_eggs_per_female,
            use_fixed_egg_count=self.use_fixed_egg_count,
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
        )

        # 2️⃣ Create population with PopulationConfig and hooks
        pop = AgeStructuredPopulation(
            species=self.species,
            population_config=pop_config,
            name=self.name,
            hooks=self._hooks
        )

        # 3️⃣ Apply all presets in order
        pop_any = cast(Any, pop)
        for preset in self._presets:
            pop_any.apply_preset(preset)

        # 4️⃣ Apply fitness settings directly to PopulationConfig (after presets)
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
                        genotype_idx = pop.index_registry.genotype_to_index[genotype]
                        target_age = pop.new_adult_age - 1
                        viability_updates = self._iter_viability_updates(
                            values=values,
                            n_ages=self.n_ages,
                            default_age=target_age,
                        )
                        for sex_idx, age_idx, raw_val in viability_updates:
                            val = raw_val
                            if is_multiply:
                                current = pop.config.viability_fitness[sex_idx, age_idx, genotype_idx]
                                val *= current
                            pop.config.set_viability_fitness(sex_idx, genotype_idx, val, age=age_idx)

            elif method_name == 'fecundity':
                fecundity_map = cast(FecundityMap, args[0])
                for genotype_selector, values in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop.index_registry.genotype_to_index[genotype]

                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                val = float(value)
                                if is_multiply:
                                    current = pop.config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_fecundity_fitness(sex_idx, genotype_idx, val)
                        else:
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop.config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_fecundity_fitness(sex_idx, genotype_idx, val)

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
                            f_idx = pop.index_registry.genotype_to_index[f_genotype]
                            m_idx = pop.index_registry.genotype_to_index[m_genotype]
                            val = float(preference)
                            if is_multiply:
                                current = pop.config.sexual_selection_fitness[f_idx, m_idx]
                                val *= current
                            pop.config.set_sexual_selection_fitness(f_idx, m_idx, val)

            elif method_name == 'zygote':
                zygote_map = cast(ZygoteFitnessMap, args[0])
                for genotype_selector, values in zygote_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='zygote',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop.index_registry.genotype_to_index[genotype]

                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                # For zygote fitness, we don't support age-specific values
                                # value should be a float, not AgeScalarMap
                                if isinstance(value, dict):
                                    raise TypeError("Zygote fitness does not support age-specific values. Use a float value instead.")
                                val = float(value)
                                if is_multiply:
                                    current = pop.config.zygote_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_zygote_fitness(sex_idx, genotype_idx, val)
                        else:
                            # values is a float, not AgeScalarMap for zygote fitness
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop.config.zygote_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_zygote_fitness(sex_idx, genotype_idx, val)

        return pop


class DiscreteGenerationPopulationBuilder(PopulationBuilderBase):
    """Builder for DiscreteGenerationPopulation.

    For populations with discrete, non-overlapping generations.

    Note:
        This builder fixes ``n_ages=2`` and ``new_adult_age=1``.
        In discrete kernels, juvenile competition strength is computed from
        total age-0 abundance directly.
    """

    def __init__(self, species: Species):
        super().__init__(species)

        self.name: str = "DiscreteGenerationPop"
        self.is_stochastic: bool = True
        self.use_continuous_sampling: bool = False
        self.use_fixed_egg_count: bool = False

        self.initial_individual_count: Optional[
            Dict[
                str,
                Dict[
                    Union[Genotype, str],
                    Union[List[float], Tuple[float, ...], ArrayF64, Dict[int, float], int, float],
                ],
            ]
        ] = None

        self.expected_eggs_per_female: float = 50.0
        self.sex_ratio: float = 0.5

        self.female_age0_survival: float = 1.0
        self.male_age0_survival: float = 1.0
        self.adult_survival: float = 0.0

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

    def presets(self, *preset_list: Any) -> "DiscreteGenerationPopulationBuilder":
        """Add preset preset packages (applied during build).

        Args:
            *preset_list (Any): Variable number of preset objects to apply.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        if preset_list:
            self._presets = list(preset_list)
        return self

    def fitness(
        self,
        viability: Optional[ViabilityMap] = None,
        fecundity: Optional[FecundityMap] = None,
        sexual_selection: Optional[SexualSelectionMap] = None,
        zygote: Optional[ViabilityMap] = None,
        mode: str = "replace",
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure fitness via population methods (applied after presets).

        Args:
            viability (Optional[Dict]): Genotype selectors to scalar or per-sex values.
            fecundity (Optional[Dict]): Genotype selectors to fecundity values.
            sexual_selection (Optional[Dict]): Flat or nested mating preference mapping.
            zygote (Optional[Dict]): Genotype selectors to zygote fitness values.
                Zygote fitness represents the probability that a zygote survives to become
                an individual, applied during reproduction stage before survival and competition.
            mode (str): 'replace' or 'multiply'.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        if viability is not None:
            self._fitness_operations.append(("viability", (viability,), {'mode': mode}))

        if fecundity is not None:
            self._fitness_operations.append(("fecundity", (fecundity,), {'mode': mode}))

        if sexual_selection is not None:
            self._fitness_operations.append(("sexual_selection", (sexual_selection,), {'mode': mode}))

        if zygote is not None:
            self._fitness_operations.append(("zygote", (zygote,), {'mode': mode}))

        return self

    @staticmethod
    def _iter_sexual_selection_entries(
        sexual_selection: Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
    ) -> Iterable[Tuple[GenotypeSelector, GenotypeSelector, float]]:
        """Normalize sexual selection mapping into (female_selector, male_selector, value).

        Args:
            sexual_selection (Dict): The sexual selection mapping to iterate over.

        Yields:
            Tuple[GenotypeSelector, GenotypeSelector, float]: Resolved selection entries.
        """
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
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure custom modifier functions.

        Args:
            gamete_modifiers (Optional[List]): (hook_id, name, modifier_func) for gametes.
            zygote_modifiers (Optional[List]): (hook_id, name, modifier_func) for zygotes.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        if gamete_modifiers is not None:
            self.gamete_modifiers = gamete_modifiers
        if zygote_modifiers is not None:
            self.zygote_modifiers = zygote_modifiers
        return self

    def setup(
        self,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        use_continuous_sampling: bool = False,
        use_fixed_egg_count: bool = False
    ) -> 'DiscreteGenerationPopulationBuilder':
        """Configure basic population settings.

        Args:
            name (str): Human-readable population name.
            stochastic (bool): Whether to use stochastic sampling.
            use_continuous_sampling (bool): If True, use Dirichlet; else standard sampling.
            use_fixed_egg_count (bool): If True, egg count is fixed; if False, Poisson.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        self.name = name
        self.is_stochastic = stochastic
        self.use_continuous_sampling = use_continuous_sampling
        self.use_fixed_egg_count = use_fixed_egg_count
        return self

    def initial_state(
        self,
        individual_count: Dict[
            str,
            Dict[
                Union[Genotype, str],
                Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float],
            ],
        ],
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure the initial population state.

        Args:
            individual_count (Dict): Initial abundance mapping grouped by sex and genotype.
                Value can be an age-indexed sequence/map or a scalar.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        self.initial_individual_count = individual_count
        return self

    def reproduction(
        self,
        eggs_per_female: float = 50.0,
        sex_ratio: float = 0.5,
        female_adult_mating_rate: float = 1.0,
        male_adult_mating_rate: float = 1.0,
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure reproduction and mating parameters.

        Args:
            eggs_per_female (float): Expected offspring produced per adult female.
            sex_ratio (float): Proportion of female offspring in [0, 1].
            female_adult_mating_rate (float): Adult female mating rate.
            male_adult_mating_rate (float): Adult male mating rate.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        self.expected_eggs_per_female = eggs_per_female
        self.sex_ratio = sex_ratio
        self.female_adult_mating_rate = female_adult_mating_rate
        self.male_adult_mating_rate = male_adult_mating_rate
        return self

    def survival(
        self,
        female_age0_survival: float = 1.0,
        male_age0_survival: float = 1.0,
        adult_survival: float = 0.0,
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure survival probabilities for juvenile and adult stages.

        Args:
            female_age0_survival (float): Female survival probability from age-0 stage.
            male_age0_survival (float): Male survival probability from age-0 stage.
            adult_survival (float): Adult survival probability to the next step.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        self.female_age0_survival = female_age0_survival
        self.male_age0_survival = male_age0_survival
        self.adult_survival = adult_survival
        return self

    def competition(
        self,
        juvenile_growth_mode: Union[int, str] = "logistic",
        low_density_growth_rate: float = 1.0,
        carrying_capacity: Optional[int] = None,
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure juvenile growth mode and density-dependence parameters.

        Args:
            juvenile_growth_mode (Union[int, str]): Growth model identifier.
            low_density_growth_rate (float): Per-step growth factor at low density.
            carrying_capacity (Optional[int]): Optional carrying capacity.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
        self.juvenile_growth_mode = juvenile_growth_mode
        self.low_density_growth_rate = low_density_growth_rate
        self.carrying_capacity = carrying_capacity
        return self

    def hooks(
        self,
        *hook_items: Union[HookFn, HookMap]
    ) -> "DiscreteGenerationPopulationBuilder":
        """Register lifecycle hooks for simulation events.

        Args:
            *hook_items (Union[Callable, Dict]): Functions decorated with @hook or mappings
                to hook registrations.

        Returns:
            DiscreteGenerationPopulationBuilder: Self for chaining.
        """
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

    def build(self) -> "DiscreteGenerationPopulation":
        """Build and return the configured DiscreteGenerationPopulation.

        Returns:
            DiscreteGenerationPopulation: A fully configured population instance.

        Raises:
            ValueError: If initial_individual_count is not set.
        """
        from natal.discrete_generation_population import DiscreteGenerationPopulation

        if self.initial_individual_count is None:
            raise ValueError(
                "initial_individual_count is required. "
                "Use .initial_state() before .build()"
            )

        initial_individual_count = PopulationConfigBuilder.resolve_discrete_initial_individual_count(
            species=self.species,
            distribution=self.initial_individual_count,
        )

        female_survival = [self.female_age0_survival, self.adult_survival]
        male_survival = [self.male_age0_survival, self.adult_survival]

        female_mating = np.array([0.0, self.female_adult_mating_rate], dtype=np.float64)
        male_mating = np.array([0.0, self.male_adult_mating_rate], dtype=np.float64)

        female_relative_fertility = np.array([0.0, 1.0], dtype=np.float64)

        pop_config = PopulationConfigBuilder.build(
            species=self.species,
            n_ages=2,
            new_adult_age=1,
            is_stochastic=self.is_stochastic,
            use_continuous_sampling=self.use_continuous_sampling,
            female_age_based_survival_rates=female_survival,
            male_age_based_survival_rates=male_survival,
            female_age_based_mating_rates=female_mating,
            male_age_based_mating_rates=male_mating,
            female_age_based_relative_fertility=female_relative_fertility,
            expected_eggs_per_female=self.expected_eggs_per_female,
            use_fixed_egg_count=self.use_fixed_egg_count,
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
        )

        pop = DiscreteGenerationPopulation(
            species=self.species,
            population_config=pop_config,
            name=self.name,
            hooks=self._hooks,
        )

        pop_any = cast(Any, pop)
        for preset in self._presets:
            pop_any.apply_preset(preset)

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
                        genotype_idx = pop.index_registry.genotype_to_index[genotype]
                        new_adult_age = 1
                        target_age = new_adult_age - 1
                        viability_updates = self._iter_viability_updates(
                            values=values,
                            n_ages=2,
                            default_age=target_age,
                        )
                        for sex_idx, age_idx, raw_val in viability_updates:
                            val = raw_val
                            if is_multiply:
                                current = pop.config.viability_fitness[sex_idx, age_idx, genotype_idx]
                                val *= current
                            pop.config.set_viability_fitness(sex_idx, genotype_idx, val, age=age_idx)

            elif method_name == 'fecundity':
                fecundity_map = cast(FecundityMap, args[0])
                for genotype_selector, values in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop.index_registry.genotype_to_index[genotype]
                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                val = float(value)
                                if is_multiply:
                                    current = pop.config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_fecundity_fitness(sex_idx, genotype_idx, val)
                        else:
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop.config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_fecundity_fitness(sex_idx, genotype_idx, val)

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
                            f_idx = pop.index_registry.genotype_to_index[f_genotype]
                            m_idx = pop.index_registry.genotype_to_index[m_genotype]
                            val = float(preference)
                            if is_multiply:
                                current = pop.config.sexual_selection_fitness[f_idx, m_idx]
                                val *= current
                            pop.config.set_sexual_selection_fitness(f_idx, m_idx, val)

            elif method_name == 'zygote':
                zygote_map = cast(ZygoteFitnessMap, args[0])
                for genotype_selector, values in zygote_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='zygote',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop.index_registry.genotype_to_index[genotype]

                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                # For zygote fitness, we don't support age-specific values
                                # value should be a float, not AgeScalarMap
                                if isinstance(value, dict):
                                    raise TypeError("Zygote fitness does not support age-specific values. Use a float value instead.")
                                val = float(value)
                                if is_multiply:
                                    current = pop.config.zygote_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_zygote_fitness(sex_idx, genotype_idx, val)
                        else:
                            # values is a float, not AgeScalarMap for zygote fitness
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop.config.zygote_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop.config.set_zygote_fitness(sex_idx, genotype_idx, val)

        return pop
