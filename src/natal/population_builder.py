"""Builder for constructing population instances with fluent API.

This module provides PopulationBuilder classes for streamlined, chainable population
construction. It separates configuration management from object instantiation,
preventing parameter bloat and enabling clear, readable code.

Example::

    pop = (AgeStructuredPopulation.builder(species)
        .setup(name="Pop1", stochastic=True)
        .age_structure(n_ages=10, new_adult_age=2)
        .survival(female_rates=[...], male_rates=[...])
        .build())
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple, TYPE_CHECKING, Iterable
import numpy as np
from numpy.typing import NDArray

from natal.genetic_structures import Species
from natal.genetic_entities import Genotype, HaploidGenome
from natal.type_def import Sex
from natal.helpers import resolve_sex_label
from natal.population_config import (
    PopulationConfig,
    build_population_config,
    initialize_gamete_map, 
    initialize_zygote_map,
    NO_COMPETITION, FIXED, LOGISTIC, CONCAVE, BEVERTON_HOLT, LINEAR
)

if TYPE_CHECKING:
    from natal.age_structured_population import AgeStructuredPopulation
    from natal.discrete_generation_population import DiscreteGenerationPopulation

__all__ = ["AgeStructuredPopulationBuilder", "DiscreteGenerationPopulationBuilder"]

GenotypeSelectorAtom = Union[Genotype, str]
GenotypeSelector = Union[GenotypeSelectorAtom, Tuple[GenotypeSelectorAtom, ...]]

class PopulationConfigBuilder:
    """Internal builder for constructing PopulationConfig from builder parameters.
    
    Handles all low-level configuration details and array initialization.
    This class encapsulates the complexity of converting high-level builder
    parameters into a complete PopulationConfig instance.
    """
    
    # TODO: initial 状态在哪里？
    @staticmethod
    def build(
        species: Species,
        # Basic settings
        n_ages: int,
        new_adult_age: int,
        is_stochastic: bool,
        use_dirichlet_sampling: bool,
        # Survival & Mating
        female_age_based_survival_rates: Optional[Any],
        male_age_based_survival_rates: Optional[Any],
        female_age_based_mating_rates: Optional[NDArray],
        male_age_based_mating_rates: Optional[NDArray],
        female_age_based_relative_fertility: Optional[NDArray],
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
        carrying_capacity: Optional[float],
        old_juvenile_carrying_capacity: Optional[int],
        expected_num_adult_females: Optional[int],
        equilibrium_individual_distribution: Optional[NDArray],
        # Modifiers
        gamete_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]],
        zygote_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]],
        # Generation time
        generation_time: Optional[int],
        # Initial state arrays (already parsed by builder)
        initial_individual_count: Optional[NDArray[np.float64]] = None,
        initial_sperm_storage: Optional[NDArray[np.float64]] = None,
    ) -> PopulationConfig:
        """Construct a complete PopulationConfig from builder parameters.
        
        Args:
            species: Genetic architecture.
            All builder parameters...
        
        Returns:
            Fully initialized PopulationConfig instance.
        
        Raises:
            ValueError: If configuration is invalid.
        """
        print("⏳ Building population...")

        # ===== Validation =====
        if n_ages <= 1:
            raise ValueError(f"n_ages must be at least 2, got {n_ages}")
        if new_adult_age < 0 or new_adult_age >= n_ages:
            raise ValueError(f"new_adult_age must be in [0, {n_ages}), got {new_adult_age}")
        
        # ===== Extract genotypes =====
        gamete_labels = species.gamete_labels or ["default"]
        genotypes = species.get_all_genotypes()
        haploid_genotypes = species.get_all_haploid_genotypes()
        
        n_genotypes = len(genotypes)
        n_haplogenotypes = len(haploid_genotypes)
        n_glabs = len(gamete_labels)
        
        if not isinstance(gamete_labels, (list, tuple)):
            raise TypeError("gamete_labels must be a list or tuple of strings")
        
        # ===== Setup modifiers =====
        gamete_modifiers_list = list(gamete_modifiers) if gamete_modifiers else []
        zygote_modifiers_list = list(zygote_modifiers) if zygote_modifiers else []
        
        gamete_modifiers_list.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))
        zygote_modifiers_list.sort(key=lambda x: x[0] if x[0] is not None else float('inf'))
        
        # Build modifier tensor wrappers (placeholder - would need mapping functions)
        gamete_tensor_mods = PopulationConfigBuilder._build_modifier_tensors(
            gamete_modifiers_list, "gamete"
        )
        zygote_tensor_mods = PopulationConfigBuilder._build_modifier_tensors(
            zygote_modifiers_list, "zygote"
        )
        
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
        if isinstance(juvenile_growth_mode, int):
            if juvenile_growth_mode not in [NO_COMPETITION, FIXED, LOGISTIC, CONCAVE]:
                raise ValueError(f"Invalid juvenile_growth_mode int: {juvenile_growth_mode}")
            juvenile_growth_mode_int = juvenile_growth_mode
        elif isinstance(juvenile_growth_mode, str):
            mode_map = {
                'NO_COMPETITION': NO_COMPETITION,
                'FIXED': FIXED,
                'LOGISTIC': LOGISTIC,
                'CONCAVE': CONCAVE,
                'BEVERTON_HOLT': BEVERTON_HOLT,
                'LINEAR': LINEAR
            }
            if juvenile_growth_mode.upper() not in mode_map:
                raise ValueError(f"Invalid juvenile_growth_mode str: {juvenile_growth_mode}")
            juvenile_growth_mode_int = mode_map[juvenile_growth_mode.upper()]
        else:
            raise TypeError(f"juvenile_growth_mode must be int or str, got {type(juvenile_growth_mode)}")
        
        # ===== Compute carrying capacity =====
        if old_juvenile_carrying_capacity is not None:
            resolved_carrying_capacity = float(old_juvenile_carrying_capacity)
        elif carrying_capacity is not None:
            resolved_carrying_capacity = float(carrying_capacity)
        elif expected_num_adult_females is not None:
            resolved_carrying_capacity = float(expected_num_adult_females) * expected_eggs_per_female
        else:
            resolved_carrying_capacity = 1000.0
        
        print("🔧 Initializing population configuration...")

        # ===== Create and return PopulationConfig =====
        cfg = build_population_config(
            n_genotypes=n_genotypes,
            n_haploid_genotypes=n_haplogenotypes,
            n_sexes=2,
            n_ages=n_ages,
            n_glabs=n_glabs,
            is_stochastic=is_stochastic,
            use_dirichlet_sampling=use_dirichlet_sampling,
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
            old_juvenile_carrying_capacity=old_juvenile_carrying_capacity,
            expected_num_adult_females=expected_num_adult_females,
            equilibrium_individual_distribution=equilibrium_individual_distribution,
            genotype_to_gametes_map=gamete_map,
            gametes_to_zygote_map=zygote_map,
            generation_time=generation_time,
            initial_individual_count=initial_individual_count,
        )

        if initial_sperm_storage is not None:
            cfg = cfg._replace(initial_sperm_storage=initial_sperm_storage.copy())

        print("✅ Population configuration initialized")

        return cfg
    
    @staticmethod
    def _get_all_haploid_genotypes(species: Species) -> List[HaploidGenome]:
        """Extract all haploid genomes from Species-level genotype iterators."""
        return list(species.iter_haploid_genotypes())
    
    @staticmethod
    def _resolve_survival_param(
        param: Optional[Any],
        expected_length: int,
        default: List[float]
    ) -> NDArray[np.float64]:
        """Resolve flexible survival spec into a 1D float array.

        Supports:
        - None -> default
        - numeric scalar -> fill all ages
        - sequence/ndarray -> truncate or pad with 0; trailing None sentinel supported
        - dict[int, float] -> sparse age map with unspecified ages defaulting to 1.0
        - callable(age) -> float
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
            out = np.ones(expected_length, dtype=np.float64)
            for age, value in param.items():
                if not isinstance(age, int):
                    raise TypeError("Age keys in survival dict must be int")
                if age < 0 or age >= expected_length:
                    raise ValueError(f"Age {age} out of range [0, {expected_length})")
                fval = float(value)
                if fval < 0:
                    raise ValueError("Survival rates must be non-negative")
                out[age] = fval
            return out

        if callable(param):
            vals = np.empty(expected_length, dtype=np.float64)
            for age in range(expected_length):
                try:
                    vals[age] = float(param(age))
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
                vals = []
                for value in obj_arr[:-1]:
                    if value is None:
                        raise TypeError("None only allowed as final sentinel in survival list")
                    vals.append(float(value))
                out = np.empty(expected_length, dtype=np.float64)
                prefix = min(len(vals), expected_length)
                if prefix > 0:
                    out[:prefix] = np.asarray(vals[:prefix], dtype=np.float64)
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
        if isinstance(sex_key, Sex):
            return int(sex_key.value)
        if isinstance(sex_key, str):
            return resolve_sex_label(sex_key)
        raise TypeError(f"Sex key must be str or Sex, got {type(sex_key)}")

    @staticmethod
    def _resolve_genotype_index(
        species: Species,
        genotype_key: Union[Genotype, str],
        genotype_to_index: Dict[Genotype, int],
    ) -> int:
        if isinstance(genotype_key, str):
            genotype = species.get_genotype_from_str(genotype_key)
        elif isinstance(genotype_key, Genotype):
            genotype = genotype_key
        else:
            raise TypeError(f"Genotype key must be Genotype or str, got {type(genotype_key)}")
        if genotype.species is not species:
            raise ValueError("Genotype must belong to this species")
        return int(genotype_to_index[genotype])

    @staticmethod
    def _resolve_age_counts_age_structured(
        age_data: Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float],
        n_ages: int,
        new_adult_age: int,
    ) -> Dict[int, float]:
        if isinstance(age_data, dict):
            out: Dict[int, float] = {}
            for age, count in age_data.items():
                if not isinstance(age, int):
                    raise TypeError(f"Age must be int, got {type(age)}")
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

        if isinstance(age_data, (int, float)) and not isinstance(age_data, bool):
            fcount = float(age_data)
            if fcount < 0:
                raise ValueError(f"Count must be non-negative, got {fcount}")
            if fcount <= 0:
                return {}
            return {age: fcount for age in range(new_adult_age, n_ages)}

        raise TypeError(
            f"Age data must be List/Tuple/NDArray, Dict, or numeric scalar, got {type(age_data)}"
        )

    @staticmethod
    def resolve_age_structured_initial_individual_count(
        species: Species,
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float]]],
        n_ages: int,
        new_adult_age: int,
    ) -> NDArray[np.float64]:
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
        sperm_storage: Dict[
            Union[Genotype, str],
            Dict[Union[Genotype, str], Union[Dict[int, float], List[float], Tuple[float, ...], NDArray[np.float64], int, float]],
        ],
        n_ages: int,
        new_adult_age: int,
    ) -> NDArray[np.float64]:
        genotypes = species.get_all_genotypes()
        genotype_to_index = {gt: idx for idx, gt in enumerate(genotypes)}
        out = np.zeros((n_ages, len(genotypes), len(genotypes)), dtype=np.float64)

        for female_key, male_dict in sperm_storage.items():
            female_idx = PopulationConfigBuilder._resolve_genotype_index(
                species, female_key, genotype_to_index
            )
            if not isinstance(male_dict, dict):
                raise TypeError(f"Sperm storage value must be dict, got {type(male_dict)}")
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
            unsupported_keys = [k for k in age_data.keys() if k not in (0, 1)]
            if unsupported_keys:
                raise ValueError(
                    f"Discrete initial dict supports only age keys 0 and 1, got {unsupported_keys}"
                )
            age0 = float(age_data.get(0, 0.0))
            age1 = float(age_data.get(1, 0.0))
            if age0 < 0 or age1 < 0:
                raise ValueError("Count must be non-negative")
            return age0, age1

        raise TypeError(f"Unsupported age_data type: {type(age_data)}")

    @staticmethod
    def resolve_discrete_initial_individual_count(
        species: Species,
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[float], Tuple[float, ...], NDArray[np.float64], Dict[int, float], int, float]]],
    ) -> NDArray[np.float64]:
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
    def _build_modifier_tensors(modifiers: List, modifier_type: str) -> List:
        """Convert modifier tuples to tensor modifier format (placeholder).
        
        In the actual implementation, this would transform modifier functions
        into their tensor-compatible form. For now, return as-is.
        """
        return modifiers if modifiers else []



class PopulationBuilderBase:
    """Abstract base builder with common chainable methods."""
    
    def __init__(self, species: Species):
        """Initialize builder with required species.
        
        Args:
            species: Genetic architecture for the population.
        """
        self.species = species
        self._presets: List[Any] = []
    
    def add_preset(self, preset: Any) -> 'PopulationBuilderBase':
        """Add a gene drive preset to apply during build.
        
        Presets are applied in the order they are added.
        
        Args:
            preset: A GeneDrivePreset or similar modification system.
        
        Returns:
            Self for chaining.
        """
        self._presets.append(preset)
        return self
    
    def build(self):
        """Build and return the configured Population.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class AgeStructuredPopulationBuilder(PopulationBuilderBase):
    """Builder for AgeStructuredPopulation with organized group methods.
    
    Parameters are grouped into 10 logical categories reflecting ecological
    and demographic structure:
    
    - setup() - Basic settings (name, stochasticity, gamete labels)
    - age_structure() - Age classes, generation time, equilibrium distribution
    - initial_state() - Initial population and sperm storage distribution
    - survival() - Age-based survival rates
    - reproduction() - Mating rates, fertility, eggs, sex ratio, sperm storage
    - competition() - Carrying capacity, growth mode, density-dependent parameters
    - presets() - Preset modification packages to apply during build
    - fitness() - Fitness tensors (viability, fecundity, sexual selection)
    - modifiers() - Custom gamete and zygote modifier functions
    - hooks() - Event hook registrations
    
    Notes:
        Fitness and modifiers are applied AFTER presets during build().
        This allows presets to set base values, which can then be overridden.
    
    Example::
    
        pop = (AgeStructuredPopulation.builder(species)
            .setup(name="Pop1", stochastic=True)
            .age_structure(n_ages=10, new_adult_age=2, generation_time=25)
            .survival(female_rates=[...], male_rates=[...])
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(juvenile_growth_mode='logistic', carrying_capacity=1000)
            .initial_state(distribution={...})
            .fitness(viability=np.ones((2,10,5)))
            .add_preset(HomingModificationDrive(...))
            .build())
    """
    
    def __init__(self, species: Species):
        """Initialize builder.
        
        Args:
            species: Genetic architecture for the population.
        """
        super().__init__(species)
        # Store builder parameters directly
        self.name: str = "AgeStructuredPop"
        self.is_stochastic: bool = True
        self.use_dirichlet_sampling: bool = False
        self.use_fixed_egg_count: bool = False
        
        # Age structure
        self.n_ages: int = 8
        self.new_adult_age: int = 2
        self.generation_time: Optional[int] = None
        self.equilibrium_individual_distribution: Optional[NDArray] = None
        
        # Initial state (required)
        self.initial_individual_count: Optional[Dict] = None
        self.initial_sperm_storage: Optional[Dict] = None
        
        # Survival and mating
        self.female_age_based_survival_rates: Optional[NDArray] = None
        self.male_age_based_survival_rates: Optional[NDArray] = None
        self.female_age_based_mating_rates: Optional[NDArray] = None
        self.male_age_based_mating_rates: Optional[NDArray] = None
        self.female_age_based_relative_fertility: Optional[NDArray] = None
        
        # Reproduction
        self.expected_eggs_per_female: float = 50.0
        self.sex_ratio: float = 0.5
        self.use_sperm_storage: bool = False
        self.sperm_displacement_rate: float = 0.0
        
        # Competition
        self.relative_competition_factor: float = 1.0
        self.juvenile_growth_mode: Union[int, str] = LOGISTIC
        self.low_density_growth_rate: float = 1.0
        self.old_juvenile_carrying_capacity: Optional[int] = None
        self.expected_num_adult_females: Optional[int] = None
        
        # Fitness and modifiers (delayed until build)
        self._fitness_operations: List[Tuple[str, tuple, dict]] = []
        self.gamete_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None
        self.zygote_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None
        
        # Hooks
        self._hooks: Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]] = {}
    
    def setup(
        self,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        use_dirichlet_sampling: bool = False,
        use_fixed_egg_count: bool = False
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure basic population settings.
        
        Args:
            name: Human-readable population name.
            stochastic: Whether to use stochastic sampling.
            use_dirichlet_sampling: If True, use Dirichlet; else Binomial/Multinomial sampling.
            use_fixed_egg_count: If True, egg count is fixed; if False, Poisson distributed.
        
        Returns:
            Self for chaining.
        """
        self.name = name
        self.is_stochastic = stochastic
        self.use_dirichlet_sampling = use_dirichlet_sampling
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
            n_ages: Number of age classes.
            new_adult_age: Age at which individuals become adults (and can reproduce).
            generation_time: Optional time for one generation (years, etc.).
            equilibrium_distribution: Optional equilibrium distribution (sex, age) for scaling.
        
        Returns:
            Self for chaining.
        """
        self.n_ages = n_ages
        self.new_adult_age = new_adult_age
        if generation_time is not None:
            self.generation_time = generation_time
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = equilibrium_distribution
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
        sperm_storage: Optional[
            Dict[
                Union[Genotype, str],
                Dict[Union[Genotype, str], Union[Dict[int, float], List[float], Tuple[float, ...], NDArray[np.float64], int, float]],
            ]
        ] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure initial population state and sperm storage.
        
        Args:
            individual_count: Initial population distribution (required).
                Format: {sex: {genotype: counts_by_age}}
            sperm_storage: Optional initial sperm storage state.
        
        Returns:
            Self for chaining.
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
            female_rates: Per-age female survival rates.
            male_rates: Per-age male survival rates.
            generation_time: Optional time scale for generation (overlaps with age_structure).
            equilibrium_distribution: Optional equilibrium distribution (overlaps with age_structure and competition).
        
        Returns:
            Self for chaining.
        """
        if female_age_based_survival_rates is not None:
            self.female_age_based_survival_rates = female_age_based_survival_rates
        if male_age_based_survival_rates is not None:
            self.male_age_based_survival_rates = male_age_based_survival_rates
        if generation_time is not None:
            self.generation_time = generation_time
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = equilibrium_distribution
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
            female_mating_rates: Per-age female mating rates.
            male_mating_rates: Per-age male mating rates.
            female_fertility: Optional per-age relative fertility multiplier for females.
            eggs_per_female: Baseline eggs (or offspring) per adult female.
            use_fixed_egg_count: If True, egg count is fixed; else Poisson distributed.
            sex_ratio: Proportion of offspring that are female (0-1).
            use_sperm_storage: Whether to model sperm storage mechanics.
            sperm_displacement_rate: Rate of sperm displacement upon remating (0-1).
        
        Returns:
            Self for chaining.
        """
        if female_age_based_mating_rates is not None:
            self.female_age_based_mating_rates = female_age_based_mating_rates
        if male_age_based_mating_rates is not None:
            self.male_age_based_mating_rates = male_age_based_mating_rates
        if female_age_based_relative_fertility is not None:
            self.female_age_based_relative_fertility = female_age_based_relative_fertility
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
        old_juvenile_carrying_capacity: Optional[int] = None,
        expected_num_adult_females: Optional[int] = None,
        equilibrium_distribution: Optional[Union[List[float], NDArray[np.float64]]] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure competition, carrying capacity, and density-dependent parameters.
        
        Args:
            competition_strength: Intensity of density-dependent competition.
            juvenile_growth_mode: Growth model ("logistic", "fixed", "concave", etc.).
            low_density_growth_rate: Growth rate at low density (intrinsic lambda).
            old_juvenile_carrying_capacity: Optional carrying capacity for older juveniles.
            expected_num_adult_females: Optional equilibrium number of adult females.
            equilibrium_distribution: Optional equilibrium distribution (overlaps with other categories).
        
        Returns:
            Self for chaining.
        """
        self.relative_competition_factor = competition_strength
        self.juvenile_growth_mode = juvenile_growth_mode
        self.low_density_growth_rate = low_density_growth_rate
        if old_juvenile_carrying_capacity is not None:
            self.old_juvenile_carrying_capacity = old_juvenile_carrying_capacity
        if expected_num_adult_females is not None:
            self.expected_num_adult_females = expected_num_adult_females
        if equilibrium_distribution is not None:
            self.equilibrium_individual_distribution = equilibrium_distribution
        return self
    
    def presets(self, *preset_list: Any) -> 'AgeStructuredPopulationBuilder':
        """Add preset preset packages (applied during build).
        
        Presets are preset configurations that may include fitness tensors,
        modifiers, and other modifications. They are applied first, then
        overridden by explicit fitness(), modifiers(), and hooks() settings
        if provided.
        
        Args:
            *preset_list: Variable number of preset objects to apply.
        
        Returns:
            Self for chaining.
        """
        if preset_list:
            self._presets = list(preset_list)
        return self
    
    def fitness(
        self,
        viability: Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]] = None,
        fecundity: Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]] = None,
        sexual_selection: Optional[Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]] = None,
        mode: str = "replace",
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure fitness via population methods (applied after presets).
        
        Fitness is set using the population's set_viability(), set_fecundity(),
        and set_sexual_selection() methods AFTER presets are applied. This allows
        presets to set base fitness values which can then be overridden.
        
        Args:
            viability: Dict mapping genotype -> {sex: value} or genotype -> value.
                If value is a dict with 'female'/'male' keys, applies per-sex.
                If value is a float, applies to both sexes.
                - Example: {'WT|WT': 1.0, 'Drive|WT': {'female': 0.9, 'male': 0.8}}
            
            fecundity: Dict mapping genotype -> fitness value (float or dict).
                - Float: Applies to both sexes.
                - Dict: {sex: value} applies per-sex.
                - Example: {'WT|WT': 1.0, 'Drive|WT': {'female': 0.5, 'male': 1.0}}
            
            sexual_selection: Nested mapping of female_selector -> {male_selector: value}.
                - female_selector can be omitted by passing flat form {male_selector: value},
                    which applies to all female genotypes.
                - selectors support Genotype, exact genotype string, pattern string,
                    and tuple unions of these.
                - Example nested: {'WT|WT': {'WT|WT': 1.0, 'Drive|WT': 0.8}}
                - Example all-female: {'Drive|WT': 0.8}
            
            mode: Scaling mode. 'replace' (default) overwrites existing values.
                'multiply' scales existing values by the provided factor.
        
        Returns:
            Self for chaining.
        """
        if viability is not None:
            self._fitness_operations.append(('viability', (viability,), {'mode': mode}))
        
        if fecundity is not None:
            self._fitness_operations.append(('fecundity', (fecundity,), {'mode': mode}))
        
        if sexual_selection is not None:
            self._fitness_operations.append(('sexual_selection', (sexual_selection,), {'mode': mode}))
        
        return self

    @staticmethod
    def _iter_sexual_selection_entries(
        sexual_selection: Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
    ) -> Iterable[Tuple[GenotypeSelector, GenotypeSelector, float]]:
        """Normalize sexual selection mapping into (female_selector, male_selector, value)."""
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
            entries.append(("*", male_selector, float(value)))
        return entries

    def modifiers(
        self,
        gamete_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None,
        zygote_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure custom modifier functions (applied after presets).
        
        Modifiers are registered AFTER presets are applied, allowing presets
        to establish base state which can then be modified.
        
        Args:
            gamete_modifiers: List of (hook_id, name, modifier_func) tuples for gamete phase.
            zygote_modifiers: List of (hook_id, name, modifier_func) tuples for zygote phase.
        
        Returns:
            Self for chaining.
        """
        if gamete_modifiers is not None:
            self.gamete_modifiers = gamete_modifiers
        if zygote_modifiers is not None:
            self.zygote_modifiers = zygote_modifiers
        return self
    
    def hooks(
        self,
        *hook_items: Union[Callable, Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]]]
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure event hook registrations.
        
        Args:
            *hook_items: Functions decorated with @hook or mappings of event names 
                to hook registrations.
        
        Returns:
            Self for chaining.
        """
        for item in hook_items:
            if isinstance(item, dict):
                for event, registrations in item.items():
                    if event not in self._hooks:
                        self._hooks[event] = []
                    self._hooks[event].extend(registrations)
            elif callable(item):
                meta = getattr(item, '_hook_meta', {})
                event = meta.get('event') or getattr(item, '_hook_event', None)
                if not event:
                    raise ValueError(
                        f"Hook '{getattr(item, '__name__', str(item))}' missing event. "
                        "Please specify with @hook(event='...')"
                    )
                
                priority = meta.get('priority', getattr(item, '_hook_priority', 0))
                name = getattr(item, '__name__', None)
                
                if event not in self._hooks:
                    self._hooks[event] = []
                self._hooks[event].append((item, name, priority))
            else:
                raise TypeError(f"Unsupported hook type: {type(item)}")
                
        return self
    
    def build(self) -> 'AgeStructuredPopulation':
        """Build and return the configured AgeStructuredPopulation.
        
        Execution order:
          1. Validates required configuration
          2. Creates PopulationConfig via PopulationConfigBuilder
          3. Creates AgeStructuredPopulation with PopulationConfig
          4. Applies all presets in order
          5. Applies fitness settings directly to PopulationConfig tensors
          6. Returns fully configured population
        
        Notes:
            PopulationConfig is immutable after population creation.
            Fitness must be set during build phase via this method.
        
        Returns:
            Initialized AgeStructuredPopulation instance.
        
        Raises:
            ValueError: If required config is missing (e.g., initial_individual_count).
        """
        # Import here to avoid circular imports
        from natal.age_structured_population import AgeStructuredPopulation
        from natal.genetic_entities import Genotype
        
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
            use_dirichlet_sampling=self.use_dirichlet_sampling,
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
            carrying_capacity=None,
            old_juvenile_carrying_capacity=self.old_juvenile_carrying_capacity,
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
        for preset in self._presets:
            pop.apply_preset(preset)
        
        # 4️⃣ Apply fitness settings directly to PopulationConfig (after presets)
        for operation in self._fitness_operations:
            method_name, args, kwargs = operation
            mode = kwargs.get('mode', 'replace')
            is_multiply = (mode == 'multiply')
            
            if method_name == 'viability':
                viability_map = args[0]
                for genotype_selector, values in viability_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='viability',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_registry.genotype_to_index[genotype]
                        target_age = pop.new_adult_age - 1

                        if isinstance(values, dict):
                            # Per-sex values: {'female': 1.0, 'male': 0.9}
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                val = float(value)
                                if is_multiply:
                                    current = pop._config.viability_fitness[sex_idx, target_age, genotype_idx]
                                    val *= current
                                pop._config.set_viability_fitness(sex_idx, genotype_idx, val)
                        else:
                            # Single value for both sexes
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop._config.viability_fitness[sex_idx, target_age, genotype_idx]
                                    val *= current
                                pop._config.set_viability_fitness(sex_idx, genotype_idx, val)
            
            elif method_name == 'fecundity':
                fecundity_map = args[0]
                for genotype_selector, values in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_registry.genotype_to_index[genotype]

                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                val = float(value)
                                if is_multiply:
                                    current = pop._config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop._config.set_fecundity_fitness(sex_idx, genotype_idx, val)
                        else:
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop._config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop._config.set_fecundity_fitness(sex_idx, genotype_idx, val)
            
            elif method_name == 'sexual_selection':
                preferences = args[0]
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
                            f_idx = pop._index_registry.genotype_to_index[f_genotype]
                            m_idx = pop._index_registry.genotype_to_index[m_genotype]
                            val = float(preference)
                            if is_multiply:
                                current = pop._config.sexual_selection_fitness[f_idx, m_idx]
                                val *= current
                            pop._config.set_sexual_selection_fitness(f_idx, m_idx, val)
        
        return pop


class DiscreteGenerationPopulationBuilder(PopulationBuilderBase):
    """Builder for DiscreteGenerationPopulation.
    
    For populations with discrete, non-overlapping generations.

    Notes:
        This builder fixes ``n_ages=2`` and ``new_adult_age=1``.
        In discrete kernels, juvenile competition strength is computed from
        total age-0 abundance directly.
    """

    def __init__(self, species: Species):
        super().__init__(species)

        self.name: str = "DiscreteGenerationPop"
        self.is_stochastic: bool = True
        self.use_dirichlet_sampling: bool = False
        self.use_fixed_egg_count: bool = False

        self.initial_individual_count: Optional[Dict] = None

        self.expected_eggs_per_female: float = 50.0
        self.sex_ratio: float = 0.5

        self.female_age0_survival: float = 1.0
        self.male_age0_survival: float = 1.0
        self.adult_survival: float = 0.0

        self.female_adult_mating_rate: float = 1.0
        self.male_adult_mating_rate: float = 1.0

        self.juvenile_growth_mode: Union[int, str] = LOGISTIC
        self.low_density_growth_rate: float = 1.0
        self.carrying_capacity: Optional[int] = None
        self.equilibrium_individual_distribution: Optional[NDArray] = None

        self.gamete_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None
        self.zygote_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None
        self._fitness_operations: List[Tuple[str, tuple, dict]] = []
        self._hooks: Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]] = {}

    def presets(self, *preset_list: Any) -> "DiscreteGenerationPopulationBuilder":
        """Add preset preset packages (applied during build).

        Args:
            *preset_list: Variable number of preset objects to apply.

        Returns:
            Self for chaining.
        """
        if preset_list:
            self._presets = list(preset_list)
        return self

    def fitness(
        self,
        viability: Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]] = None,
        fecundity: Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]] = None,
        sexual_selection: Optional[Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]] = None,
        mode: str = "replace",
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure fitness via population methods (applied after presets).

        Args:
            viability: Mapping from genotype selectors to scalar or per-sex values.
            fecundity: Mapping from genotype selectors to fecundity values (scalar or per-sex).
            sexual_selection: Flat or nested mating preference mapping.
            mode: 'replace' or 'multiply'.

        Returns:
            Self for chaining.
        """
        if viability is not None:
            self._fitness_operations.append(("viability", (viability,), {'mode': mode}))

        if fecundity is not None:
            self._fitness_operations.append(("fecundity", (fecundity,), {'mode': mode}))

        if sexual_selection is not None:
            self._fitness_operations.append(("sexual_selection", (sexual_selection,), {'mode': mode}))

        return self

    @staticmethod
    def _iter_sexual_selection_entries(
        sexual_selection: Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]
    ) -> Iterable[Tuple[GenotypeSelector, GenotypeSelector, float]]:
        """Normalize sexual selection mapping into (female_selector, male_selector, value)."""
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
            entries.append(("*", male_selector, float(value)))
        return entries

    def modifiers(
        self,
        gamete_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None,
        zygote_modifiers: Optional[List[Tuple[int, Optional[str], Callable]]] = None,
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure custom modifier functions.

        Args:
            gamete_modifiers: List of (hook_id, name, modifier_func) tuples for gamete phase.
            zygote_modifiers: List of (hook_id, name, modifier_func) tuples for zygote phase.

        Returns:
            Self for chaining.
        """
        if gamete_modifiers is not None:
            self.gamete_modifiers = gamete_modifiers
        if zygote_modifiers is not None:
            self.zygote_modifiers = zygote_modifiers
        return self

    def setup(
        self,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        use_dirichlet_sampling: bool = False,
        use_fixed_egg_count: bool = False
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure basic population settings.
        
        Args:
            name: Human-readable population name.
            stochastic: Whether to use stochastic sampling.
            use_dirichlet_sampling: If True, use Dirichlet; else Binomial/Multinomial sampling.
            use_fixed_egg_count: If True, egg count is fixed; if False, Poisson distributed.
        
        Returns:
            Self for chaining.
        """
        self.name = name
        self.is_stochastic = stochastic
        self.use_dirichlet_sampling = use_dirichlet_sampling
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
            individual_count: Initial abundance mapping grouped by sex and genotype.
                The innermost value can be an age-indexed sequence/map, or a scalar
                (``int``/``float``) interpreted by the discrete-generation initializer.

        Returns:
            Self for chaining.
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
            eggs_per_female: Expected offspring produced per adult female.
                Defaults to ``50.0``.
            sex_ratio: Proportion of female offspring in ``[0, 1]``.
                Defaults to ``0.5``.
            female_adult_mating_rate: Adult female mating participation rate.
                Defaults to ``1.0``.
            male_adult_mating_rate: Adult male mating participation rate.
                Defaults to ``1.0``.

        Returns:
            Self for chaining.
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
            female_age0_survival: Female survival probability from age-0 stage.
                Defaults to ``1.0``.
            male_age0_survival: Male survival probability from age-0 stage.
                Defaults to ``1.0``.
            adult_survival: Adult survival probability to the next step.
                Defaults to ``0.0`` for non-overlapping generations.

        Returns:
            Self for chaining.
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
            juvenile_growth_mode: Growth model identifier (int constant or mode string,
                such as ``"logistic"``). Defaults to ``"logistic"``.
            low_density_growth_rate: Per-step growth factor at low density.
                Defaults to ``1.0``.
            carrying_capacity: Optional carrying capacity used by density-dependent
                juvenile growth models. Defaults to ``None``.

        Returns:
            Self for chaining.
        """
        self.juvenile_growth_mode = juvenile_growth_mode
        self.low_density_growth_rate = low_density_growth_rate
        self.carrying_capacity = carrying_capacity
        return self

    def hooks(
        self,
        *hook_items: Union[Callable, Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]]]
    ) -> "DiscreteGenerationPopulationBuilder":
        """Register lifecycle hooks for simulation events.

        Args:
            *hook_items: Functions decorated with @hook or mappings of event names 
                to hook registrations.

        Returns:
            Self for chaining.
        """
        for item in hook_items:
            if isinstance(item, dict):
                for event, registrations in item.items():
                    if event not in self._hooks:
                        self._hooks[event] = []
                    self._hooks[event].extend(registrations)
            elif callable(item):
                meta = getattr(item, '_hook_meta', {})
                event = meta.get('event') or getattr(item, '_hook_event', None)
                if not event:
                    raise ValueError(
                        f"Hook '{getattr(item, '__name__', str(item))}' missing event. "
                        "Please specify with @hook(event='...')"
                    )
                
                priority = meta.get('priority', getattr(item, '_hook_priority', 0))
                name = getattr(item, '__name__', None)
                
                if event not in self._hooks:
                    self._hooks[event] = []
                self._hooks[event].append((item, name, priority))
            else:
                if item is not None:
                    raise TypeError(f"Unsupported hook type: {type(item)}")
        return self

    def build(self) -> "DiscreteGenerationPopulation":
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
            use_dirichlet_sampling=self.use_dirichlet_sampling,
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
            carrying_capacity=self.carrying_capacity,
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

        for preset in self._presets:
            pop.apply_preset(preset)

        for operation in self._fitness_operations:
            method_name, args, kwargs = operation
            mode = kwargs.get('mode', 'replace')
            is_multiply = (mode == 'multiply')

            if method_name == 'viability':
                viability_map = args[0]
                for genotype_selector, values in viability_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='viability',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_registry.genotype_to_index[genotype]
                        target_age = 0 # Discrete generation only has age 0 selection
                        
                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                val = float(value)
                                if is_multiply:
                                    current = pop._config.viability_fitness[sex_idx, target_age, genotype_idx]
                                    val *= current
                                pop._config.set_viability_fitness(sex_idx, genotype_idx, val)
                        else:
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop._config.viability_fitness[sex_idx, target_age, genotype_idx]
                                    val *= current
                                pop._config.set_viability_fitness(sex_idx, genotype_idx, val)

            elif method_name == 'fecundity':
                fecundity_map = args[0]
                for genotype_selector, values in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_registry.genotype_to_index[genotype]
                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                val = float(value)
                                if is_multiply:
                                    current = pop._config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop._config.set_fecundity_fitness(sex_idx, genotype_idx, val)
                        else:
                            for sex_idx in (0, 1):
                                val = float(values)
                                if is_multiply:
                                    current = pop._config.fecundity_fitness[sex_idx, genotype_idx]
                                    val *= current
                                pop._config.set_fecundity_fitness(sex_idx, genotype_idx, val)

            elif method_name == 'sexual_selection':
                preferences = args[0]
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
                            f_idx = pop._index_registry.genotype_to_index[f_genotype]
                            m_idx = pop._index_registry.genotype_to_index[m_genotype]
                            val = float(preference)
                            if is_multiply:
                                current = pop._config.sexual_selection_fitness[f_idx, m_idx]
                                val *= current
                            pop._config.set_sexual_selection_fitness(f_idx, m_idx, val)

        return pop
