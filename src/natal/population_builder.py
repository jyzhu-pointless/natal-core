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
        female_age_based_survival_rates: Optional[NDArray],
        male_age_based_survival_rates: Optional[NDArray],
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
        
        female_survival = PopulationConfigBuilder._resolve_array_param(
            female_age_based_survival_rates, n_ages, _default_female
        )
        male_survival = PopulationConfigBuilder._resolve_array_param(
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
            generation_time=generation_time
        )

        print("✅ Population configuration initialized")

        return cfg
    
    @staticmethod
    def _get_all_haploid_genotypes(species: Species) -> List[HaploidGenome]:
        """Extract all haploid genomes from Species-level genotype iterators."""
        return list(species.iter_haploid_genotypes())
    
    @staticmethod
    def _resolve_array_param(
        param: Optional[Union[List, NDArray]],
        expected_length: int,
        default: List[float]
    ) -> NDArray[np.float64]:
        """Resolve array parameter with fallback to default."""
        if param is not None:
            arr = np.asarray(param, dtype=np.float64)
            if len(arr) != expected_length:
                raise ValueError(f"Parameter length {len(arr)} != {expected_length}")
            return arr
        else:
            # Use default, resized to expected_length
            default_arr = np.array(default[:expected_length], dtype=np.float64)
            if len(default_arr) < expected_length:
                default_arr = np.pad(default_arr, (0, expected_length - len(default_arr)), constant_values=0)
            return default_arr
    
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
        self._recipes: List[Any] = []
    
    def add_recipe(self, recipe: Any) -> 'PopulationBuilderBase':
        """Add a gene drive recipe to apply during build.
        
        Recipes are applied in the order they are added.
        
        Args:
            recipe: A GeneDriveRecipe or similar modification system.
        
        Returns:
            Self for chaining.
        """
        self._recipes.append(recipe)
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
    - recipes() - Preset modification packages to apply during build
    - fitness() - Fitness tensors (viability, fecundity, sexual selection)
    - modifiers() - Custom gamete and zygote modifier functions
    - hooks() - Event hook registrations
    
    Notes:
        Fitness and modifiers are applied AFTER recipes during build().
        This allows recipes to set base values, which can then be overridden.
    
    Example::
    
        pop = (AgeStructuredPopulation.builder(species)
            .setup(name="Pop1", stochastic=True)
            .age_structure(n_ages=10, new_adult_age=2, generation_time=25)
            .survival(female_rates=[...], male_rates=[...])
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .competition(juvenile_growth_mode='logistic', carrying_capacity=1000)
            .initial_state(distribution={...})
            .fitness(viability=np.ones((2,10,5)))
            .add_recipe(HomingModificationDrive(...))
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
        individual_count: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]],
        sperm_storage: Optional[Dict[Union[Genotype, str], Dict[Union[Genotype, str], Union[Dict[int, float], List[float], float]]]] = None
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
        female_age_based_survival_rates: Optional[Union[List[float], NDArray[np.float64]]] = None,
        male_age_based_survival_rates: Optional[Union[List[float], NDArray[np.float64]]] = None,
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
    
    def recipes(self, *recipe_list: Any) -> 'AgeStructuredPopulationBuilder':
        """Add preset recipe packages (applied during build).
        
        Recipes are preset configurations that may include fitness tensors,
        modifiers, and other modifications. They are applied first, then
        overridden by explicit fitness(), modifiers(), and hooks() settings
        if provided.
        
        Args:
            *recipe_list: Variable number of recipe objects to apply.
        
        Returns:
            Self for chaining.
        """
        if recipe_list:
            self._recipes = list(recipe_list)
        return self
    
    def fitness(
        self,
        viability: Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]] = None,
        fecundity: Optional[Dict[GenotypeSelector, float]] = None,
        sexual_selection: Optional[Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]] = None
    ) -> 'AgeStructuredPopulationBuilder':
        """Configure fitness via population methods (applied after recipes).
        
        Fitness is set using the population's set_viability(), set_fecundity(),
        and set_sexual_selection() methods AFTER recipes are applied. This allows
        recipes to set base fitness values which can then be overridden.
        
        Args:
            viability: Dict mapping genotype -> {sex: value} or genotype -> value.
                If value is a dict with 'female'/'male' keys, applies per-sex.
                If value is a float, applies to both sexes.
                - Example: {'WT|WT': 1.0, 'Drive|WT': {'female': 0.9, 'male': 0.8}}
            
            fecundity: Dict mapping genotype -> fitness value.
                Applied to both sexes unless genotype includes sex qualifiers.
                - Example: {'WT|WT': 1.0, 'Drive|WT': 0.5}
            
            sexual_selection: Nested mapping of female_selector -> {male_selector: value}.
                - female_selector can be omitted by passing flat form {male_selector: value},
                    which applies to all female genotypes.
                - selectors support Genotype, exact genotype string, pattern string,
                    and tuple unions of these.
                - Example nested: {'WT|WT': {'WT|WT': 1.0, 'Drive|WT': 0.8}}
                - Example all-female: {'Drive|WT': 0.8}
        
        Returns:
            Self for chaining.
        """
        if viability is not None:
            self._fitness_operations.append(('viability', (viability,), {}))
        
        if fecundity is not None:
            self._fitness_operations.append(('fecundity', (fecundity,), {}))
        
        if sexual_selection is not None:
            self._fitness_operations.append(('sexual_selection', (sexual_selection,), {}))
        
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
        """Configure custom modifier functions (applied after recipes).
        
        Modifiers are registered AFTER recipes are applied, allowing recipes
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
          4. Applies all recipes in order
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
        )
        
        # 2️⃣ Create population with PopulationConfig and hooks
        pop = AgeStructuredPopulation(
            species=self.species,
            population_config=pop_config,
            name=self.name,
            initial_individual_count=self.initial_individual_count,
            initial_sperm_storage=self.initial_sperm_storage,
            hooks=self._hooks
        )
        
        # 3️⃣ Apply all recipes in order
        for recipe in self._recipes:
            pop.apply_recipe(recipe)
        
        # 4️⃣ Apply fitness settings directly to PopulationConfig (after recipes)
        for operation in self._fitness_operations:
            method_name, args, kwargs = operation
            
            if method_name == 'viability':
                viability_map = args[0]
                for genotype_selector, values in viability_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='viability',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_core.genotype_to_index[genotype]

                        if isinstance(values, dict):
                            # Per-sex values: {'female': 1.0, 'male': 0.9}
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                pop._config.set_viability_fitness(sex_idx, genotype_idx, value)
                        else:
                            # Single value for both sexes
                            pop._config.set_viability_fitness(0, genotype_idx, values)  # Female
                            pop._config.set_viability_fitness(1, genotype_idx, values)  # Male
            
            elif method_name == 'fecundity':
                fecundity_map = args[0]
                for genotype_selector, value in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_core.genotype_to_index[genotype]

                        # Fecundity applies to both sexes
                        pop._config.set_fecundity_fitness(0, genotype_idx, value)  # Female
                        pop._config.set_fecundity_fitness(1, genotype_idx, value)  # Male
            
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
                            f_idx = pop._index_core.genotype_to_index[f_genotype]
                            m_idx = pop._index_core.genotype_to_index[m_genotype]
                            pop._config.set_sexual_selection_fitness(f_idx, m_idx, preference)
        
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

    def recipes(self, *recipe_list: Any) -> "DiscreteGenerationPopulationBuilder":
        """Add preset recipe packages (applied during build).

        Args:
            *recipe_list: Variable number of recipe objects to apply.

        Returns:
            Self for chaining.
        """
        if recipe_list:
            self._recipes = list(recipe_list)
        return self

    def fitness(
        self,
        viability: Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]] = None,
        fecundity: Optional[Dict[GenotypeSelector, float]] = None,
        sexual_selection: Optional[Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]] = None,
    ) -> "DiscreteGenerationPopulationBuilder":
        """Configure fitness via population methods (applied after recipes).

        Args:
            viability: Mapping from genotype selectors to scalar or per-sex values.
            fecundity: Mapping from genotype selectors to fecundity values.
            sexual_selection: Flat or nested mating preference mapping.

        Returns:
            Self for chaining.
        """
        if viability is not None:
            self._fitness_operations.append(("viability", (viability,), {}))

        if fecundity is not None:
            self._fitness_operations.append(("fecundity", (fecundity,), {}))

        if sexual_selection is not None:
            self._fitness_operations.append(("sexual_selection", (sexual_selection,), {}))

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
        individual_count: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]],
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

        female_survival = np.array(
            [self.female_age0_survival, self.adult_survival], dtype=np.float64
        )
        male_survival = np.array(
            [self.male_age0_survival, self.adult_survival], dtype=np.float64
        )

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
        )

        pop = DiscreteGenerationPopulation(
            species=self.species,
            population_config=pop_config,
            name=self.name,
            initial_individual_count=self.initial_individual_count,
            hooks=self._hooks,
        )

        for recipe in self._recipes:
            pop.apply_recipe(recipe)

        for operation in self._fitness_operations:
            method_name, args, kwargs = operation

            if method_name == 'viability':
                viability_map = args[0]
                for genotype_selector, values in viability_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='viability',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_core.genotype_to_index[genotype]

                        if isinstance(values, dict):
                            for sex_label, value in values.items():
                                sex_idx = resolve_sex_label(sex_label)
                                pop._config.set_viability_fitness(sex_idx, genotype_idx, value)
                        else:
                            pop._config.set_viability_fitness(0, genotype_idx, values)
                            pop._config.set_viability_fitness(1, genotype_idx, values)

            elif method_name == 'fecundity':
                fecundity_map = args[0]
                for genotype_selector, value in fecundity_map.items():
                    matched_genotypes = pop.species.resolve_genotype_selectors(
                        selector=genotype_selector,
                        context='fecundity',
                    )

                    for genotype in matched_genotypes:
                        genotype_idx = pop._index_core.genotype_to_index[genotype]
                        pop._config.set_fecundity_fitness(0, genotype_idx, value)
                        pop._config.set_fecundity_fitness(1, genotype_idx, value)

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
                            f_idx = pop._index_core.genotype_to_index[f_genotype]
                            m_idx = pop._index_core.genotype_to_index[m_genotype]
                            pop._config.set_sexual_selection_fitness(f_idx, m_idx, preference)

        return pop
