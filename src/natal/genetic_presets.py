""" Genetic presets for mutations, gene drives, and allele conversions.

This module provides a framework for defining reusable genetic modifications including:
- Gene drives (e.g., CRISPR/Cas9 homing drives)
- General mutations (point mutations, insertions, deletions)
- Allele conversion rules and fitness effects
- Complex genetic constructions

Each preset can modify population genetics through three mechanisms:
  - gamete_modifier: modifies gamete frequencies during gametogenesis
  - zygote_modifier: modifies embryo genotypes after fertilization
  - fitness_patch: declarative specification of viability/fecundity/sexual selection effects

Presets are applied to populations using the preferred API:
  population.apply_preset(preset)  # Modern API
  
Legacy API (deprecated):
  preset.apply(population)  # For backwards compatibility

The module also provides a generic allele conversion rule system:
  - GameteAlleleConversionRule: defines allele conversions with probabilities
  - GameteConversionRuleSet: manages multiple conversion rules
  - ZygoteAlleleConversionRule: defines conversions in embryos
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Callable, Union, TYPE_CHECKING, List, Set, Literal
import numpy as np
from natal.modifiers import GameteModifier, ZygoteModifier
from natal.genetic_entities import Gene, Genotype, HaploidGenotype
from natal.genetic_structures import Species
from natal.population_config import extract_gamete_frequencies
from natal.gamete_allele_conversion import GameteAlleleConversionRule, GameteConversionRuleSet
from natal.zygote_allele_conversion import ZygoteAlleleConversionRule, ZygoteConversionRuleSet
from natal.type_def import Sex, Age
from natal.helpers import resolve_sex_label

if TYPE_CHECKING:
    from natal.base_population import BasePopulation

__all__ = [
    "GeneticPreset",      # Abstract base class for custom presets
    "HomingDrive",        # Built-in gene drive preset
    "apply_preset_to_population",  # Core application function
]

# Temporary type alias
_AlleleSpecifier = Union[Gene, str]
_SexSpecifier = Union[Sex, str]
_SexSpecificRates = Union[float, Tuple[float, float], Dict[_SexSpecifier, float]]

# Defines how a specific allele scales fitness
# e.g., if "Dr" allele has viability_scaling = 0.8, then:
# "WT|WT" -> 1.0 viability
# "Dr::WT" -> 0.8 viability
# "Dr|Dr" -> 0.64 viability (multiplicative)
_ViabilityScalingConfig = Union[
    float,             # both sex, at the largest juvenile age
    Dict[Age, float],  # both sex, age-specific
    Dict[              # sex-specific
        _SexSpecifier, 
        Union[float, Dict[Age, float]]
    ]
]
_FecundityScalingConfig = Union[
    float,                        # both sex
    Dict[_SexSpecifier, float]    # sex-specific
]
_SexualSelectionScalingConfig = Union[
    float,                        # applies to all female genotypes
    Tuple[float, float]           # (male selected by default, male selected by allele carriers)
]

PresetFitnessPatch = Dict[str, Any]

def _normalize_sex_key(sex_key: _SexSpecifier) -> int:
    """Normalize sex key to integer index used by PopulationConfig.

    Accepted values:
    - Sex enum members (Sex.FEMALE, Sex.MALE)
    - string aliases: female/f, male/m (case-insensitive)
    """
    return resolve_sex_label(sex_key)

def _count_allele_copies(genotype: Genotype, target_gene: Gene) -> int:
    """Count copies of a target allele in a diploid genotype.

    This assumes gene names are unique within a species and therefore map to a
    single locus. Copy counting is done by checking maternal/paternal alleles at
    that locus only (0/1/2), instead of scanning all genes in the genotype.
    """
    mat_gene, pat_gene = genotype.get_alleles_at_locus(target_gene.locus)
    return int(mat_gene is target_gene) + int(pat_gene is target_gene)

def _count_combined_allele_copies(genotype: Genotype, target_genes: List[Gene]) -> int:
    """Count total copies of a list of alleles in a genotype."""
    total = 0
    # Optimization: Usually these alleles are at the same locus.
    # We could optimize, but summing individual counts is safe and correct.
    for gene in target_genes:
        total += _count_allele_copies(genotype, gene)
    # Cap at 2 for diploid systems if they are alleles of the same locus,
    # but logic holds generally (e.g. 2 means homozygous-equivalent cost).
    return total

def _calculate_allele_effect(scale: Union[float, Tuple[float, float]], copies: int, mode: str = "multiplicative") -> float:
    """Calculate fitness factor based on allele copies and scaling mode."""
    if mode == "custom":
        if not isinstance(scale, (tuple, list)) or len(scale) != 2:
            raise ValueError("For 'custom' fitness mode, scaling value must be a tuple (heterozygous_fitness, homozygous_fitness).")
        if copies == 0:
            return 1.0
        elif copies == 1:
            return float(scale[0])
        elif copies == 2:
            return float(scale[1])
        return 1.0

    if isinstance(scale, (tuple, list)):
        raise ValueError(f"Tuple scaling value {scale} is only valid when mode='custom'.")

    scale_val = float(scale)
    if mode == "multiplicative":
        return scale_val ** copies
    elif mode == "dominant":  # TODO: 存在问题：resistance和drive都存在时scale了两遍，不符合dominant
        return scale_val if copies > 0 else 1.0
    elif mode == "recessive":
        return scale_val if copies == 2 else 1.0
    else:
        raise ValueError(f"Unknown fitness scaling mode: '{mode}'. Expected 'multiplicative', 'dominant', 'recessive', or 'custom'.")

def _make_fitness_patch_given_allele_scaling(
    allele_name: Union[str, List[str], Tuple[str, ...]],
    viability_scaling: Optional[_ViabilityScalingConfig] = None,
    fecundity_scaling: Optional[_FecundityScalingConfig] = None,
    sexual_selection_scaling: Optional[_SexualSelectionScalingConfig] = None,
    viability_mode: str = "multiplicative",
    fecundity_mode: str = "multiplicative",
    sexual_selection_mode: str = "multiplicative",
) -> PresetFitnessPatch:
    """Helper to create a fitness patch dict for a single allele's scaling effects."""
    # Dictionary keys must be hashable. Lists are not, so we convert to tuple.
    if isinstance(allele_name, list):
        key = tuple(allele_name)
    else:
        key = allele_name
        
    patch: PresetFitnessPatch = {}
    
    if viability_scaling is not None:
        patch['viability_allele'] = {key: (viability_scaling, viability_mode)}
    
    if fecundity_scaling is not None:
        patch['fecundity_allele'] = {key: (fecundity_scaling, fecundity_mode)}
    
    if sexual_selection_scaling is not None:
        patch['sexual_selection_allele'] = {key: (sexual_selection_scaling, sexual_selection_mode)}
    
    return patch

def _apply_viability_allele_scaling(
    population: 'BasePopulation',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _ViabilityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven viability scaling using multiplicative copy-number effect."""
    viability_arr = population._config.viability_fitness
    default_age = int(population._config.new_adult_age) - 1
    
    # Resolve one or more alleles
    target_genes = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in viability_allele patch.")
        target_genes.append(gene)

    for genotype in all_genotypes:
        genotype_idx = population._index_registry.genotype_to_index[genotype]
        copies = _count_combined_allele_copies(genotype, target_genes)
        if copies == 0:
            continue

        if isinstance(config, (int, float, tuple, list)):
            factor = _calculate_allele_effect(config, copies, mode)
            for sex_idx in (0, 1):
                current = float(viability_arr[sex_idx, default_age, genotype_idx])
                population._config.set_viability_fitness(sex_idx, genotype_idx, current * factor, default_age)
            continue

        if not isinstance(config, dict):
            raise TypeError(f"Invalid viability allele config for '{allele_name}': {type(config).__name__}")

        if config and all(isinstance(age_key, int) for age_key in config.keys()):
            for age, scale in config.items():
                factor = _calculate_allele_effect(scale, copies, mode)
                for sex_idx in (0, 1):
                    current = float(viability_arr[sex_idx, int(age), genotype_idx])
                    population._config.set_viability_fitness(sex_idx, genotype_idx, current * factor, int(age))
            continue

        for sex_key, sex_config in config.items():
            sex_idx = _normalize_sex_key(sex_key)
            if isinstance(sex_config, (int, float, tuple, list)):
                factor = _calculate_allele_effect(sex_config, copies, mode)
                current = float(viability_arr[sex_idx, default_age, genotype_idx])
                population._config.set_viability_fitness(sex_idx, genotype_idx, current * factor, default_age)
            elif isinstance(sex_config, dict):
                for age, scale in sex_config.items():
                    factor = _calculate_allele_effect(scale, copies, mode)
                    current = float(viability_arr[sex_idx, int(age), genotype_idx])
                    population._config.set_viability_fitness(sex_idx, genotype_idx, current * factor, int(age))
            else:
                raise TypeError(
                    f"Invalid viability allele sex config for '{allele_name}', sex '{sex_key}': "
                    f"{type(sex_config).__name__}"
                )

def _apply_fecundity_allele_scaling(
    population: 'BasePopulation',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _FecundityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven fecundity scaling using multiplicative copy-number effect."""
    fecundity_arr = population._config.fecundity_fitness
    
    # Resolve one or more alleles
    target_genes = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in fecundity_allele patch.")
        target_genes.append(gene)

    for genotype in all_genotypes:
        genotype_idx = population._index_registry.genotype_to_index[genotype]
        copies = _count_combined_allele_copies(genotype, target_genes)
        if copies == 0:
            continue

        if isinstance(config, (int, float, tuple, list)):
            factor = _calculate_allele_effect(config, copies, mode)
            for sex_idx in (0, 1):
                current = float(fecundity_arr[sex_idx, genotype_idx])
                population._config.set_fecundity_fitness(sex_idx, genotype_idx, current * factor)
            continue

        if not isinstance(config, dict):
            raise TypeError(f"Invalid fecundity allele config for '{allele_name}': {type(config).__name__}")

        for sex_key, scale in config.items():
            sex_idx = _normalize_sex_key(sex_key)
            factor = _calculate_allele_effect(scale, copies, mode)
            current = float(fecundity_arr[sex_idx, genotype_idx])
            population._config.set_fecundity_fitness(sex_idx, genotype_idx, current * factor)

def _apply_sexual_selection_allele_scaling(
    population: 'BasePopulation',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _SexualSelectionScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven sexual-selection scaling.

    - float: multiplicative by male allele copy-number for all female genotypes.
    - tuple(default, carrier): binary by male carrier status (copy > 0).
    """
    sex_sel_arr = population._config.sexual_selection_fitness
    
    # Resolve one or more alleles
    target_genes = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in sexual_selection_allele patch.")
        target_genes.append(gene)

    for f_genotype in all_genotypes:
        f_idx = population._index_registry.genotype_to_index[f_genotype]
        for m_genotype in all_genotypes:
            m_idx = population._index_registry.genotype_to_index[m_genotype]
            copies = _count_combined_allele_copies(m_genotype, target_genes)

            if isinstance(config, tuple):
                if len(config) != 2:
                    raise ValueError(
                        f"sexual_selection allele tuple for '{allele_name}' must have length 2, got {len(config)}"
                    )
                factor = float(config[1] if copies > 0 else config[0])
            else:
                factor = _calculate_allele_effect(config, copies, mode)

            current = float(sex_sel_arr[f_idx, m_idx])
            population._config.set_sexual_selection_fitness(f_idx, m_idx, current * factor)

def _apply_preset_fitness_patch(population: 'BasePopulation', patch: PresetFitnessPatch) -> None:
    """Apply a declarative preset fitness patch to population config tensors.

    Patch schema (all keys optional):
    - viability: Dict[genotype_selector, _ViabilityScalingConfig]
    - fecundity: Dict[genotype_selector, _FecundityScalingConfig]
    - sexual_selection: Dict[female_selector, Union[float, Dict[male_selector, float]]]
    """
    if not patch:
        return

    if population._config is None:
        return

    all_genotypes = list(population._index_registry.genotype_to_index.keys())

    viability_patch = patch.get('viability', {})
    for selector, config in viability_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.viability',
        )
        for genotype in matched:
            genotype_idx = population._index_registry.genotype_to_index[genotype]

            # scalar: both sexes at default viability age
            if isinstance(config, (int, float)):
                population._config.set_viability_fitness(0, genotype_idx, float(config))
                population._config.set_viability_fitness(1, genotype_idx, float(config))
                continue

            if not isinstance(config, dict):
                raise TypeError(f"Invalid viability config for selector '{selector}': {type(config).__name__}")

            # age-specific for both sexes: {age: scale}
            if config and all(isinstance(age_key, int) for age_key in config.keys()):
                for age, scale in config.items():
                    population._config.set_viability_fitness(0, genotype_idx, float(scale), int(age))
                    population._config.set_viability_fitness(1, genotype_idx, float(scale), int(age))
                continue

            # sex-specific: {sex: float | {age: scale}}
            for sex_key, sex_config in config.items():
                sex_idx = _normalize_sex_key(sex_key)
                if isinstance(sex_config, (int, float)):
                    population._config.set_viability_fitness(sex_idx, genotype_idx, float(sex_config))
                elif isinstance(sex_config, dict):
                    for age, scale in sex_config.items():
                        population._config.set_viability_fitness(sex_idx, genotype_idx, float(scale), int(age))
                else:
                    raise TypeError(
                        f"Invalid viability sex config for selector '{selector}', sex '{sex_key}': "
                        f"{type(sex_config).__name__}"
                    )

    fecundity_patch = patch.get('fecundity', {})
    for selector, config in fecundity_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.fecundity',
        )
        for genotype in matched:
            genotype_idx = population._index_registry.genotype_to_index[genotype]

            if isinstance(config, (int, float)):
                population._config.set_fecundity_fitness(0, genotype_idx, float(config))
                population._config.set_fecundity_fitness(1, genotype_idx, float(config))
                continue

            if not isinstance(config, dict):
                raise TypeError(f"Invalid fecundity config for selector '{selector}': {type(config).__name__}")

            for sex_key, scale in config.items():
                sex_idx = _normalize_sex_key(sex_key)
                population._config.set_fecundity_fitness(sex_idx, genotype_idx, float(scale))

    sexual_selection_patch = patch.get('sexual_selection', {})
    for female_selector, male_config in sexual_selection_patch.items():
        female_matched = population.species.resolve_genotype_selectors(
            selector=female_selector,
            all_genotypes=all_genotypes,
            context='preset.sexual_selection(female)',
        )

        # Allow shorthand: female_selector -> scalar means all-male targets
        if isinstance(male_config, (int, float)):
            male_map = {'*': float(male_config)}
        else:
            male_map = male_config

        if not isinstance(male_map, dict):
            raise TypeError(
                f"Invalid sexual_selection config for female selector '{female_selector}': "
                f"{type(male_config).__name__}"
            )

        for male_selector, scale in male_map.items():
            male_matched = population.species.resolve_genotype_selectors(
                selector=male_selector,
                all_genotypes=all_genotypes,
                context='preset.sexual_selection(male)',
            )
            for f_genotype in female_matched:
                f_idx = population._index_registry.genotype_to_index[f_genotype]
                for m_genotype in male_matched:
                    m_idx = population._index_registry.genotype_to_index[m_genotype]
                    population._config.set_sexual_selection_fitness(f_idx, m_idx, float(scale))

    # Allele-based patches: preset defines per-allele effect, framework expands to genotypes.
    viability_allele_patch = patch.get('viability_allele', {})
    for allele_name, val in viability_allele_patch.items():
        if isinstance(val, tuple) and len(val) == 2:
            config, mode = val
        else:
            config, mode = val, "multiplicative"
        _apply_viability_allele_scaling(population, all_genotypes, allele_name, config, mode)

    fecundity_allele_patch = patch.get('fecundity_allele', {})
    for allele_name, val in fecundity_allele_patch.items():
        if isinstance(val, tuple) and len(val) == 2:
            config, mode = val
        else:
            config, mode = val, "multiplicative"
        _apply_fecundity_allele_scaling(population, all_genotypes, allele_name, config, mode)

    sexual_selection_allele_patch = patch.get('sexual_selection_allele', {})
    for allele_name, val in sexual_selection_allele_patch.items():
        if isinstance(val, tuple) and len(val) == 2:
            config, mode = val
        else:
            config, mode = val, "multiplicative"
        _apply_sexual_selection_allele_scaling(population, all_genotypes, allele_name, config, mode)

def apply_preset_to_population(population: 'BasePopulation', preset: 'GeneticPreset') -> None:
    """Apply a genetic preset to a population by registering its modifiers and fitness effects.
    
    This function handles the mechanical application of a preset to a population,
    including:
    1. Species binding and validation
    2. Registration of gamete modifiers
    3. Registration of zygote modifiers  
    4. Application of fitness patches
    
    Args:
        population: The BasePopulation instance to modify.
        preset: The GeneticPreset instance to apply.
    
    Note:
        This is typically called through the modern API:
        ``population.apply_preset(preset)``
        
        The legacy API ``preset.apply(population)`` is deprecated but still supported.
    
    Raises:
        ValueError: If preset is bound to a different species than the population
        RuntimeError: If preset has no bound species
    """
    preset.bind_species(population.species)

    gamete_mod = preset.gamete_modifier(population)
    zygote_mod = preset.zygote_modifier(population)

    if gamete_mod is not None:
        population.add_gamete_modifier(
            gamete_mod,
            name=f"{preset.name}/gamete",
            refresh=False,
        )
    
    if zygote_mod is not None:
        population.add_zygote_modifier(
            zygote_mod,
            name=f"{preset.name}/zygote",
            refresh=False,
        )

    if gamete_mod is not None or zygote_mod is not None:
        population._refresh_modifier_maps()

    # Preferred path: declarative fitness patch
    patch = preset.fitness_patch()
    if patch:
        _apply_preset_fitness_patch(population, patch)
        return

class GeneticPreset(ABC):
    """Abstract base for genetic modification presets including gene drives, mutations, and allele conversions.
    
    A preset bundles gamete modifiers, zygote modifiers, and fitness effects
    that form a cohesive genetic system. This can include:
    - Gene drives (e.g., CRISPR/Cas9 homing drives)
    - General mutations (point mutations, insertions, deletions)
    - Complex allele conversion systems
    
    Presets should implement:
      - gamete_modifier(): returns GameteModifier callable or None
      - zygote_modifier(): returns ZygoteModifier callable or None  
      - fitness_patch(): returns declarative fitness configuration dict or None
    
    All methods are optional (can return None). At least one method should be implemented
    for the preset to have any effect.
    
    Usage:
        # Modern API (recommended)
        population.apply_preset(preset)
        
        # Legacy API (deprecated)
        preset.apply(population)
    """
    
    def __init__(
        self, 
        name: str = "",
        species: Optional[Species] = None,
    ):
        """Initialize the preset.
        
        Args:
            name: Optional human-readable name for the preset.
            species: Optional species bound at construction time. If provided,
                applying this preset to a population with a different species
                will raise an error.
        """
        self.name = name or self.__class__.__name__
        self.hook_id: Optional[int] = None
        self._bound_species: Optional[Species] = species

    def bind_species(self, species: Species) -> None:
        """Bind this preset instance to a concrete species.

        This enables delayed species injection: users can construct presets
        without passing species, and binding happens automatically when the
        preset is applied to a population.
        """
        if self._bound_species is None:
            self._bound_species = species
            return

        if self._bound_species is species:
            return

        raise ValueError(
            f"Preset '{self.name}' is already bound to species "
            f"'{self._bound_species.name}' and cannot be applied to population species '{species.name}'."
        )

    def _require_bound_species(self) -> Species:
        """Return the bound species or raise if preset has not been injected yet."""
        if self._bound_species is None:
            raise RuntimeError(
                f"Preset '{self.name}' is not bound to a species. "
                "Apply it through population.apply_preset(...) or builder.presets(...).build()."
            )
        return self._bound_species

    def _resolve_bound_gene(self, allele_name: str) -> Gene:
        """Resolve an allele name into a Gene using the currently bound species."""
        species = self._require_bound_species()
        gene = species.gene_index.get(allele_name)
        if gene is None:
            raise ValueError(
                f"Allele '{allele_name}' not found in species '{species.name}' "
                f"for preset '{self.name}'."
            )
        return gene
    
    @abstractmethod
    def gamete_modifier(self, population: 'BasePopulation') -> Optional[GameteModifier]:
        """Return a gamete modifier or None.
        
        The modifier should return:
        
            Dict[(sex_idx, genotype_idx) -> Dict[compressed_hg_glab_idx -> freq]]
        
        where compressed_hg_glab_idx is an integer index into the compressed
        haploid genotype space.
        """
        return None
    
    @abstractmethod
    def zygote_modifier(self, population: 'BasePopulation') -> Optional[ZygoteModifier]:
        """Return a zygote modifier or None.
        
        The modifier should return:
        
            Dict[(c1, c2) -> (idx_modified | Genotype | Dict[idx -> prob])]
        
        where c1, c2 are compressed coordinate pairs representing the parental
        diploid genotypes.
        """
        return None
    
    @abstractmethod
    def fitness_patch(self) -> Optional[PresetFitnessPatch]:
        """Return a declarative fitness patch dict to modify population config tensors.
        
        The patch can specify modifications to viability, fecundity, and sexual
        selection fitness using a structured schema. This allows the framework
        to apply complex fitness effects without requiring the preset to directly
        manipulate config tensors.
        
        If a patch is provided, it takes precedence over legacy tensor modifier
        methods (viability_fitness_modifier, etc.) for fitness effect
        """
        return None
    
    def _resolve_allele_name(self, allele: _AlleleSpecifier) -> str:
        """Helper to resolve allele inputs to their string names."""
        if isinstance(allele, Gene):
            return allele.name
        elif isinstance(allele, str):
            return allele
        else:
            raise TypeError("Allele must be a Gene instance or a string name.")

    def _resolve_rates(
        self, rate: _SexSpecificRates
    ) -> Tuple[float, float]:
        """Helper to resolve rate inputs into a tuple of (female_rate, male_rate)."""
        if isinstance(rate, (int, float)):
            return (rate, rate)
        elif isinstance(rate, tuple):
            return rate
        elif isinstance(rate, dict):
            female_rate = rate.get(Sex.FEMALE) or rate.get("female") or rate.get("f") or rate.get("F") or 0.0
            male_rate = rate.get(Sex.MALE) or rate.get("male") or rate.get("m") or rate.get("M") or 0.0
            return (female_rate, male_rate)
        else:
            raise TypeError("Rate must be a float, tuple of floats, or dict with sex keys.")

    def apply(self, population) -> None:
        """Register this preset onto a population (DEPRECATED).
        
        .. deprecated:: 
            Use population.apply_preset(preset) instead.
            This method is kept for backwards compatibility and may be removed in future versions.
        
        Args:
            population: The BasePopulation instance to modify.
            
        See Also:
            :meth:`natal.base_population.BasePopulation.apply_preset` - Preferred modern API
        """
        apply_preset_to_population(population, self)

class HomingDrive(GeneticPreset):
    """Homing-based gene drive (e.g., CRISPR/Cas9 homing drives).
    
    This preset implements a homing gene drive that spreads through homology-directed
    repair (HDR) converting wild-type alleles into drive alleles in heterozygotes.
    It can also generate resistance alleles through non-homologous end joining (NHEJ).
    
    Key features:
    - Drive conversion in heterozygotes (WT → Drive)
    - Germline resistance formation (WT → Resistance)
    - Embryo resistance formation
    - Maternal/paternal Cas9 deposition
    - Functional vs non-functional resistance alleles
    - Sex-specific rates for all processes
    
    The drive operates through a sequential cascade:
    1. Homing conversion (WT → Drive)
    2. Resistance formation in remaining WT alleles
    3. Optional functional resistance split
    
    Usage:
        drive = HomingDrive(
            name="MyDrive",
            drive_allele="Drive",
            target_allele="WT",
            resistance_allele="Resistance",
            drive_conversion_rate=0.95,
            late_germline_resistance_formation_rate=0.03
        )
        population.apply_preset(drive)
    """
    
    def __init__(
        self,
        name: str,
        drive_allele: _AlleleSpecifier,
        target_allele: _AlleleSpecifier,
        resistance_allele: Optional[_AlleleSpecifier] = None,
        functional_resistance_allele: Optional[_AlleleSpecifier] = None,
        cas9_allele: Optional[_AlleleSpecifier] = None,
        drive_conversion_rate: _SexSpecificRates = 0.5,
        late_germline_resistance_formation_rate: _SexSpecificRates = 0.0,
        embryo_resistance_formation_rate: _SexSpecificRates = 0.0,
        functional_resistance_ratio: float = 0.0,
        fecundity_scaling: _FecundityScalingConfig = 1.0,
        viability_scaling: _ViabilityScalingConfig = 1.0,
        sexual_selection_scaling: _SexualSelectionScalingConfig = 1.0,
        viability_mode: str = "multiplicative",
        fecundity_mode: str = "multiplicative",
        sexual_selection_mode: str = "multiplicative",
        cas9_deposition_glab: Optional[str] = None,
        species: Optional[Species] = None,
        use_paternal_deposition: bool = False,
    ):
        """Initialize a homing-based gene drive (e.g., CRISPR/Cas9 homing drives).
    
        This drive spreads via homology-directed repair (HDR) converting wild-type alleles into drive alleles in heterozygotes.
        It can also generate resistance alleles through non-homologous end joining (NHEJ).
        
        Args:
            name (str): Name of the gene drive.
            drive_allele (str or Gene): The allele carrying the drive cassette.
            target_allele (str or Gene): The wild-type allele targeted by the drive.
            resistance_allele (str or Gene, optional): The non-functional resistance allele formed by NHEJ.
            functional_resistance_allele (str or Gene, optional): The functional resistance allele 
                formed by in-frame NHEJ. If not provided, assume no functional resistance.
            cas9_allele (str or Gene, optional): The allele carrying Cas9 for cleavage, used
                when modeling a split drive where Cas9 is separate from the drive locus.
            drive_conversion_rate (float or dict): Probability of drive conversion caused by Cas9 cleavage
                and homology-directed repair in heterozygotes. Can be a single float (applies to both sexes), 
                a dict with sex keys, or a tuple (female_rate, male_rate) for sex-specific rates.
            late_germline_resistance_formation_rate (float or dict): Probability of resistance formation 
                *after* drive conversion in the germline. Can be a single float (applies to both sexes), 
                a dict with sex keys, or a tuple (female_rate, male_rate) for sex-specific rates.
            embryo_resistance_formation_rate (float or dict): Probability of resistance formation 
                in embryos due to maternal/paternal Cas9 deposition. Can be a single float, dict, or tuple.
            functional_resistance_ratio (float): Proportion of resistance alleles that are functional 
                (in-frame mutations). Range: 0.0 (all non-functional) to 1.0 (all functional).
            fecundity_scaling (float or dict): Fitness cost multiplier for drive carriers affecting fecundity.
                Applied multiplicatively based on allele copy number.
            viability_scaling (float or dict): Fitness cost multiplier for drive carriers affecting viability.
                Applied multiplicatively based on allele copy number.
            sexual_selection_scaling (float or tuple): Fitness cost affecting sexual selection.
                Can be a single float or tuple (default_selection, carrier_selection).
            viability_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            fecundity_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            sexual_selection_mode (str): Scaling mode for scalar sexual_selection_scaling.
                Note: if sexual_selection_scaling is a tuple, mode is ignored.
            cas9_deposition_glab (str, optional): Gamete label for Cas9 deposition tracking.
                Used for maternal/paternal effect modeling.
            species (Species, optional): Species to bind at construction time. If None, 
                will be bound when applied to population.
            use_paternal_deposition (bool): Whether to enable paternal Cas9 deposition.
                If True, fathers can deposit Cas9 in embryos.
        
        Example:
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT", 
            ...     resistance_allele="R2",
            ...     drive_conversion_rate=0.95,
            ...     late_germline_resistance_formation_rate=0.03
            ... )
            >>> population.apply_preset(drive)
        """
        self._str_drive_allele = self._resolve_allele_name(drive_allele)
        self._str_target_allele = self._resolve_allele_name(target_allele)
        self._str_resistance_allele = (self._resolve_allele_name(resistance_allele)
            if resistance_allele else None)
        self._str_functional_resistance_allele = (self._resolve_allele_name(functional_resistance_allele) 
            if functional_resistance_allele else None)
        self._str_cas9_allele = self._resolve_allele_name(cas9_allele) if cas9_allele else None
        
        self.drive_conversion_rate = self._resolve_rates(drive_conversion_rate)
        self.late_germline_resistance_formation_rate = self._resolve_rates(late_germline_resistance_formation_rate)
        self.embryo_resistance_formation_rate = self._resolve_rates(embryo_resistance_formation_rate)
        self.functional_resistance_ratio = float(functional_resistance_ratio)

        # Store declarative fitness scaling configs.
        self.fecundity_scaling = fecundity_scaling
        self.viability_scaling = viability_scaling
        self.sexual_selection_scaling = sexual_selection_scaling
        
        self.viability_mode = viability_mode
        self.fecundity_mode = fecundity_mode
        self.sexual_selection_mode = sexual_selection_mode

        self.cas9_deposition_glab = str(cas9_deposition_glab) if cas9_deposition_glab else None
        self.use_paternal_deposition = bool(use_paternal_deposition)
        
        super().__init__(name=name, species=species)

    def fitness_patch(self) -> PresetFitnessPatch:
        """Return declarative fitness patch for homing drive scaling configs."""
        # Combine drive and non-functional resistance alleles into a single group.
        # This ensures that a "Drive|Resistance" genotype is treated as having 
        # 2 copies of the "disrupted" allele class, which is crucial for correct
        # dominant/recessive scaling logic.
        alleles = [self._str_drive_allele]
        if self._str_resistance_allele:
            alleles.append(self._str_resistance_allele)
        
        patch = _make_fitness_patch_given_allele_scaling(
            alleles,
            self.viability_scaling,
            self.fecundity_scaling,
            self.sexual_selection_scaling,
            self.viability_mode,
            self.fecundity_mode,
            self.sexual_selection_mode,
        )
        
        return patch
    
    def _instantiate_allele(self, allele_name: str, population: 'BasePopulation') -> Gene:
        """Helper to get Gene object for an allele name from the population's species."""
        gene = population.species.gene_index.get(allele_name)
        if gene is None:
            raise ValueError(f"Allele '{allele_name}' not found in species '{population.species.name}'.")
        return gene

    @property
    def drive_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_drive_allele)

    @property
    def target_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_target_allele)

    @property
    def resistance_genotype(self) -> Gene:
        return self._resolve_bound_gene(self._str_resistance_allele)

    @property
    def functional_resistance_allele(self) -> Optional[Gene]:
        if self._str_functional_resistance_allele is None:
            return None
        return self._resolve_bound_gene(self._str_functional_resistance_allele)

    @property
    def cas9_allele(self) -> Optional[Gene]:
        if self._str_cas9_allele is None:
            return None
        return self._resolve_bound_gene(self._str_cas9_allele)

    def gamete_modifier(self, population: 'BasePopulation') -> Optional[GameteModifier]:
        """Implement homing in heterozygous parents, germline resistance, and Cas9 deposition.
        
        In heterozygotes (drive/wild-type), gametes are biased towards drive.
        """
        def drive_carrier_filter(gt: Genotype) -> bool:
            from natal.genetic_presets import _count_allele_copies
            has_drive = _count_allele_copies(gt, self.drive_allele) > 0
            if self.cas9_allele:
                has_cas9 = _count_allele_copies(gt, self.cas9_allele) > 0
                return has_drive and has_cas9
            return has_drive
        
        # RuleSet compiles these rules into a Sequential Cascade.
        # This means the target pool shrinks after every rule. 
        # So Rule 2 (Resistance) only acts on the targets that FAILED Rule 1 (Homing).
        rule_set = GameteConversionRuleSet(f"{self.name}_Homing")
        for sex in (Sex.FEMALE, Sex.MALE):
            homing_rate = self.drive_conversion_rate[sex]
            res_rate = self.late_germline_resistance_formation_rate[sex]

            # 1. Homing (Target -> Drive)
            # Example: If homing_rate is 0.7, 70% of targets become Drive. 30% pass to the next rule.
            if homing_rate > 0:
                rule_set.add_allele_convert(
                    from_allele=self.target_allele,
                    to_allele=self.drive_allele,
                    rate=homing_rate,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                )
            
            # 2. Germline Resistance (Target -> Resistance)
            # This operates ON THE REMAINDER of the target alleles (e.g. the 30% that survived Homing).
            if res_rate > 0:
                if self.functional_resistance_allele and self.functional_resistance_ratio > 0:
                    # 2a. Functional resistance
                    # Applying absolute `res_rate * func_res_ratio` directly works because GameteAlleleConversionRule
                    # calculates rates against the *current* target pool. So if 30% targets are left, and this 
                    # rate is 0.1, it converts 10% of that 30% (overall 3% of origin).
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.functional_resistance_allele,
                        rate=res_rate * self.functional_resistance_ratio,
                        sex_filter=sex,
                        genotype_filter=drive_carrier_filter,
                    )
                    
                    # 2b. Non-functional resistance
                    # The functional rule above removed `res_rate * func_res_ratio` from the available targets.
                    # To hit the correct math for the *remaining* non-functional portion, we divide the 
                    # non-functional rate by whatever remains of the target pool after the functional edits.
                    target_remaining = 1.0 - (res_rate * self.functional_resistance_ratio)
                    adjusted_nf_rate = ((res_rate * (1.0 - self.functional_resistance_ratio)) 
                                        / target_remaining) if target_remaining > 0 else 0.0
                    if adjusted_nf_rate > 0:
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele,
                            to_allele=self.resistance_genotype,
                            rate=adjusted_nf_rate,
                            sex_filter=sex,
                            genotype_filter=drive_carrier_filter,
                        )
                else:
                    # Generic resistance (no functional/non-functional split)
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.resistance_genotype,
                        rate=res_rate,
                        sex_filter=sex,
                        genotype_filter=drive_carrier_filter,
                    )

            # 3. Gamete labeling for maternal Cas9 deposition
            # Instead of editing alleles, this tags the entire output gamete from drive-carrying females 
            # with `cas9_deposition_glab`. The zygote modifier will read this tag to apply embryo resistance.
            if sex == Sex.FEMALE or self.use_paternal_deposition:
                rule_set.add_hg_convert(
                    hg_match=lambda hg: True,
                    to_haploid_genotype=lambda hg: hg,
                    rate=1.0,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                    target_glab=self.cas9_deposition_glab
                )
            
            print(f"ruleset: {rule_set}")
        
        return rule_set.to_gamete_modifier(population) if rule_set.rules else None
    
    def zygote_modifier(self, population: 'BasePopulation') -> Optional[ZygoteModifier]:
        """Implement embryo resistance.
        
        Crosses between drive and wild-type show cleavage resistance, where
        resistance alleles are created instead of drive homozygotes being lethal.
        """
        def drive_carrier_filter(gt: Genotype) -> bool:
            from natal.genetic_presets import _count_allele_copies
            has_drive = _count_allele_copies(gt, self.drive_allele) > 0
            if self.split and self.cas9_allele:
                has_cas9 = _count_allele_copies(gt, self.cas9_allele) > 0
                return has_drive and has_cas9
            return has_drive
            
        rule_set = ZygoteConversionRuleSet(f"{self.name}_EmbryoResistance")

        for sex in (Sex.FEMALE, Sex.MALE):
            # Base embryo resistance on the zygote having the drive allele,
            # converting target alleles to resistance alleles.
            rate = self.embryo_resistance_formation_rate[sex]
            if rate > 0:
                if self.cas9_deposition_glab:
                    # Target maternal deposition directly using Gamete label
                    # This ensures embryo resistance ONLY happens if the egg was labeled `cas9_deposition_glab`
                    # during gametogenesis (meaning the mother actually had the Cas9/Drive allele).
                    
                    func_res_ratio = self.functional_resistance_ratio
                    if self.functional_resistance_allele and func_res_ratio > 0:
                        # 1. Functional resistance
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele,
                            to_allele=self.functional_resistance_allele,
                            rate=rate * func_res_ratio,
                            maternal_glab=self.cas9_deposition_glab if sex == Sex.FEMALE else None,
                            paternal_glab=self.cas9_deposition_glab if sex == Sex.MALE and self.use_paternal_deposition else None,
                        )
                        # 2. Non-functional resistance on remaining targets
                        # Just like gametogenesis, we must adjust the rate to account for targets already removed 
                        # by the functional resistance rule.
                        target_remaining = 1.0 - (rate * func_res_ratio)
                        nf_rate = (rate * (1.0 - func_res_ratio)) / target_remaining if target_remaining > 0 else 0.0
                        if nf_rate > 0:
                            rule_set.add_allele_convert(
                                from_allele=self.target_allele,
                                to_allele=self.resistance_genotype,
                                rate=nf_rate,
                                maternal_glab=self.cas9_deposition_glab if sex == Sex.FEMALE else None,
                                paternal_glab=self.cas9_deposition_glab if sex == Sex.MALE and self.use_paternal_deposition else None,
                            )
                    else:
                        # Generic resistance (no functional split) triggered by maternal glab
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele,
                            to_allele=self.resistance_genotype,
                            rate=rate,
                            maternal_glab=self.cas9_deposition_glab if sex == Sex.FEMALE else None,
                            paternal_glab=self.cas9_deposition_glab if sex == Sex.MALE and self.use_paternal_deposition else None,
                        )
            
        return rule_set.to_zygote_modifier(population) if rule_set.rules else None