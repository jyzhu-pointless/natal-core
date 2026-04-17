"""Genetic presets for mutations, gene drives, and allele conversions.

This module provides a framework for defining reusable genetic modifications including:
- Gene drives (e.g., CRISPR/Cas9 homing drives)
- General mutations (point mutations, insertions, deletions)
- Allele conversion rules and fitness effects
- Complex genetic constructions

Each preset can modify population genetics through three mechanisms:
- gamete_modifier: modifies gamete frequencies during gametogenesis
- zygote_modifier: modifies embryo genotypes after fertilization
- fitness_patch: declarative specification of viability/fecundity/sexual selection effects
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Self,
    Tuple,
    TypeGuard,
    Union,
    cast,
)

from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.genetic_entities import Gene, Genotype
from natal.genetic_structures import Species
from natal.helpers import resolve_sex_label
from natal.modifiers import GameteModifier, ZygoteModifier
from natal.type_def import Age, Sex
from natal.zygote_allele_conversion import ZygoteConversionRuleSet

if TYPE_CHECKING:
    from natal.base_population import BasePopulation

__all__ = [
    "GeneticPreset",      # Abstract base class for custom presets
    "HomingDrive",        # Built-in gene drive preset
    "ToxinAntidoteDrive", # Toxin-Antidote gene drive preset
    "apply_preset_to_population",  # Core application function
]

# Temporary type alias
_AlleleSpecifier = Union[Gene, str]
_SexSpecifier = Union[Sex, int, str]
_SexSpecificRates = Union[float, Tuple[float, float], Dict[_SexSpecifier, float]]
_AlleleScalingMode = Literal["multiplicative", "dominant", "recessive", "custom"]

# Defines how a specific allele scales fitness
# e.g., if "Dr" allele has viability_scaling = 0.8, then:
# "WT|WT" -> 1.0 viability
# "Dr::WT" -> 0.8 viability
# "Dr|Dr" -> 0.64 viability (multiplicative)
# Viability patch config for one allele key.
# Supported shapes:
# 1) float
#    -> apply at default viability age for both sexes
# 2) (het, hom) tuple
#    -> only meaningful with mode="custom"
# 3) {age: scale}
#    -> apply to both sexes by age
# 4) {sex: scale or {age: scale}}
#    -> sex-specific, optionally age-specific
_ViabilityScalingConfig = Union[
    float,             # both sex, at the largest juvenile age
    Tuple[float, float],
    Dict[Age, Union[float, Tuple[float, float]]],  # both sex, age-specific
    Dict[  # sex-specific
        _SexSpecifier,
        Union[float, Tuple[float, float], Dict[Age, Union[float, Tuple[float, float]]]],
    ],
]
# Fecundity patch config for one allele key.
# Supported shapes:
# 1) float
# 2) (het, hom) tuple for mode="custom"
# 3) {sex: scale}
_FecundityScalingConfig = Union[
    float,  # both sex
    Tuple[float, float],
    Dict[_SexSpecifier, Union[float, Tuple[float, float]]],  # sex-specific
]
# Sexual-selection patch config for one allele key.
# float: copy-number based scaling (by mode)
# tuple(default, carrier): binary carrier rule
_SexualSelectionScalingConfig = Union[
    float,                        # applies to all female genotypes
    Tuple[float, float]           # (male selected by default, male selected by allele carriers)
]

# Zygote patch config for one allele key.
# Supported shapes:
# 1) float
# 2) (het, hom) tuple for mode="custom"
# 3) {sex: scale}
_ZygoteViabilityScalingConfig = Union[
    float,  # both sex
    Tuple[float, float],
    Dict[_SexSpecifier, Union[float, Tuple[float, float]]],  # sex-specific
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


# Public alias for cross-function and cross-module reuse.
count_allele_copies = _count_allele_copies

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

def _calculate_allele_effect(
    scale: Union[float, Tuple[float, float]],
    copies: int,
    mode: str = "multiplicative"
) -> float:
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
    elif mode == "dominant":
        return scale_val if copies > 0 else 1.0
    elif mode == "recessive":
        return scale_val if copies == 2 else 1.0
    else:
        raise ValueError(f"Unknown fitness scaling mode: '{mode}'. "
                         "Expected 'multiplicative', 'dominant', 'recessive', or 'custom'.")

def _is_effect_scale(value: object) -> TypeGuard[Union[float, Tuple[float, float]]]:
    """Narrow runtime config value to the scale type accepted by _calculate_allele_effect."""
    if isinstance(value, (int, float)):
        return True
    if not isinstance(value, tuple):
        return False
    pair = _as_pair(cast(object, value))
    if pair is None:
        return False
    return isinstance(pair[0], (int, float)) and isinstance(pair[1], (int, float))


def _is_viability_age_map(config: Mapping[object, object]) -> TypeGuard[Dict[Age, Union[float, Tuple[float, float]]]]:
    return all(isinstance(age_key, int) and _is_effect_scale(scale) for age_key, scale in config.items())


def _is_simple_age_scale_map(config: Mapping[object, object]) -> TypeGuard[Dict[int, Union[int, float]]]:
    return all(isinstance(age_key, int) and isinstance(scale, (int, float)) for age_key, scale in config.items())


def _as_pair(value: object) -> Optional[Tuple[object, object]]:
    if not isinstance(value, tuple):
        return None
    items = cast(Tuple[object, ...], value)
    if len(items) != 2:
        return None
    return items[0], items[1]


def _coerce_sex_specifier(value: object) -> _SexSpecifier:
    if isinstance(value, (Sex, int, str)):
        return value
    raise TypeError(f"Invalid sex key type: {type(value).__name__}")


def _coerce_selector(value: object) -> Union[Genotype, str, Tuple[Union[Genotype, str], ...]]:
    if isinstance(value, (Genotype, str)):
        return value
    if isinstance(value, tuple):
        tuple_value = cast(Tuple[object, ...], value)
        if all(isinstance(v, (Genotype, str)) for v in tuple_value):
            return cast(Tuple[Union[Genotype, str], ...], tuple_value)
    raise TypeError(f"Invalid selector type: {type(cast(object, value)).__name__}")


def _split_config_mode(value: object) -> Tuple[object, str]:
    pair = _as_pair(value)
    if pair is not None and isinstance(pair[1], str):
        return pair[0], pair[1]
    return value, "multiplicative"


def _is_viability_scaling_config(value: object) -> TypeGuard[_ViabilityScalingConfig]:
    if isinstance(value, (int, float)) or _is_effect_scale(value):
        return True
    if not isinstance(value, Mapping):
        return False
    config_map = cast(Mapping[object, object], value)
    if _is_viability_age_map(config_map):
        return True
    for sex_key, sex_config in config_map.items():
        if not isinstance(sex_key, (Sex, int, str)):
            return False
        if _is_effect_scale(sex_config):
            continue
        if isinstance(sex_config, Mapping) and _is_viability_age_map(cast(Mapping[object, object], sex_config)):
            continue
        return False
    return True


def _is_fecundity_scaling_config(value: object) -> TypeGuard[_FecundityScalingConfig]:
    if isinstance(value, (int, float)) or _is_effect_scale(value):
        return True
    if not isinstance(value, Mapping):
        return False
    config_map = cast(Mapping[object, object], value)
    return all(isinstance(sex_key, (Sex, int, str)) and _is_effect_scale(scale) for sex_key, scale in config_map.items())


def _is_sexual_selection_scaling_config(value: object) -> TypeGuard[_SexualSelectionScalingConfig]:
    if isinstance(value, (int, float)):
        return True
    pair = _as_pair(value)
    return pair is not None and isinstance(pair[0], (int, float)) and isinstance(pair[1], (int, float))

def _is_zygote_viability_scaling_config(value: object) -> TypeGuard[_ZygoteViabilityScalingConfig]:
    """Type guard for zygote viability scaling configuration."""
    if isinstance(value, (int, float)):
        return True
    pair = _as_pair(value)
    if pair is not None and isinstance(pair[0], (int, float)) and isinstance(pair[1], (int, float)):
        return True
    if not isinstance(value, Mapping):
        return False
    config_map = cast(Mapping[object, object], value)
    return all(isinstance(sex_key, (Sex, int, str)) and _is_effect_scale(scale) for sex_key, scale in config_map.items())

def _make_fitness_patch_given_allele_scaling(
    allele_name: Union[str, List[str], Tuple[str, ...]],
    viability_scaling: Optional[_ViabilityScalingConfig] = None,
    fecundity_scaling: Optional[_FecundityScalingConfig] = None,
    sexual_selection_scaling: Optional[_SexualSelectionScalingConfig] = None,
    zygote_scaling: Optional[_ZygoteViabilityScalingConfig] = None,
    viability_mode: _AlleleScalingMode = "multiplicative",
    fecundity_mode: _AlleleScalingMode = "multiplicative",
    sexual_selection_mode: str = "multiplicative",
    zygote_mode: _AlleleScalingMode = "multiplicative",
) -> PresetFitnessPatch:
    """Helper to create a fitness patch dict for a single allele's scaling effects.

    This function supports all four fitness types: viability, fecundity, sexual selection,
    and zygote fitness. Zygote fitness represents the probability that a zygote survives
    to become an individual, applied during reproduction stage before survival and competition.

    Args:
        allele_name: Name or list of allele names to apply scaling to.
        viability_scaling: Viability fitness scaling configuration.
        fecundity_scaling: Fecundity fitness scaling configuration.
        sexual_selection_scaling: Sexual selection scaling configuration.
        zygote_scaling: Zygote fitness scaling configuration.
        viability_mode: Scaling mode for viability fitness.
        fecundity_mode: Scaling mode for fecundity fitness.
        sexual_selection_mode: Scaling mode for sexual selection.
        zygote_mode: Scaling mode for zygote fitness.

    Returns:
        PresetFitnessPatch: Dictionary containing fitness patch configurations.
    """
    # Dictionary keys must be hashable. Lists are not, so we convert to tuple.
    if isinstance(allele_name, list):
        key = tuple(allele_name)
    else:
        key = allele_name

    patch: PresetFitnessPatch = {}

    if viability_scaling is not None:
        patch['viability_per_allele'] = {key: (viability_scaling, viability_mode)}

    if fecundity_scaling is not None:
        patch['fecundity_per_allele'] = {key: (fecundity_scaling, fecundity_mode)}

    if sexual_selection_scaling is not None:
        patch['sexual_selection_per_allele'] = {key: (sexual_selection_scaling, sexual_selection_mode)}

    if zygote_scaling is not None:
        patch['zygote_per_allele'] = {key: (zygote_scaling, zygote_mode)}

    return patch

def _apply_viability_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _ViabilityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven viability scaling using multiplicative copy-number effect."""
    # viability tensor layout:
    #   viability_fitness[sex_idx, age_idx, genotype_idx]
    # This function multiplies existing values in-place via setter calls,
    # so multiple presets/patches compose multiplicatively.
    viability_arr = population.config.viability_fitness
    default_age = int(population.config.new_adult_age) - 1

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in viability_per_allele patch.")
        target_genes.append(gene)

    for genotype in all_genotypes:
        genotype_idx = population.index_registry.genotype_to_index[genotype]
        copies = _count_combined_allele_copies(genotype, target_genes)
        if copies == 0:
            # No target allele copies in this genotype: no effect.
            continue

        if isinstance(config, (int, float, tuple, list)):
            # Scalar/custom tuple branch:
            # apply same factor to both sexes at default viability age.
            factor = _calculate_allele_effect(config, copies, mode)
            for sex_idx in (0, 1):
                current = float(viability_arr[sex_idx, default_age, genotype_idx])
                population.config.set_viability_fitness(sex_idx, genotype_idx, current * factor, default_age)
            continue

        config_map = cast(Mapping[object, object], config)

        if _is_viability_age_map(config_map):
            # Age-map branch: config treated as {age: scale} for both sexes.
            for age, scale in config_map.items():
                factor = _calculate_allele_effect(scale, copies, mode)
                for sex_idx in (0, 1):
                    current = float(viability_arr[sex_idx, age, genotype_idx])
                    population.config.set_viability_fitness(sex_idx, genotype_idx, current * factor, age)
            continue

        for sex_key, sex_config in config_map.items():
            # Sex-map branch:
            # sex_config can be either:
            #   - direct scale for default age
            #   - nested {age: scale}
            sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
            if _is_effect_scale(sex_config):
                factor = _calculate_allele_effect(sex_config, copies, mode)
                current = float(viability_arr[sex_idx, default_age, genotype_idx])
                population.config.set_viability_fitness(sex_idx, genotype_idx, current * factor, default_age)
            elif isinstance(sex_config, Mapping):
                sex_age_map = cast(Mapping[object, object], sex_config)
                for age, scale in sex_age_map.items():
                    if not isinstance(age, int):
                        raise TypeError(
                            f"Invalid viability sex-age key for '{allele_name}', sex '{sex_key}': {type(age).__name__}"
                        )
                    if not _is_effect_scale(scale):
                        raise TypeError(
                            f"Invalid viability sex-age scale for '{allele_name}', sex '{sex_key}', age {age}: "
                            f"{type(scale).__name__}"
                        )
                    factor = _calculate_allele_effect(scale, copies, mode)
                    current = float(viability_arr[sex_idx, int(age), genotype_idx])
                    population.config.set_viability_fitness(sex_idx, genotype_idx, current * factor, int(age))
            else:
                raise TypeError(
                    f"Invalid viability allele sex config for '{allele_name}', sex '{sex_key}': "
                    f"{type(sex_config).__name__}"
                )

def _apply_fecundity_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _FecundityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven fecundity scaling using multiplicative copy-number effect."""
    # fecundity tensor layout:
    #   fecundity_fitness[sex_idx, genotype_idx]
    # As with viability, this function multiplies current values.
    fecundity_arr = population.config.fecundity_fitness

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in fecundity_per_allele patch.")
        target_genes.append(gene)

    for genotype in all_genotypes:
        genotype_idx = population.index_registry.genotype_to_index[genotype]
        copies = _count_combined_allele_copies(genotype, target_genes)
        if copies == 0:
            continue

        if isinstance(config, (int, float, tuple, list)):
            # Global branch (both sexes).
            factor = _calculate_allele_effect(config, copies, mode)
            for sex_idx in (0, 1):
                current = float(fecundity_arr[sex_idx, genotype_idx])
                population.config.set_fecundity_fitness(sex_idx, genotype_idx, current * factor)
            continue

        config_map = cast(Mapping[object, object], config)
        for sex_key, scale in config_map.items():
            # Sex-specific branch.
            sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
            if not _is_effect_scale(scale):
                raise TypeError(
                    f"Invalid fecundity sex scale for '{allele_name}', sex '{sex_key}': {type(scale).__name__}"
                )
            factor = _calculate_allele_effect(scale, copies, mode)
            current = float(fecundity_arr[sex_idx, genotype_idx])
            population.config.set_fecundity_fitness(sex_idx, genotype_idx, current * factor)

def _apply_sexual_selection_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _SexualSelectionScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven sexual-selection scaling.

    - float: multiplicative by male allele copy-number for all female genotypes.
    - tuple(default, carrier): binary by male carrier status (copy > 0).
    """
    # sexual-selection tensor layout:
    #   sexual_selection_fitness[female_genotype_idx, male_genotype_idx]
    # Effect is computed from male allele copies, then applied per pair.
    sex_sel_arr = population.config.sexual_selection_fitness

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in sexual_selection_per_allele patch.")
        target_genes.append(gene)

    for f_genotype in all_genotypes:
        f_idx = population.index_registry.genotype_to_index[f_genotype]
        for m_genotype in all_genotypes:
            m_idx = population.index_registry.genotype_to_index[m_genotype]
            copies = _count_combined_allele_copies(m_genotype, target_genes)

            if isinstance(config, tuple):
                # Binary carrier logic:
                # config[0] for non-carriers, config[1] for carriers.
                if len(config) != 2:
                    raise ValueError(
                        f"sexual_selection allele tuple for '{allele_name}' must have length 2, got {len(config)}"
                    )
                factor = float(config[1] if copies > 0 else config[0])
            else:
                # Copy-number-based logic via mode.
                factor = _calculate_allele_effect(config, copies, mode)

            current = float(sex_sel_arr[f_idx, m_idx])
            population.config.set_sexual_selection_fitness(f_idx, m_idx, current * factor)

def _apply_preset_fitness_patch(population: 'BasePopulation[Any]', patch: PresetFitnessPatch) -> None:
    """Apply a declarative preset fitness patch to population config tensors.

    Patch schema (all keys optional):
    - viability: Dict[genotype_selector, _ViabilityScalingConfig]
    - fecundity: Dict[genotype_selector, _FecundityScalingConfig]
    - sexual_selection: Dict[female_selector, Union[float, Dict[male_selector, float]]]
    """
    if not patch:
        return

    all_genotypes = list(population.index_registry.genotype_to_index.keys())

    # ----------------------------------------------------------------------
    # 1) Selector-based viability patch
    #
    # Input examples:
    # - {"A1|A1": 0.8}
    # - {"A1|A1": {0: 0.9, 1: 0.8}}
    # - {"A1|A1": {"female": 0.9, "male": {0: 0.95}}}
    # ----------------------------------------------------------------------
    viability_patch = patch.get('viability', {})
    for selector, config in viability_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.viability',
        )
        for genotype in matched:
            genotype_idx = population.index_registry.genotype_to_index[genotype]

            # scalar: both sexes at default viability age
            if isinstance(config, (int, float)):
                population.config.set_viability_fitness(0, genotype_idx, float(config))
                population.config.set_viability_fitness(1, genotype_idx, float(config))
                continue

            # age-specific for both sexes: {age: scale}
            config_map = cast(Mapping[object, object], config)
            if _is_simple_age_scale_map(config_map):
                for age, scale in config_map.items():
                    population.config.set_viability_fitness(0, genotype_idx, float(scale), int(age))
                    population.config.set_viability_fitness(1, genotype_idx, float(scale), int(age))
                continue

            # sex-specific: {sex: float | {age: scale}}
            for sex_key, sex_config in config_map.items():
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if isinstance(sex_config, (int, float)):
                    population.config.set_viability_fitness(sex_idx, genotype_idx, float(sex_config))
                elif isinstance(sex_config, Mapping):
                    sex_age_map = cast(Mapping[object, object], sex_config)
                    for age, scale in sex_age_map.items():
                        if not isinstance(age, int) or not isinstance(scale, (int, float)):
                            raise TypeError(
                                f"Invalid viability sex-age config for selector '{selector}', sex '{sex_key}'"
                            )
                        population.config.set_viability_fitness(sex_idx, genotype_idx, float(scale), int(age))
                else:
                    raise TypeError(
                        f"Invalid viability sex config for selector '{selector}', sex '{sex_key}': "
                        f"{type(sex_config).__name__}"
                    )

    # ----------------------------------------------------------------------
    # 2) Selector-based fecundity patch
    #
    # Input examples:
    # - {"A1|A1": 0.8}
    # - {"A1|A1": {"female": 0.9, "male": 0.7}}
    # ----------------------------------------------------------------------
    fecundity_patch = patch.get('fecundity', {})
    for selector, config in fecundity_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.fecundity',
        )
        for genotype in matched:
            genotype_idx = population.index_registry.genotype_to_index[genotype]

            if isinstance(config, (int, float)):
                population.config.set_fecundity_fitness(0, genotype_idx, float(config))
                population.config.set_fecundity_fitness(1, genotype_idx, float(config))
                continue

            config_map = cast(Mapping[object, object], config)
            for sex_key, scale in config_map.items():
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if not isinstance(scale, (int, float)):
                    raise TypeError(
                        f"Invalid fecundity sex scale for selector '{selector}', sex '{sex_key}'"
                    )
                population.config.set_fecundity_fitness(sex_idx, genotype_idx, float(scale))

    # ----------------------------------------------------------------------
    # 3) Selector-based sexual-selection patch
    #
    # Input examples:
    # - {"female_selector": 0.9}  # shorthand for all males
    # - {"female_selector": {"male_selector": 1.2}}
    # ----------------------------------------------------------------------
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
            male_map = cast(Mapping[object, object], male_config)

        for male_selector, scale in male_map.items():
            if not isinstance(scale, (int, float)):
                raise TypeError(
                    f"Invalid sexual_selection scale for female selector '{female_selector}'"
                )
            male_matched = population.species.resolve_genotype_selectors(
                selector=_coerce_selector(male_selector),
                all_genotypes=all_genotypes,
                context='preset.sexual_selection(male)',
            )
            for f_genotype in female_matched:
                f_idx = population.index_registry.genotype_to_index[f_genotype]
                for m_genotype in male_matched:
                    m_idx = population.index_registry.genotype_to_index[m_genotype]
                    population.config.set_sexual_selection_fitness(f_idx, m_idx, float(scale))

    # ----------------------------------------------------------------------
    # 4) Allele-based patches
    #
    # This layer expands allele-centric config into genotype-level writes:
    # 1) resolve allele name(s) to Gene objects
    # 2) count target copies per genotype (0/1/2)
    # 3) convert copies -> factor according to mode
    # 4) multiply corresponding tensor cells
    # ----------------------------------------------------------------------
    viability_per_allele_patch = patch.get('viability_per_allele', {})
    for allele_name, val in viability_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_viability_scaling_config(config):
            raise TypeError(f"Invalid viability_per_allele config for '{allele_name}'")
        _apply_viability_allele_scaling(population, all_genotypes, allele_name, config, mode)

    fecundity_per_allele_patch = patch.get('fecundity_per_allele', {})
    for allele_name, val in fecundity_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_fecundity_scaling_config(config):
            raise TypeError(f"Invalid fecundity_per_allele config for '{allele_name}'")
        _apply_fecundity_allele_scaling(population, all_genotypes, allele_name, config, mode)

    sexual_selection_per_allele_patch = patch.get('sexual_selection_per_allele', {})
    for allele_name, val in sexual_selection_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_sexual_selection_scaling_config(config):
            raise TypeError(f"Invalid sexual_selection_per_allele config for '{allele_name}'")
        _apply_sexual_selection_allele_scaling(population, all_genotypes, allele_name, config, mode)

    # 5) Zygote fitness patch
    zygote_patch = patch.get('zygote', {})
    for selector, config in zygote_patch.items():
        matched = population.species.resolve_genotype_selectors(
            selector=selector,
            all_genotypes=all_genotypes,
            context='preset.zygote',
        )
        for genotype in matched:
            genotype_idx = population.index_registry.genotype_to_index[genotype]
            if isinstance(config, (int, float)):
                population.config.set_zygote_fitness(0, genotype_idx, float(config))
                population.config.set_zygote_fitness(1, genotype_idx, float(config))
            elif isinstance(config, Mapping):
                config_map = cast(Mapping[object, object], config)
                for sex_key, sex_config in config_map.items():
                    sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                    if isinstance(sex_config, (int, float)):
                        population.config.set_zygote_fitness(sex_idx, genotype_idx, float(sex_config))

    # 6) Zygote allele-based fitness patch
    zygote_per_allele_patch = patch.get('zygote_per_allele', {})
    for allele_name, val in zygote_per_allele_patch.items():
        config, mode = _split_config_mode(val)
        if not _is_zygote_viability_scaling_config(config):
            raise TypeError(f"Invalid zygote_per_allele config for '{allele_name}'")
        _apply_zygote_viability_allele_scaling(population, all_genotypes, allele_name, config, mode)

def _apply_zygote_viability_allele_scaling(
    population: 'BasePopulation[Any]',
    all_genotypes: List[Genotype],
    allele_name: Union[str, Tuple[str, ...]],
    config: _ZygoteViabilityScalingConfig,
    mode: str = "multiplicative",
) -> None:
    """Apply allele-driven zygote viability scaling using multiplicative copy-number effect."""
    # zygote tensor layout:
    #   zygote_fitness[sex_idx, genotype_idx]
    # This function multiplies existing values in-place via setter calls,
    # so multiple presets/patches compose multiplicatively.
    zygote_arr = population.config.zygote_fitness

    # Resolve one or more alleles
    target_genes: List[Gene] = []
    names = allele_name if isinstance(allele_name, tuple) else str(allele_name).split('+')

    for name in names:
        gene = population.species.gene_index.get(name.strip())
        if gene is None:
            raise ValueError(f"Unknown allele '{name}' in zygote_per_allele patch.")
        target_genes.append(gene)

    # Compute scaling factors for each genotype
    for genotype in all_genotypes:
        genotype_idx = population.index_registry.genotype_to_index[genotype]
        copy_count: int = _count_combined_allele_copies(genotype, target_genes)

        # Apply scaling based on copy count
        if copy_count == 0:
            continue  # No effect for zero copies

        # Get scaling factor for this copy count
        if isinstance(config, (int, float)):
            # Single value: apply to all copies
            scale_per_copy = float(config)
            total_scale: float = scale_per_copy ** copy_count
            for sex_idx in range(2):
                current = zygote_arr[sex_idx, genotype_idx]
                new_value: float = current * total_scale
                population.config.set_zygote_fitness(sex_idx, genotype_idx, new_value)
        elif isinstance(config, Mapping):
            # Sex-specific or age-specific config
            config_map = cast(Mapping[object, object], config)
            for sex_key, sex_config in config_map.items():
                sex_idx = _normalize_sex_key(_coerce_sex_specifier(sex_key))
                if isinstance(sex_config, (int, float)):
                    # Single value per sex
                    scale_per_copy = float(sex_config)
                    total_scale: float = scale_per_copy ** copy_count
                    current = zygote_arr[sex_idx, genotype_idx]
                    new_value: float = current * total_scale
                    population.config.set_zygote_fitness(sex_idx, genotype_idx, new_value)
                elif isinstance(sex_config, Mapping):
                    # Age-specific config (not applicable to zygote fitness)
                    raise TypeError(
                        f"Age-specific config not supported for zygote_allele: {sex_config}"
                    )
                else:
                    raise TypeError(
                        f"Invalid zygote allele sex config for '{allele_name}', sex '{sex_key}': "
                        f"{type(sex_config).__name__}"
                    )
        else:
            raise TypeError(
                f"Invalid zygote allele config for '{allele_name}': {type(config).__name__}"
            )


def apply_preset_to_population(population: 'BasePopulation[Any]', preset: 'GeneticPreset') -> None:
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
        population.refresh_modifier_maps()

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

    Examples:
        >>> population.apply_preset(preset)

    Attributes:
        name (str): Human-readable preset name.
        hook_id (Optional[int]): Optional identifier used when registering modifiers.
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
    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Return a gamete modifier or None.

        The modifier should return:

            Dict[(sex_idx, genotype_idx) -> Dict[compressed_hg_glab_idx -> freq]]

        where compressed_hg_glab_idx is an integer index into the compressed
        haploid genotype space.
        """
        return None

    @abstractmethod
    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Return a zygote modifier or None.

        The modifier should return:

            Dict[(c1, c2) -> (idx_modified | Genotype | Dict[idx -> prob])]

        where c1, c2 are compressed coordinate pairs representing the parental
        diploid genotypes.
        """
        return None

    def fitness_patch(self) -> Optional[PresetFitnessPatch]:
        """Return declarative fitness patch.

        Returns:
            Fitness patch from custom function if set, otherwise None.
            Subclasses should override this method for built-in behavior.
        """
        if self._custom_fitness_patch is not None:
            return self._custom_fitness_patch()
        return None

    def with_fitness_patch(
        self,
        patch_func: Callable[[], Optional[PresetFitnessPatch]]
    ) -> Self:
        """Set a custom fitness patch function and return self for chaining.

        This allows dynamic modification of fitness effects at runtime
        without subclassing, using a fluent interface.

        Args:
            patch_func: Callable that returns a PresetFitnessPatch or None.

        Returns:
            Self for method chaining.

        Example:
            >>> preset = (HomingDrive(...)
            ...     .with_fitness_patch(lambda: {
            ...         'viability_allele': {'Drive': (0.8, 'dominant')}
            ...     }))
            >>> population.apply_preset(preset)

            >>> # Also works with complex custom logic
            >>> def conditional_patch():
            ...     if some_condition:
            ...         return {'fecundity_allele': {'Mut': (0.5, 'recessive')}}
            ...     return None
            >>>
            >>> preset = HomingDrive(...).with_fitness_patch(conditional_patch)

        Note:
            This overrides any fitness patch defined in subclasses.
            To preserve subclass behavior while adding modifications,
            subclass and call super().fitness_patch() instead.
        """
        if not callable(patch_func):
            raise TypeError(f"patch_func must be callable, got {type(patch_func)}")
        self._custom_fitness_patch = patch_func
        return self

    def clear_fitness_patch(self) -> 'GeneticPreset':
        """Remove any custom fitness patch, restoring default behavior.

        Returns:
            Self for method chaining.
        """
        self._custom_fitness_patch = None
        return self

    def _resolve_allele_name(self, allele: _AlleleSpecifier) -> str:
        """Helper to resolve allele inputs to their string names."""
        if isinstance(allele, Gene):
            return allele.name
        return allele

    def _resolve_rates(
        self, rate: _SexSpecificRates
    ) -> Tuple[float, float]:
        """Helper to resolve rate inputs into a tuple of (female_rate, male_rate)."""
        if isinstance(rate, (int, float)):
            return (rate, rate)
        if isinstance(rate, tuple):
            return rate
        female_rate = rate.get(Sex.FEMALE) or rate.get("female") or rate.get("f") or rate.get("F") or 0.0
        male_rate = rate.get(Sex.MALE) or rate.get("male") or rate.get("m") or rate.get("M") or 0.0
        return (female_rate, male_rate)

    def apply(self, population: 'BasePopulation[Any]') -> None:
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

    Key features include drive conversion in heterozygotes, germline/embryo
    resistance formation, optional parental Cas9 deposition, and sex-specific
    rate control.

    The drive operates through a sequential cascade:
    1. Homing conversion (WT -> Drive)
    2. Resistance formation in remaining WT alleles
    3. Optional functional resistance split

    Attributes:
        drive_conversion_rate (Tuple[float, float]): Female/male homing rates.
        late_germline_resistance_formation_rate (Tuple[float, float]): Female/male
            late germline resistance rates.
        embryo_resistance_formation_rate (Tuple[float, float]): Female/male embryo
            resistance rates.

    Examples:
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
        zygote_scaling: _ZygoteViabilityScalingConfig = 1.0,
        viability_mode: _AlleleScalingMode = "multiplicative",
        fecundity_mode: _AlleleScalingMode = "multiplicative",
        sexual_selection_mode: _AlleleScalingMode = "multiplicative",
        zygote_mode: _AlleleScalingMode = "multiplicative",
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
            fecundity_scaling (float or dict): Fitness multiplier for drive carriers affecting fecundity.
                Applied multiplicatively based on allele copy number.
            viability_scaling (float or dict): Fitness multiplier for drive carriers affecting viability.
                Applied multiplicatively based on allele copy number.
            sexual_selection_scaling (float or tuple): Fitness multiplier affecting sexual selection.
                Can be a single float or tuple (default_selection, carrier_selection).
            zygote_scaling (float or dict): Fitness multiplier affecting survival of zygotes before
                competition takes place. Applied multiplicatively based on allele copy number.
            viability_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            fecundity_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            sexual_selection_mode (str): Scaling mode for scalar sexual_selection_scaling.
                Note: if sexual_selection_scaling is a tuple, mode is ignored.
            zygote_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            cas9_deposition_glab (str, optional): Gamete label for Cas9 deposition tracking.
                Used for maternal/paternal effect modeling.
            species (Species, optional): Species to bind at construction time. If None,
                will be bound when applied to population.
            use_paternal_deposition (bool): Whether to enable paternal Cas9 deposition.
                If True, fathers can deposit Cas9 in embryos.

        Examples:
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
        self.zygote_scaling = zygote_scaling

        self.viability_mode: _AlleleScalingMode = viability_mode
        self.fecundity_mode: _AlleleScalingMode = fecundity_mode
        self.sexual_selection_mode: _AlleleScalingMode = sexual_selection_mode
        self.zygote_mode: _AlleleScalingMode = zygote_mode

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
            self.zygote_scaling,
            self.viability_mode,
            self.fecundity_mode,
            self.sexual_selection_mode,
            self.zygote_mode,
        )

        return patch

    def _instantiate_allele(self, allele_name: str, population: 'BasePopulation[Any]') -> Gene:
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
        if self._str_resistance_allele is None:
            raise ValueError(f"Resistance allele not defined in HomingDrive '{self.name}'.")
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

    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Implement homing in heterozygous parents, germline resistance, and Cas9 deposition.

        In heterozygotes (drive/wild-type), gametes are biased towards drive.
        """
        def drive_carrier_filter(gt: Genotype) -> bool:
            from natal.genetic_presets import count_allele_copies

            has_drive = count_allele_copies(gt, self.drive_allele) > 0
            if self.cas9_allele:
                has_cas9 = count_allele_copies(gt, self.cas9_allele) > 0
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
            # Examples: If homing_rate is 0.7, 70% of targets become Drive. 30% pass to the next rule.
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

        return rule_set.to_gamete_modifier(population) if rule_set.rules else None

    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Implement embryo resistance.

        Cleavage in the embryo (due to deposited Cas9 or zygotic expression)
        converts wild-type alleles into resistance alleles.
        """
        rule_set = ZygoteConversionRuleSet(f"{self.name}_EmbryoResistance")

        def zygote_has_cas9(gt: Genotype) -> bool:
            """Check if the zygote itself carries the Cas9 source (somatic cleavage)."""
            from natal.genetic_presets import count_allele_copies

            target = self.cas9_allele if self.cas9_allele else self.drive_allele
            return count_allele_copies(gt, target) > 0

        for sex in (Sex.FEMALE, Sex.MALE):
            rate = self.embryo_resistance_formation_rate[sex]
            if rate > 0:
                m_glab = None
                p_glab = None
                g_filter = None

                if self.cas9_deposition_glab:
                    # Label-based deposition (Maternal/Paternal effect)
                    if sex == Sex.FEMALE:
                        m_glab = self.cas9_deposition_glab
                    elif self.use_paternal_deposition:
                        p_glab = self.cas9_deposition_glab
                    else:
                        # Male rate > 0 but no paternal deposition -> somatic/zygotic expression
                        g_filter = zygote_has_cas9
                else:
                    # No labels provided -> cleavage depends on zygote's own Cas9 alleles
                    g_filter = zygote_has_cas9

                # Skip if no filter is active to avoid global mutation bug
                if m_glab is None and p_glab is None and g_filter is None:
                    continue

                func_res_ratio = self.functional_resistance_ratio
                if self.functional_resistance_allele and func_res_ratio > 0:
                    # 1. Functional resistance
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.functional_resistance_allele,
                        rate=rate * func_res_ratio,
                        maternal_glab=m_glab,
                        paternal_glab=p_glab,
                        genotype_filter=g_filter,
                    )
                    # 2. Non-functional resistance on remaining targets
                    target_remaining = 1.0 - (rate * func_res_ratio)
                    nf_rate = (rate * (1.0 - func_res_ratio)) / target_remaining if target_remaining > 0 else 0.0
                    if nf_rate > 0:
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele,
                            to_allele=self.resistance_genotype,
                            rate=nf_rate,
                            maternal_glab=m_glab,
                            paternal_glab=p_glab,
                            genotype_filter=g_filter,
                        )
                else:
                    # Generic resistance (no functional split)
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.resistance_genotype,
                        rate=rate,
                        maternal_glab=m_glab,
                        paternal_glab=p_glab,
                        genotype_filter=g_filter,
                    )

        return rule_set.to_zygote_modifier(population) if rule_set.rules else None


class ToxinAntidoteDrive(GeneticPreset):
    """Toxin-Antidote gene drive (e.g., TARE, TADE).

    This preset implements a toxin-antidote gene drive system where a "drive" allele
    disrupts a "target" allele into a "disrupted" version. The "disrupted" allele
    typically carries a high fitness cost (the toxin effect), while the "drive"
    allele itself often provides a functional rescue (the antidote).

    Key features include germline disruption, embryo disruption, and
    configurable fitness costs for the disrupted allele.

    In a typical TARE (Toxin-Antidote Recessive Embryo lethality) configuration,
    the disrupted allele is set to be recessive lethal (viability_scaling=0.0,
    viability_mode="recessive").

    Attributes:
        conversion_rate (Tuple[float, float]): Female/male germline disruption rates.
        embryo_disruption_rate (Tuple[float, float]): Female/male embryo disruption rates.
        viability_mode (_AlleleScalingMode): Scaling mode for viability effects.
        fecundity_mode (_AlleleScalingMode): Scaling mode for fecundity effects.
    """

    def __init__(
        self,
        name: str,
        drive_allele: _AlleleSpecifier,
        target_allele: _AlleleSpecifier,
        disrupted_allele: _AlleleSpecifier,
        conversion_rate: _SexSpecificRates = 0.8,
        embryo_disruption_rate: _SexSpecificRates = 0.0,
        viability_scaling: _ViabilityScalingConfig = 1.0,
        fecundity_scaling: _FecundityScalingConfig = 1.0,
        sexual_selection_scaling: Optional[_SexualSelectionScalingConfig] = None,
        zygote_viability_scaling: _ZygoteViabilityScalingConfig = 0.0,
        viability_mode: _AlleleScalingMode = "recessive",
        fecundity_mode: _AlleleScalingMode = "recessive",
        sexual_selection_mode: _AlleleScalingMode = "recessive",
        zygote_viability_mode: _AlleleScalingMode = "recessive",
        cas9_deposition_glab: Optional[str] = None,
        species: Optional[Species] = None,
        use_paternal_deposition: bool = False,
    ):
        """Initialize a toxin-antidote gene drive.

        Args:
            name: Name of the gene drive.
            drive_allele: The allele carrying the antidote and disruption machinery.
            target_allele: The wild-type allele targeted for disruption.
            disrupted_allele: The resulting non-functional/disrupted allele.
            conversion_rate: Probability of target disruption in the germline.
            embryo_disruption_rate: Probability of target disruption in embryos.
            viability_scaling: Fitness multiplier for the disrupted allele.
            fecundity_scaling: Fecundity multiplier for the disrupted allele.
            sexual_selection_scaling: Optional sexual-selection multiplier for the disrupted allele.
                Supports a scalar copy-number effect or a tuple
                ``(default_male, carrier_male)``.
            zygote_viability_scaling: Zygote viability scaling configuration for the disrupted allele.
                Applied during reproduction stage before survival; represents probability that a zygote
                survives to become an individual.
            viability_mode: Scaling mode for viability (default "recessive").
            fecundity_mode: Scaling mode for fecundity (default "recessive").
            sexual_selection_mode: Scaling mode for scalar sexual-selection values.
                Ignored when ``sexual_selection_scaling`` is a tuple.
            zygote_viability_mode: Scaling mode for zygote viability (default "multiplicative").
            cas9_deposition_glab: Gamete label for Cas9 deposition tracking.
            species: Optional species to bind at construction.
            use_paternal_deposition: Whether to enable paternal Cas9 deposition.
        """
        self._str_drive_allele = self._resolve_allele_name(drive_allele)
        self._str_target_allele = self._resolve_allele_name(target_allele)
        self._str_disrupted_allele = self._resolve_allele_name(disrupted_allele)

        self.conversion_rate = self._resolve_rates(conversion_rate)
        self.embryo_disruption_rate = self._resolve_rates(embryo_disruption_rate)

        self.viability_scaling = viability_scaling
        self.fecundity_scaling = fecundity_scaling
        self.sexual_selection_scaling = sexual_selection_scaling
        self.zygote_viability_scaling = zygote_viability_scaling
        self.viability_mode: _AlleleScalingMode = viability_mode
        self.fecundity_mode: _AlleleScalingMode = fecundity_mode
        self.sexual_selection_mode = sexual_selection_mode
        self.zygote_viability_mode: _AlleleScalingMode = zygote_viability_mode

        self.cas9_deposition_glab = str(cas9_deposition_glab) if cas9_deposition_glab else None
        self.use_paternal_deposition = bool(use_paternal_deposition)

        super().__init__(name=name, species=species)

    def fitness_patch(self) -> PresetFitnessPatch:
        """Return declarative fitness patch for the disrupted allele."""
        return _make_fitness_patch_given_allele_scaling(
            self._str_disrupted_allele,
            self.viability_scaling,
            self.fecundity_scaling,
            self.sexual_selection_scaling,
            self.zygote_viability_scaling,  # zygote_viability_scaling
            self.viability_mode,
            self.fecundity_mode,
            self.sexual_selection_mode,
            self.zygote_viability_mode,  # zygote_viability_mode
        )

    @property
    def drive_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_drive_allele)

    @property
    def target_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_target_allele)

    @property
    def disrupted_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_disrupted_allele)

    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Implement target disruption in the germline of drive carriers."""
        def drive_carrier_filter(gt: Genotype) -> bool:
            return _count_allele_copies(gt, self.drive_allele) > 0

        rule_set = GameteConversionRuleSet(f"{self.name}_GermlineDisruption")
        for sex in (Sex.FEMALE, Sex.MALE):
            rate = self.conversion_rate[sex]
            if rate > 0:
                rule_set.add_allele_convert(
                    from_allele=self.target_allele,
                    to_allele=self.disrupted_allele,
                    rate=rate,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                )

            if self.cas9_deposition_glab and (sex == Sex.FEMALE or self.use_paternal_deposition):
                rule_set.add_hg_convert(
                    hg_match=lambda hg: True,
                    to_haploid_genotype=lambda hg: hg,
                    rate=1.0,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                    target_glab=self.cas9_deposition_glab
                )

        return rule_set.to_gamete_modifier(population) if rule_set.rules else None

    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Implement target disruption in embryos."""
        rule_set = ZygoteConversionRuleSet(f"{self.name}_EmbryoDisruption")

        def zygote_has_drive(gt: Genotype) -> bool:
            return _count_allele_copies(gt, self.drive_allele) > 0

        for sex in (Sex.FEMALE, Sex.MALE):
            rate = self.embryo_disruption_rate[sex]
            if rate > 0:
                m_glab = self.cas9_deposition_glab if sex == Sex.FEMALE else None
                p_glab = self.cas9_deposition_glab if (sex == Sex.MALE and self.use_paternal_deposition) else None
                g_filter = None if (m_glab or p_glab) else zygote_has_drive

                if m_glab or p_glab or g_filter:
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.disrupted_allele,
                        rate=rate,
                        maternal_glab=m_glab,
                        paternal_glab=p_glab,
                        genotype_filter=g_filter,
                    )

        return rule_set.to_zygote_modifier(population) if rule_set.rules else None
