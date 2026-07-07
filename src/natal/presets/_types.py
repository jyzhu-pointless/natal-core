"""Type aliases, TypeGuard functions, and helper functions for genetic presets.

Private module — not part of the public API.
"""

# pyright: reportUnusedFunction=false

from collections.abc import Mapping
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TypeGuard,
    Union,
    cast,
)

from natal.genetics import Gene, Genotype
from natal.utils.helpers import resolve_sex_label
from natal.utils.types import Age, Sex

# Temporary type alias
_AlleleSpecifier = Union[Gene, str]
AlleleSpecifier = _AlleleSpecifier
_SexSpecifier = Union[Sex, int, str]
SexSpecifier = _SexSpecifier
_SexSpecificRates = Union[float, Tuple[float, float], Dict[_SexSpecifier, float]]
SexSpecificRates = _SexSpecificRates
_AlleleScalingMode = Literal["multiplicative", "dominant", "recessive", "custom"]
AlleleScalingMode = _AlleleScalingMode

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
ViabilityScalingConfig = _ViabilityScalingConfig

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
FecundityScalingConfig = _FecundityScalingConfig

# Sexual-selection patch config for one allele key.
# float: copy-number based scaling (by mode)
# tuple(default, carrier): binary carrier rule
_SexualSelectionScalingConfig = Union[
    float,                        # applies to all female genotypes
    Tuple[float, float]           # (male selected by default, male selected by allele carriers)
]
SexualSelectionScalingConfig = _SexualSelectionScalingConfig

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
ZygoteViabilityScalingConfig = _ZygoteViabilityScalingConfig

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
    """Type guard: check if *config* maps age integers to effect scales.

    Args:
        config: The value to check.

    Returns:
        True if every key is an int and every value is a valid effect scale.
    """
    return all(isinstance(age_key, int) and _is_effect_scale(scale) for age_key, scale in config.items())


def _is_simple_age_scale_map(config: Mapping[object, object]) -> TypeGuard[Dict[int, Union[int, float]]]:
    """Type guard: check if *config* maps age integers to simple numeric scales.

    Args:
        config: The value to check.

    Returns:
        True if every key is an int and every value is a number.
    """
    return all(isinstance(age_key, int) and isinstance(scale, (int, float)) for age_key, scale in config.items())


def _as_pair(value: object) -> Optional[Tuple[object, object]]:
    """Safely extract a 2-tuple from an unknown value.

    Args:
        value: The value to convert.

    Returns:
        A 2-tuple if *value* is a tuple of length 2, else None.
    """
    if not isinstance(value, tuple):
        return None
    items = cast(Tuple[object, ...], value)
    if len(items) != 2:
        return None
    return items[0], items[1]


def _coerce_sex_specifier(value: object) -> _SexSpecifier:
    """Coerce an unknown value to a valid sex specifier (Sex, int, or str).

    Args:
        value: The value to coerce.

    Returns:
        The validated sex specifier.

    Raises:
        TypeError: If *value* is not a Sex, int, or str.
    """
    if isinstance(value, (Sex, int, str)):
        return value
    raise TypeError(f"Invalid sex key type: {type(value).__name__}")


def _coerce_selector(value: object) -> Union[Genotype, str, Tuple[Union[Genotype, str], ...]]:
    """Coerce an unknown value to a genotype selector.

    Args:
        value: The value to coerce.

    Returns:
        The validated selector (Genotype, str, or tuple thereof).

    Raises:
        TypeError: If *value* is not a valid selector type.
    """
    if isinstance(value, (Genotype, str)):
        return value
    if isinstance(value, tuple):
        tuple_value = cast(Tuple[object, ...], value)
        if all(isinstance(v, (Genotype, str)) for v in tuple_value):
            return cast(Tuple[Union[Genotype, str], ...], tuple_value)
    raise TypeError(f"Invalid selector type: {type(cast(object, value)).__name__}")


def _split_config_mode(value: object) -> Tuple[object, str]:
    """Split a scaling config value from its mode specifier.

    If *value* is a 2-tuple ``(scaling_value, mode_str)``, return
    both parts.  Otherwise return ``(value, "multiplicative")``.

    Args:
        value: The scaling config, optionally paired with a mode.

    Returns:
        A tuple of ``(scaling_value, mode_str)``.
    """
    pair = _as_pair(value)
    if pair is not None and isinstance(pair[1], str):
        return pair[0], pair[1]
    return value, "multiplicative"


def _is_viability_scaling_config(value: object) -> TypeGuard[_ViabilityScalingConfig]:
    """Type guard: check if *value* is a valid viability scaling config.

    Accepts:
    - A single number or effect scale pair.
    - An age-keyed dict of effect scales.
    - A sex-keyed dict where each value is an effect scale or age-keyed dict.

    Args:
        value: The value to check.

    Returns:
        True if *value* matches the ViabilityScalingConfig shape.
    """
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
    """Type guard: check if *value* is a valid fecundity scaling config.

    Args:
        value: The value to check.

    Returns:
        True if *value* is a number, effect scale, or sex-keyed dict
        of effect scales.
    """
    if isinstance(value, (int, float)) or _is_effect_scale(value):
        return True
    if not isinstance(value, Mapping):
        return False
    config_map = cast(Mapping[object, object], value)
    return all(isinstance(sex_key, (Sex, int, str)) and _is_effect_scale(scale) for sex_key, scale in config_map.items())


def _is_sexual_selection_scaling_config(value: object) -> TypeGuard[_SexualSelectionScalingConfig]:
    """Type guard: check if *value* is a valid sexual selection scaling config.

    Args:
        value: The value to check.

    Returns:
        True if *value* is a number or an effect scale pair.
    """
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
