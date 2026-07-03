"""Modifier system for population simulations.

This subpackage defines protocols and helper functions for constructing and
wrapping modifiers that alter gamete or zygote production in the simulation.
"""

from natal.modifiers.module import (  # noqa: F401
    GameteModifier,
    GenotypeFilter,
    GlabSelector,
    ZygoteModifier,
    apply_comp_map,
    build_modifier_wrappers,
    evaluate_genotype_filter,
    normalize_zygote_val,
    parse_zygote_key,
    wrap_gamete_modifier,
    wrap_zygote_modifier,
    write_zygote_mapping,
)

__all__ = [
    "apply_comp_map",
    "build_modifier_wrappers",
    "evaluate_genotype_filter",
    "GameteModifier",
    "GenotypeFilter",
    "GlabSelector",
    "normalize_zygote_val",
    "parse_zygote_key",
    "wrap_gamete_modifier",
    "wrap_zygote_modifier",
    "write_zygote_mapping",
    "ZygoteModifier",
]
