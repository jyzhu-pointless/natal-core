"""Modifier system for population simulations.

This subpackage defines protocols and helper functions for constructing and
wrapping modifiers that alter gamete or zygote production in the simulation.
"""

from natal.modifiers.gamete_conversion import (  # noqa: F401
    GameteAlleleConversionRule,
    GameteConversionRuleSet,
    GameteGlabConversionRule,
    GameteGtypeConversionRule,
    GameteHaploidGenomeConversionRule,  # backward compat alias
)
from natal.modifiers.module import (  # noqa: F401
    GameteModifier,
    GenotypeFilter,
    GlabSelector,
    ZygoteModifier,
    build_modifier_wrappers,
    evaluate_genotype_filter,
    wrap_gamete_modifier,
    wrap_zygote_modifier,
)
from natal.modifiers.zygote_conversion import (  # noqa: F401
    ZygoteAlleleConversionRule,
    ZygoteConversionRuleSet,
    ZygoteGenotypeConversionRule,  # backward compat alias
    ZygoteGlabRedirectRule,
    ZygoteZtypeConversionRule,
)

__all__ = [
    "build_modifier_wrappers",
    "evaluate_genotype_filter",
    "GameteAlleleConversionRule",
    "GameteConversionRuleSet",
    "GameteGlabConversionRule",
    "GameteGtypeConversionRule",
    "GameteHaploidGenomeConversionRule",
    "GameteModifier",
    "GenotypeFilter",
    "GlabSelector",
    "wrap_gamete_modifier",
    "wrap_zygote_modifier",
    "ZygoteAlleleConversionRule",
    "ZygoteConversionRuleSet",
    "ZygoteGenotypeConversionRule",
    "ZygoteGlabRedirectRule",
    "ZygoteModifier",
    "ZygoteZtypeConversionRule",
]
