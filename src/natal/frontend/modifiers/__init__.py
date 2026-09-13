"""Modifier system for population simulations.

This subpackage defines protocols and helper functions for constructing and
wrapping modifiers that alter gamete or zygote production in the simulation.
"""

from natal.frontend.modifiers.conversion_rules import (  # noqa: F401
    GameteAlleleConversionRule,
    GameteGtypeConversionRule,
    ZygoteAlleleConversionRule,
    ZygoteZtypeConversionRule,
)
from natal.frontend.modifiers.gamete_conversion import (  # noqa: F401
    GameteConversionRuleSet,
)
from natal.frontend.modifiers.module import (  # noqa: F401
    GameteModifier,
    GenotypeFilter,
    GlabSelector,
    ZygoteModifier,
    build_modifier_wrappers,
    evaluate_genotype_filter,
    wrap_gamete_modifier,
    wrap_zygote_modifier,
)
from natal.frontend.modifiers.zygote_conversion import (  # noqa: F401
    ZygoteConversionRuleSet,
)

__all__ = [
    "build_modifier_wrappers",
    "evaluate_genotype_filter",
    "GameteAlleleConversionRule",
    "GameteConversionRuleSet",
    "GameteGtypeConversionRule",
    "GameteModifier",
    "GenotypeFilter",
    "GlabSelector",
    "wrap_gamete_modifier",
    "wrap_zygote_modifier",
    "ZygoteAlleleConversionRule",
    "ZygoteConversionRuleSet",
    "ZygoteModifier",
    "ZygoteZtypeConversionRule",
]
