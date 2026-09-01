"""Fitness patch construction and application — backward-compat shim.

Imported from :mod:`natal.frontend.fitness._patch` where the canonical
implementation lives.  This shim will be removed in a future clean-up.
"""

# pyright: reportPrivateUsage=false, reportUnusedImport=false

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

# make_fitness_patch_given_allele_scaling stays here — it's a
# pure dict-construction helper with no population dependency.
from natal.frontend.presets._types import PresetFitnessPatch
from natal.frontend.presets._types import _AlleleScalingMode as _AlleleScalingMode
from natal.frontend.presets._types import (
    _FecundityScalingConfig as _FecundityScalingConfig,
)
from natal.frontend.presets._types import (
    _SexualSelectionScalingConfig as _SexualSelectionScalingConfig,
)
from natal.frontend.presets._types import (
    _ViabilityScalingConfig as _ViabilityScalingConfig,
)
from natal.frontend.presets._types import (
    _ZygoteViabilityScalingConfig as _ZygoteViabilityScalingConfig,
)

# The ``fitness._patch`` re-exports below are deferred through module-level
# ``__getattr__``: ``_patch`` imports type helpers from ``presets._types`` at
# module level, so an eager import here would make ``presets/__init__`` fail
# whenever ``fitness._patch`` is imported first (partially-initialized module).
# ``make_fitness_patch_given_allele_scaling`` above does NOT depend on
# ``_patch`` and stays eagerly defined.
_PATCH_REEXPORTS = frozenset({
    "_apply_fecundity_allele_scaling",
    "_apply_fecundity_slab_scaling",
    "_apply_sexual_selection_allele_scaling",
    "_apply_sexual_selection_slab_scaling",
    "_apply_viability_allele_scaling",
    "_apply_viability_slab_scaling",
    "_apply_zygote_slab_scaling",
    "_apply_zygote_viability_allele_scaling",
    "apply_preset_fitness_patch",
})

if TYPE_CHECKING:
    # Static-only import: precise signatures for type checkers; the runtime
    # export stays deferred through ``__getattr__`` (cycle-safe).
    pass


def __getattr__(name: str) -> object:
    """Lazily re-export the canonical implementations from ``fitness._patch``."""
    if name in _PATCH_REEXPORTS:
        from natal.frontend.fitness import _patch as _patch_module

        return getattr(_patch_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_fitness_patch_given_allele_scaling(
    allele_name: Union[str, List[str], Tuple[str, ...]],
    viability_scaling: Optional[_ViabilityScalingConfig] = None,
    fecundity_scaling: Optional[_FecundityScalingConfig] = None,
    sexual_selection_scaling: Optional[_SexualSelectionScalingConfig] = None,
    zygote_viability_scaling: Optional[_ZygoteViabilityScalingConfig] = None,
    viability_mode: _AlleleScalingMode = "multiplicative",
    fecundity_mode: _AlleleScalingMode = "multiplicative",
    sexual_selection_mode: str = "multiplicative",
    zygote_viability_mode: _AlleleScalingMode = "multiplicative",
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
        zygote_viability_scaling: Zygote fitness scaling configuration.
        viability_mode: Scaling mode for viability fitness.
        fecundity_mode: Scaling mode for fecundity fitness.
        sexual_selection_mode: Scaling mode for sexual selection.
        zygote_viability_mode: Scaling mode for zygote fitness.

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

    if zygote_viability_scaling is not None:
        patch['zygote_per_allele'] = {key: (zygote_viability_scaling, zygote_viability_mode)}

    return patch
