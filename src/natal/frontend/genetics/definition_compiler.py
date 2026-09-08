"""Compile normalized declarations into isolated, reusable model products."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from natal.frontend.data.config import ModelDraft
    from natal.frontend.data.definition import ModelDefinition, SpatialInputs
    from natal.frontend.genetics.compile import GameteList, ZygoteList
    from natal.frontend.patterns import IndividualSelector
    from natal.frontend.presets import GeneticPreset
    from natal.frontend.registry.index import IndexRegistry

FITNESS_FIELDS = (
    "viability_fitness", "fecundity_fitness",
    "sexual_selection_fitness", "zygote_viability_fitness",
)


@dataclass(frozen=True)
class NormalizedModel:
    """Normalized inputs, separate from the cached products of genetic recipes.

    Attributes:
        settings: Ecological declarations, initial state, switches, and shapes.
        registry: Explicit active type layout.
        presets: Registered user recipes; opaque resources remain caller-owned.
        manual_gamete: Explicit gamete modifier declarations.
        manual_zygote: Explicit zygote modifier declarations.
        fitness_base: Initial fitness arrays, before any user recipe or patch.
        fitness_steps: Ordered explicit fitness patches, including their modes.
        compilation_key: Identity token connecting a declaration to its products.
        hook_calls: Event registrations and their default dispatch options.
        observation_groups: Named selectors projected by the observation compiler.
        observation_collapse_age: Whether projections sum the age axis.
        history_mode: Raw-state or observation recording policy.
        history_max_rows: Retention bound, or the population default.
        compress: Whether to prune unreachable active genetic types.
        declared_zygote_types: Explicit types retained during pruning.
    """

    settings: ModelDraft
    registry: IndexRegistry
    presets: tuple[GeneticPreset, ...]
    manual_gamete: tuple[tuple[int, str | None, object], ...]
    manual_zygote: tuple[tuple[int, str | None, object], ...]
    fitness_base: tuple[NDArray[np.float64], ...]
    fitness_steps: tuple[tuple[int, dict[str, object]], ...]
    compilation_key: object
    hook_calls: tuple[tuple[tuple[object, ...], dict[str, object]], ...] = ()
    observation_groups: Mapping[str, IndividualSelector] | None = None
    observation_collapse_age: bool = False
    history_mode: Literal["raw", "observation"] = "raw"
    history_max_rows: int | None = None
    compress: bool = False
    declared_zygote_types: frozenset[str] | frozenset[int] | None = None
    spatial: SpatialInputs | None = None


@dataclass(frozen=True)
class CompiledModel:
    """One completed candidate; its products are reused when build finalizes it."""

    config: ModelDraft
    compilation_key: object
    registry: IndexRegistry
    gamete_modifiers: GameteList
    zygote_modifiers: ZygoteList


def copy_registry(registry: IndexRegistry) -> IndexRegistry:
    """Copy active index containers while preserving interned genetic identities."""
    from natal.frontend.registry.index import IndexRegistry

    result = IndexRegistry()
    result.slab_labels = list(registry.slab_labels)
    result.glab_labels = list(registry.glab_labels)
    for genotype, label in registry.index_to_ztype:
        result.register_ztype(genotype, label)
    for haplotype, label in registry.index_to_gtype:
        result.register_gtype(haplotype, label)
    return result


def snapshot_inputs(inputs: NormalizedModel) -> NormalizedModel:
    """Detach NATAL-owned inputs without copying user recipe resources."""
    from natal.frontend.data.definition import snapshot_spatial_inputs

    return NormalizedModel(
        inputs.settings._replace(
            **{name: value.copy() for name, value in inputs.settings._asdict().items() if isinstance(value, np.ndarray)},
            custom=deepcopy(inputs.settings.custom),
        ), copy_registry(inputs.registry), inputs.presets,
        inputs.manual_gamete, inputs.manual_zygote,
        tuple(array.copy() for array in inputs.fitness_base),
        deepcopy(inputs.fitness_steps), inputs.compilation_key,
        tuple((tuple(items), dict(options)) for items, options in inputs.hook_calls),
        None if inputs.observation_groups is None else dict(inputs.observation_groups),
        inputs.observation_collapse_age, inputs.history_mode, inputs.history_max_rows,
        inputs.compress, inputs.declared_zygote_types,
        None if inputs.spatial is None else snapshot_spatial_inputs(inputs.spatial),
    )


def compile_definition(
    definition: ModelDefinition, *, cached: CompiledModel | None = None,
) -> CompiledModel:
    """Compile one isolated declaration, or finalize its already compiled products.

    Args:
        definition: Normalized model inputs and the ordered user declarations.
        cached: Products from the same prepared declaration. Build uses this
            cache so finalization never repeats opaque recipe execution.

    Returns:
        Fully compiled candidate data, ready for validation and one native commit.
    """
    from typing import cast

    from natal.frontend.configurator import Configurator

    inputs = definition.normalized
    if inputs is None:
        raise ValueError("Compilation requires normalized model declarations.")
    if cached is not None:
        if cached.compilation_key is not inputs.compilation_key:
            raise ValueError("Compiled products belong to a different declaration.")
        genetic_fields = (*FITNESS_FIELDS, "zygotes_to_gametes_map", "gametes_to_zygotes_map", "offspring_tensor")
        config = inputs.settings._replace(**{name: getattr(cached.config, name).copy() for name in genetic_fields})
        return CompiledModel(
            config, inputs.compilation_key, inputs.registry,
            list(cached.gamete_modifiers), list(cached.zygote_modifiers),
        )
    candidate = Configurator(inputs.settings, species=definition.species)
    candidate._registry = inputs.registry  # pyright: ignore[reportPrivateUsage]  # the compiler owns this isolated candidate.
    for name, base in zip(FITNESS_FIELDS, inputs.fitness_base):
        target: NDArray[np.float64] = getattr(candidate.config, name)
        target[...] = base if base.shape == target.shape else np.ones_like(target)
    gametes: GameteList = []
    zygotes: ZygoteList = []
    bindings = [(preset, preset._bound_species) for preset in inputs.presets]  # pyright: ignore[reportPrivateUsage]  # restore user bindings if compilation fails.
    try:
        from natal.frontend.fitness import apply_preset_fitness_patch

        ordered = sorted(inputs.presets, key=lambda item: item.priority)
        for position in range(len(ordered) + 1):
            for before, step in inputs.fitness_steps:
                if min(before, len(ordered)) == position:
                    candidate.fitness(**step)  # pyright: ignore[reportArgumentType]  # normalized validated fitness keyword arguments.
            if position == len(ordered):
                break
            preset = ordered[position]
            preset.bind_species(definition.species)
            gamete = preset.gamete_modifier(candidate)
            zygote = preset.zygote_modifier(candidate)
            if gamete is not None:
                gametes.append((len(gametes), f"{preset.name}/gamete", gamete))
            if zygote is not None:
                zygotes.append((len(zygotes), f"{preset.name}/zygote", zygote))
            patch = preset.fitness_patch()
            if patch:
                apply_preset_fitness_patch(candidate, patch)
        gametes.extend(cast("GameteList", list(inputs.manual_gamete)))
        zygotes.extend(cast("ZygoteList", list(inputs.manual_zygote)))
        candidate._compile_candidate_maps(gametes, zygotes)  # pyright: ignore[reportPrivateUsage]  # apply every collected modifier exactly once.
    except BaseException:
        for preset, binding in bindings:
            preset._bound_species = binding  # pyright: ignore[reportPrivateUsage]  # failed compilation publishes no NATAL binding changes.
        raise
    return CompiledModel(candidate.config, inputs.compilation_key, inputs.registry, gametes, zygotes)
