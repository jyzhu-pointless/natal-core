"""Compile declarations into isolated, reusable model products.

The compiler is a set of plain functions with explicit inputs: it never
constructs a builder and never writes builder state. Products travel back
to the caller as a transient ``CompiledProducts`` package; the receiving
builder accepts them in one internal operation.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Mapping, NamedTuple, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from natal.frontend.data.config import ModelDraft
    from natal.frontend.data.definition import ModelDefinition
    from natal.frontend.genetics.compile import GameteList, ZygoteList
    from natal.frontend.genetics.structures.species import Species
    from natal.frontend.registry.index import IndexRegistry

FITNESS_FIELDS = (
    "viability_fitness", "fecundity_fitness",
    "sexual_selection_fitness", "zygote_viability_fitness",
)
# Derived genetic products a compile (re)produces; everything else on the
# draft is declaration input that a finalize step must not recompute.
GENETIC_PRODUCT_FIELDS = (
    *FITNESS_FIELDS, "zygotes_to_gametes_map", "gametes_to_zygotes_map", "offspring_tensor",
)


class CompiledProducts(NamedTuple):
    """One completed candidate compile, returned to its accepting builder.

    A transient return package, not long-lived state: the builder that
    accepts it publishes ``config``/``registry``/modifier products as its
    own build state in one internal operation.
    """

    config: ModelDraft
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


def detach_draft(draft: ModelDraft) -> ModelDraft:
    """Copy owned draft arrays and custom slots without touching opaque resources.

    The copy detaches a draft from every previous holder so later in-place
    writes cannot alias; recipes, callables, and other user resources are
    never copied.
    """
    return draft._replace(
        **{name: value.copy() for name, value in draft._asdict().items() if isinstance(value, np.ndarray)},
        custom=deepcopy(draft.custom),
    )


class _CompileHost:
    """Read surface handed to recipes while a candidate compile runs.

    The compile function owns the working draft; this view only exposes
    the :class:`~natal.frontend.genetics.compile.RecipeHost` protocol so
    user recipes cannot tell which host drives the compilation.
    """

    def __init__(
        self, species: Species, registry: IndexRegistry, draft: ModelDraft,
    ) -> None:
        """Bind the host to the candidate's isolated working state."""
        self._species = species
        self._registry = registry
        self.draft = draft  # mutated by the compile loop as steps replace the draft

    @property
    def species(self) -> Species:
        """The species whose architecture the candidate compiles against."""
        return self._species

    @property
    def config(self) -> ModelDraft:
        """The candidate's current working draft."""
        return self.draft

    @property
    def registry(self) -> IndexRegistry:
        """The candidate's index registry."""
        return self._registry

    @property
    def index_registry(self) -> IndexRegistry:
        """Alias of :attr:`registry` (recipe-host protocol member)."""
        return self._registry


def _apply_fitness_step(host: _CompileHost, step: Mapping[str, object]) -> None:
    """Re-apply one declared explicit fitness patch to the working draft.

    Uses the same route-table writer spelling as the ``fitness()`` chain
    method, so a cold rebuild resolves patterns exactly as the original
    declaration did.
    """
    from natal.frontend.configurator._writers import DraftWriter

    writes = {name: value for name, value in step.items() if name != "mode"}
    if not writes:
        return
    mode = cast("str", step.get("mode", "replace"))
    writer = DraftWriter(
        host.draft, on_replace=None, species=host.species, registry=host.registry,
    )
    writer.apply(writes, mode=mode)
    host.draft = writer.draft


def compile_definition(definition: ModelDefinition) -> CompiledProducts:
    """Compile one isolated declaration into its genetic products.

    The recipes expand once against an isolated working copy of the
    declaration's inputs: fitness arrays are re-seeded from the raw
    baseline, explicit fitness patches and preset recipes apply in their
    declared order, manual modifiers are appended, and the inheritance
    maps rebuild from the Mendelian baseline. The declaration and its
    owner are never mutated; a failed compile publishes nothing and
    restores preset species bindings.

    Args:
        definition: The full frozen declaration to compile.

    Returns:
        Fully compiled candidate products, ready for validation and one
        native commit.

    Raises:
        ValueError: If the declaration carries no normalized draft.
    """
    from natal.frontend.fitness import apply_preset_fitness_patch

    draft = definition.draft
    registry = definition.registry
    if draft is None or registry is None:
        raise ValueError("Compilation requires normalized model declarations.")
    species = definition.species
    host = _CompileHost(species, registry, draft)
    for name, base in zip(FITNESS_FIELDS, definition.fitness_base):
        target: NDArray[np.float64] = getattr(host.draft, name)
        target[...] = base if base.shape == target.shape else np.ones_like(target)
    gametes: GameteList = []
    zygotes: ZygoteList = []
    presets = definition.presets
    fitness_steps = definition.fitness_steps
    bindings = [(preset, preset._bound_species) for preset in presets]  # pyright: ignore[reportPrivateUsage]  # restore user bindings if compilation fails.
    try:
        ordered = sorted(presets, key=lambda item: item.priority)
        for position in range(len(ordered) + 1):
            for before, step in fitness_steps:
                if min(before, len(ordered)) == position:
                    _apply_fitness_step(host, step)  # normalized validated fitness keyword arguments.
            if position == len(ordered):
                break
            preset = ordered[position]
            preset.bind_species(species)
            gamete = preset.gamete_modifier(host)
            zygote = preset.zygote_modifier(host)
            if gamete is not None:
                gametes.append((len(gametes), f"{preset.name}/gamete", gamete))
            if zygote is not None:
                zygotes.append((len(zygotes), f"{preset.name}/zygote", zygote))
            patch = preset.fitness_patch()
            if patch:
                apply_preset_fitness_patch(host, patch)
        gametes.extend(cast("GameteList", list(definition.manual_gamete)))
        zygotes.extend(cast("ZygoteList", list(definition.manual_zygote)))
        from natal.frontend.configurator._registry_builder import rebuild_config_maps

        host.draft, _applied = rebuild_config_maps(
            species, host.draft, registry,
            gamete_modifiers=gametes, zygote_modifiers=zygotes,
            compress=False, host=host,
        )  # apply every collected modifier exactly once.
    except BaseException:
        for preset, binding in bindings:
            preset._bound_species = binding  # pyright: ignore[reportPrivateUsage]  # failed compilation publishes no NATAL binding changes.
        raise
    return CompiledProducts(host.draft, registry, gametes, zygotes)
