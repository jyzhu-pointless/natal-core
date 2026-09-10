"""The unified modifier-map compiler.

One spelling of "apply the accumulated modifier recipes to a Mendelian
baseline": :func:`compile_modifier_maps` chains the wrapper callables
over the baseline tables and derives the offspring tensor.  Both
historical rebuild paths — the population-side refresh and the
build-side ``rebuild_config_maps`` — funnel through here, so the two
entry points cannot drift apart (the parity safety net in
``tests/test_compile_unification.py`` pins them bit-for-bit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics.matrices import recompute_offspring_tensor

if TYPE_CHECKING:
    from natal.frontend.genetics.structures.species import Species
    from natal.frontend.model.draft import ModelDraft
    from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
    from natal.frontend.registry.index import IndexRegistry

    # Modifier list entries are (id, name, callable) triples; gamete and
    # zygote callables have disjoint shapes, so the compiler takes one
    # concrete alias per side (matching build_modifier_wrappers).
    GameteList = list[Tuple[int, Optional[str], GameteModifier]]
    ZygoteList = list[Tuple[int, Optional[str], ZygoteModifier]]

__all__: list[str] = ["RecipeHost", "compile_modifier_maps", "next_modifier_id"]


class RecipeHost(Protocol):
    """Read surface a recipe (preset / rule-set / fitness patch) may inspect.

    ``BasePopulation`` satisfies this protocol structurally, and so does
    the build-side candidate: a ``PopulationBuilder`` mid-compile.  Recipes run
    exactly once against whichever host drives the compilation — there is
    no adapter that impersonates a population.
    """

    @property
    def species(self) -> Species: ...

    @property
    def config(self) -> ModelDraft: ...

    @property
    def registry(self) -> IndexRegistry: ...

    @property
    def index_registry(self) -> IndexRegistry: ...


def project_mendelian_maps(
    species: Species, registry: IndexRegistry,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Project Mendelian tables onto the candidate's exact active axes.

    Args:
        species: Registered genetic structure providing the full baseline.
        registry: Ordered active zygote and gamete types, possibly sparse.

    Returns:
        Isolated meiosis and fertilization arrays aligned to the registry.
    """
    from natal.frontend.builder._registry_builder import build_registry

    full = build_registry(species)
    zindices = {key: index for index, key in enumerate(full.index_to_ztype)}
    gindices = {key: index for index, key in enumerate(full.index_to_gtype)}
    zactive = [zindices[key] for key in registry.index_to_ztype]
    gactive = [gindices[key] for key in registry.index_to_gtype]
    baseline = species.get_config_blueprint()
    meiosis = baseline["zygotes_to_gametes_map"][:, zactive, :][:, :, gactive]
    fertilization = baseline["gametes_to_zygotes_map"][gactive, :, :][:, gactive, :][:, :, zactive]
    return meiosis, fertilization


def next_modifier_id(
    modifiers: GameteList | ZygoteList,
) -> int:
    """Return the next auto-assigned modifier ID for a candidate list.

    Args:
        modifiers: Existing ``(id, name, callable)`` triples.

    Returns:
        ``max(id) + 1``, or ``0`` when the list is empty.
    """
    ids = [mid for mid, _, _ in modifiers]
    return (max(ids) + 1) if ids else 0


def compile_modifier_maps(
    baseline_z2g: NDArray[np.float64],
    baseline_g2z: NDArray[np.float64],
    *,
    gamete_modifiers: GameteList,
    zygote_modifiers: ZygoteList,
    registry: IndexRegistry,
    population: Optional[RecipeHost],
) -> tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """Apply modifier recipes to a baseline and derive the offspring tensor.

    Single owner of the modifier-application order (gamete wrappers in
    list order, then zygote wrappers in list order) and of the offspring
    derivation — both rebuild entry points call this, so the compile
    semantics exist exactly once.

    Args:
        baseline_z2g: Mendelian (or override) meiosis table of shape
            ``(2, n_ztypes, n_gtypes)`` matching the *registry's* active
            axes; the table is copied before any wrapper touches it.
        baseline_g2z: Fusion table of shape
            ``(n_gtypes, n_gtypes, n_ztypes)`` on the same axes.
        gamete_modifiers: (id, name, callable) triples, preset-derived
            first then manual, as accumulated by the caller.
        zygote_modifiers: The zygote-side twin of *gamete_modifiers*.
        registry: The registry whose active axes the tables address.
        population: Optional host (live population or build-side
            candidate) handed to recipe factories; wrappers built from
            pre-compiled callables receive it unchanged.

    Returns:
        ``(z2g, g2z, offspring_tensor)`` — fresh contiguous tables with
        the modifier recipes applied and the derived tensor recomputed.
    """
    from natal.frontend.modifiers.module import build_modifier_wrappers

    gamete_funcs, zygote_funcs = build_modifier_wrappers(
        gamete_modifiers=gamete_modifiers,
        zygote_modifiers=zygote_modifiers,
        population=population,
        registry=registry,
    )

    z2g = np.array(baseline_z2g, dtype=np.float64, copy=True)
    g2z = np.array(baseline_g2z, dtype=np.float64, copy=True)
    for fn in gamete_funcs:
        z2g = fn(z2g)
    for fn in zygote_funcs:
        g2z = fn(g2z)
    z2g = np.ascontiguousarray(z2g)
    g2z = np.ascontiguousarray(g2z)
    offspring = recompute_offspring_tensor(z2g, g2z)
    return z2g, g2z, offspring
