"""The unified modifier-map compiler.

One spelling of "apply the accumulated modifier recipes to a Mendelian
baseline": :func:`compile_modifier_maps` chains the wrapper callables
over complete baseline tables. Publication derives the offspring tensor
only after the runtime axes have been selected.  Both
historical rebuild paths — the population-side refresh and the
build-side ``rebuild_config_maps`` — funnel through here, so the two
entry points cannot drift apart (the parity safety net in
``tests/test_compile_unification.py`` pins them bit-for-bit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray

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
    """Acquire isolated Mendelian tables for a complete unpublished catalog.

    Args:
        species: Registered genetic structure providing the full baseline.
        registry: Complete zygote and gamete catalog in species order.

    Returns:
        Isolated meiosis and fertilization arrays aligned to the registry.

    Raises:
        ValueError: If the registry is published, incomplete, or reordered.
    """
    from natal.frontend.builder._registry_builder import build_registry

    # A published registry has already pruned axes, so the full species
    # baseline no longer lines up with it.
    if registry.published:
        raise ValueError("Mendelian projection requires an unpublished full registry.")
    full = build_registry(species)
    # Tables are addressed positionally: any missing or reordered ztype/gtype
    # entry makes the baseline arrays address the wrong biology.
    if (registry.index_to_ztype != full.index_to_ztype
            or registry.index_to_gtype != full.index_to_gtype):
        raise ValueError("Mendelian projection requires the complete species registry.")
    baseline = species.get_config_blueprint()
    # copy=True detaches the result from the species' cached blueprint so
    # downstream modifiers may mutate it in place. Order is (meiosis = z2g,
    # fusion = g2z), matching compile_modifier_maps' parameter order.
    return (
        np.array(baseline["zygotes_to_gametes_map"], dtype=np.float64, copy=True),
        np.array(baseline["gametes_to_zygotes_map"], dtype=np.float64, copy=True),
    )


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
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply modifier recipes to a baseline.

    The compiler owns modifier application order (gamete wrappers in list
    order, then zygote wrappers in list order). Offspring derivation belongs
    to the runtime consumer and is deliberately absent here.

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
        population: Optional unpublished compilation host handed to
            recipe factories; wrappers built from
            pre-compiled callables receive it unchanged.

    Returns:
        ``(z2g, g2z)`` — fresh contiguous tables with modifier recipes applied.
    """
    from natal.frontend.modifiers.module import build_modifier_wrappers

    # Resolve the declarations into tensor-level callables once; the host is
    # the only population-like surface recipes may read.
    gamete_funcs, zygote_funcs = build_modifier_wrappers(
        gamete_modifiers=gamete_modifiers,
        zygote_modifiers=zygote_modifiers,
        population=population,
        registry=registry,
    )

    # Copy both baselines: wrappers may mutate in place, and the caller's
    # (often cached) arrays must stay untouched.
    z2g = np.array(baseline_z2g, dtype=np.float64, copy=True)
    g2z = np.array(baseline_g2z, dtype=np.float64, copy=True)
    # Application order is part of the contract: every gamete wrapper in list
    # order first, then every zygote wrapper. Modifiers need not commute.
    for fn in gamete_funcs:
        z2g = fn(z2g)
    for fn in zygote_funcs:
        g2z = fn(g2z)
    # Rust consumers read C-contiguous buffers; a wrapper may hand back a view.
    z2g = np.ascontiguousarray(z2g)
    g2z = np.ascontiguousarray(g2z)
    return z2g, g2z
