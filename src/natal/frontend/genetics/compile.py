"""The unified modifier-map compiler (plan 5.1, slice 5).

One spelling of "apply the accumulated modifier recipes to a Mendelian
baseline": :func:`compile_modifier_maps` chains the wrapper callables
over the baseline tables and derives the offspring tensor.  Both
historical rebuild paths — the population-side refresh and the
build-side ``rebuild_config_maps`` — funnel through here, so the two
entry points cannot drift apart (the parity safety net in
``tests/test_compile_unification.py`` pins them bit-for-bit).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from natal.frontend.data._engine import recompute_offspring_tensor

if TYPE_CHECKING:
    from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
    from natal.frontend.population.base import BasePopulation
    from natal.frontend.registry.index import IndexRegistry

    # Modifier list entries are (id, name, callable) triples; gamete and
    # zygote callables have disjoint shapes, so the compiler takes one
    # concrete alias per side (matching build_modifier_wrappers).
    GameteList = list[Tuple[int, Optional[str], GameteModifier]]
    ZygoteList = list[Tuple[int, Optional[str], ZygoteModifier]]

__all__: list[str] = ["compile_modifier_maps"]


def compile_modifier_maps(
    baseline_z2g: NDArray[np.float64],
    baseline_g2z: NDArray[np.float64],
    *,
    gamete_modifiers: GameteList,
    zygote_modifiers: ZygoteList,
    registry: IndexRegistry,
    # Any: BasePopulation is generic over its state container; the
    # compiler only passes the host through to recipe factories.
    population: Optional[BasePopulation[Any]],
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
        population: Optional live population handed to recipe factories
            (runtime refresh passes the population; the build path
            passes ``None`` and uses pre-built wrappers).

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
