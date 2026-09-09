"""State fixtures committed through the owning spatial session.

Hook plans are fixed at build time, so a test-setup state edit goes
through the same native channel as ``reset()``: the stacked session
``set_state`` handoff.  The clock is preserved by passing the container's
current tick.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

import natal as nt


def set_deme_state(population: nt.SpatialPopulation, index: int, payload: Mapping[str, object]) -> None:
    """Commit a one-shot state edit without moving the owning session's clock."""
    assert payload.get("n_tick", population.tick) == population.tick
    ind_all, sperm_all = population._stack_deme_state_arrays()  # noqa: SLF001 — test fixture uses the container's own stacked write path
    ind_all[index][:] = np.asarray(payload["individual_count"], dtype=np.float64)
    if "sperm_storage" in payload:
        sperm_all[index][:] = np.asarray(payload["sperm_storage"], dtype=np.float64)
    population._rust_spatial_backend.set_state(  # noqa: SLF001  # native state handoff, identical to reset()
        ind_all, sperm_all, int(population.tick)
    )
    population._rust_states_dirty = True  # noqa: SLF001  # refresh deme caches from the session
