"""State fixtures committed through the managed deme's public transaction."""
from __future__ import annotations

from typing import Mapping

import numpy as np

import natal as nt


def set_deme_state(population: nt.SpatialPopulation, index: int, payload: Mapping[str, object]) -> None:
    """Commit a one-shot state edit without moving the owning session's clock."""
    assert payload.get("n_tick", population.tick) == population.tick
    applied = False
    def write(ctx: nt.TickContext) -> int:
        nonlocal applied
        if not applied:
            ctx.state.individual_count[:] = np.asarray(payload["individual_count"], dtype=np.float64)
            if "sperm_storage" in payload:
                ctx.state.sperm_storage[:] = np.asarray(payload["sperm_storage"], dtype=np.float64)
            applied = True
        return 0
    population.register_hooks(write, event="first", deme=index)
    population.trigger_event("first", deme_id=index)
    assert applied
