"""Composition-based spatial population container.

`SpatialPopulation` intentionally does NOT inherit from ``BasePopulation``.
Each deme is represented by one concrete ``BasePopulation`` subclass instance.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.spatial_topology import GridTopology, build_adjacency_matrix

__all__ = ["SpatialPopulation"]


class SpatialPopulation:
    """Spatial container composed of per-deme population objects.

    This class models spatial structure via composition: every deme is one
    already-initialized ``BasePopulation`` subclass instance.
    """

    def __init__(
        self,
        demes: Sequence[BasePopulation],
        *,
        topology: Optional[GridTopology] = None,
        adjacency: Optional[NDArray[np.float64]] = None,
        migration_rate: float = 0.0,
        name: str = "SpatialPopulation",
    ) -> None:
        if not demes:
            raise ValueError("demes must contain at least one BasePopulation instance")

        self._demes: List[BasePopulation] = list(demes)
        for idx, deme in enumerate(self._demes):
            if not isinstance(deme, BasePopulation):
                raise TypeError(f"deme[{idx}] must be a BasePopulation subclass instance")

        first_species = self._demes[0].species
        for idx, deme in enumerate(self._demes[1:], start=1):
            if deme.species is not first_species:
                raise ValueError(
                    f"deme[{idx}] species does not match deme[0]; all demes must share the same Species object"
                )

        n_demes = len(self._demes)
        if adjacency is None:
            if topology is None:
                adjacency = np.eye(n_demes, dtype=np.float64)
            else:
                if topology.n_demes != n_demes:
                    raise ValueError(
                        f"topology.n_demes ({topology.n_demes}) must match number of demes ({n_demes})"
                    )
                adjacency = build_adjacency_matrix(topology)
        else:
            adjacency = np.asarray(adjacency, dtype=np.float64)

        if adjacency.shape != (n_demes, n_demes):
            raise ValueError(
                f"adjacency shape mismatch: expected ({n_demes}, {n_demes}), got {adjacency.shape}"
            )

        self._name = name
        self._topology = topology
        self._adjacency = adjacency
        self._migration_rate = float(migration_rate)
        self._tick = int(self._demes[0].tick)

        for idx, deme in enumerate(self._demes[1:], start=1):
            if int(deme.tick) != self._tick:
                raise ValueError(
                    f"deme[{idx}] tick ({deme.tick}) does not match deme[0] tick ({self._tick})"
                )

    @property
    def name(self) -> str:
        return self._name

    @property
    def demes(self) -> Sequence[BasePopulation]:
        return tuple(self._demes)

    @property
    def n_demes(self) -> int:
        return len(self._demes)

    @property
    def species(self):
        return self._demes[0].species

    @property
    def adjacency(self) -> NDArray[np.float64]:
        return self._adjacency

    @property
    def migration_rate(self) -> float:
        return self._migration_rate

    @migration_rate.setter
    def migration_rate(self, value: float) -> None:
        self._migration_rate = float(value)

    def deme(self, idx: int) -> BasePopulation:
        return self._demes[idx]

    @property
    def tick(self) -> int:
        return self._tick

    def _stack_deme_state_arrays(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind_all = np.stack([deme.state.individual_count for deme in self._demes], axis=0)
        
        # Handle potential absence of sperm_storage (e.g. DiscreteGenerationPopulation)
        sperm_list = []
        for deme in self._demes:
            s = getattr(deme.state, "sperm_storage", None)
            if s is None:
                # Create a dummy array if storage is missing
                cfg = deme._config
                s = np.zeros((cfg.n_ages, cfg.n_genotypes, cfg.n_genotypes), dtype=np.float64)
            sperm_list.append(s)
        
        sperm_all = np.stack(sperm_list, axis=0)
        return ind_all, sperm_all

    def _apply_stacked_state(self, ind_all: NDArray[np.float64], sperm_all: NDArray[np.float64], tick: int) -> None:
        for deme_id, deme in enumerate(self._demes):
            new_fields = {
                "n_tick": int(tick),
                "individual_count": ind_all[deme_id],
            }
            if hasattr(deme.state, "sperm_storage"):
                new_fields["sperm_storage"] = sperm_all[deme_id]
            
            deme._state = deme.state._replace(**new_fields)  # type: ignore[attr-defined]
            deme.tick = int(tick)
        self._tick = int(tick)

    def _shared_config(self) -> Any:
        """Return one shared config for spatial kernels.

        Current spatial kernel wrappers expect one config object for all demes.
        """
        export_fn = getattr(self._demes[0], "export_config", None)
        if not callable(export_fn):
            raise TypeError("deme[0] does not implement export_config()")
        cfg = export_fn()
        for idx, deme in enumerate(self._demes[1:], start=1):
            deme_export = getattr(deme, "export_config", None)
            if not callable(deme_export):
                raise TypeError(f"deme[{idx}] does not implement export_config()")
            if deme_export() is not cfg:
                raise ValueError(
                    f"deme[{idx}] uses a different config object; current spatial runner requires a shared config"
                )
        return cfg

    def run_tick(self) -> "SpatialPopulation":
        """Run one spatial tick via generated spatial wrapper."""
        for idx, deme in enumerate(self._demes):
            if getattr(deme, "_finished", False):
                raise RuntimeError(f"deme[{idx}] has finished; cannot run spatial tick")

        hooks = self._demes[0].get_compiled_event_hooks()
        assert hooks.run_spatial_tick_fn is not None, "hooks.run_spatial_tick_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        run_tick_fn = hooks.run_spatial_tick_fn
        registry = hooks.registry
        config = self._shared_config()
        has_sperm = hasattr(self._demes[0].state, "sperm_storage")
        ind_all, sperm_all = self._stack_deme_state_arrays()

        final_state_tuple, result = run_tick_fn(
            ind_all,
            sperm_all,
            config, # Note: wrapper template needs updating to handle has_sperm
            registry,
            int(self._tick),
            self._adjacency,
            float(self._migration_rate),
        )

        self._apply_stacked_state(final_state_tuple[0], final_state_tuple[1], int(final_state_tuple[2]))

        if int(result) != 0:
            for deme in self._demes:
                deme._finished = True  # type: ignore[attr-defined]
                deme.trigger_event("finish")
        return self

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        finish: bool = False,
    ) -> "SpatialPopulation":
        """Run multiple spatial ticks via generated spatial wrapper."""
        if n_steps < 0:
            raise ValueError("n_steps must be >= 0")

        for idx, deme in enumerate(self._demes):
            if getattr(deme, "_finished", False):
                raise RuntimeError(f"deme[{idx}] has finished; cannot run spatial simulation")

        hooks = self._demes[0].get_compiled_event_hooks()
        assert hooks.run_spatial_fn is not None, "hooks.run_spatial_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        run_fn = hooks.run_spatial_fn
        registry = hooks.registry
        config = self._shared_config()
        ind_all, sperm_all = self._stack_deme_state_arrays()

        final_state_tuple, _history, was_stopped = run_fn(
            ind_all,
            sperm_all,
            config,
            registry,
            int(self._tick),
            int(n_steps),
            self._adjacency,
            float(self._migration_rate),
            int(record_every),
        )

        self._apply_stacked_state(final_state_tuple[0], final_state_tuple[1], int(final_state_tuple[2]))

        if bool(was_stopped):
            for deme in self._demes:
                deme._finished = True  # type: ignore[attr-defined]
                deme.trigger_event("finish")
        elif finish:
            for deme in self._demes:
                deme.finish_simulation()

        return self
