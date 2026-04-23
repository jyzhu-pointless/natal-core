"""Discrete-generation population model.

This module provides a lightweight non-overlapping generation model that keeps
n_ages=2:
- age 0: offspring/zygotes produced in current tick
- age 1: reproducing adults

The simulation flow remains split as:
first hook -> reproduction -> early hook -> survival -> late hook -> aging
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.base_population import BasePopulation
from natal.genetic_entities import Genotype
from natal.genetic_structures import Species
from natal.population_config import PopulationConfig
from natal.population_state import (
    DiscretePopulationState,
    parse_flattened_discrete_state,
)
from natal.type_def import Sex

if TYPE_CHECKING:
    from natal.index_registry import IndexRegistry
    from natal.population_builder import DiscreteGenerationPopulationBuilder


__all__ = ["DiscreteGenerationPopulation"]


class DiscreteGenerationPopulation(BasePopulation[DiscretePopulationState]):
    """Population with strict non-overlapping generations.

    Maintains exactly two age classes:
    - age 0: newly produced offspring
    - age 1: reproducing adults

    Attributes:
        state (DiscretePopulationState): Current discrete population state.
        config (PopulationConfig): Active normalized configuration with two-age layout.
        history (List[Tuple[int, np.ndarray]]): Flattened snapshots indexed by tick.
    """

    @staticmethod
    def _normalize_config(population_config: PopulationConfig) -> PopulationConfig:
        """Normalize configuration fields required by the discrete model.

        Args:
            population_config: Source configuration to normalize.

        Returns:
            A configuration fixed to two age classes with age-1 adults.
        """
        config_hook_slot = int(getattr(population_config, "hook_slot", 0))
        if config_hook_slot <= 0:
            config_hook_slot = int(population_config.hook_slot)

        return population_config._replace(
            n_ages=2,
            new_adult_age=1,
            adult_ages=np.array([1], dtype=np.int64),
            hook_slot=np.int32(config_hook_slot),
        )

    def __init__(
        self,
        species: Species,
        population_config: PopulationConfig,
        name: Optional[str] = None,
        initial_individual_count: Optional[
            Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]]
        ] = None,
        hooks: Optional[Dict[str, List[Tuple[Any, Optional[str], Optional[int]]]]] = None,
    ):
        if name is None:
            name = "DiscreteGenerationPop"

        super().__init__(species, name, hooks=hooks or {})

        self._config = self._normalize_config(population_config)

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        n_sexes = self._config_nn.n_sexes
        n_genotypes = self._config_nn.n_genotypes
        n_ages = self._config_nn.n_ages

        self._state = DiscretePopulationState.create(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_genotypes=n_genotypes,
            n_tick=0,
            individual_count=np.zeros((n_sexes, n_ages, n_genotypes), dtype=np.float64),
        )

        cfg_init_ind = self._config_nn.get_scaled_initial_individual_count()
        if cfg_init_ind.shape == self._state_nn.individual_count.shape:
            self._state_nn.individual_count[:] = cfg_init_ind

        self._history_shape = (
            1 + n_sexes * n_ages * n_genotypes,
        )

        if initial_individual_count is not None:
            self._state_nn.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        self._initial_population_snapshot = (
            self._state_nn.individual_count.copy(),
            None,
            None,
        )

        self._finalize_hooks()

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        use_continuous_sampling: bool = False,
        use_fixed_egg_count: bool = False,
    ) -> DiscreteGenerationPopulationBuilder:
        """Create and preconfigure a discrete-generation population builder.

        This is a convenience forwarding entry point. Parameter semantics and
        defaults are the same as ``DiscreteGenerationPopulationBuilder.setup``.

        Args:
            species: Species definition used to initialize the builder.
            name: Population name passed through to ``builder.setup``.
            stochastic: Whether to use stochastic sampling. Passed through to ``builder.setup``.
            use_continuous_sampling: If True, use Dirichlet; else Binomial/Multinomial sampling.
                Passed through to ``builder.setup``.
            use_fixed_egg_count: If True, egg count is fixed; if False, Poisson distributed.
                Passed through to ``builder.setup``.

        Returns:
            A configured ``DiscreteGenerationPopulationBuilder`` for fluent chaining.

        Examples:
            ``DiscreteGenerationPopulation.setup(species).initial_state(...).build()``
        """
        from natal.population_builder import DiscreteGenerationPopulationBuilder

        builder = DiscreteGenerationPopulationBuilder(species)
        builder.setup(
            name=name,
            stochastic=stochastic,
            use_continuous_sampling=use_continuous_sampling,
            use_fixed_egg_count=use_fixed_egg_count,
        )
        return builder

    def _resolve_age_distribution(
        self,
        age_data: Union[List[int], Dict[int, int], int, float],
    ) -> Tuple[float, float]:
        """Resolve user-provided initial data into (age0, age1).

        Rules include:
            - scalar x -> (0, x)
            - [x] -> (0, x)
            - [x0, x1] -> (x0, x1)
            - {0: x0, 1: x1} -> (x0, x1), missing keys default to 0
        """
        if isinstance(age_data, (int, float)):
            return 0.0, float(age_data)

        if isinstance(age_data, list):
            if len(age_data) == 0:
                return 0.0, 0.0
            if len(age_data) == 1:
                return 0.0, float(age_data[0])
            if len(age_data) == 2:
                return float(age_data[0]), float(age_data[1])
            raise ValueError(
                f"Discrete initial list must have length <= 2, got {len(age_data)}"
            )

        unsupported_keys = [k for k in age_data.keys() if k not in (0, 1)]
        if unsupported_keys:
            raise ValueError(
                f"Discrete initial dict supports only age keys 0 and 1, got {unsupported_keys}"
            )
        return float(age_data.get(0, 0.0)), float(age_data.get(1, 0.0))

    def _distribute_initial_population(
        self,
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]],
    ) -> None:
        self._state_nn.individual_count.fill(0.0)

        for sex_key, genotype_dist in distribution.items():
            sex_key_norm = sex_key.lower().strip()
            if sex_key_norm == "female":
                sex_idx = int(Sex.FEMALE.value)
            elif sex_key_norm == "male":
                sex_idx = int(Sex.MALE.value)
            else:
                raise ValueError(f"Sex must be 'female' or 'male', got '{sex_key}'")

            for genotype_key, age_data in genotype_dist.items():
                genotype = self._resolve_genotype_key(genotype_key)
                genotype_idx = self._registry_nn.genotype_to_index[genotype]
                age0_count, age1_count = self._resolve_age_distribution(age_data)
                self._state_nn.individual_count[sex_idx, 0, genotype_idx] = age0_count
                self._state_nn.individual_count[sex_idx, 1, genotype_idx] = age1_count

    def run(
        self,
        n_steps: int = 1,
        record_every: Optional[int] = None,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> DiscreteGenerationPopulation:
        """Run multi-step evolution using optimized simulation kernels.

        Args:
            n_steps: Number of steps to evolve.
            record_every: Interval for recording snapshots.
                If None, uses self.record_every. If 0, no snapshots are recorded.
            finish: Whether to mark the population as finished after the run.
            clear_history_on_start: Whether to clear existing history before starting.

        Returns:
            DiscreteGenerationPopulation: Self for chaining.

        Raises:
            RuntimeError: If the population is already finished and cannot continue.
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot run() again after finish=True."
            )

        if record_every is None:
            record_every = self.record_every

        if self.should_use_python_dispatch():
            from natal.hooks.executor import run_discrete_with_hooks

            return cast(
                DiscreteGenerationPopulation,
                run_discrete_with_hooks(
                n_steps=n_steps,
                record_every=record_every,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
                population=self,
            ),
            )

        hooks = self.get_compiled_event_hooks()

        # run_discrete_fn and registry are always initialized by get_compiled_event_hooks().
        assert hooks.run_discrete_fn is not None, "hooks.run_discrete_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        run_fn = cast(
            Callable[..., Tuple[Tuple[NDArray[np.float64], int], Optional[NDArray[np.float64]], bool]],
            hooks.run_discrete_fn,
        )
        registry = hooks.registry

        final_state_tuple, history_new, was_stopped = run_fn(
            state=self._state_nn,
            config=self._config_nn,
            registry=registry,
            n_ticks=n_steps,
            record_interval=record_every,
        )

        self._state = DiscretePopulationState(
            n_tick=int(final_state_tuple[1]),
            individual_count=final_state_tuple[0],
        )
        self._tick = int(final_state_tuple[1])

        self._process_kernel_history(history_new, clear_history_on_start)

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def run_tick(self) -> DiscreteGenerationPopulation:
        """Execute a single simulation tick.

        Overrides BasePopulation.run_tick to use the accelerated run() pipeline
        which correctly handles compiled hooks.
        """
        return self.run(n_steps=1, record_every=self.record_every)

    def reset(self) -> None:
        """Reset the population to its initial state."""
        self._tick = 0
        self._history = []
        self._finished = False
        if hasattr(self, '_initial_population_snapshot'):
            ind_copy, _, _ = self._initial_population_snapshot

            # Recreate state with initial data
            self._state = DiscretePopulationState.create(
                n_sexes=self._config_nn.n_sexes,
                n_ages=self._config_nn.n_ages,
                n_genotypes=self._config_nn.n_genotypes,
                n_tick=0,
                individual_count=ind_copy.copy(),
            )

    def get_total_count(self) -> int:
        return int(round(np.sum(self._state_nn.individual_count)))

    def get_female_count(self) -> int:
        return int(round(np.sum(self._state_nn.individual_count[int(Sex.FEMALE.value)])))

    def get_male_count(self) -> int:
        return int(round(np.sum(self._state_nn.individual_count[int(Sex.MALE.value)])))

    def get_history(self) -> np.ndarray:
        if len(self._history) == 0:
            return np.zeros((0, self._history_shape[0]), dtype=np.float64)
        return np.array([rec[1] for rec in self._history], dtype=np.float64)

    def clear_history(self) -> None:
        self._history.clear()

    def create_history_snapshot(self) -> None:
        flattened = self._state_nn.flatten_all()
        self._history.append((self._tick, flattened.copy()))
        self._enforce_history_limit()

    def export_state(self) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """Export the current state and optional history.

        Returns:
            A tuple of ``(state_flat, history)`` where ``state_flat`` contains
            tick and individual counts, and ``history`` is either ``None`` or a
            stacked history array.
        """
        state_flat = self._state_nn.flatten_all()
        history = self.get_history() if self._history else None
        return state_flat, history

    def export_config(self) -> PopulationConfig:
        """Export the current population configuration.

        Returns:
            The active ``PopulationConfig`` used by the population.
        """
        return self._config_nn

    def import_config(self, config: PopulationConfig) -> None:
        """Import a population configuration into the discrete model.

        Args:
            config: Configuration object to install.
        """
        self._config = self._normalize_config(config)

    def import_state(
        self,
        state: Union[DiscretePopulationState, NDArray[np.float64], Dict[str, np.ndarray]],
        history: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Import state and optional history records.

        Args:
            state: ``DiscretePopulationState``, flattened state array, or a mapping
                containing ``individual_count``.
            history: Optional 2D history array previously returned by ``export_state``.
        """
        assert isinstance(state, (np.ndarray, DiscretePopulationState, dict)), \
            "state must be a DiscretePopulationState, flattened ndarray, or dict"
        if isinstance(state, np.ndarray):
            state_obj = parse_flattened_discrete_state(
                state,
                n_sexes=self._config_nn.n_sexes,
                n_ages=self._config_nn.n_ages,
                n_genotypes=self._config_nn.n_genotypes,
            )
        elif isinstance(state, DiscretePopulationState):
            state_obj = state
        else:
            state_obj = DiscretePopulationState(
                n_tick=int(state.get("n_tick", self._tick)),
                individual_count=np.asarray(state["individual_count"], dtype=np.float64),
            )

        self._state = DiscretePopulationState(
            n_tick=int(state_obj.n_tick),
            individual_count=state_obj.individual_count.copy(),
        )
        self._tick = int(state_obj.n_tick)

        if history is not None and history.shape[0] > 0:
            self.clear_history()
            for row_idx in range(history.shape[0]):
                flat = history[row_idx, :]
                tick = int(flat[0])
                self._history.append((tick, flat.copy()))

    @property
    def _state_nn(self) -> DiscretePopulationState:
        """Non-optional state accessor for subclass internals."""
        return self._require_state()

    @property
    def _config_nn(self) -> PopulationConfig:
        """Non-optional config accessor for subclass internals."""
        return self._require_config()

    @property
    def _registry_nn(self) -> IndexRegistry:
        """Non-optional registry accessor for subclass internals."""
        return self._require_registry()

    def __repr__(self) -> str:
        status = "Finished" if self._finished else "Active"
        return f"<DiscreteGenerationPopulation(name='{self.name}', tick={self.tick}, status={status})>"
