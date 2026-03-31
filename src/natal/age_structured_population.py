"""Age-structured population models.

This module implements age-structured (overlapping generation) population
models and utilities for survival, reproduction, juvenile recruitment, and
fitness management.

Attributes:
    __all__ (list[str]): List of public symbols exported by this module.

.. _Google Python Style Guide:
   http://google.github.io/styleguide/pyguide.html
"""

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

import natal.kernels.simulation_kernels as sk
from natal.base_population import BasePopulation, HookRegistrationMap
from natal.genetic_entities import Genotype
from natal.genetic_structures import Species
from natal.index_registry import IndexRegistry
from natal.population_config import PopulationConfig
from natal.population_state import PopulationState
from natal.type_def import Sex

if TYPE_CHECKING:
    from natal.population_builder import AgeStructuredPopulationBuilder

__all__ = ["AgeStructuredPopulation"]

# Type alias for hooks
HookCallback = Callable[..., object]
# =============================================================================
# Age-structured population model (based on BasePopulation)
# =============================================================================

class AgeStructuredPopulation(BasePopulation[PopulationState]):
    """Age-structured population model (overlapping generations).

    An age-structured population built on ``BasePopulation`` and
    ``PopulationState``. Supports age-dependent survival and fecundity,
    juvenile recruitment modes, optional sperm-storage mechanics, and a
    hook/modifier system for user extensions.

    Attributes:
        snapshots (dict): Storage for custom state snapshots.
    """

    def __init__(
        self,
        species: Species,
        population_config: PopulationConfig,
        name: Optional[str] = None,
        initial_individual_count: Optional[Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]]] = None,
        initial_sperm_storage: Optional[Dict[Union[Genotype, str], Dict[Union[Genotype, str], Union[Dict[int, float], List[float], float]]]] = None,
        hooks: Optional[HookRegistrationMap] = None,
    ):
        """Initialize an age-structured population instance using a PopulationConfig.

        Args:
            species: Species object describing genetic architecture.
            population_config: Fully initialized PopulationConfig instance.
            name: Human-readable population name. If None, uses "AgeStructuredPop".
            initial_individual_count: Initial population distribution.
                Format: {sex: {genotype: counts_by_age}}
            initial_sperm_storage: Initial sperm storage state (if supported).
            hooks: Event hook registrations to apply.

        Examples:
            >>> pop_config = PopulationConfigBuilder.build(species, ...)
            >>> pop = AgeStructuredPopulation(
            ...     species,
            ...     pop_config,
            ...     name="MyPop",
            ...     initial_individual_count={...}
            ... )
        """
        if name is None:
            name = "AgeStructuredPop"

        hooks_map: HookRegistrationMap = hooks or {}
        super().__init__(species, name, hooks=hooks_map)

        config_hook_slot = int(getattr(population_config, "hook_slot", 0))
        if config_hook_slot <= 0:
            config_hook_slot = self.hook_slot
        self._config = population_config._replace(hook_slot=np.int32(config_hook_slot))

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        self._state = PopulationState.create(
            n_genotypes=population_config.n_genotypes,
            n_sexes=population_config.n_sexes,
            n_ages=population_config.n_ages,
        )

        # Initialize from builder-injected config arrays if available.
        cfg_init_ind = population_config.get_scaled_initial_individual_count()
        if cfg_init_ind.shape == self._state_nn.individual_count.shape:
            self._state_nn.individual_count[:] = cfg_init_ind
        cfg_init_sperm = population_config.get_scaled_initial_sperm_storage()
        if cfg_init_sperm.shape == self._state_nn.sperm_storage.shape:
            self._state_nn.sperm_storage[:] = cfg_init_sperm

        self.snapshots = {}

        if initial_individual_count is not None:
            self._state_nn.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        if initial_sperm_storage is not None:
            # TODO: add population_config.use_sperm_storage
            self._distribute_initial_sperm_storage(species, initial_sperm_storage)

        self._initial_population_snapshot = (
            self._state_nn.individual_count.copy(),
            self._state_nn.sperm_storage.copy(),
            None,
        )

        self._initialize_registry()
        self._finalize_hooks()

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        use_dirichlet_sampling: bool = False,
        gamete_labels: Optional[List[str]] = None,
        use_fixed_egg_count: bool = False,
    ) -> 'AgeStructuredPopulationBuilder':
        """Create and preconfigure an age-structured population builder.

        Args:
            species: Species definition used to initialize the builder.
            name: Population name.
            stochastic: Whether to use stochastic sampling.
            use_dirichlet_sampling: Whether to use Dirichlet sampling.
            gamete_labels: Optional labels for gamete tracking.
            use_fixed_egg_count: Whether egg count is deterministic.

        Returns:
            A configured ``AgeStructuredPopulationBuilder`` for fluent chaining.

        Examples:
            ``AgeStructuredPopulation.setup(species).age_structure(...).initial_state(...).build()``
        """
        from natal.population_builder import AgeStructuredPopulationBuilder
        builder = AgeStructuredPopulationBuilder(species)
        builder.setup(
            name=name,
            stochastic=stochastic,
            use_dirichlet_sampling=use_dirichlet_sampling,
            use_fixed_egg_count=use_fixed_egg_count
        )
        return builder

    def _distribute_initial_population(
        self,
        distribution: Mapping[str, Mapping[Union[Genotype, str], object]]
    ) -> None:
        """Distribute initial population from a specification dictionary.

        Args:
            distribution: Format {sex: {genotype: age_counts}}
                where age_counts can be a list or dict of age -> count.

        Raises:
            ValueError: If sex key is invalid.
            TypeError: If age data is not a list or dict.
        """
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

                if isinstance(age_data, list):
                    for age, raw_count in enumerate(cast(List[object], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Age count must be numeric, got {type(raw_count)}")
                        count = float(raw_count)
                        if age < self._config_nn.n_ages and count > 0:
                            self._state_nn.individual_count[sex_idx, age, genotype_idx] = count
                elif isinstance(age_data, dict):
                    for age_raw, raw_count in cast(Dict[object, object], age_data).items():
                        if not isinstance(age_raw, int):
                            raise TypeError(f"Age key must be int, got {type(age_raw)}")
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Age count must be numeric, got {type(raw_count)}")
                        age = age_raw
                        count = float(raw_count)
                        if age < self._config_nn.n_ages and count > 0:
                            self._state_nn.individual_count[sex_idx, age, genotype_idx] = count
                else:
                    raise TypeError(f"age_data must be a list or dict, got {type(age_data)}")

    def _distribute_initial_sperm_storage(
        self,
        species: Species,
        sperm_storage_dist: Mapping[Any, Mapping[Any, object]]
    ) -> None:
        """Populate the internal sperm storage from user-provided initial distribution.

        Note:
            Supported formats for age_data (innermost value):
            - Dict[int, float]: Sparse mapping {age: count, ...}
            - List[float]: Dense list [count_age0, count_age1, ...]
            - float/int: Scalar value applied to all adult ages (>= new_adult_age)

        Args:
            species: Species object for genotype parsing.
            sperm_storage_dist: Mapping of {female_genotype: {male_genotype: age_data}}.

        Raises:
            TypeError: If genotype keys or age data have incorrect types.
            ValueError: If sperm counts or ages are out of range.
        """
        self._state_nn.sperm_storage.fill(0.0)
        for female_key, male_dict in sperm_storage_dist.items():
            if isinstance(female_key, str):
                female_genotype = species.get_genotype_from_str(female_key)
            elif isinstance(female_key, Genotype):
                female_genotype = female_key
            else:
                raise TypeError(f"Female genotype key must be Genotype or str, got {type(female_key)}")

            female_idx = self._registry_nn.genotype_to_index[female_genotype]

            for male_key, age_data in male_dict.items():
                # Parse male genotype
                if isinstance(male_key, str):
                    male_genotype = species.get_genotype_from_str(male_key)
                elif isinstance(male_key, Genotype):
                    male_genotype = male_key
                else:
                    raise TypeError(f"Male genotype key must be Genotype or str, got {type(male_key)}")

                male_idx = self._registry_nn.genotype_to_index[male_genotype]

                # Parse age_data: supports multiple formats
                if isinstance(age_data, dict):
                    # Dict format: {age: count, ...}
                    for age_raw, raw_count in cast(Dict[object, object], age_data).items():
                        if not isinstance(age_raw, int):
                            raise TypeError(f"Age must be int, got {type(age_raw)}")
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Sperm count must be numeric, got {type(raw_count)}")
                        age = age_raw
                        count = float(raw_count)
                        if age < 0 or age >= self.n_ages:
                            raise ValueError(f"Age {age} out of range [0, {self.n_ages})")
                        if count < 0:
                            raise ValueError(f"Sperm count must be non-negative, got {count}")
                        if count > 0:
                            self._state_nn.sperm_storage[age, female_idx, male_idx] = count

                elif isinstance(age_data, list):
                    # List format: [count_age0, count_age1, ...]
                    for age, raw_count in enumerate(cast(List[object], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Sperm count must be numeric, got {type(raw_count)}")
                        count = float(raw_count)
                        if age >= self.n_ages:
                            break
                        if count < 0:
                            raise ValueError(f"Sperm count must be non-negative, got {count}")
                        if count > 0:
                            self._state_nn.sperm_storage[age, female_idx, male_idx] = count

                elif isinstance(age_data, tuple):
                    # Tuple format: (count_age0, count_age1, ...)
                    for age, raw_count in enumerate(cast(Tuple[object, ...], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Sperm count must be numeric, got {type(raw_count)}")
                        count = float(raw_count)
                        if age >= self.n_ages:
                            break
                        if count < 0:
                            raise ValueError(f"Sperm count must be non-negative, got {count}")
                        if count > 0:
                            self._state_nn.sperm_storage[age, female_idx, male_idx] = count

                elif isinstance(age_data, (int, float)) and not isinstance(age_data, bool):
                    # Scalar format: apply to all adult ages
                    if age_data < 0:
                        raise ValueError(f"Sperm count must be non-negative, got {age_data}")
                    if age_data > 0:
                        for age in range(self.new_adult_age, self.n_ages):
                            self._state_nn.sperm_storage[age, female_idx, male_idx] = float(age_data)
                else:
                    raise TypeError(f"Age data must be Dict, List, or numeric scalar, got {type(age_data)}")

    @property
    def state(self) -> PopulationState:
        """PopulationState: The current state container for the population."""
        return self._state_nn

    @property
    def _state_nn(self) -> PopulationState:
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

    def reset(self) -> None:
        """Reset the population to its initial state.

        Restores individual counts and sperm storage to original values.
        """
        self._tick = 0
        self._history = []
        self._finished = False
        if hasattr(self, '_initial_population_snapshot'):
            ind_copy, sperm_copy, _ = self._initial_population_snapshot

            self._state = PopulationState.create(
                n_genotypes=self._config_nn.n_genotypes,
                n_sexes=self._config_nn.n_sexes,
                n_ages=self._config_nn.n_ages,
                n_tick=0,
                individual_count=ind_copy.copy(),
                sperm_storage=sperm_copy.copy(),
            )

    @property
    def n_ages(self) -> int:
        """int: Number of age classes in this population."""
        return self._config_nn.n_ages

    @property
    def new_adult_age(self) -> int:
        """int: Minimum age at which individuals are considered adults."""
        return self._config_nn.new_adult_age

    def get_total_count(self) -> int:
        """Return the total number of individuals in the population.

        Returns:
            float: Grand total across all sexes, ages, and genotypes.
        """
        return self._state_nn.individual_count.sum()

    def get_female_count(self) -> int:
        """Return the total number of female individuals.

        Returns:
            float: Sum of all female individual counts.
        """
        return self._state_nn.individual_count[Sex.FEMALE.value, :, :].sum()

    def get_male_count(self) -> int:
        """Return the total number of male individuals.

        Returns:
            float: Sum of all male individual counts.
        """
        return self._state_nn.individual_count[Sex.MALE.value, :, :].sum()

    def get_adult_count(self, sex: str = 'both') -> int:
        """Return the number of adult individuals for the given sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'`` (aliases accepted).

        Returns:
            float: Total number of adults for the requested sex(es).

        Raises:
            ValueError: If the sex identifier is not recognized.
        """
        if sex not in ('female', 'male', 'both', 'F', 'M'):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")

        total = 0

        if sex in ('female', 'F', 'both'):
            total += self._state_nn.individual_count[Sex.FEMALE.value, self.new_adult_age:self.n_ages, :].sum()

        if sex in ('male', 'M', 'both'):
            total += self._state_nn.individual_count[Sex.MALE.value, self.new_adult_age:self.n_ages, :].sum()

        return int(total)


    def _get_fecundity(self, genotype: Genotype, sex: Sex) -> float:
        """Internal helper: return fecundity for a genotype and sex.

        Args:
            genotype: Target genotype.
            sex: Target sex.

        Returns:
            float: The fecundity fitness value.
        """
        genotype_idx = self._registry_nn.genotype_to_index[genotype]
        sex_idx = int(sex.value)
        return self._config_nn.fecundity_fitness[sex_idx, genotype_idx]

    def _get_sexual_preference(self, female_genotype: Genotype, male_genotype: Genotype) -> float:
        """Internal helper: return sexual preference value for a genotype pair.

        Args:
            female_genotype: Genotype of the female.
            male_genotype: Genotype of the male.

        Returns:
            float: The sexual selection fitness weight.
        """
        f_idx = self._registry_nn.genotype_to_index[female_genotype]
        m_idx = self._registry_nn.genotype_to_index[male_genotype]
        return self._config_nn.sexual_selection_fitness[f_idx, m_idx]

    # ========================================================================
    # State export/import (simulation_kernels interface)
    # ========================================================================

    def export_config(self) -> 'PopulationConfig':
        """Export population configuration to Config jitclass.

        Returns:
            PopulationConfig: A copy of the current population configuration.
        """
        return self._config_nn

    def import_config(self, config: 'PopulationConfig') -> None:
        """Import configuration into the population.

        Args:
            config: Config jitclass instance.
        """
        # Configuration is usually read-only (used by run_tick),
        # kept here for completeness.
        self._config = config

    def create_history_snapshot(self) -> None:
        """Record current state to history records.

        Saves the current tick and a flattened copy of state to _history.
        """
        flattened = self._state_nn.flatten_all()
        self._history.append((self._tick, flattened.copy()))

    def get_history(self) -> np.ndarray:
        """Return history records as a 2D NumPy array.

        Returns:
            np.ndarray: Float64 array with shape
                ``(n_snapshots, 1 + n_sexes*n_ages*n_genotypes + n_ages*n_genotypes^2)``,
                where each row is a flattened snapshot state.

        Raises:
            ValueError: If no history is recorded.
        """
        if len(self._history) == 0:
            raise ValueError("No history recorded")

        # Stack flattened data of all snapshots
        flat_array = np.array([rec[1] for rec in self._history], dtype=np.float64)
        return flat_array

    def clear_history(self) -> None:
        """Clear history records."""
        self._history.clear()

    def export_state(self) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """Export population state as a flattened array.

        Returns:
            Tuple[NDArray, Optional[NDArray]]: (state_flat, history).
                state_flat: [n_tick, ind_count.ravel(), sperm_storage.ravel()]
                history: Optional array of shape (n_snapshots, flatten_size).
        """
        state_flat = self._state_nn.flatten_all()
        history = self.get_history() if self._history else None
        return state_flat, history

    def import_state(self, state: Union['PopulationState', NDArray[np.float64], Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]],
                     history: Optional[np.ndarray] = None) -> None:
        """Import state and optional history records.

        Args:
            state: Flattened array, PopulationState object, or data dictionary.
            history: Optional history 2D array.
        """
        from natal.population_state import PopulationState, parse_flattened_state

        if isinstance(state, np.ndarray):
            # Reconstruct state from flattened array
            n_sexes, n_ages, n_genotypes = self._state_nn.individual_count.shape
            state_obj = parse_flattened_state(state, n_sexes, n_ages, n_genotypes)
            self._state_nn.individual_count[:] = state_obj.individual_count
            self._state_nn.sperm_storage[:] = state_obj.sperm_storage
            self._state = PopulationState(
                n_tick=state_obj.n_tick,
                individual_count=self._state_nn.individual_count,
                sperm_storage=self._state_nn.sperm_storage,
            )
        elif isinstance(state, dict):
            self._state_nn.individual_count[:] = state['individual_count']
            self._state_nn.sperm_storage[:] = state['sperm_storage']
        elif isinstance(state, PopulationState):
            self._state_nn.individual_count[:] = state.individual_count
            self._state_nn.sperm_storage[:] = state.sperm_storage
            self._state = PopulationState(
                n_tick=state.n_tick,
                individual_count=self._state_nn.individual_count,
                sperm_storage=self._state_nn.sperm_storage,
            )
        else:
            self._state_nn.individual_count[:] = state[0]
            self._state_nn.sperm_storage[:] = state[1]

        if history is not None and history.shape[0] > 0:
            self.clear_history()
            for row_idx in range(history.shape[0]):
                flat = history[row_idx, :]
                tick = int(flat[0])
                self._history.append((tick, flat.copy()))

    # ========================================================================
    # History restoration helpers
    # ========================================================================

    def get_history_as_objects(self, indices: Optional[List[int]] = None) -> List[Tuple[int, PopulationState]]:
        """Convert selected flattened snapshots back to PopulationState objects.

        Args:
            indices: List of snapshot indices to convert. If None, converts all.

        Returns:
            List[Tuple[int, PopulationState]]: List of (tick, state) tuples.

        Raises:
            IndexError: If an index is out of range.
        """
        if indices is None:
            indices = list(range(len(self._history)))

        from natal.population_state import parse_flattened_state
        result: List[Tuple[int, PopulationState]] = []
        for idx in indices:
            if idx < 0 or idx >= len(self._history):
                raise IndexError(f"History index {idx} out of range [0, {len(self._history)})")

            tick, flattened = self._history[idx]
            state = parse_flattened_state(
                flattened,
                n_sexes=2,
                n_ages=self._config_nn.n_ages,
                n_genotypes=len(self._registry_nn.index_to_genotype)
            )
            result.append((tick, state))
        return result

    def restore_checkpoint(self, tick: int) -> None:
        """Restore the population to a specific history tick.

        Args:
            tick: The target tick number.

        Raises:
            ValueError: If no record is found for the specified tick.
        """
        from natal.population_state import parse_flattened_state

        for t, flattened in self._history:
            if t == tick:
                state = parse_flattened_state(
                    flattened,
                    n_sexes=2,
                    n_ages=self._config_nn.n_ages,
                    n_genotypes=len(self._registry_nn.index_to_genotype)
                )
                # Copy state data directly.
                self._state_nn.individual_count[:] = state.individual_count
                self._state_nn.sperm_storage[:] = state.sperm_storage
                self._tick = tick
                return

        raise ValueError(f"No history record found for tick {tick}")

    # ========================================================================
    # Hooks system
    # ========================================================================

    # [Allowed hook events]
    #
    #     Before simulation:  [initialization]
    #                                |
    #                                v
    #     For tick in T:    |-------------------------------------------------------------------------|
    #                       |     [first] -->  reproduction  --> [early] -->  survival  --> [late]    |
    #                       |        ^                                                         |      |
    #                       |        |<--------------------------------------------------------|      |
    #                       |-------------------------------------------------------------------------|
    #                                |
    #                                v
    #     After simulation:      [finish]
    #

    # ========================================================================
    # Evolution logic
    # ========================================================================

    def _get_kernel_config(self) -> Tuple[Any, ...]:
        """Build configuration tuple for simulation kernels.

        Returns:
            tuple: A Numba-compatible configuration tuple.
        """
        return sk.export_config(self)  # type: ignore

    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        finish: bool = False,
        clear_history_on_start: bool = False
    ) -> 'AgeStructuredPopulation':
        """Run multi-step evolution using optimized simulation kernels.

        Args:
            n_steps: Number of steps to evolve.
            record_every: Interval for recording snapshots (0 to disable).
            finish: Whether to mark the population as finished after the run.
            clear_history_on_start: Whether to clear existing history before starting.

        Returns:
            AgeStructuredPopulation: Self for chaining.

        Raises:
            RuntimeError: If the population is already finished and cannot continue.
        """

        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot run() again after finish=True."
            )

        config = sk.export_config(self)

        hooks = self.get_compiled_event_hooks()

        # run_fn and registry are always initialized by get_compiled_event_hooks()
        assert hooks.run_fn is not None, "hooks.run_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"

        run_fn: Callable[..., Any] = hooks.run_fn
        registry = hooks.registry

        # Directly call the fixed-signature runner for multi-step evolution.
        final_state_tuple, history_new, was_stopped = run_fn(
            state=self._state_nn,
            config=config,
            registry=registry,
            n_ticks=n_steps,
            record_interval=record_every,
        )

        # Process final state (tuple format: ind_count, sperm, tick)
        self._state = PopulationState(
            n_tick=int(final_state_tuple[2]),
            individual_count=final_state_tuple[0],
            sperm_storage=final_state_tuple[1],
        )
        self._tick = int(final_state_tuple[2])

        # history_new is a 2D NDArray (n_snapshots, history_size)
        self._process_kernel_history(history_new, clear_history_on_start)

        # If terminated early by hooks, set _finished flag
        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            # Otherwise, if finish parameter is True, actively trigger finish
            self.finish_simulation()

        return self

    def run_tick(self) -> 'AgeStructuredPopulation':
        """
        Execute a single tick of evolution.

        Returns:
            AgeStructuredPopulation: Self for chaining.

        Raises:
            RuntimeError: If the population is already finished and cannot continue.
        """
        return self.run(n_steps=1, record_every=self.record_every, clear_history_on_start=False)

    def get_age_distribution(self, sex: str = 'both') -> np.ndarray:
        """Return the age distribution for the requested sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'``.

        Returns:
            NDArray[np.float64]: Age distribution array with shape (n_ages,).

        Raises:
            ValueError: If sex identifier is invalid.
        """
        if sex not in ('female', 'male', 'both', 'F', 'M'):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")

        # Access directly from PopulationState
        if sex in ('female', 'F'):
            return self._state_nn.individual_count[Sex.FEMALE.value, :, :].sum(axis=1)
        elif sex in ('male', 'M'):
            return self._state_nn.individual_count[Sex.MALE.value, :, :].sum(axis=1)
        else:
            return self._state_nn.individual_count.sum(axis=(0, 2))

    def get_genotype_count(self, genotype: Genotype) -> Tuple[int, int]:
        """Return total counts for a genotype as (female_count, male_count).

        Args:
            genotype: Target genotype instance.

        Returns:
            Tuple[int,int]: ``(female_count, male_count)`` across all ages.
        """
        genotype_idx = self._registry_nn.genotype_to_index[genotype]
        female_count = self._state_nn.individual_count[Sex.FEMALE.value, :, genotype_idx].sum()
        male_count = self._state_nn.individual_count[Sex.MALE.value, :, genotype_idx].sum()
        return (female_count, male_count)

    @property
    def genotypes_present(self) -> Set[Genotype]:
        """Set[Genotype]: Returns the set of genotypes with count > 0."""
        present: Set[Genotype] = set()
        for genotype_idx, genotype in enumerate(self._registry_nn.index_to_genotype):
            total_count = self._state_nn.individual_count[:, :, genotype_idx].sum()
            if total_count > 0:
                present.add(genotype)
        return present

    def __repr__(self) -> str:
        """Return a compact string representation of the population."""
        return (f"AgeStructuredPopulation(name='{self.name}', n_ages={self.n_ages}, "
                f"total_count={self.get_total_count()}, "
                f"adult_females={self.get_adult_count('female')}, "
                f"adult_males={self.get_adult_count('male')})")
