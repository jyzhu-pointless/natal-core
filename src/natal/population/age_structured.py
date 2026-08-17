"""Age-structured population models.

This module implements age-structured (overlapping generation) population
models and utilities for survival, reproduction, juvenile recruitment, and
fitness management.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

import natal.engine.lifecycle as lifecycle_engine
import natal.numba.utils as _numba_utils
from natal.data import PopulationConfig, PopulationState
from natal.genetics import Genotype, Species
from natal.hooks.types import CompiledHookDescriptor
from natal.population.base import BasePopulation, HookRegistrationMap
from natal.registry.index import IndexRegistry
from natal.utils.types import Sex

if TYPE_CHECKING:
    from natal.configurator import (
        AgeStructuredConfigurator,
    )
    from natal.engine.backends.rust_backend import RustLifecycleBackend

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
        snapshots (Dict[str, object]): Storage for custom state snapshots.
    """

    def __init__(
        self,
        species: Species,
        population_config: PopulationConfig,
        name: Optional[str] = None,
        index_registry: Optional[IndexRegistry] = None,
        initial_individual_count: Optional[Mapping[str, Mapping[Union[Genotype, str], Union[List[int], Dict[int, int]]]]] = None,
        initial_sperm_storage: Optional[Mapping[Union[Genotype, str], Mapping[Union[Genotype, str], Union[Dict[int, float], List[float], float]]]] = None,
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

        if index_registry is not None:
            self._index_registry = index_registry

        config_hook_slot = int(getattr(population_config, "hook_slot", 0))
        if config_hook_slot <= 0:
            config_hook_slot = self.hook_slot
        self._config = population_config._replace(hook_slot=np.int32(config_hook_slot))

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        self._state = PopulationState.create(
            n_ztypes=population_config.n_ztypes,
            n_sexes=population_config.n_sexes,
            n_ages=population_config.n_ages,
        )

        # Initialize from builder-injected config arrays if available.
        cfg_init_ind = population_config.initial_individual_count
        if cfg_init_ind.shape == self.state.individual_count.shape:
            self.state.individual_count[:] = cfg_init_ind
        cfg_init_sperm = population_config.initial_sperm_storage
        if cfg_init_sperm.shape == self.state.sperm_storage.shape:
            self.state.sperm_storage[:] = cfg_init_sperm

        self.snapshots = {}
        self._rust_lifecycle_backend: RustLifecycleBackend | None = None

        if initial_individual_count is not None:
            self.state.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        if initial_sperm_storage is not None:
            # TODO: add population_config.use_sperm_storage
            self._distribute_initial_sperm_storage(species, initial_sperm_storage)

        self._initial_population_snapshot = (
            self.state.individual_count.copy(),
            self.state.sperm_storage.copy(),
            None,
        )

        self._initialize_registry()
        self._finalize_hooks()

        # Build self-describing history schema (frozen at construction).
        self._init_history_schema(
            kind="age_structured",
            n_demes=1,
            has_sperm_storage=True,
        )

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        *,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
        declared_genotypes: Sequence[str] | Sequence[int] | None = None,  # deprecated alias
    ) -> AgeStructuredConfigurator:
        """Start building an age-structured population with overlapping generations.

        This is the fluent entry point for constructing an
        ``AgeStructuredPopulation``.  It returns an ``AgeStructuredConfigurator``
        that you configure by chaining domain methods (``initial_state()``,
        ``reproduction()``, ``competition()``, etc.) and finalize with
        ``build()``.

        Args:
            species: Species object describing the population's genetic
                architecture (chromosomes, loci, alleles).
            name: Human-readable name for the population.
                Defaults to ``"AgeStructuredPop"``.
            stochastic: If ``False``, use deterministic (median) outcomes for
                reproduction and survival. Defaults to ``True``.
            continuous_sampling: If ``True``, sample from continuous
                distributions instead of discrete counts.
                Defaults to ``False``.
            fixed_egg_count: If ``True``, disable Poisson noise on egg counts
                so each female produces exactly the specified number of eggs.
                Defaults to ``False``.
            compress: If ``True``, enable full index compression at build
                time, pruning unreachable GTypes and ZTypes to shrink
                internal arrays. Defaults to ``False``.
            declared_zygote_types: Optional sequence of genotype strings
                (``"WT|WT"``) or integer indices that are treated as
                reachable even if absent from the initial state.  Use this
                to prevent compression from pruning genotypes that may
                appear later via hooks or runtime presets.
            declared_genotypes: Deprecated alias for
                *declared_zygote_types*.

        Returns:
            ``AgeStructuredConfigurator`` ready for domain-method chaining.
            Call ``.build()`` to produce an ``AgeStructuredPopulation``.

        Raises:
            ValueError: If both ``declared_zygote_types`` and
                ``declared_genotypes`` (deprecated alias) are specified
                simultaneously.
        """
        from natal.configurator import Configurator

        if declared_genotypes is not None:
            if declared_zygote_types is not None:
                raise ValueError(
                    "Cannot specify both declared_zygote_types and "
                    "declared_genotypes (deprecated alias)."
                )
            declared_zygote_types = declared_genotypes
        return Configurator.for_age_structured(species).setup(
            name=name,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            fixed_egg_count=fixed_egg_count,
            compress=compress,
            declared_zygote_types=declared_zygote_types,
        )

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
        self.state.individual_count.fill(0.0)
        for sex_key, genotype_dist in distribution.items():
            sex_key_norm = sex_key.lower().strip()
            if sex_key_norm == "female":
                sex_idx = int(Sex.FEMALE.value)
            elif sex_key_norm == "male":
                sex_idx = int(Sex.MALE.value)
            else:
                raise ValueError(f"Sex must be 'female' or 'male', got '{sex_key}'")

            for genotype_key, age_data in genotype_dist.items():
                from natal.patterns import (
                    GenotypePatternParser,
                    ZygoteTypePattern,
                )

                if isinstance(genotype_key, str):
                    pattern = ZygoteTypePattern.from_slab_key(genotype_key, self.species)
                else:
                    parser = GenotypePatternParser(self.species)
                    pattern = ZygoteTypePattern(
                        parser.parse(str(genotype_key)), slab=None
                    )

                z_idx = self.registry.resolve_default_ztype_index(pattern)

                if isinstance(age_data, list):
                    for age, raw_count in enumerate(cast(List[object], age_data)):
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Age count must be numeric, got {type(raw_count)}")
                        count = float(raw_count)
                        if age < self.config.n_ages and count > 0:
                            self.state.individual_count[sex_idx, age, z_idx] = count
                elif isinstance(age_data, dict):
                    for age_raw, raw_count in cast(Dict[object, object], age_data).items():
                        if not isinstance(age_raw, int):
                            raise TypeError(f"Age key must be int, got {type(age_raw)}")
                        if not isinstance(raw_count, (int, float)) or isinstance(raw_count, bool):
                            raise TypeError(f"Age count must be numeric, got {type(raw_count)}")
                        age = age_raw
                        count = float(raw_count)
                        if age < self.config.n_ages and count > 0:
                            self.state.individual_count[sex_idx, age, z_idx] = count
                else:
                    raise TypeError(f"age_data must be a list or dict, got {type(age_data)}")

    def _distribute_initial_sperm_storage(
        self,
        species: Species,
        sperm_storage_dist: Mapping[
            Union[Genotype, str],
            Mapping[Union[Genotype, str], Union[Dict[int, float], List[float], Tuple[float, ...], float, int]],
        ]
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
        self.state.sperm_storage.fill(0.0)
        from natal.patterns import GenotypePatternParser, ZygoteTypePattern

        for female_key, male_dict in sperm_storage_dist.items():
            assert isinstance(female_key, (str, Genotype)), \
                f"Female genotype key must be Genotype or str, got {type(female_key)}"

            if isinstance(female_key, str):
                female_pattern = ZygoteTypePattern.from_slab_key(female_key, species)
            else:
                parser = GenotypePatternParser(species)
                female_pattern = ZygoteTypePattern(
                    parser.parse(str(female_key)), slab=None
                )

            f_z = self.registry.resolve_default_ztype_index(female_pattern)

            for male_key, age_data in male_dict.items():
                assert isinstance(male_key, (str, Genotype)), \
                    f"Male genotype key must be Genotype or str, got {type(male_key)}"

                if isinstance(male_key, str):
                    male_pattern = ZygoteTypePattern.from_slab_key(male_key, species)
                else:
                    parser = GenotypePatternParser(species)
                    male_pattern = ZygoteTypePattern(
                        parser.parse(str(male_key)), slab=None
                    )

                m_z = self.registry.resolve_default_ztype_index(male_pattern)

                assert isinstance(age_data, (dict, list, tuple, int, float)), \
                    f"Age data must be Dict, List, or numeric scalar, got {type(age_data)}"

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
                            self.state.sperm_storage[age, f_z, m_z] = count

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
                            self.state.sperm_storage[age, f_z, m_z] = count

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
                            self.state.sperm_storage[age, f_z, m_z] = count

                else:
                    # Scalar format: apply to all adult ages
                    if age_data < 0:
                        raise ValueError(f"Sperm count must be non-negative, got {age_data}")
                    if age_data > 0:
                        for age in range(self.new_adult_age, self.n_ages):
                            self.state.sperm_storage[age, f_z, m_z] = float(age_data)

    @property
    def state(self) -> PopulationState:
        """PopulationState: The current state container for the population."""
        if self._state is None:
            raise AttributeError("Population state has not been initialized.")
        return self._state

    def reset(self) -> None:
        """Reset the population to its initial state.

        Restores individual counts and sperm storage to original values.
        """
        self._tick = 0
        if self._history_obj is not None:
            self._history_obj.clear()
        self._finished = False
        if hasattr(self, '_initial_population_snapshot'):
            ind_copy, sperm_copy, _ = self._initial_population_snapshot

            self._state = PopulationState.create(
                n_ztypes=self.config.n_ztypes,
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_tick=0,
                individual_count=ind_copy.copy(),
                sperm_storage=sperm_copy.copy(),
            )

    @property
    def n_ages(self) -> int:
        """int: Number of age classes in this population."""
        return self.config.n_ages

    @property
    def new_adult_age(self) -> int:
        """int: Minimum age at which individuals are considered adults."""
        return self.config.new_adult_age

    def get_total_count(self) -> int:
        """Return the total number of individuals in the population.

        Returns:
            float: Grand total across all sexes, ages, and genotypes.
        """
        return self.state.individual_count.sum()

    def get_female_count(self) -> int:
        """Return the total number of female individuals.

        Returns:
            float: Sum of all female individual counts.
        """
        return self.state.individual_count[Sex.FEMALE.value, :, :].sum()

    def get_male_count(self) -> int:
        """Return the total number of male individuals.

        Returns:
            float: Sum of all male individual counts.
        """
        return self.state.individual_count[Sex.MALE.value, :, :].sum()

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
            total += self.state.individual_count[Sex.FEMALE.value, self.new_adult_age:self.n_ages, :].sum()

        if sex in ('male', 'M', 'both'):
            total += self.state.individual_count[Sex.MALE.value, self.new_adult_age:self.n_ages, :].sum()

        return int(total)

    @property
    def config(self) -> PopulationConfig:
        """PopulationConfig: The current configuration."""
        return cast(PopulationConfig, super().config)

    # ========================================================================
    # State export/import (simulator interface)
    # ========================================================================

    def export_config(self) -> PopulationConfig:
        """Export population configuration to Config jitclass.

        Returns:
            PopulationConfig: A copy of the current population configuration.
        """
        return self.config

    def import_config(self, config: PopulationConfig) -> None:
        """Import configuration into the population.

        Args:
            config: Config jitclass instance.
        """
        # Configuration is usually read-only (used by run_tick),
        # kept here for completeness.
        self._config = config

    def clear_history(self) -> None:
        """Clear history records."""
        self.history.clear()

    def export_state(self) -> NDArray[np.float64]:
        """Export population state as a flattened array.

        Returns:
            NDArray: Flattened state array ``[n_tick, ind_count.ravel(), sperm_storage.ravel()]``.
        """
        return self.state.flatten_all()

    def import_state(self, state: Union[PopulationState, NDArray[np.float64], Dict[str, np.ndarray], Tuple[np.ndarray, np.ndarray]]) -> None:
        """Import state and reset the history timeline.

        All validation happens before any mutation — a failed import leaves the
        population unchanged.

        Args:
            state: Flattened array, PopulationState object, or data dictionary.
        """
        from natal.data import PopulationState, parse_flattened_state

        n_sexes, n_ages, n_ztypes = self.state.individual_count.shape

        # ── Phase 1: parse and validate all inputs ──
        if isinstance(state, np.ndarray):
            state_obj = parse_flattened_state(state, n_sexes, n_ages, n_ztypes)
        elif isinstance(state, PopulationState):
            state_obj = state
        elif isinstance(state, dict):
            state_obj = PopulationState(
                n_tick=int(state.get("n_tick", self._tick)),
                individual_count=np.asarray(state["individual_count"], dtype=np.float64),
                sperm_storage=np.asarray(state["sperm_storage"], dtype=np.float64),
            )
        else:
            if len(state) != 2:
                raise ValueError(f"Tuple state must have length 2, got {len(state)}")
            state_obj = PopulationState(
                n_tick=self._tick,
                individual_count=np.asarray(state[0], dtype=np.float64),
                sperm_storage=np.asarray(state[1], dtype=np.float64),
            )

        expected_individual_shape = self.state.individual_count.shape
        if state_obj.individual_count.shape != expected_individual_shape:
            raise ValueError(
                "individual_count shape mismatch: expected "
                f"{expected_individual_shape}, got {state_obj.individual_count.shape}"
            )
        expected_sperm_shape = self.state.sperm_storage.shape
        if state_obj.sperm_storage.shape != expected_sperm_shape:
            raise ValueError(
                "sperm_storage shape mismatch: expected "
                f"{expected_sperm_shape}, got {state_obj.sperm_storage.shape}"
            )

        # ── Phase 2: commit atomically ──
        self.state.individual_count[:] = state_obj.individual_count
        self.state.sperm_storage[:] = state_obj.sperm_storage
        self._state = PopulationState(
            n_tick=state_obj.n_tick,
            individual_count=self.state.individual_count,
            sperm_storage=self.state.sperm_storage,
        )
        self._tick = int(state_obj.n_tick)
        self.clear_history()

    # ========================================================================
    # History restoration helpers
    # ========================================================================

    def restore_checkpoint(self, tick: int) -> None:
        """Restore the population to a specific raw-history tick.

        Args:
            tick: The target tick number.

        Raises:
            ValueError: If no record is found for the specified tick.
        """
        super().restore_checkpoint(tick)

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
        """Build configuration tuple for simulation engine.

        Returns:
            tuple: A Numba-compatible configuration tuple.
        """
        return self.export_config()

    def enable_rust_backend(self, seed: int = 0) -> AgeStructuredPopulation:
        """Enable the Rust lifecycle backend for subsequent runs.

        The Rust backend executes CSR declarative hooks only.  If this
        population contains custom hook callables, this method raises so the
        caller can keep using the Numba path; ``run()`` also falls back to
        Numba automatically when custom hooks are present.

        The backend snapshots the current configuration and hook program.
        Call this after all hook registration and config updates; runtime
        ``pop.update()`` changes after enabling require disabling and
        re-enabling the backend.

        Args:
            seed: Seed for the Rust RNG used in stochastic simulations.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the Rust extension is unavailable or custom hooks
                are registered.
        """
        from natal.engine.backends.rust_backend import (
            RustLifecycleBackend,
            rust_backend_available,
        )

        if not rust_backend_available():
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "before enabling the Rust backend."
            )
        if self._has_non_csr_hooks():
            raise RuntimeError(
                "Rust backend only supports CSR declarative hooks. "
                "Keep the Numba backend for custom hook callables."
            )
        self._rust_lifecycle_backend = RustLifecycleBackend(
            self.config,
            self._build_hook_program(),
            seed=seed,
        )
        return self

    def disable_rust_backend(self) -> AgeStructuredPopulation:
        """Disable the Rust backend and return to the Numba/Python path.

        Returns:
            Self for chaining.
        """
        self._rust_lifecycle_backend = None
        return self

    @property
    def using_rust_backend(self) -> bool:
        """Return whether the Rust lifecycle backend is currently enabled.

        Returns:
            True when ``enable_rust_backend()`` has been called and no custom
            hooks were added afterwards.
        """
        return self._rust_lifecycle_backend is not None and not self._has_non_csr_hooks()

    def _has_non_csr_hooks(self) -> bool:
        """Return whether any compiled hook needs a Python/Numba callable."""
        descriptors = cast(
            List[CompiledHookDescriptor],
            getattr(self, "compiled_hook_descriptors", []),
        )
        return any(
            desc.plan is None
            and (desc.njit_fn is not None or desc.py_wrapper is not None)
            for desc in descriptors
        )

    def _run_rust_lifecycle(
        self,
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> AgeStructuredPopulation:
        """Run the Rust batch kernel and commit its history rows."""
        backend = self._rust_lifecycle_backend
        if backend is None:
            raise RuntimeError("Rust backend is not enabled; call enable_rust_backend() first.")

        observation_mask = self._observation_mask
        final_state, history_new, was_stopped = backend.run(
            self.state,
            n_steps=n_steps,
            record_every=record_every,
            observation_mask=observation_mask,
        )

        self._state = final_state
        self._tick = int(final_state.n_tick)
        self._process_kernel_history(history_new, clear_history_on_start)

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def run(
        self,
        n_steps: int,
        record_every: Optional[int] = None,
        finish: bool = False,
        clear_history_on_start: bool = False
    ) -> AgeStructuredPopulation:
        """Run multi-step evolution using the unified lifecycle engine.

        Args:
            n_steps: Number of steps to evolve.
            record_every: Interval for recording snapshots.
                If None, uses self.record_every. If 0, no snapshots are recorded.
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

        self._running = True
        try:
            if record_every is None:
                record_every = self.record_every

            if self._rust_lifecycle_backend is not None and not self._has_non_csr_hooks():
                return self._run_rust_lifecycle(
                    n_steps=n_steps,
                    record_every=record_every,
                    finish=finish,
                    clear_history_on_start=clear_history_on_start,
                )

            config = self.export_config()
            wrappers = self.get_compiled_event_hooks()
            assert wrappers.hooks.registry is not None, "hooks.registry should always be initialized"

            if _numba_utils.NUMBA_ENABLED and wrappers.run_fn is not None:
                obs_mask = self._observation_mask
                n_obs = len(self._observation.labels) if self._observation is not None else 0

                final_state_tuple, history_new, was_stopped = wrappers.run_fn(
                    state=self.state,
                    config=config,
                    registry=wrappers.hooks.registry,
                    n_ticks=n_steps,
                    record_interval=record_every,
                    observation_mask=obs_mask,
                    n_obs_groups=n_obs,
                )

                self._state = PopulationState(
                    n_tick=int(final_state_tuple[2]),
                    individual_count=final_state_tuple[0],
                    sperm_storage=final_state_tuple[1],
                )
                self._tick = int(final_state_tuple[2])
                self._process_kernel_history(history_new, clear_history_on_start)
            else:
                return self._run_python_lifecycle(
                    tick_fn=lifecycle_engine.run_structured_tick,
                    n_steps=n_steps,
                    record_every=record_every,
                    finish=finish,
                    clear_history_on_start=clear_history_on_start,
                )

            if was_stopped:
                self._finished = True
                self.trigger_event("finish")
            elif finish:
                self.finish_simulation()

            return self
        finally:
            self._running = False

    def _run_python_lifecycle(
        self,
        tick_fn: Callable[..., tuple[PopulationState, int]],
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> AgeStructuredPopulation:
        """Run the pure-Python unified lifecycle loop.

        Hook execution is delegated to ``trigger_event`` so CSR declarative
        hooks, njit hooks, Python wrapper hooks, and legacy plain callbacks
        keep their existing dispatch semantics.  The CSR registry passed to
        the lifecycle loop is therefore the empty program.

        Args:
            tick_fn: Unified single-tick function.
            n_steps: Number of ticks to execute.
            record_every: Recording interval.  ``0`` disables recording.
            finish: Whether to finish the population after the run.
            clear_history_on_start: Whether to clear history first.

        Returns:
            This population after the run.
        """
        self.ensure_hook_executor()
        registry = self._create_empty_hook_program()

        def first_hook(state: PopulationState, config: PopulationConfig, deme_id: int) -> int:
            """Execute the ``first`` event against *state*."""
            _ = config, deme_id
            self._state = state
            self._tick = int(state.n_tick)
            return self.trigger_event("first", deme_id=deme_id)

        def early_hook(state: PopulationState, config: PopulationConfig, deme_id: int) -> int:
            """Execute the ``early`` event against *state*."""
            _ = config, deme_id
            self._state = state
            self._tick = int(state.n_tick)
            return self.trigger_event("early", deme_id=deme_id)

        def late_hook(state: PopulationState, config: PopulationConfig, deme_id: int) -> int:
            """Execute the ``late`` event against *state*."""
            _ = config, deme_id
            self._state = state
            self._tick = int(state.n_tick)
            return self.trigger_event("late", deme_id=deme_id)

        if clear_history_on_start:
            self.clear_history()

        if record_every > 0 and (self.tick % record_every == 0):
            self._record_current_snapshot(allow_existing=True)

        def record_fn(state: PopulationState) -> None:
            """Record *state* through the normal History path."""
            self._state = state
            self._tick = int(state.n_tick)
            self._record_current_snapshot(allow_existing=True)

        final_state, was_stopped = lifecycle_engine.run(
            tick_fn=tick_fn,
            state=self.state,
            config=self.config,
            registry=registry,
            first_hook=first_hook,
            early_hook=early_hook,
            late_hook=late_hook,
            deme_id=-1,
            n_steps=n_steps,
            record_every=record_every,
            record_fn=record_fn,
        )
        self._state = final_state
        self._tick = int(final_state.n_tick)

        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            self.finish_simulation()

        return self

    def run_tick(self) -> AgeStructuredPopulation:
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
            return self.state.individual_count[Sex.FEMALE.value, :, :].sum(axis=1)
        elif sex in ('male', 'M'):
            return self.state.individual_count[Sex.MALE.value, :, :].sum(axis=1)
        else:
            return self.state.individual_count.sum(axis=(0, 2))

    def get_genotype_count(self, genotype: Genotype) -> Tuple[int, int]:
        """Return total counts for a genotype as (female_count, male_count).

        .. deprecated::
            Use ``self.registry.ztype_index()`` + manual array sum instead.
        """
        import warnings
        warnings.warn(
            "get_genotype_count is deprecated; use registry + manual sum",
            DeprecationWarning, stacklevel=2,
        )
        genotype_idx = self.registry.ztype_index(genotype, self.registry.slab_labels[0])
        female_count = self.state.individual_count[Sex.FEMALE.value, :, genotype_idx].sum()
        male_count = self.state.individual_count[Sex.MALE.value, :, genotype_idx].sum()
        return (female_count, male_count)

    @property
    def genotypes_present(self) -> Set[Genotype]:
        """Set[Genotype]: Returns the set of genotypes with count > 0.

        .. deprecated::
            Use ``self.registry.index_to_genotype`` + manual count check
            instead.
        """
        import warnings
        warnings.warn(
            "genotypes_present is deprecated; use registry + manual count check",
            DeprecationWarning, stacklevel=2,
        )
        present: Set[Genotype] = set()
        for z_idx, (genotype, _slab) in enumerate(self.registry.index_to_ztype):
            total_count = self.state.individual_count[:, :, z_idx].sum()
            if total_count > 0:
                present.add(genotype)
        return present

    def update(self) -> AgeStructuredConfigurator:
        """Return an ``AgeStructuredConfigurator`` for modifying this population's config."""
        return cast('AgeStructuredConfigurator', self._create_configurator())

    def __repr__(self) -> str:
        """Return a compact string representation of the population."""
        return (f"AgeStructuredPopulation(name='{self.name}', n_ages={self.n_ages}, "
                f"total_count={self.get_total_count()}, "
                f"adult_females={self.get_adult_count('female')}, "
                f"adult_males={self.get_adult_count('male')})")
