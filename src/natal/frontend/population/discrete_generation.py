"""Discrete-generation population model.

Non-overlapping generations with n_ages=2:
- age 0: offspring produced in the current tick
- age 1: reproducing adults

Simulation flow:
first hook → reproduction → early hook → survival → late hook → aging
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

import natal.backends.reference.lifecycle as lifecycle_engine
from natal.frontend.data import (
    DiscretePopulationState,
    ModelDraft,
    parse_flattened_discrete_state,
)
from natal.frontend.genetics import Genotype, Species
from natal.frontend.population.base import BasePopulation
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.types import Sex

if TYPE_CHECKING:
    from natal.backends.rust.rust_backend import RustDiscreteLifecycleBackend
    from natal.contracts.params import Params
    from natal.frontend.configurator import Configurator

__all__ = ["DiscreteGenerationPopulation"]


def _require_discrete_config(config: object) -> ModelDraft:
    """Validate that *config* satisfies the discrete-generation invariants.

    Since the draft merge there is a single ``ModelDraft`` schema for
    both granularities; what distinguishes a discrete-generation draft
    is its normalized shape, not its type.  This helper enforces that
    shape at every entry point that stores a draft on a
    ``DiscreteGenerationPopulation`` (``__init__``, ``import_config``):
    non-``ModelDraft`` objects are rejected with ``TypeError``, and
    drafts violating the discrete-generation invariants with
    ``ValueError``.

    The discrete-generation engine hardcodes a 2-age lifecycle
    (age 0 = offspring, age 1 = reproducing adult; adults are replaced
    every tick).  A draft with ``n_ages != 2``, ``new_adult_age != 1``,
    or ``adult_ages != [1]`` would run but produce silently wrong
    dynamics, so it is rejected up front rather than silently
    normalized.

    Args:
        config: The candidate draft object.

    Returns:
        *config* itself, narrowed to ``ModelDraft``.

    Raises:
        TypeError: If *config* is not a ``ModelDraft``.
        ValueError: If *config* is a ``ModelDraft`` but ``n_ages != 2``,
            ``new_adult_age != 1``, or ``adult_ages != [1]``.
    """
    if not isinstance(config, ModelDraft):
        raise TypeError(
            f"DiscreteGenerationPopulation requires a discrete-generation "
            f"ModelDraft, got {type(config).__name__}. Build one via "
            f"Configurator.for_discrete() or build_discrete_engine_config()."
        )
    if config.n_ages != 2 or config.new_adult_age != 1:
        raise ValueError(
            f"The discrete-generation draft must satisfy the "
            f"discrete-generation invariants: n_ages == 2 and "
            f"new_adult_age == 1, got n_ages={config.n_ages}, "
            f"new_adult_age={config.new_adult_age}. "
            f"The discrete engine hardcodes a 2-age lifecycle."
        )
    expected_adult_ages = np.array([1], dtype=np.int64)
    if not np.array_equal(config.adult_ages, expected_adult_ages):
        raise ValueError(
            f"The discrete-generation draft adult_ages must be [1], got "
            f"{config.adult_ages!r}."
        )
    # Non-overlapping generations: adults never survive a tick.  This is
    # the data marker distinguishing the discrete normalization from a
    # 2-age overlapping (age-structured) draft.
    if float(config.age_based_survival_rates[:, 1].sum()) != 0.0:
        raise ValueError(
            "The discrete-generation draft requires zero adult survival "
            f"(non-overlapping generations), got {config.age_based_survival_rates.tolist()}."
        )
    return config


class DiscreteGenerationPopulation(BasePopulation[DiscretePopulationState]):
    """Population with strict non-overlapping generations."""

    def __init__(
        self,
        species: Species,
        population_config: ModelDraft,
        name: Optional[str] = None,
        index_registry: Optional[IndexRegistry] = None,
        initial_individual_count: Optional[
            Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]]
        ] = None,
        hook_items: Optional[List[object]] = None,
    ):
        """Initialize a discrete-generation population.

        Constructs the population from a species definition and a
        discrete-normalized ``ModelDraft``, sets up genotype registries
        and the initial age-by-genotype distribution, and registers
        hooks for event-driven intervention.

        Args:
            species: Genetic architecture describing loci, alleles and
                chromosome structure.
            population_config: A fully initialized
                ``ModelDraft`` in the discrete normalization.  A draft
                violating the invariants is rejected with ``ValueError``;
                build a discrete draft via ``Configurator.for_discrete()``
                via ``Configurator.for_discrete()`` or
                ``build_discrete_engine_config()``.
            name: Human-readable population name.  Defaults to
                ``"DiscreteGenerationPop"``.
            index_registry: Optional shared registry for index compression.
            initial_individual_count: Optional per-sex, per-genotype
                initial distribution that overrides the config default.
            hook_items: Hook registrations (``Op`` objects, ``@hook``-
                decorated functions, or single-parameter callables).

        Raises:
            TypeError: If *population_config* is not a ``ModelDraft``.
            ValueError: If *population_config* violates the discrete
                generation invariants (``n_ages == 2``, ``new_adult_age
                == 1``, ``adult_ages == [1]``).
        """
        if name is None:
            name = "DiscreteGenerationPop"

        super().__init__(species, name, hook_items=hook_items)

        if index_registry is not None:
            self._index_registry = index_registry

        self._config = _require_discrete_config(population_config)

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        n_sexes = self.config.n_sexes
        n_ztypes = self.config.n_ztypes
        n_ages = self.config.n_ages

        # Create an empty state first so we can check whether the config's
        # default initial_individual_count has compatible dimensions --
        # presets from genetic_presets.py often pre-size this to match.
        self._state = DiscretePopulationState.create(
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            n_tick=0,
            individual_count=np.zeros((n_sexes, n_ages, n_ztypes), dtype=np.float64),
        )

        cfg_init_ind = self.config.initial_individual_count
        if cfg_init_ind.shape == self.state.individual_count.shape:
            self.state.individual_count[:] = cfg_init_ind

        # An explicit distribution overrides the config default. We zero out
        # the array first because _distribute_initial_population accumulates.
        if initial_individual_count is not None:
            self.state.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        self._python_backend = False
        # True while a Rust batch run executes; in-hook writes defer to the
        # next run (session borrow held by the engine).
        self._rust_run_active = False
        self._rust_lifecycle_backend: RustDiscreteLifecycleBackend | None = None
        self._rust_backend_seed: int | None = None
        # Contract params materialized from the draft at enable time and
        # re-materialized on each dirty sync; the source object handed to
        # the session's directed refresh_params pull.
        self._contract_params: Params | None = None

        # Keep a pristine copy so reset() can restore the starting state.
        self._initial_population_snapshot = (
            self.state.individual_count.copy(),
            None,
            None,
        )

        self._finalize_hooks()

        # Build self-describing history schema (frozen at construction).
        self._init_history_schema(
            kind="discrete_generation",
            n_demes=1,
            has_sperm_storage=False,
        )

    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "DiscreteGenerationPop",
        stochastic: bool = True,
        continuous_sampling: bool = False,
        fixed_egg_count: bool = False,
        backend: Literal["auto", "rust", "python"] = "auto",
        *,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
        declared_genotypes: Sequence[str] | Sequence[int] | None = None,  # deprecated alias
    ) -> Configurator:
        """Fluent population construction entry point.

        Returns the unified ``Configurator`` wrapping a
        discrete-normalized draft.  Chain domain methods and end with
        ``.build()`` to create a Population.
        """
        from natal.frontend.configurator import Configurator

        if declared_genotypes is not None:
            if declared_zygote_types is not None:
                raise ValueError(
                    "Cannot specify both declared_zygote_types and "
                    "declared_genotypes (deprecated alias)."
                )
            declared_zygote_types = declared_genotypes
        return Configurator.from_species(species, discrete=True).setup(
            name=name,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            fixed_egg_count=fixed_egg_count,
            backend=backend,
            compress=compress,
            declared_zygote_types=declared_zygote_types,
        )

    def _resolve_age_distribution(
        self,
        age_data: Union[List[int], Dict[int, int], int, float],
    ) -> Tuple[float, float]:
        """Resolve age distribution data into ``(age_0, age_1)`` counts.

        Accepts int/float (all to age 1), list of length 0–2, or dict
        with keys 0 and/or 1.

        Args:
            age_data: Age distribution specification.

        Returns:
            A tuple of ``(age_0_count, age_1_count)``.

        Raises:
            ValueError: If *age_data* format is unsupported.
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
            raise ValueError(f"Discrete initial list must have length <= 2, got {len(age_data)}")
        unsupported_keys = [k for k in age_data.keys() if k not in (0, 1)]
        if unsupported_keys:
            raise ValueError(f"Discrete initial dict supports only age keys 0 and 1, got {unsupported_keys}")
        return float(age_data.get(0, 0.0)), float(age_data.get(1, 0.0))

    def _distribute_initial_population(
        self,
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]]],
    ) -> None:
        """Distribute individuals across genotypes and ages from a nested dict.

        Args:
            distribution: Dict mapping sex -> {genotype -> age_distribution}.

        Raises:
            ValueError: If sex key is not ``"female"`` or ``"male"``.
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
                from natal.frontend.patterns import (
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
                age0_count, age1_count = self._resolve_age_distribution(age_data)
                self.state.individual_count[sex_idx, 0, z_idx] = age0_count
                self.state.individual_count[sex_idx, 1, z_idx] = age1_count

    def enable_rust_backend(self, seed: int = 0) -> DiscreteGenerationPopulation:
        """Enable the Rust backend for subsequent runs.

        CSR declarative hooks travel to Rust inside the ``HookProgram``;
        single-parameter Python callbacks are bridged through the
        session's ``python_callbacks`` channel (fired at event boundaries
        after the CSR hooks ran, state copies written back per call).
        Call this after all hook registration and config updates.  The
        current config is materialized into the contract pair once; the
        session owns its copies.  Later value changes flow through the
        dirty-set bridge: write paths mark contract fields and the next
        ``run()`` pulls exactly those fields into the live session — no
        rebuild, no RNG reset.  Structural changes (hooks, blueprint
        flags) or direct out-of-band array edits still go through
        :meth:`refresh_rust_backend`, the explicit full-refresh escape
        hatch.

        Args:
            seed: Seed for the Rust RNG.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the Rust extension is unavailable.
        """
        from natal.backends.rust.rust_backend import (
            RustDiscreteLifecycleBackend,
            rust_backend_available,
        )
        from natal.contracts.materialize import materialize

        if not rust_backend_available():
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "before enabling the Rust backend."
            )
        hook_program = self._build_hook_program()
        self._run_program = self._run_program._replace(hooks=hook_program)
        backend = RustDiscreteLifecycleBackend(
            self.config,
            hook_program,
            seed=seed,
        )
        self._register_rust_callbacks(backend)
        self._rust_lifecycle_backend = backend
        self._rust_backend_seed = seed
        self._contract_params = materialize(self.config).params
        self._rust_dirty.clear()
        return self

    def disable_rust_backend(self) -> DiscreteGenerationPopulation:
        """Disable the Rust backend and return to the reference path.

        Returns:
            Self for chaining.
        """
        self._rust_lifecycle_backend = None
        self._rust_backend_seed = None
        self._contract_params = None
        self._rust_dirty.clear()
        return self

    def refresh_rust_backend(self) -> DiscreteGenerationPopulation:
        """Rebuild the Rust backend from the current config and hooks.

        Explicit full-refresh escape hatch: rebuilds the session from a
        fresh materialization (RNG resets to the original seed).  Value-only
        changes do not need this — the dirty-set bridge syncs them before
        the next run automatically.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the backend was never enabled.
        """
        if self._rust_backend_seed is None:
            raise RuntimeError("Rust backend is not enabled; call enable_rust_backend() first.")
        return self.enable_rust_backend(seed=self._rust_backend_seed)

    @property
    def using_rust_backend(self) -> bool:
        """Return whether the Rust backend is currently active.

        Returns:
            True when ``enable_rust_backend()`` has been called.
        """
        return getattr(self, "_rust_lifecycle_backend", None) is not None

    def _sync_rust_backend(self) -> None:
        """Drain the dirty set into the live Rust session.

        Dirty non-empty: re-materialize the contract params from the draft
        and pull exactly the dirty fields into the session (directed
        refresh — the session object and its RNG survive).  The
        ``__hooks__``/``__blueprint__`` sentinels route to a full backend
        rebuild because hooks and blueprint flags are session structure,
        not values.
        """
        if self._rust_backend_seed is None or not self._rust_dirty:
            return
        if (
            self._rust_lifecycle_backend is None
            or "__hooks__" in self._rust_dirty
            or "__blueprint__" in self._rust_dirty
        ):
            self.refresh_rust_backend()
            self._rust_dirty.clear()
            return
        from natal.contracts.materialize import materialize

        self._contract_params = materialize(self.config).params
        # backend is non-None here: the None case rebuilt above and returned.
        self._rust_lifecycle_backend.refresh_params(
            sorted(self._rust_dirty), self._contract_params
        )
        self._rust_dirty.clear()

    def _run_rust_lifecycle(
        self,
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> DiscreteGenerationPopulation:
        """Run the Rust batch kernel and commit its history rows."""
        self._sync_rust_backend()
        backend = self._rust_lifecycle_backend
        if backend is None:
            raise RuntimeError(
                "Rust backend is not enabled; call enable_rust_backend() first."
            )

        # In-hook writes during the batch defer session pushes to the next run.
        self._rust_run_active = True
        try:
            final_state, history_new, was_stopped = backend.run(
                self.state,
                n_steps=n_steps,
                record_every=record_every,
                observation_mask=self._observation_mask,
            )
        finally:
            self._rust_run_active = False

        # Merge the session's set_param writes (params_log rows under their
        # own commit ticks + final draft values) so the Rust run path keeps
        # the same audit trail and draft visibility as the Python channel.
        self._absorb_rust_eco_journal(backend.drain_eco_journal())

        self._state = final_state
        self._tick = int(final_state.n_tick)
        self._process_kernel_history(history_new, clear_history_on_start)

        if was_stopped:
            self._finished = True
            self.trigger_event("finish", deme_id=self._deme_id)
        elif finish:
            self.finish_simulation()

        return self

    def run(
        self,
        n_steps: int = 1,
        record_every: Optional[int] = None,
        finish: bool = False,
        clear_history_on_start: bool = False,
    ) -> DiscreteGenerationPopulation:
        """Run the population for *n_steps* ticks.

        Uses the reference lifecycle orchestration and the
        pure-Python unified lifecycle loop otherwise.

        Args:
            n_steps: Number of ticks to simulate.
            record_every: Interval for recording history snapshots.
                Defaults to ``self.record_every``.
            finish: If True, trigger the finish event after the run.
            clear_history_on_start: If True, clear history before running.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the population has already finished.
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. Cannot run() again after finish=True."
            )

        self._running = True
        try:
            record_every_resolved = (
                record_every if record_every is not None else self.record_every
            )

            if getattr(self, "_rust_lifecycle_backend", None) is not None:
                return self._run_rust_lifecycle(
                    n_steps=n_steps,
                    record_every=record_every_resolved,
                    finish=finish,
                    clear_history_on_start=clear_history_on_start,
                )

            # Non-Rust path: the pure-Python reference lifecycle, where the
            # CSR interpreter and the Python callbacks alternate per event.
            # set_param ops also run here because their writes must reach the
            # parameter write channel (route dispatch / audit log).
            tick_fn = (
                lifecycle_engine.run_wf_tick
                if getattr(self.config, "extreme_speed_mode", 0) > 0
                else lifecycle_engine.run_discrete_tick
            )
            return self._run_python_lifecycle(
                tick_fn=tick_fn,
                n_steps=n_steps,
                record_every=record_every_resolved,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
            )
        finally:
            self._running = False

    def _run_python_lifecycle(
        self,
        tick_fn: Callable[..., tuple[DiscretePopulationState, int, ModelDraft]],
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> DiscreteGenerationPopulation:
        """Run the pure-Python unified lifecycle loop.

        Hook execution is delegated to ``trigger_event`` so all hook types
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
        from natal.frontend.hooks.types import empty_hook_program

        registry = empty_hook_program()

        def refresh_config(_config: ModelDraft) -> ModelDraft:
            """Return the population's current config (write-channel rebind)."""
            return self.config

        def first_hook(
            state: DiscretePopulationState,
            config: ModelDraft,
            deme_id: int,
        ) -> int:
            """Execute the ``first`` event against *state*."""
            _ = config, deme_id
            self._state = state
            self._tick = int(state.n_tick)
            return self.trigger_event("first", deme_id=deme_id)

        def early_hook(
            state: DiscretePopulationState,
            config: ModelDraft,
            deme_id: int,
        ) -> int:
            """Execute the ``early`` event against *state*."""
            _ = config, deme_id
            self._state = state
            self._tick = int(state.n_tick)
            return self.trigger_event("early", deme_id=deme_id)

        def late_hook(
            state: DiscretePopulationState,
            config: ModelDraft,
            deme_id: int,
        ) -> int:
            """Execute the ``late`` event against *state*."""
            _ = config, deme_id
            self._state = state
            self._tick = int(state.n_tick)
            return self.trigger_event("late", deme_id=deme_id)

        if clear_history_on_start:
            self.clear_history()

        if record_every > 0 and (self.tick % record_every == 0):
            self._record_current_snapshot(allow_existing=True)

        def record_fn(state: DiscretePopulationState) -> None:
            """Record *state* through the normal History path."""
            self._state = state
            self._tick = int(state.n_tick)
            self._record_current_snapshot(allow_existing=True)

        input_config = self.config
        final_state, was_stopped, config = lifecycle_engine.run(
            tick_fn=tick_fn,
            state=self.state,
            config=self.config,
            registry=registry,
            first_hook=first_hook,
            early_hook=early_hook,
            late_hook=late_hook,
            deme_id=self._deme_id,
            n_steps=n_steps,
            record_every=record_every,
            record_fn=record_fn,
            config_refresh=refresh_config,
        )
        self._state = final_state
        self._tick = int(final_state.n_tick)
        # The lifecycle rebuilds the config only when its in-kernel CSR
        # flush fired; hook-executor writes already republished through
        # set_config, so rebinding the stale lifecycle config here would
        # clobber them.
        if config is not input_config:
            self.set_config(config)

        if was_stopped:
            self._finished = True
            self.trigger_event("finish", deme_id=self._deme_id)
        elif finish:
            self.finish_simulation()

        return self

    def run_tick(self) -> DiscreteGenerationPopulation:
        """Run a single simulation tick.

        Returns:
            Self for chaining.
        """
        return self.run(n_steps=1, record_every=self.record_every)

    def reset(self) -> None:
        """Reset tick, history, and population state to initial values."""
        self._tick = 0
        if self._history_obj is not None:
            self._history_obj.clear()
        self._finished = False
        # Guard against calls before __init__ finishes (e.g. during
        # BasePopulation.__init__ -> _initialize -> reset chain).
        if hasattr(self, '_initial_population_snapshot'):
            ind_copy, _, _ = self._initial_population_snapshot
            self._state = DiscretePopulationState.create(
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_ztypes=self.config.n_ztypes,
                n_tick=0,
                individual_count=ind_copy.copy(),
            )

    def get_total_count(self) -> int:
        """Return the total number of individuals across all categories."""
        return int(round(np.sum(self.state.individual_count)))

    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        return int(round(np.sum(self.state.individual_count[int(Sex.FEMALE.value)])))

    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        return int(round(np.sum(self.state.individual_count[int(Sex.MALE.value)])))

    def clear_history(self) -> None:
        """Remove all recorded history snapshots."""
        self.history.clear()

    def export_state(self) -> NDArray[np.float64]:
        """Export the current state as a flat array.

        Returns:
            NDArray: Flattened state array.
        """
        return self.state.flatten_all()

    @property
    def config(self) -> ModelDraft:
        """ModelDraft: The current configuration."""
        return super().config

    def export_config(self) -> ModelDraft:
        """Return a copy of the current configuration."""
        return self.config

    def import_config(self, config: ModelDraft) -> None:
        """Replace the current configuration with *config*.

        Args:
            config: A discrete-normalized ``ModelDraft`` to install on this
                population.  Other types are
                rejected with ``TypeError``.

        Raises:
            TypeError: If *config* is not a ``ModelDraft``.
            ValueError: If *config* violates the discrete-generation
                invariants (``n_ages == 2``, ``new_adult_age == 1``,
                ``adult_ages == [1]``).
        """
        self._config = _require_discrete_config(config)

    def import_state(
        self,
        state: Union[DiscretePopulationState, NDArray[np.float64], Dict[str, np.ndarray]],
    ) -> None:
        """Replace the current state and reset the history timeline.

        All validation happens before any mutation — a failed import leaves the
        population unchanged.

        Args:
            state: New state as a ``DiscretePopulationState``, flat ndarray,
                or dict with ``individual_count`` key.
        """
        # ── Phase 1: parse and validate all inputs ──
        if isinstance(state, np.ndarray):
            state_obj = parse_flattened_discrete_state(
                state,
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_ztypes=self.config.n_ztypes,
            )
        elif isinstance(state, DiscretePopulationState):
            state_obj = state
        else:
            state_obj = DiscretePopulationState(
                n_tick=int(state.get("n_tick", self._tick)),
                individual_count=np.asarray(state["individual_count"], dtype=np.float64),
            )

        expected_shape = self.state.individual_count.shape
        if state_obj.individual_count.shape != expected_shape:
            raise ValueError(
                "individual_count shape mismatch: expected "
                f"{expected_shape}, got {state_obj.individual_count.shape}"
            )

        # ── Phase 2: commit atomically ──
        self._state = DiscretePopulationState(
            n_tick=int(state_obj.n_tick),
            individual_count=state_obj.individual_count.copy(),
        )
        self._tick = int(state_obj.n_tick)
        self.clear_history()

    @property
    def state(self) -> DiscretePopulationState:
        """DiscretePopulationState: The current population state.

        Raises:
            AttributeError: If the state has not been initialized.
        """
        if self._state is None:
            raise AttributeError("Population state has not been initialized.")
        return self._state

    def update(self) -> Configurator:
        """Return a ``Configurator`` for modifying this population's config."""
        return self._create_configurator()

    def __repr__(self) -> str:
        """Return a string summary of the discrete-generation population."""
        status = "Finished" if self._finished else "Active"
        return f"<DiscreteGenerationPopulation(name='{self.name}', tick={self.tick}, status={status})>"
