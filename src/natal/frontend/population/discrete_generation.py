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
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from natal.frontend.data import (
    DiscretePopulationState,
    parse_flattened_discrete_state,
)
from natal.frontend.genetics import Genotype, Species
from natal.frontend.model import (
    ModelDraft,
)
from natal.frontend.population.base import BasePopulation
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.types import Sex

if TYPE_CHECKING:
    from natal.backends.rust.rust_backend import RustDiscreteLifecycleBackend
    from natal.frontend.builder import PopulationBuilder, RuntimeUpdater
    from natal.frontend.hooks import CompiledHookDescriptor

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
            f"PopulationBuilder.for_discrete() or build_discrete_engine_config()."
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
            Dict[
                str,
                Dict[
                    Union[Genotype, str], Union[List[int], Dict[int, int], int, float]
                ],
            ]
        ] = None,
        hook_descriptors: Sequence[CompiledHookDescriptor] = (),
    ):
        """Initialize a discrete-generation population.

        Constructs the population from a species definition and a
        discrete-normalized ``ModelDraft``, sets up genotype registries
        and the initial age-by-genotype distribution, and installs the
        compiled hook plan injected by the builder.

        Args:
            species: Genetic architecture describing loci, alleles and
                chromosome structure.
            population_config: A fully initialized
                ``ModelDraft`` in the discrete normalization.  A draft
                violating the invariants is rejected with ``ValueError``;
                build a discrete draft via ``PopulationBuilder.for_discrete()``
                via ``PopulationBuilder.for_discrete()`` or
                ``build_discrete_engine_config()``.
            name: Human-readable population name.  Defaults to
                ``"DiscreteGenerationPop"``.
            index_registry: Optional shared registry for index compression.
            initial_individual_count: Optional per-sex, per-genotype
                initial distribution that overrides the config default.
            hook_descriptors: Compiled hook plan (declarative CSR
                descriptors and Python callbacks) injected exactly once by
                the builder; there is no post-construction registration.

        Raises:
            TypeError: If *population_config* is not a ``ModelDraft``.
            ValueError: If *population_config* violates the discrete
                generation invariants (``n_ages == 2``, ``new_adult_age
                == 1``, ``adult_ages == [1]``).
        """
        if name is None:
            name = "DiscreteGenerationPop"

        super().__init__(species, name, hook_descriptors=hook_descriptors)

        if index_registry is not None:
            self._index_registry = index_registry

        self._config = _require_discrete_config(population_config)

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()

        self._initialize_registry()

        n_sexes = self._config.n_sexes
        n_ztypes = self._config.n_ztypes
        n_ages = self._config.n_ages

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

        cfg_init_ind = self._config.initial_individual_count
        if cfg_init_ind.shape == self._live_state().individual_count.shape:
            self._live_state().individual_count[:] = cfg_init_ind

        # An explicit distribution overrides the config default. We zero out
        # the array first because _distribute_initial_population accumulates.
        if initial_individual_count is not None:
            self._live_state().individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)

        self._rust_run_active = False
        self._rust_lifecycle_backend: RustDiscreteLifecycleBackend | None = None
        self._rust_backend_seed: int | None = None
        # Structural changes (blueprint flags, modifier maps) rebuild the
        # session before the next run; value changes go straight to the
        # session through the writers and the run-boundary ecology flush.
        self._rust_needs_rebuild: bool = False

        # Keep a pristine copy so reset() can restore the starting state.
        self._initial_population_snapshot = (
            self._live_state().individual_count.copy(),
            None,
            None,
        )

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
        *,
        compress: bool = False,
        declared_zygote_types: Sequence[str] | Sequence[int] | None = None,
        declared_genotypes: Sequence[str]
        | Sequence[int]
        | None = None,  # deprecated alias
        extreme_speed_mode: int | None = None,
    ) -> PopulationBuilder:
        """Fluent population construction entry point.

        Returns the unified ``PopulationBuilder`` wrapping a
        discrete-normalized draft.  Chain domain methods and end with
        ``.build()`` to create a Population.
        """
        from natal.frontend.builder import PopulationBuilder

        if declared_genotypes is not None:
            if declared_zygote_types is not None:
                raise ValueError(
                    "Cannot specify both declared_zygote_types and "
                    "declared_genotypes (deprecated alias)."
                )
            declared_zygote_types = declared_genotypes
        return PopulationBuilder.from_species(species, discrete=True).setup(
            name=name,
            stochastic=stochastic,
            continuous_sampling=continuous_sampling,
            fixed_egg_count=fixed_egg_count,
            compress=compress,
            declared_zygote_types=declared_zygote_types,
            extreme_speed_mode=extreme_speed_mode,
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
        distribution: Dict[
            str,
            Dict[Union[Genotype, str], Union[List[int], Dict[int, int], int, float]],
        ],
    ) -> None:
        """Distribute individuals across genotypes and ages from a nested dict.

        Args:
            distribution: Dict mapping sex -> {genotype -> age_distribution}.

        Raises:
            ValueError: If sex key is not ``"female"`` or ``"male"``.
        """
        self._live_state().individual_count.fill(0.0)
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
                    pattern = ZygoteTypePattern.from_slab_key(
                        genotype_key, self.species
                    )
                else:
                    parser = GenotypePatternParser(self.species)
                    pattern = ZygoteTypePattern(
                        parser.parse(str(genotype_key)), slab=None
                    )

                z_idx = self.registry.resolve_default_ztype_index(pattern)
                age0_count, age1_count = self._resolve_age_distribution(age_data)
                self._live_state().individual_count[sex_idx, 0, z_idx] = age0_count
                self._live_state().individual_count[sex_idx, 1, z_idx] = age1_count

    def _initialize_session(self, seed: int = 0) -> DiscreteGenerationPopulation:
        """Enable the Rust backend for subsequent runs.

        CSR declarative hooks travel to Rust inside the ``HookProgram``;
        single-parameter Python callbacks are bridged through the
        session's ``python_callbacks`` channel (fired at event boundaries
        interleaved with the CSR slots by priority, state copies written back per call).
        Call this after config updates.  The
        current config is materialized into the contract pair once; the
        session owns its copies.  Later value changes flow through the
        dirty-set bridge: write paths mark contract fields and the next
        ``run()`` pulls exactly those fields into the live session — no
        rebuild, no RNG reset.  Structural changes (blueprint
        flags) rebuild the session before the next run.

        Args:
            seed: Seed for the Rust RNG.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the Rust extension is unavailable.
        """
        self._require_standalone_owner("_initialize_session")
        from natal.backends.rust.rust_backend import (
            RustDiscreteLifecycleBackend,
            rust_backend_available,
        )

        if not rust_backend_available():
            raise RuntimeError(
                "natal._engine_rs is not available; build it with `maturin develop` "
                "before enabling the Rust backend."
            )
        hook_program = self._hook_program
        backend = RustDiscreteLifecycleBackend(
            # Materialization copies the first owned draft itself; only an
            # existing session needs a current native parameter snapshot.
            self._config if self._rust_lifecycle_backend is None and self._config is not None else self.config,
            hook_program,
            seed=seed,
        )
        self._register_rust_callbacks(backend)
        # Capture the state BEFORE the field switch: under a structural
        # rebuild the lazy pull must read the OLD session, not the freshly
        # constructed one whose state is the blueprint initial population
        # again.
        state_to_install = self._live_state()
        self._rust_lifecycle_backend = backend
        self._rust_backend_seed = seed
        self._rust_needs_rebuild = False
        # The session owns the state from here on: install the
        # live Python state so the freshly seeded RNG continues from the
        # population's current counts and tick.
        backend.set_state(state_to_install)
        self._state_cache_stale = False
        return self

    def _run_startup_sync(self) -> None:
        """Install changed execution flags without replacing state or RNG."""
        if not self._rust_needs_rebuild:
            return
        backend = self._rust_lifecycle_backend
        if backend is None:
            raise RuntimeError("The population session has not been initialized.")
        config = self.config
        backend.configure_program(self._hook_program, config)
        self._register_rust_callbacks(backend)
        self._rust_needs_rebuild = False

    def _run_rust_lifecycle(
        self,
        n_steps: int,
        record_every: int,
        finish: bool,
        clear_history_on_start: bool,
    ) -> DiscreteGenerationPopulation:
        """Run the Rust batch kernel and commit its history rows."""
        self._run_startup_sync()
        backend = self._rust_lifecycle_backend
        if backend is None:
            # Instances that skipped ``build()`` (clones, direct
            # construction) lazily create their engine session at the
            # first run boundary; ``_initialize_session`` installs the
            # current live state into the fresh session.
            self._initialize_session(seed=int(self._rust_backend_seed or 0))
            backend = self._rust_lifecycle_backend
            assert backend is not None  # enable either returns or raises

        # Raw-mode runs keep a full record-aligned checkpoint per recorded
        # tick inside the session (state + RNG + ecology), so the public
        # restore_checkpoint rolls back everything, not just counts.
        history_obj = getattr(self, "_history_obj", None)
        checkpoint_every = (
            record_every
            if record_every > 0
            and history_obj is not None
            and history_obj.schema.mode == "raw"
            else 0
        )

        if history_obj is not None and clear_history_on_start:
            self.clear_history()
        if history_obj is not None:
            # Observation selector, history ownership, and the checkpoint
            # pruner bind once per (population, History); later runs skip.
            self._bind_history_recording(backend)

        self._rust_run_active = True
        try:
            _final_tick, _history_new, was_stopped = backend.run(
                n_steps=n_steps,
                record_every=record_every,
                observation_mask=self._observation_mask,
                checkpoint_every=checkpoint_every,
            )
        finally:
            self._rust_run_active = False

        # The session owns the state, the tick, and the finished marker:
        # ``pop.tick``/``is_finished`` read them natively, and the cached
        # container refreshes lazily on the next read.
        self._mark_state_cache_stale()
        # Bound native HistoryStore receives records during the session run.

        if was_stopped:
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
        """Run the population for *n_steps* ticks through the Rust engine.

        Args:
            n_steps: Number of ticks to simulate.
            record_every: Interval for recording history snapshots.
                Defaults to ``self.record_every``.
            finish: If True, trigger the finish event after the run.
            clear_history_on_start: If True, clear history before running.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the population has already finished or the
                native engine extension is unavailable (the session is
                created by ``build()`` or lazily at the first run).
        """
        self._require_standalone_owner("run")
        if getattr(self, "_running", False):
            raise RuntimeError("Nested run is forbidden")
        if self.is_failed:
            raise RuntimeError("Population has failed; restore a checkpoint or reset before run")
        if self.is_finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. Cannot run() again after finish=True."
            )

        self._running = True
        try:
            record_every_resolved = (
                record_every if record_every is not None else self.record_every
            )

            return self._run_rust_lifecycle(
                n_steps=n_steps,
                record_every=record_every_resolved,
                finish=finish,
                clear_history_on_start=clear_history_on_start,
            )
        except BaseException:
            # The session itself marks the execution Failed when the native
            # run or an in-run callback aborts; only the cached state needs
            # a staleness mark here.
            self._mark_state_cache_stale()
            raise
        finally:
            self._running = False

    def run_tick(self) -> DiscreteGenerationPopulation:
        """Run a single simulation tick.

        Returns:
            Self for chaining.
        """
        return self.run(n_steps=1, record_every=self.record_every)

    def reset(self) -> None:
        """Reset tick, history, and population state to initial values."""
        self._require_standalone_owner("reset")
        self._tick = 0
        if self._history_obj is not None:
            self._history_obj.clear()
        # Guard against calls before __init__ finishes (e.g. during
        # BasePopulation.__init__ -> _initialize -> reset chain).
        if hasattr(self, "_initial_population_snapshot"):
            ind_copy, _, _ = self._initial_population_snapshot
            self._state = DiscretePopulationState.create(
                n_sexes=self.config.n_sexes,
                n_ages=self.config.n_ages,
                n_ztypes=self.config.n_ztypes,
                n_tick=0,
                individual_count=ind_copy.copy(),
            )
        backend = self._rust_lifecycle_backend
        if backend is not None and self._state is not None:
            # The session owns the runtime state: the reset container
            # becomes the new session state.
            backend.set_state(self._state)
            backend.reseed(int(self._rust_backend_seed or 0))
            backend.clear_checkpoints()
            self._state_cache_stale = False

    def _native_counts(self) -> tuple[float, float, float] | None:
        """Sum per-sex counts natively when a live session owns the state.

        Returns:
            ``(total, female, male)`` from the session, or ``None`` when
            no session exists yet (the local container is then
            authoritative and stays the numpy-sum fallback source).
        """
        backend = self._rust_lifecycle_backend
        if backend is None:
            return None
        return backend.counts()

    def get_total_count(self) -> int:
        """Return the total number of individuals across all categories."""
        counts = self._native_counts()
        if counts is not None:
            return int(round(counts[0]))
        return int(round(np.sum(self._live_state().individual_count)))

    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        counts = self._native_counts()
        if counts is not None:
            return int(round(counts[1]))
        return int(
            round(self._live_state().individual_count[int(Sex.FEMALE.value)].sum())
        )

    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        counts = self._native_counts()
        if counts is not None:
            return int(round(counts[2]))
        return int(
            round(self._live_state().individual_count[int(Sex.MALE.value)].sum())
        )

    def clear_history(self) -> None:
        """Clear history rows and the paired session checkpoints."""
        self._require_standalone_owner("clear_history")
        super().clear_history()

    def export_state(self) -> NDArray[np.float64]:
        """Export the current state as a flat array.

        Returns:
            NDArray: Flattened state array.
        """
        return self._live_state().flatten_all()

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
        self._require_standalone_owner("import_config")
        self._install_config(_require_discrete_config(config))

    def import_state(
        self,
        state: Union[
            DiscretePopulationState, NDArray[np.float64], Dict[str, np.ndarray]
        ],
    ) -> None:
        """Replace the current state and reset the history timeline.

        All validation happens before any mutation — a failed import leaves the
        population unchanged.

        Args:
            state: New state as a ``DiscretePopulationState``, flat ndarray,
                or dict with ``individual_count`` key.
        """
        self._require_standalone_owner("import_state")
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
                n_tick=int(state.get("n_tick", self.tick)),
                individual_count=np.asarray(
                    state["individual_count"], dtype=np.float64
                ),
            )

        expected_shape = self._live_state().individual_count.shape
        if state_obj.individual_count.shape != expected_shape:
            raise ValueError(
                "individual_count shape mismatch: expected "
                f"{expected_shape}, got {state_obj.individual_count.shape}"
            )

        self._validate_import_values(state_obj)
        candidate = DiscretePopulationState(
            n_tick=int(state_obj.n_tick),
            individual_count=state_obj.individual_count.copy(),
        )
        backend = self._rust_lifecycle_backend
        if backend is not None:
            # Publish Python views only after the native transaction accepts
            # the complete candidate; a rejected import preserves the timeline.
            backend.set_state(candidate)
        self._state = candidate
        self._tick = candidate.n_tick
        self._state_cache_stale = False
        self.clear_history()

    def _refresh_state_cache_from_session(self) -> None:
        """Pull the session-owned state into the local cache."""
        backend = self._rust_lifecycle_backend
        if backend is None:
            return
        tick, ind_flat = backend.state_snapshot()
        n_ztypes = int(self.config.n_ztypes)
        self._state = DiscretePopulationState(
            n_tick=int(tick),
            individual_count=ind_flat.reshape(2, 2, n_ztypes).copy(),
        )
        self._state_cache_stale = False

    def _snapshot_state(self) -> DiscretePopulationState:
        """Copy the live state container for the public :attr:`state` snapshot.

        Returns:
            A fresh ``DiscretePopulationState`` with a copied count
            array; writes through it never reach the engine.
        """
        src = self._live_state()  # lazily pulls the session snapshot under Rust
        assert src is not None  # the base property guards initialization
        return DiscretePopulationState(
            n_tick=int(src.n_tick),
            individual_count=src.individual_count.copy(),
        )

    def update(self) -> RuntimeUpdater:
        """Return a ``RuntimeUpdater`` for modifying this population's parameters."""
        return self._create_updater()

    def __repr__(self) -> str:
        """Return a string summary of the discrete-generation population."""
        status = "Finished" if self.is_finished else "Active"
        return f"<DiscreteGenerationPopulation(name='{self.name}', tick={self.tick}, status={status})>"
