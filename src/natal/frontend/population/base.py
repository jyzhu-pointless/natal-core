"""Base population model helpers and abstractions.

This module provides the abstract base class and utilities for population
models (discrete-generation and age-structured). The base class defines
common interfaces, evolution methods, history management, and helpers
that are implemented by concrete population classes.

This module provides a common abstraction layer for population models while
keeping internal state representations compatible with the NumPy-based engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.frontend.data import (
    DiscretePopulationState,
    ModelDefinition,
    ModelDraft,
    PopulationState,
)
from natal.frontend.genetics import Genotype, HaploidGenotype, Species
from natal.frontend.hooks._compile import build_hook_program
from natal.frontend.hooks.types import (
    EVENT_ID_MAP,
    RESULT_CONTINUE,
    CompiledHookDescriptor,
    HookProgram,
)
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.registry.index import IndexRegistry

"""Runtime fields pulled into the session by the run-boundary flush
every ecology scalar, every vector column, the custom slots,
and the genetics tensors.  In-run writes (hook callbacks, deferred
pushes) land in the draft only while the session owns its borrow; the
flush re-materializes the contract params and pulls this whole list, so
the next run starts from the user-visible draft values.  The list is a
writable subset of the contract fields — values keep the session alive;
structural changes (modifier maps rebuilt at a different width) still
rebuild through ``_rust_needs_rebuild``.
"""
RUNTIME_FLUSH_FIELDS: tuple[str, ...] = (
    "carrying_capacity",
    "eggs_per_female",
    "sex_ratio",
    "sperm_displacement_rate",
    "low_density_growth_rate",
    "growth_mode",
    "external_expected_eggs",
    "survival_rates",
    "mating_rates",
    "reproduction_rates",
    "fertility",
    "competition_weights",
    "equilibrium_distribution",
    "migration_rate",
    "custom_slots",
    "viability_fitness",
    "fecundity_fitness",
    "sexual_selection_fitness",
    "zygote_viability_fitness",
    "offspring_tensor",
    "meiosis_map",
    "female_ztype_compatibility",
    "male_ztype_compatibility",
)


T_State = TypeVar("T_State", bound=Union[PopulationState, DiscretePopulationState])

if TYPE_CHECKING:
    from typing import Protocol, Self

    from natal._engine_rs import HistoryStore, ParameterLog
    from natal.frontend.builder import RuntimeUpdater
    from natal.frontend.builder._writers import AuditValue, SessionChannel
    from natal.frontend.hooks.tick_context import HookRunner, TickContext
    from natal.frontend.output._recording import RecordingPlan
    from natal.frontend.output.history import History
    from natal.frontend.output.observation import Observation, ObservationResult
    from natal.frontend.population._params_view import ParamsView
    from natal.frontend.presets import GeneticPreset

    class _CallbackBridge(Protocol):
        """Structural type of the Rust backend adapter's callback channel."""

        def set_python_callbacks(
            self,
            first: List[Callable[..., int]],
            early: List[Callable[..., int]],
            late: List[Callable[..., int]],
            finish: List[Callable[..., int]] | None = None,
        ) -> None:
            """Register per-event callback lists."""
            ...

    class _RecordingBackend(Protocol):
        """Structural type of the backend surfaces the History binds to."""

        def bind_history(self, store: HistoryStore, log: ParameterLog) -> None:
            """Attach the native history store and its parameter log."""
            ...

        def retain_checkpoints_from(self, from_tick: int) -> None:
            """Drop checkpoints older than *from_tick* (eviction pair)."""
            ...

# A parameter snapshot row: (tick, parameter name, old value, new value).
ParamChange = Tuple[int, str, float, float]

class BasePopulation(ABC, Generic[T_State]):
    """Abstract base class for population models.

    The base class unifies common behavior for different population model
    implementations (for example, discrete-generation and age-structured
    models). It manages the species/genetic architecture,
    indexing, the frozen hook plan, and modifier pipelines.

    Attributes:
        ALLOWED_EVENTS (List[str]): Event names supported by the hook system.
        species (Species): Genetic architecture descriptor for this population.
        name (str): Human-readable population name.
        tick (int): Current simulation tick.
        registry (IndexRegistry): Index registry for genotype/haplotype mappings.
        config (ModelDraft): Active static draft/config container.
        state (T_State): Active population state container.
        history (List[Tuple[int, np.ndarray]]): Recorded state snapshots by tick.
        compiled_hook_descriptors (tuple[CompiledHookDescriptor, ...]): Read-only
            snapshot of the compiled hook descriptors (CSR plans and Python
            callbacks) injected at build time, in declaration order.
            Homogeneous demes cloned from the same template share the
            underlying tuple via identity.
    """

    # Allowed hook events (subclasses may extend this list).
    ALLOWED_EVENTS = [
        "initialization",
        "first",
        "early",
        "late",
        "finish",
    ]

    # Set to True on demes of a SpatialPopulation: their genetics draft
    # tables start out shared, so in-place genetics writes would leak
    # across demes.  ParamsView.tensor_write refuses genetics writes on
    # such populations; the sanctioned channel is DemeSlice.write_genetics,
    # which forks the variant first.
    _shares_genetics_draft: bool = False

    # Frozen declaration snapshot, attached by
    # PopulationBuilder.build(); None until then (e.g. clones built via __new__).
    _definition: ModelDefinition | None = None

    # Cached projection mask for :meth:`observe`.  The Observation rule and
    # the population layout are both frozen at build time, so the mask is
    # compiled once and reused; clones rebuild it lazily (their rule object
    # is shared, so the rebuilt mask is value-identical).
    _observation_query_mask: Optional[np.ndarray] = None

    # Runtime reconfiguration log: the
    # build-time definition stays frozen; every committed preset
    # reconfiguration appends here so the post-build history of genetic
    # rule changes is replayable next to the frozen snapshot.  Annotation
    # only — the list is created per-instance (never a class-level
    # default, which every instance would share).
    # object: reconfigure_preset's **changes values are heterogeneous
    # user input (floats, dicts, ...), mirrored verbatim per entry.
    _reconfiguration_log: list[tuple[int, str, dict[str, object]]]

    _initial_population_snapshot: tuple[NDArray[np.float64], NDArray[np.float64] | None, None]
    _runtime_config_reader: Callable[[ModelDraft], ModelDraft] | None = None
    _runtime_state_reader: Callable[[], None] | None = None
    _current_definition: ModelDefinition | None = None
    _runtime_parameter_writer: SessionChannel | None = None
    # The active callback's event scope, registered by the native bridge
    # for the duration of one callback invocation.  Parameter reads and
    # writes resolve their transaction, candidate, pending log, and
    # metadata from it; no population field is ever swapped for a
    # callback.  ``None`` outside callbacks.
    _active_event: TickContext | None = None

    # Run-state authority is the native session: ``tick``, ``is_finished``,
    # and ``is_failed`` read the session's execution state (or, inside a
    # callback, the event context's tick).  ``_tick`` below is only the
    # session-less fallback for populations that never created a session.

    # Owning-container read channels for managed spatial demes: demes have
    # no private session, so their lifecycle status and tick project the
    # shared spatial session through these injected readers.  The
    # annotations carry ``None`` defaults so duck-typed hosts without the
    # channels degrade instead of raising.
    _runtime_execution_state_reader: Callable[[], tuple[str, int]] | None = None
    _runtime_tick_reader: Callable[[], int] | None = None

    # The (History, backend) pair whose native recording surfaces were
    # already bound once (observation selector, history store, checkpoint
    # pruner).  Holding the objects themselves (not ids) keeps the
    # identity comparison safe against garbage collection.
    _history_binding: tuple[History, object] | None = None

    # Subclass-owned native session factory (each model creates its own
    # Rust session type there); the base recording and manual-event
    # paths call it lazily.  Annotation only — duck-typed hosts without
    # a session never declare it, so readers probe via ``getattr``.
    # ``Callable[..., object]`` because each model returns its own
    # concrete population type.
    _initialize_session: Callable[..., object]

    def __init__(
        self,
        species: Species,
        name: str = "Population",
        hook_descriptors: Sequence[CompiledHookDescriptor] = (),
    ):
        """Initialize the base population.

        Args:
            species: Genetic architecture specifying chromosomes, loci, and alleles.
            name: Optional population name (default: "Population").
            hook_descriptors: Compiled hook plan (declarative CSR
                descriptors and Python callbacks) injected exactly once by
                the builder.  Populations never register hooks after
                construction; the packed program is fixed for this
                instance's lifetime.

        Note:
            Registry and genotypes are initialized lazily via Template Method.
            Subclasses must implement _create_registry() and _get_genotypes().
        """
        self._species = species
        self._name = name
        self._tick = 0
        # Deme index this population executes as: 0 for panmictic models,
        # the live deme index when a SpatialPopulation manages this object
        # as one of its demes.  Hooks read it via pop.deme_id, so the Rust
        # per-deme kernel and the Python reference lifecycle must agree.
        self._deme_id: int = 0
        # DELAYED: Registry will be created via _initialize_registry()
        self._index_registry: Optional[IndexRegistry] = None

        # Compiled hook plan: immutable descriptor tuple plus the CSR
        # program packed once at injection (callbacks interleaved with
        # declarative slots in one stable priority order).  Clones share
        # both via identity; nothing re-registers or reorders them.
        self._hook_descriptors: tuple[CompiledHookDescriptor, ...] = tuple(
            hook_descriptors
        )
        self._hook_program: HookProgram = build_hook_program(
            self._hook_descriptors, order_by_priority=True
        )

        # Frozen recording plan (installed by the builder at the end of
        # build(); None on clones until they copy the template's plan).
        self._recording_plan: Optional[RecordingPlan] = None

        # Self-describing History data model (frozen at build time).
        self._history_obj: Optional[History] = None

        # History config
        self.record_every: int = 1
        self.max_history: int = 5000  # Default rolling window size

        # Presets with priority IDs.  Writes go to _presets; derived
        # modifier lists are rebuilt from _presets + _manual_* on demand.
        self._presets: list[GeneticPreset] = []

        # Directly-added modifiers (manual, not from presets).
        self._manual_gamete: list[tuple[int, str | None, GameteModifier]] = []
        self._manual_zygote: list[tuple[int, str | None, ZygoteModifier]] = []

        # Derived modifier lists — rebuilt by refresh_modifiers().
        self._gamete_modifiers: list[tuple[int, str | None, GameteModifier]] = []
        self._zygote_modifiers: list[tuple[int, str | None, ZygoteModifier]] = []

        # Callback runner used to bridge Python callbacks into Rust sessions.
        self._hook_runner: Optional[HookRunner] = None

        # Static data container.
        self._config: Optional[ModelDraft] = None

        # PopulationState container.
        self._state: Optional[T_State] = None

        # Session-less fallback clock: the native session owns the
        # authoritative tick whenever one exists; this field only answers
        # queries for populations that never initialized a session (raw
        # construction, clones before their first run).
        self._tick = 0

        # One-time native recording binding (see the class annotation).
        self._history_binding = None

        # Re-entrancy guard flag.
        self._running = False

        # Observation-based history recording.
        self._observation: Optional[Observation] = None
        self._observation_mask: Optional[np.ndarray] = None

        # Parameter snapshot log (tick, name, old, new) appended by the
        # runtime writers on every committed scalar change.
        from natal._engine_rs import ParameterLog

        self._params_log = ParameterLog()

        # Rust dirty-set bridge: contract field names whose draft values
        # changed after the Rust session was built.  The next run() pulls
        # exactly these fields into the session (no rebuild, no RNG reset).
        # The sentinel "__blueprint__" forces a full backend rebuild instead.
        # Session ownership flags live in the model subclasses (their
        # backends are concrete types); base-class consumers reach them via
        # getattr so the annotation stays unclaimed here.

    @property
    def compiled_hook_descriptors(self) -> tuple[CompiledHookDescriptor, ...]:
        """Read-only snapshot of the hook plan installed at build time.

        The descriptor tuple is fixed at construction; there is no
        post-construction registration.  Declarative arrays were packed
        into the CSR program once at injection, so mutating the returned
        descriptors (or the declaration objects they were compiled from)
        cannot alter the installed plan.
        """
        return self._hook_descriptors

    def _clone(
        self,
        name: str,
        config: ModelDraft | None = None,
    ) -> Self:
        """Create a lightweight functional copy sharing compiled state and config.

        Used by ``SpatialPopulationBuilder`` to efficiently clone template demes without
        re-running hook compilation or preset application. The clone shares
        compiled hooks, index registry, modifier pipelines, and config arrays
        with the template. Only state arrays and history are independent.

        The *config* is stored by reference (no copy, no conversion): for a
        homogeneous spatial build every clone shares the same config object,
        which is what gives ``_dispatch_scalar`` its identity-based dedup
        invariant.  Subclasses must not re-normalize or ``_replace`` the
        config here — that would split shells and break the dedup.

        Args:
            name: Unique name for the clone.
            config: Optional config to use (default: template's config).
                Accepts a ``ModelDraft``; no conversion is performed.

        Returns:
            A new population instance of the same type with shared compiled state.
        """
        cls = type(self)
        clone = cls.__new__(cls)

        # --- rust dirty bridge (independent per deme) ---
        # Session-ownership attributes: clones start backend-less
        # with a fresh cache flag — __new__ skips every initializer, so a
        # missing attribute here would crash reset()/state reads later.
        # object.__setattr__ matches the __new__-host idiom used above.
        for _attr, _value in (
            ("_rust_lifecycle_backend", None),
            ("_rust_backend_seed", None),
            ("_rust_run_active", False),
            ("_state_cache_stale", False),
            ("_rust_needs_rebuild", False),
            ("_history_binding", None),
        ):
            object.__setattr__(clone, _attr, _value)

        # --- runtime provenance (independent per clone) ---
        clone._reconfiguration_log = []

        from natal._engine_rs import ParameterLog

        clone._params_log = ParameterLog()

        # --- shared identity ---
        clone._species = self._species
        clone._name = name
        # The fallback clock must match the template even when the template
        # already owns a session (the clone lazily creates its own).
        clone._tick = int(self.tick)
        # Clones are built via __new__ (no __init__), so the deme index
        # must be copied explicitly; a spatial deme's clone keeps its id
        # until SpatialPopulation.__init__ restamps the whole list.
        clone._deme_id = int(self._deme_id)

        # --- shared hooks (compiled once at build, read-only afterwards) ---
        # Clones share the descriptor tuple, the packed CSR program, and
        # the callback runner by identity — no re-registration, no
        # reordering, no re-execution of declarations.
        clone._hook_descriptors = self._hook_descriptors
        clone._hook_program = self._hook_program
        clone._hook_runner = self._hook_runner
        clone._recording_plan = self._recording_plan

        # --- shared registry ---
        clone._index_registry = self._index_registry

        # --- shared presets & modifiers ---
        clone._presets = list(self._presets)
        clone._manual_gamete = list(self._manual_gamete)
        clone._manual_zygote = list(self._manual_zygote)
        # Copy derived lists so clone starts with valid modifier state.
        # Without this, add_gamete_modifier(refresh=True) on a clone
        # would start from an empty list, dropping all preset modifiers.
        clone._gamete_modifiers = list(self._gamete_modifiers)
        clone._zygote_modifiers = list(self._zygote_modifiers)

        # --- config (shared reference for homogeneous, group-specific for heterogeneous) ---
        resolved_config = config if config is not None else self._config
        if resolved_config is None:
            raise ValueError("Cannot clone: population config is not initialized")
        clone._config = resolved_config

        # --- subclass-specific genotype caches ---
        for _attr in ('_genotypes_list', '_haploid_genotypes_list'):
            _val = getattr(self, _attr, None)
            if _val is not None:
                object.__setattr__(clone, _attr, _val)

        # --- fresh state: copy data from template ---
        template_state = self._live_state()
        state_cls = type(template_state)
        new_state = state_cls.create(
            n_ztypes=resolved_config.n_ztypes,
            n_sexes=resolved_config.n_sexes,
            n_ages=resolved_config.n_ages,
        )
        object.__setattr__(clone, '_state', new_state)
        # Live containers on both sides: the copy below must populate the
        # clone's engine arrays, not a snapshot.
        clone_state_nn = clone._live_state()
        self_state_nn = template_state
        clone_state_nn.individual_count[:] = self_state_nn.individual_count
        # sperm_storage only exists on PopulationState (age-structured), not
        # on DiscretePopulationState — use getattr for type-safe access.
        clone_sperm = getattr(clone_state_nn, 'sperm_storage', None)
        self_sperm = getattr(self_state_nn, 'sperm_storage', None)
        if clone_sperm is not None and self_sperm is not None:
            clone_sperm[:] = self_sperm

        # --- snapshot (handle both age-structured and discrete-generation formats) ---
        snap = getattr(self, '_initial_population_snapshot', None)
        if snap is not None:
            object.__setattr__(clone, '_initial_population_snapshot', (
                snap[0].copy(),
                snap[1].copy() if snap[1] is not None else None,
                snap[2],
            ))

        # --- runtime output policy (immutable schema, independent rows) ---
        clone._observation = self._observation
        clone._observation_mask = (
            None
            if self._observation_mask is None
            else self._observation_mask.copy()
        )
        source_history = self._history_obj
        if source_history is None:
            clone._history_obj = None
        else:
            from natal.frontend.output.history import History

            clone._history_obj = History(
                source_history.schema,
                max_rows=source_history.max_rows,
            )
        clone._running = False
        clone.record_every = int(self.record_every)
        clone.max_history = int(self.max_history)

        # subclass-specific runtime state (e.g. AgeStructuredPopulation.snapshots)
        if hasattr(self, 'snapshots'):
            object.__setattr__(clone, 'snapshots', {})

        return clone

    # ========================================================================
    # Lifecycle ownership
    # ========================================================================

    def _require_standalone_owner(self, operation: str) -> None:
        """Reject independent lifecycle control after transfer to a spatial owner."""
        if getattr(self, "_runtime_parameter_writer", None) is not None:
            raise RuntimeError(
                f"A managed deme cannot {operation} independently; "
                "use the owning SpatialPopulation for lifecycle and history control."
            )

    # ========================================================================
    # Registry and Genotype Initialization
    # ========================================================================

    def _initialize_registry(self) -> None:
        """Template method: Initialize registry and register all genotypes.

        If a registry was already provided (e.g. from PopulationBuilder, possibly
        compressed), it is reused.  Otherwise a fresh registry is created
        and populated from the Species.
        """
        # If a registry was already injected (e.g. compressed by PopulationBuilder),
        # keep it — don't overwrite with a fresh one.
        if self._index_registry is not None:
            return

        self._index_registry = self._create_registry()

        # Set somatic (slab) labels before registering genotypes —
        # register_genotype() auto-cross-products with slab_labels.
        raw_slabs = cast(Optional[List[str]], getattr(self._species, "somatic_labels", None))
        slabs = raw_slabs or ["default"]
        self._index_registry.slab_labels = slabs

        # Set gamete (glab) labels before registering haplogenotypes —
        # register_haplogenotype() auto-cross-products with glab_labels.
        raw_glabs = cast(Optional[List[str]], getattr(self._species, "gamete_labels", None))
        glabs = raw_glabs or ["default"]
        self._index_registry.glab_labels = glabs

        genotypes = self._get_genotypes()
        for genotype in genotypes:
            self._index_registry.register_genotype(genotype)

        haplogenotypes = self._get_haplogenotypes()
        if haplogenotypes:
            for hg in haplogenotypes:
                self._index_registry.register_haplogenotype(hg)

    # Helpers
    def _create_registry(self) -> IndexRegistry:
        """Create a new IndexRegistry for this population.

        Returns:
            A fresh IndexRegistry instance.
        """
        return IndexRegistry()

    def _get_genotypes(self) -> List[Genotype]:
        """Retrieve all diploid genotypes defined by the species.

        Returns:
            List of all Genotype objects.
        """
        return self.species.get_all_genotypes()

    def _get_haplogenotypes(self) -> Optional[List[HaploidGenotype]]:
        """Retrieve all haploid genotypes defined by the species.

        Returns:
            List of all HaploidGenotype objects, or None if not available.
        """
        return self.species.get_all_haploid_genotypes()

    # ========================================================================
    # Basic properties
    # ========================================================================

    @property
    def species(self) -> Species:
        """The species/genetic architecture for this population."""
        return self._species

    @property
    def name(self) -> str:
        """The human-readable name of the population."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the population name."""
        self._name = value

    def _native_lifecycle_state(self) -> tuple[str, int] | None:
        """Return the owning session's ``(status, tick)``, or ``None``.

        The native session is the single authority for the execution
        status and the clock.  Managed spatial demes have no private
        session; their status and tick project the shared spatial session
        through the injected container readers.  Inside a callback (or
        while a run holds the session borrow) no native call is made —
        the caller resolves the value from the event context or the
        session-less fallback instead, so a hook can never re-enter the
        borrowed session.

        Returns:
            The native ``(status name, tick)``, or ``None`` when no
            session answer is available right now.
        """
        if self._active_event is not None or getattr(self, "_rust_run_active", False):
            return None
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None:
            state_reader = getattr(backend, "execution_state", None)
            tick_reader = getattr(backend, "current_tick", None)
            if callable(state_reader) and callable(tick_reader):
                # cast: the backend attribute is subclass-owned and every
                # adapter (panmictic, discrete) exposes the same
                # ``(status, phase)`` / tick pair, which the base class
                # cannot see statically.
                status, _phase = cast("tuple[str, int]", state_reader())
                return (str(status), int(cast("int", tick_reader())))
            return None
        state_reader = getattr(self, "_runtime_execution_state_reader", None)
        tick_reader = getattr(self, "_runtime_tick_reader", None)
        if state_reader is not None and tick_reader is not None:
            status, _phase = state_reader()
            return (str(status), int(tick_reader()))
        return None

    def _native_tick(self) -> int | None:
        """Return the session tick, or ``None`` without a live session."""
        state = self._native_lifecycle_state()
        if state is not None:
            return state[1]
        return None

    @property
    def tick(self) -> int:
        """The current simulation tick or generation index (read-only).

        The tick is read from the owning native session; inside a hook
        callback it is the tick the event fired at.  Only populations
        that never initialized a session fall back to the local clock.
        Assigning raises instead (see the setter).
        """
        event = self._active_event
        if event is not None:
            return int(event.tick)
        native = self._native_tick()
        if native is not None:
            return native
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:
        """Reject clock changes outside an owning lifecycle operation."""
        self._require_standalone_owner("set tick")
        raise RuntimeError(
            "tick is read-only; use run(), reset(), or restore_checkpoint() "
            "to change the simulation clock."
        )

    @property
    def registry(self) -> IndexRegistry:
        """IndexRegistry instance managing genotype, haplotype, and label indices."""
        return self.index_registry

    @property
    def index_registry(self) -> IndexRegistry:
        """Public accessor for the internal IndexRegistry.

        ``registry`` is a retained alias reading the same field.

        Raises:
            AttributeError: If the registry has not been initialized yet.
        """
        if self._index_registry is None:
            raise AttributeError("Index registry has not been initialized.")
        return self._index_registry

    @property
    def config(self) -> ModelDraft:
        """Public accessor for compiled population configuration."""
        if self._config is None:
            raise AttributeError("Population config has not been initialized.")
        from copy import deepcopy

        event = self._active_event
        if event is not None:
            # Inside a callback the committed answer is the event's
            # candidate; materialize it lazily and hand out a copy.
            candidate = event.materialize_candidate()
            if candidate is not None:
                return deepcopy(candidate)
        reader = getattr(self, "_runtime_config_reader", None)
        if reader is not None and not getattr(self, "_rust_run_active", False):
            return reader(self._config)
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None and not getattr(self, "_rust_run_active", False):
            return backend.config_snapshot(self._config)
        return deepcopy(self._config)

    def _install_config(self, config: object) -> None:
        """Validate and commit an explicit configuration import to the live session."""
        from copy import deepcopy

        from natal.contracts.materialize import materialize_params

        old = self._config
        if old is None:
            raise RuntimeError("The model has not been initialized.")
        if not isinstance(config, ModelDraft):
            raise TypeError("config must be a ModelDraft")
        layout = ("n_sexes", "n_ages", "n_ztypes", "n_gtypes", "n_glabs", "n_slabs", "new_adult_age", "adult_ages", "ztype_names", "gtype_names")
        if any(not np.array_equal(getattr(old, field), getattr(config, field)) for field in layout):
            raise ValueError("Configuration import cannot change the model's active layout.")
        candidate = deepcopy(config)
        params = materialize_params(candidate)
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None:
            backend.refresh_params(list(params.__dataclass_fields__), params)
        self._config = candidate
        self._mark_rust_dirty()

    def set_config(self, config: ModelDraft) -> None:
        """Replace this population's configuration."""
        self._config = config

    def _event_candidate(self) -> ModelDraft | None:
        """Return the active callback's candidate, materializing it lazily.

        Returns:
            The event candidate draft, or ``None`` when no callback is
            active or the callback has no native transaction.
        """
        event = self._active_event
        if event is None:
            return None
        return event.materialize_candidate()

    def _prepared_event_candidate(self) -> ModelDraft | None:
        """Return the active callback's candidate only if already prepared.

        Never materializes anything, so bare (context-free) reads stay
        lazy inside a callback.

        Returns:
            The prepared candidate draft, or ``None``.
        """
        event = self._active_event
        if event is None:
            return None
        return event.prepared_candidate()

    def _param_audit_sink(self) -> Callable[[str, AuditValue, AuditValue], None]:
        """Return the typed audit sink for runtime parameter writes.

        Returns:
            The active callback's pending-log sink during a callback, or
            the population's own :meth:`log_param_value` otherwise.
        """
        event = self._active_event
        if event is not None:
            return event.audit_sink()
        return self.log_param_value

    def _create_updater(self) -> RuntimeUpdater:
        """Create the idle-session runtime updater for this population.

        Subclass ``update()`` methods call this helper so concrete return
        types do not need ``cast()``.
        """
        from natal.frontend.builder import RuntimeUpdater

        return RuntimeUpdater(self)

    @property
    def params(self) -> ParamsView:
        """Validated parameter surface (domain A).

        Ecology attributes are writable with bounds checking
        (``pop.params.carrying_capacity = 8000``); genetics tensors are
        read-only copies with pattern-index reads
        (``pop.params.viability_fitness[nt.Sex.MALE, 2, "A|a"]``) and an
        explicit ``tensor_write`` channel.  Every write goes through the
        same writer stack as ``pop.update()`` — draft, live Rust
        session, and dirty bridge stay in sync.

        Returns:
            A :class:`~natal.frontend.population._params_view.ParamsView`
            bound to this population.

        .. versionadded:: NEXT
        """
        from natal.frontend.population._params_view import ParamsView

        return ParamsView(self)

    @property
    def params_log(self) -> Tuple[ParamChange, ...]:
        """Read-only parameter snapshot log.

        Runtime writers append one ``(tick, name, old, new)`` row per
        committed scalar change — no change, no row.  The log is the
        hook-audit trail: writes made through ``pop.params`` /
        ``pop.update()`` (including from inside hooks via the tick
        context) all land here.

        Returns:
            A tuple snapshot of the logged rows.
        """
        return tuple(self._params_log.snapshot())

    @property
    def params_log_details(self) -> Tuple[Tuple[int, str, int, str, bool | int | float | NDArray[np.float64] | None, bool | int | float | NDArray[np.float64] | None], ...]:
        """Return native commits as tick, event, deme, name, old, and new.

        The legacy params_log property remains a four-column projection;
        this query retains complete event and deme provenance and typed
        old/new values. None marks additions or deletions; arrays are copies.

        Returns:
            An immutable tuple of the current valid audit entries.
        """
        return tuple(self._params_log.details())

    def log_param_value(
        self, name: str, old: bool | int | float | NDArray[np.float64] | None,
        new: bool | int | float | NDArray[np.float64] | None, event: str = "update",
    ) -> None:
        """Record one successful typed parameter commit in native storage.

        Args:
            name: Parameter route or custom field name.
            old: Value before the commit, or None for a newly created field.
            new: Value after the commit, or None for a removed field.
            event: Responsible update operation or lifecycle event.
        """
        self._params_log.append_value(int(self.tick), name, old, new, event, self._deme_id)

    def log_param_change(self, name: str, old: float, new: float) -> None:
        """Append one parameter snapshot row at the current tick.

        Called by the runtime writers at their commit point; rows are only
        produced when the value actually changes.

        Args:
            name: Route (user-facing) parameter name.
            old: Committed value before the write.
            new: Committed value after the write.
        """
        if old != new:
            self._params_log.append_detail((int(self.tick), name, float(old), float(new)), "update", self._deme_id)

    # ========================================================================
    # Modifier and preset management
    # ========================================================================

    def _session_target(self) -> Any:
        """Resolve the idle-session commit target for this population.

        Returns:
            The resolved ``_UpdateTarget`` (typed ``Any`` here: the
            runtime target classes are internal to the builder
            package and the population only forwards them).

        Raises:
            RuntimeError: If a run holds the session borrow.
        """
        from natal.frontend.builder._runtime import idle_session_target

        return idle_session_target(self)

    def reapply_preset_fitness(self) -> None:
        """Reset fitness tensors to 1.0 and re-apply all preset fitness patches.

        Called after structural changes to presets (addition, removal, or
        reconfiguration).  Only preset-derived fitness is restored — any
        fitness values set directly via ``pop.update().fitness()`` will be
        overwritten. This explicit reset clears stored manual-fitness patches.
        """
        from natal.frontend.builder._runtime import reset_preset_fitness

        if self._config is None:
            return
        reset_preset_fitness(self._session_target())

    def refresh_modifiers(self, rebuild_maps: bool = True) -> None:
        """Rebuild derived modifier lists and maps from _presets + _manual_*.

        Presets are applied in priority order, then manual modifiers are
        appended.  Modifier maps (zygotes_to_gametes_map,
        gametes_to_zygotes_map, offspring_tensor) are rebuilt from the
        combined list.

        Args:
            rebuild_maps: If ``True`` (default), also commit the rebuilt
                maps.  Set to ``False`` when the caller plans to batch
                multiple modifier registrations and will commit once
                afterward.
        """
        from natal.frontend.builder._runtime import (
            commit_genetic_update,
            compile_runtime_candidate,
            read_declaration,
        )

        target = self._session_target()
        declaration = read_declaration(target)
        old = target.live_draft()
        # map-only updates preserve current native fitness.
        products = compile_runtime_candidate(
            target.species, old, target.registry, declaration, preserve_fitness=True,
        )
        if rebuild_maps:
            commit_genetic_update(
                target, old, products.config, declaration,
                products.gamete_modifiers, products.zygote_modifiers,
            )
        else:
            self._gamete_modifiers = list(products.gamete_modifiers)
            self._zygote_modifiers = list(products.zygote_modifiers)

    def refresh_modifier_maps(self) -> None:
        """Rebuild the three modifier maps from current modifier lists.

        Recomputes ``zygotes_to_gametes_map``,
        ``gametes_to_zygotes_map``, and the derived ``offspring_tensor``
        through the unified compiler
        (:func:`natal.frontend.genetics.compile.compile_modifier_maps`)
        — the same spelling the build path uses, so the two entry
        points cannot drift (pinned bit-for-bit by the parity safety
        net).

        .. note::

            This method is called automatically by :meth:`refresh_modifiers`
            and by individual ``add_gamete_modifier`` /
            ``add_zygote_modifier`` when ``refresh=True``.
        """
        from natal.frontend.builder._runtime import recompile_modifier_maps

        if self._config is None or self._index_registry is None:
            return
        if not self._index_registry.index_to_haplo or not self._index_registry.index_to_genotype:
            return
        recompile_modifier_maps(self._session_target())

    def add_gamete_modifier(
        self,
        modifier: GameteModifier,
        name: Optional[str] = None,
        modifier_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a gamete-level modifier.

        Args:
            modifier: A ``GameteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            modifier_id: Optional numeric priority used for ordering.
            refresh: If True (default), immediately rebuild modifier maps.
                Set to False when adding multiple modifiers in a batch;
                call :meth:`refresh_modifiers` or
                :meth:`refresh_modifier_maps` afterward to apply all at once.
        """
        from natal.frontend.builder._runtime import add_manual_modifier

        add_manual_modifier(
            self._session_target(), "gamete", modifier, name, modifier_id,
            refresh=refresh,
        )

    def add_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        name: Optional[str] = None,
        modifier_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a zygote-level modifier.

        Args:
            modifier: A ``ZygoteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            modifier_id: Optional numeric priority used for ordering.
            refresh: If True (default), immediately rebuild modifier maps.
                Set to False when adding multiple modifiers in a batch;
                call :meth:`refresh_modifiers` or
                :meth:`refresh_modifier_maps` afterward to apply all at once.
        """
        from natal.frontend.builder._runtime import add_manual_modifier

        add_manual_modifier(
            self._session_target(), "zygote", modifier, name, modifier_id,
            refresh=refresh,
        )

    def add_preset(self, preset: GeneticPreset) -> None:
        """Add a preset to this population.

        Registration is idempotent by object identity: if the exact same
        preset instance is already in ``_presets``, this is a no-op.
        This prevents double-registration when ``presets(drive)`` is
        called twice — the alternative (appending twice) would cause
        ``refresh_modifiers()`` to build two copies of the preset's
        gamete/zygote modifier, double-applying its effect in the
        offspring tensor.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).
        """
        if not any(p is preset for p in self._presets):
            self._presets.append(preset)

    def apply_preset(self, preset: GeneticPreset) -> None:
        """Apply a genetic preset to this population.

        This is the preferred API for registering presets. The preset's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).

        Examples:
            >>> from natal.frontend.presets import HomingDrive
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     drive_conversion_rate=0.95
            ... )
            >>> population.apply_preset(drive)

        See Also:
            :class:`natal.frontend.presets.GeneticPreset` - Base class for creating custom presets
            :class:`natal.frontend.presets.HomingDrive` - Built-in gene drive preset
        """
        from natal.frontend.builder._runtime import (
            apply_runtime_presets,
        )

        apply_runtime_presets(self._session_target(), (preset,))

    @classmethod
    def builder(cls, species: Species) -> Any:
        """Create a builder for this population type.

        This is the recommended way to construct populations with presets.

        Args:
            species: Genetic architecture for the population.

        Returns:
            A builder instance for this population type (typed ``Any``:
            each concrete population class finalizes the builder type).

        Raises:
            NotImplementedError: Always — concrete population classes
                override this with their own builder entry point.

        Examples:
            >>> pop = (AgeStructuredPopulation.builder(species)
            ...     .set_age_structure(n_ages=10)
            ...     .add_preset(HomingModificationDrive(...))
            ...     .build())
        """
        raise NotImplementedError(f"{cls.__name__} must implement builder()")

    def register_gamete_labels(self, labels: Optional[Sequence[str]]) -> None:
        """
        Register gamete labels in the IndexRegistry.

        Args:
            labels: Sequence of string labels to register. Labels must be
                unique in the provided sequence. Existing labels are ignored.
        """
        if not hasattr(self, "_index_registry") or self._index_registry is None:
            raise RuntimeError("IndexRegistry not initialized; cannot register gamete labels")

        if labels is None:
            return

        # Normalize and validate input
        try:
            seq = list(labels)
        except Exception as e:
            raise TypeError("labels must be a sequence of strings") from e

        # Ensure provided labels are unique
        if len(set(seq)) != len(seq):
            raise ValueError("labels must be unique")

        # Register each string label if not already present
        for lab in seq:
            if lab not in self._index_registry.glab_labels:
                self._index_registry.glab_labels.append(lab)

    @property
    def presets(self) -> List[GeneticPreset]:
        """Return a snapshot of the presets applied to this population."""
        return list(self._presets)

    @property
    def gamete_modifiers(self) -> List[tuple[int, str | None, GameteModifier]]:
        """Return a snapshot of the registered gamete modifiers."""
        return list(self._gamete_modifiers)

    @property
    def zygote_modifiers(self) -> List[tuple[int, str | None, ZygoteModifier]]:
        """Return a snapshot of the registered zygote modifiers."""
        return list(self._zygote_modifiers)

    @abstractmethod
    def update(self) -> RuntimeUpdater:
        """Return a ``RuntimeUpdater`` for modifying this population.

        All chainable domain methods (``.competition()``,
        ``.reproduction()``, …) write changes immediately — no ``.apply()``
        or ``.freeze()`` needed for simple parameter updates.  The updater
        holds only its commit target and resolves every value against the
        live session at operation time; build-only methods (``.build()``,
        ``.setup()``, ``.hooks()``, …) do not exist on it.

        Examples:

            >>> pop.update().competition(carrying_capacity=5000)
            >>> pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

        .. versionadded:: NEXT
        """
        ...

    @property
    def reconfiguration_log(
        self,
    ) -> tuple[tuple[int, str, dict[str, object]], ...]:
        """The committed runtime preset reconfigurations, in order.

        Each entry is ``(tick, preset_name, changes)`` recorded after a
        reconfiguration committed; failed attempts append nothing.  The
        build-time declaration lives on :attr:`definition` — this log is
        the post-build provenance of genetic-rule changes.

        Returns:
            The ordered reconfiguration entries.
        """
        entries: list[tuple[int, str, dict[str, object]]] = (
            self.__dict__.get("_reconfiguration_log", [])
        )
        # Each returned entry carries a fresh dict copy — callers
        # cannot mutate the recorded history through the snapshot.
        return tuple(
            (tick, name, dict(changes)) for tick, name, changes in entries
        )

    @property
    def definition(self) -> ModelDefinition:
        """The frozen declaration snapshot this population was built from.

        Returns:
            The :class:`~natal.frontend.data.ModelDefinition` captured at
            build time.

        Raises:
            AttributeError: If the population was not built through
                ``PopulationBuilder.build()`` (no snapshot exists).
        """
        if self._definition is None:
            raise AttributeError(
                "This population has no declaration snapshot; it was not "
                "built through PopulationBuilder.build()."
            )
        return self._definition

    @property
    def state(self) -> T_State:
        """Return a point-in-time snapshot of the current state container.

        Snapshot discipline: long-lived callers outside
        hooks receive copies, so writing through the returned container
        can never reach the engine's live arrays.  The in-hook writable
        loan is a separate controlled channel (:class:`TickContext`).

        Returns:
            A fresh state container holding copied arrays and the
            current tick.

        Raises:
            AttributeError: If the state has not been initialized.
        """
        if self._state is None:
            raise AttributeError("Population state has not been initialized.")
        return self._snapshot_state()

    def _snapshot_state(self) -> T_State:
        """Build the snapshot returned by :attr:`state` (subclass hook).

        Returns:
            A fresh container with copied arrays; never the live
            ``_state`` itself.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _snapshot_state()"
        )

    def _live_state(self) -> T_State:
        """Return the live state container for internal engine paths.

        The narrowed, non-optional twin of ``_state``: every internal
        caller (state install, lifecycle write-back, migration stack)
        runs after construction created the container, so this accessor
        encodes that invariant instead of scattering Optional guards.
        The public :attr:`state` stays the snapshot face.

        Returns:
            The initialized live state container.

        Raises:
            RuntimeError: If the state has not been initialized.
        """
        reader = self._runtime_state_reader
        if reader is not None:
            reader()
        elif (
            getattr(self, "_rust_lifecycle_backend", None) is not None
            and getattr(self, "_state_cache_stale", False)
        ):
            self._refresh_state_cache_from_session()
        if self._state is None:
            raise RuntimeError("Population state has not been initialized.")
        return self._state

    @staticmethod
    def _validate_import_values(state: T_State) -> None:
        """Validate state values even before a lazily initialized native owner exists."""
        if state.n_tick < 0:
            raise ValueError("tick must be nonnegative")
        arrays = [state.individual_count]
        if isinstance(state, PopulationState):
            arrays.append(state.sperm_storage)
        if any(not np.isfinite(values).all() or np.any(values < 0) for values in arrays):
            raise ValueError("state counts must be finite and nonnegative")

    def _refresh_state_cache_from_session(self) -> None:
        """Pull the session-owned state into the local cache (subclass hook).

        Subclasses with a live Rust backend rebuild their state container
        from ``backend.state_snapshot()``.  The base default is a no-op so
        duck-typed hosts without a session stay functional.
        """
        return

    def _mark_rust_dirty(self) -> None:
        """Schedule program and execution-flag configuration at the run boundary.

        The owning session, current state, checkpoints, and RNG survive this
        update. Model subclasses own the pending-program marker.
        """
        self._rust_needs_rebuild = True

    def _mark_state_cache_stale(self) -> None:
        """Flag the cached state as behind the Rust session.

        Runs, restores, and direct session writes change the session-owned
        state; the next ``_live_state()`` read pulls a fresh snapshot.
        """
        self._state_cache_stale = True

    def _restore_ecology_to_draft(self, ecology: Mapping[str, object]) -> None:
        """Write a restored checkpoint's ecology into the draft.

        The Rust session has already restored its authoritative columns.
        Keep the declaration metadata's tensor shapes and optional presence
        consistent with that boundary for subsequent queries and writers.

        Args:
            ecology: Contract-name → value mapping from
                ``backend.restore_from_checkpoint`` (scalars as floats,
                vectors as arrays).
        """
        from natal.frontend.builder._writers import contract_to_draft_field

        # Draft ecology values are heterogeneous: scalars stay floats,
        # vectors become float64 ndarrays, and the eggs sentinel becomes
        # None (the Optional declaration).
        overrides: dict[str, float | NDArray[np.float64] | None] = {}
        custom = ecology.get("custom_slots")
        if isinstance(custom, dict) and self._config is not None:
            from natal.frontend.data import build_custom_slots

            self._config = self._config._replace(custom=build_custom_slots(cast("Mapping[str, object]", custom)))
        for name, value in ecology.items():
            draft_field = contract_to_draft_field(str(name))
            if not hasattr(self._config, draft_field):
                # e.g. migration_rate on panmictic drafts (never declared).
                continue
            current: float | NDArray[np.float64] | None = getattr(
                self._config, draft_field
            )
            if isinstance(value, np.ndarray):
                restored = np.array(value, dtype=np.float64)
                if name == "equilibrium_distribution":
                    assert self._config is not None
                    overrides[draft_field] = restored.reshape(2, int(self._config.n_ages)) if restored.size else None
                elif isinstance(current, np.ndarray):
                    # The wire carries deme-flattened vectors; the draft
                    # stores the structured shape (e.g. survival as
                    # (2, n_ages)).  Restore the declared shape so routed
                    # reads and update() writes keep working.
                    overrides[draft_field] = restored.reshape(current.shape)
            elif isinstance(value, float):
                if draft_field == "external_expected_eggs" and value < 0.0:
                    # Wire sentinel ↔ draft Optional translation.
                    overrides[draft_field] = None
                else:
                    overrides[draft_field] = value
        draft = self._config
        if overrides and draft is not None:
            self._config = draft._replace(**overrides)

    @property
    def history(self) -> History:
        """Return the self-describing History owned by this population.

        Returns:
            History whose immutable schema was frozen at build time.

        Raises:
            RuntimeError: If construction has not installed History yet.
        """
        if self._history_obj is None:
            raise RuntimeError("Population History has not been initialized.")
        return self._history_obj

    def _init_history_schema(
        self,
        *,
        kind: str,
        n_demes: int = 1,
        has_sperm_storage: bool = False,
    ) -> None:
        """Install the temporary raw History schema used during construction.

        PopulationBuilder replaces this schema with the final compiled recording
        plan before returning the built population.

        Args:
            kind: One of ``"age_structured"``, ``"discrete_generation"``.
            n_demes: Number of demes (1 for panmictic).
            has_sperm_storage: Whether sperm storage arrays are present.
        """
        from natal.frontend.output.history import (
            History,
            HistorySchema,
            PopulationLayout,
        )

        state = self._live_state()
        ind = state.individual_count
        n_sexes = int(ind.shape[0])
        n_ages = int(ind.shape[1]) if ind.ndim == 3 else 1
        n_ztypes = int(ind.shape[-1])

        sex_labels = ("female", "male")[:n_sexes]

        layout = PopulationLayout.from_population(
            kind=kind,
            n_demes=n_demes,
            n_sexes=n_sexes,
            n_ages=n_ages,
            n_ztypes=n_ztypes,
            has_sperm_storage=has_sperm_storage,
            sex_labels=sex_labels,
            registry=self.index_registry,
        )

        ind_size = n_sexes * n_ages * n_ztypes
        sperm_size = n_ages * n_ztypes * n_ztypes if has_sperm_storage else 0
        row_size = 1 + ind_size * n_demes + sperm_size * n_demes

        schema = HistorySchema(
            mode="raw",
            population=layout,
            row_size=row_size,
            observation=None,
            spatial_layout=None,
        )
        self._history_obj = History(schema, max_rows=self.max_history)

    # ========================================================================
    # History recording and checkpoint restore
    # ========================================================================

    def _bind_history_recording(self, backend: _RecordingBackend) -> None:
        """Bind the native recording surfaces once per (population, History).

        Three surfaces travel together: the observation selector compiled
        into the History store, the store/log ownership handed to the
        session, and the checkpoint pruner paired with history capacity.
        They are bound at the first recording boundary (first run,
        snapshot, or explicit event) and re-bound only when the History
        object or the session changes — the Observation rule and the
        schema are frozen at build time, so re-installing them per run
        would repeat identical work.

        Args:
            backend: The live lifecycle session adapter.
        """
        history_obj = self._history_obj
        if history_obj is None:
            return
        binding = self._history_binding
        if binding is not None and binding[0] is history_obj and binding[1] is backend:
            return
        if history_obj.schema.mode == "observation":
            history_obj._configure_observation(self.observation)  # pyright: ignore[reportPrivateUsage]  # Population binds its recording selector
        backend.bind_history(history_obj._store, self._params_log)  # pyright: ignore[reportPrivateUsage]  # share native ownership
        history_obj._bind_checkpoint_pruner(backend.retain_checkpoints_from)  # pyright: ignore[reportPrivateUsage]  # capacity changes synchronously release native checkpoints.
        self._history_binding = (history_obj, backend)

    def _record_current_snapshot(self, *, allow_existing: bool) -> None:
        """Commit the current state to the unique History container.

        Args:
            allow_existing: Whether an automatic run boundary may reuse the
                already-recorded current tick without writing a second row.

        Raises:
            RuntimeError: If History, Population state, or Observation is not
                initialized.
            ValueError: If a strict snapshot repeats or precedes the latest
                tick, or an automatic boundary is stale or has a different
                payload.
        """
        history_obj = self._history_obj
        if history_obj is None:
            raise RuntimeError("History is not initialized for this population.")
        if self._state is None:
            raise RuntimeError("Population state is not initialized.")
        if history_obj.schema.mode == "observation" and self._observation is None:
            raise RuntimeError("Observation is not initialized for this population.")
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is None:
            # Directly constructed populations initialize the same native
            # owner lazily; recording never creates a second state store.
            self._initialize_session()
            backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is None:
            # Unreachable with the model implementations: their
            # _initialize_session either installs the backend or raises.
            raise RuntimeError("Population session initialization failed.")
        self._bind_history_recording(backend)
        backend.record_history(allow_existing)

    def clear_history(self) -> None:
        """Remove all rows while preserving the frozen History schema.

        Session-side record checkpoints are dropped with the rows so a
        later ``restore_checkpoint`` cannot resurrect cleared ticks.
        """
        self._require_standalone_owner("clear_history")
        if self._history_obj is not None:
            self._history_obj.clear()
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None:
            backend.clear_checkpoints()

    def record_snapshot(self) -> None:
        """Record the current stable state into history.

        Must only be called when the engine is not running (between
        ``run()`` calls). Duplicate ticks are rejected.

        Raises:
            RuntimeError: If the population is currently running.
            ValueError: If the current tick is already recorded.
        """
        self._require_standalone_owner("record_snapshot")
        if getattr(self, "_running", False):
            raise RuntimeError(
                "Cannot record snapshot while the population is running."
            )
        self._record_current_snapshot(allow_existing=False)

    def restore_checkpoint(self, tick: int) -> None:
        """Restore the population to its recorded state at *tick*.

        Only valid for raw-mode history. The Rust session
        rolls back its record-aligned checkpoint in full — counts, sperm
        storage, the ecology parameters (so a post-record parameter change
        like ``update().competition(...)`` is undone), and the RNG stream
        (a restore continues the exact stream rather than reseeding).
        The recorded execution status and phase are restored as well.
        Future history and parameter logs are truncated to the checkpoint.

        Args:
            tick: Exact tick to restore.

        Raises:
            ValueError: If mode is not ``"raw"`` or tick is not found.
        """
        self._require_standalone_owner("restore_checkpoint")
        history_obj = getattr(self, "_history_obj", None)
        if history_obj is None or history_obj.is_empty:
            raise ValueError("No history available for checkpoint restore.")
        if history_obj.schema.mode != "raw":
            raise ValueError(
                "Cannot restore population state from observation-mode "
                "history.  Record raw history to enable checkpoint "
                "restoration."
            )
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None:
            result = backend.restore_from_checkpoint(tick)
            if result is None:
                # Frozen-surface message: checkpoints are record-aligned with the
                # history rows, so the frozen "not found in history" wording stays.
                raise ValueError(f"Tick {tick} not found in history.")
            _restored_tick, ecology = result
            self._restore_ecology_to_draft(ecology)
            self._mark_state_cache_stale()
            backend.truncate_checkpoints(tick)
            history_obj.truncate(retain_until_tick=tick)
            # The native checkpoint carries the recorded execution status
            # and the restored tick, so ``tick``/``is_finished``/
            # ``is_failed`` follow the restore with no Python-side
            # reconciliation.
            return
        restored_tick, ic, ss = history_obj.restore_state(tick)
        state = self._state
        if state is None:
            raise RuntimeError("Population state is not initialized.")
        state.individual_count[:] = ic.reshape(state.individual_count.shape)
        # sperm_storage only exists on PopulationState (age-structured), not
        # on DiscretePopulationState — use getattr for type-safe access.
        sperm = getattr(state, "sperm_storage", None)
        if ss is not None and sperm is not None:
            sperm[:] = ss.reshape(sperm.shape)
        self._state = state._replace(n_tick=restored_tick)
        self._tick = restored_tick
        history_obj.truncate(retain_until_tick=tick)

    # ========================================================================
    # Core methods
    # ========================================================================

    @abstractmethod
    def run_tick(self) -> BasePopulation[T_State]:
        """Execute one simulation tick.

        Typical sequence:
        1. Check termination and re-entrancy guards.
        2. Trigger ``first`` hooks.
        3. Run reproduction step.
        4. Trigger ``early`` hooks.
        5. Run survival step.
        6. Trigger ``late`` hooks.
        7. Run aging step.
        8. Increment tick and clear running flag.

        If any hook returns ``RESULT_STOP``, remaining steps are skipped and
        the population is marked as finished.

        Returns:
            BasePopulation[T_State]: ``self`` for chaining.

        Raises:
            RuntimeError: If the population is finished or already running.
        """
        pass

    def step(self) -> BasePopulation[T_State]:
        """Alias for `BasePopulation.run_tick()`"""
        return self.run_tick()

    @abstractmethod
    def run(
        self,
        n_steps: int,
        record_every: Optional[int] = None,
        finish: bool = False,
    ) -> BasePopulation[T_State]:
        """Run multi-step evolution for *n_steps* ticks.

        Concrete population classes implement the batch execution against
        their native session.

        Args:
            n_steps: Number of ticks to simulate.
            record_every: Snapshot interval; ``None`` uses the
                population's configured ``record_every``, ``0`` disables
                recording.
            finish: Whether to mark the population as finished afterwards.

        Returns:
            Self for chaining.

        Raises:
            RuntimeError: If the population is finished, failed, or
                already running.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the population to its initial state."""

    @abstractmethod
    def get_total_count(self) -> float:
        """Return the total number of individuals in the population."""
        pass

    @property
    def observation(self) -> Observation:
        """The immutable :class:`Observation` for this population.

        PopulationBuilder installs either an explicit rule or the canonical identity
        rule before returning the built Population.

        Raises:
            RuntimeError: If the Observation has not been initialized yet.
        """
        if self._observation is None:
            raise RuntimeError("Population Observation has not been initialized.")
        return self._observation

    def observe(self) -> ObservationResult:
        """Project current state through the population's observation.

        The projection mask is compiled once from the frozen Observation
        and layout, then reused across queries (the native projection
        only reads it).

        Returns:
            ObservationResult with projected values and explicit axes.

        Raises:
            RuntimeError: When state is not available yet.
        """
        obs = self.observation
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None:
            from types import MappingProxyType

            from natal.frontend.output.observation import ObservationResult

            layout = self.history.schema.population
            mask = self._observation_query_mask
            if mask is None:
                mask = obs.build_mask(layout.n_sexes, layout.n_ages, layout.n_ztypes)
                self._observation_query_mask = mask
            tick, values = backend.observe_current(mask, [0], obs.collapse_age, False)
            shape = (obs.n_groups, layout.n_sexes)
            if not obs.collapse_age:
                shape += (layout.n_ages,)
            return ObservationResult(
                tick=tick, _values=values.reshape(shape), axes=obs.axes,
                _labels=MappingProxyType({"group": obs.labels}),
            )
        state = getattr(self, "state", None)
        if state is None:
            raise RuntimeError("Population has no state to observe.")
        ic = getattr(state, "individual_count", None)
        if ic is None:
            raise RuntimeError("Population state has no individual_count.")
        return obs.project(ic, tick=self.tick)

    # ========================================================================
    # Hook dispatch and introspection
    # ========================================================================
    # The hook plan is compiled once by the builder and injected at
    # construction: the descriptor tuple and the packed CSR program are
    # fixed for the population's lifetime.  The population owns only the
    # execution and introspection side — native Rust sessions execute
    # declarative plans and bridge Python callbacks at event boundaries
    # in one stable priority order.  There is no runtime registration
    # channel.

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """Trigger an event and execute the hooks declared for it.

        Execution order per event: CSR declarative plans and Python
        callbacks interleaved by one stable ascending-priority order
        (each callback receives a fresh
        :class:`~natal.frontend.hooks.tick_context.TickContext`).

        Args:
            event_name: Event name to trigger.
            deme_id: Deme index the hooks execute as.  0 for panmictic
                populations; the live deme index when the population is
                managed by a SpatialPopulation.

        Returns:
            int: ``RESULT_CONTINUE`` (0) to continue, ``RESULT_STOP`` (1)
            to stop.
        """
        native = getattr(self, "_runtime_parameter_writer", None)
        if native is None:
            native = getattr(self, "_rust_lifecycle_backend", None)
        if native is None:
            # Directly constructed standalone populations do not create a
            # session until the first operation that needs native execution.
            # Managed spatial demes have a runtime writer and therefore stay
            # owned by their SpatialPopulation container.
            initialize = getattr(self, "_initialize_session", None)
            if callable(initialize):
                initialize(seed=int(getattr(self, "_rust_backend_seed", 0) or 0))
                native = getattr(self, "_rust_lifecycle_backend", None)
        if native is not None and hasattr(native, "trigger_event"):
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is None:
                return RESULT_CONTINUE
            # Refresh the installed program without replacing session
            # state/RNG (manual events use the same native log as run
            # checkpoints).
            if getattr(self, "_runtime_parameter_writer", None) is None:
                self._bind_history_recording(native)
                native.configure_program(self._hook_program, self.config)
                self._register_rust_callbacks(native)
            if getattr(self, "_runtime_parameter_writer", None) is None:
                result = int(native.trigger_event(event_id, deme_id))
            else:
                result = int(native.trigger_event(event_id))
            self._mark_state_cache_stale()
            return result
        raise RuntimeError(
            "Native hook execution is unavailable; initialize a Rust session "
            "before triggering events."
        )

    def get_compiled_hooks(
        self, event: Optional[str] = None
    ) -> List[CompiledHookDescriptor]:
        """Get compiled hook descriptors, optionally filtered by event.

        Args:
            event: Optional event name to filter by.

        Returns:
            List of ``CompiledHookDescriptor`` sorted by priority.
        """
        hooks = list(self.compiled_hook_descriptors)
        if event is not None:
            hooks = [h for h in hooks if h.event == event]
        return sorted(hooks, key=lambda h: h.priority)

    def has_python_callbacks(self) -> bool:
        """Return whether the injected plan contains a Python callback."""
        return any(desc.callback is not None for desc in self.compiled_hook_descriptors)

    def has_python_hooks(self) -> bool:
        """Back-compatible alias for :meth:`has_python_callbacks`."""
        return self.has_python_callbacks()

    def _ensure_hook_runner(self) -> HookRunner:
        """Return the callback runner, building it on first use."""
        if self._hook_runner is None:
            from natal.frontend.hooks.tick_context import HookRunner

            self._hook_runner = HookRunner(self)
        runner: HookRunner = self._hook_runner
        return runner

    def _register_rust_callbacks(self, backend: _CallbackBridge) -> None:
        """Bridge Python callbacks into a Rust session.

        One adapter per in-tick event (first/early/late); each adapter has
        the Rust ``(ind, sperm, tick, deme_id) -> int`` signature and runs
        every callback of its event in priority order.  Events without
        callbacks register an empty list so Rust kernels skip the GIL
        boundary entirely.
        """
        from natal.frontend.hooks.types import (
            EVENT_EARLY,
            EVENT_FINISH,
            EVENT_FIRST,
            EVENT_LATE,
        )

        runner = self._ensure_hook_runner()
        backend.set_python_callbacks(
            runner.rust_callbacks(EVENT_FIRST),
            runner.rust_callbacks(EVENT_EARLY),
            runner.rust_callbacks(EVENT_LATE),
            runner.rust_callbacks(EVENT_FINISH),
        )

    @abstractmethod
    def get_female_count(self) -> float:
        """Return the total number of female individuals."""
        pass

    @abstractmethod
    def get_male_count(self) -> float:
        """Return the total number of male individuals."""
        pass

    # ========================================================================
    # Population queries (count aliases)
    # ========================================================================

    @property
    def total_population_size(self) -> float:
        """Total population size (alias of ``get_total_count``)."""
        return self.get_total_count()

    @property
    def total_females(self) -> float:
        """Total number of females (alias of ``get_female_count``)."""
        return self.get_female_count()

    @property
    def total_males(self) -> float:
        """Total number of males (alias of ``get_male_count``)."""
        return self.get_male_count()

    @property
    def sex_ratio(self) -> float:
        """Return the female-to-male ratio, or ``np.inf`` when male count is zero."""
        males = self.get_male_count()
        return self.get_female_count() / males if males > 0 else np.inf

    # ========================================================================
    # Simulation lifecycle status
    # ========================================================================

    @property
    def is_finished(self) -> bool:
        """Whether the owning session recorded a stopped execution.

        A run ends Stopped when a hook requested a stop or
        :meth:`finish_simulation` locked the population; the marker
        lives on the native session, so a checkpoint restore restores it
        together with the rest of the recorded boundary and ``reset()``
        clears it.  Populations that never initialized a session are
        never finished.

        Inside a hook callback the session cannot be queried (the run
        holds its borrow), so the answer comes from the event scope: the
        ``finish`` event executes while the population is being locked,
        and finish hooks observe the finished population like they did
        before the status moved into the session.
        """
        state = self._native_lifecycle_state()
        if state is not None:
            return state[0] == "Stopped"
        event = self._active_event
        if event is not None:
            return event.event == "finish"
        return False

    @property
    def is_failed(self) -> bool:
        """Whether the owning session recorded a failed execution.

        A session marks itself Failed when a run or explicit event
        aborted; ``restore_checkpoint`` or :meth:`reset` returns it to a
        runnable boundary.  Populations that never initialized a session
        are never failed.
        """
        state = self._native_lifecycle_state()
        if state is not None:
            return state[0] == "Failed"
        return False

    def finish_simulation(self) -> None:
        """
        End simulation, trigger the ``finish`` event, and lock the population.

        After calling it, ``step()``, ``run_tick()``, and ``run()`` cannot run again.
        From inside a hook callback, request the same outcome through the
        context's ``stop()`` — the stopped run marks the session Stopped
        and the ``finish`` event fires, so the population locks the same
        way; calling this method on the population while a run holds the
        session borrow would re-enter the native session.

        Raises:
            RuntimeError: If the population is already finished.

        Examples:
            >>> builder = nt.DiscreteGenerationPopulation.setup(species, name="demo")
            >>> def finish_early(ctx):
            ...     ctx.stop()
            >>> pop = builder.hooks(finish_early, event='late').build()
        """
        self._require_standalone_owner("finish_simulation")
        if self.is_finished:
            raise RuntimeError(
                f"Population '{self.name}' has already finished."
            )

        # The finished marker is the session's Stopped status.  Stop the
        # session *before* the finish event so finish hooks already
        # observe ``is_finished``, and stop even when the event itself
        # fails (the population must stay locked on either path).
        backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is None and getattr(self, "_initialize_session", None) is not None:
            # The finish event lazily creates the session below; create it
            # here so the Stopped marker lands natively first.
            self._initialize_session(seed=int(getattr(self, "_rust_backend_seed", 0) or 0))
            backend = getattr(self, "_rust_lifecycle_backend", None)
        if backend is not None:
            backend.stop()
        self.trigger_event("finish", deme_id=self._deme_id)

    # ========================================================================
    # Allele frequency computation
    # ========================================================================

    def compute_allele_frequencies(self) -> Dict[str, float]:
        """
        Compute frequencies of all alleles in the population, normalized per locus.

        Returns:
            Dict[str, float]: Mapping ``{allele_name: frequency}``.
            Frequencies are per-locus proportions in the range ``[0.0, 1.0]``.
        """
        if self._state is None or self._index_registry is None:
            return {}

        # A Rust-backed population marks its Python snapshot stale after a run.
        # Read through the live-state boundary so this query observes the
        # current session state without requiring a separate public state read.
        state = self._live_state()

        # 1. Initialize counters.
        allele_counts: Dict[str, float] = {}
        locus_totals: Dict[str, float] = {}  # locus_name -> total_count

        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                locus_totals[locus.name] = 0.0
                for gene in locus.alleles:
                    allele_counts[gene.name] = 0.0

        # 2. Aggregate genotype counts.
        # individual_count shape: (n_sexes, n_ages, n_genotypes)
        # Sum over sex and age to get total count per genotype.
        genotype_counts = state.individual_count.sum(axis=(0, 1))

        registry = self._index_registry
        for z_idx, (genotype, _slab) in enumerate(registry.index_to_ztype):
            count = genotype_counts[z_idx]
            if count <= 0:
                continue

            for chrom in self.species.chromosomes:
                for locus in chrom.loci:
                    mat, pat = genotype.get_alleles_at_locus(locus)
                    for allele in (mat, pat):
                        if allele is not None:
                            allele_counts[allele.name] += count
                            locus_totals[locus.name] += count

        # 3. Compute frequencies.
        frequencies: Dict[str, float] = {}
        for allele_name, count in allele_counts.items():
            # Lookup the locus total for this allele.
            # We do not keep a direct fast gene->locus reverse index here,
            # so we safely resolve via species.gene_index.
            gene = self.species.gene_index.get(allele_name)
            if gene and locus_totals[gene.locus.name] > 0:
                frequencies[allele_name] = count / locus_totals[gene.locus.name]
            else:
                frequencies[allele_name] = 0.0

        return frequencies

    def __repr__(self) -> str:
        """Return a string summary of the population state."""  # noqa: D400
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tick={self.tick}, "
            f"size={self.get_total_count()})"
        )
