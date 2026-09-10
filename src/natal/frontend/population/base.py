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
from natal.frontend.hooks.types import HookProgram
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.population._mixins._output import OutputMixin
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
    from typing import Self

    from natal.frontend.configurator import RuntimeUpdater
    from natal.frontend.configurator._writers import AuditValue, SessionChannel
    from natal.frontend.hooks import (
        CompiledHookDescriptor,
    )
    from natal.frontend.hooks.tick_context import HookRunner, TickContext
    from natal.frontend.output._recording import RecordingPlan
    from natal.frontend.output.history import History
    from natal.frontend.output.observation import Observation, ObservationResult
    from natal.frontend.population._params_view import ParamsView
    from natal.frontend.presets import GeneticPreset

# A parameter snapshot row: (tick, parameter name, old value, new value).
ParamChange = Tuple[int, str, float, float]

class BasePopulation(OutputMixin, ABC, Generic[T_State]):
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
    # Configurator.build(); None until then (e.g. clones built via __new__).
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

        # Evolution status flag: whether simulation is finished.
        self._finished = False

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

        Used by ``SpatialConfigurator`` to efficiently clone template demes without
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
        ):
            object.__setattr__(clone, _attr, _value)

        # --- runtime provenance (independent per clone) ---
        clone._reconfiguration_log = []

        from natal._engine_rs import ParameterLog

        clone._params_log = ParameterLog()

        # --- shared identity ---
        clone._species = self._species
        clone._name = name
        clone._tick = int(self._tick)
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
        clone._finished = False
        clone._running = False
        clone.record_every = int(self.record_every)
        clone.max_history = int(self.max_history)

        # subclass-specific runtime state (e.g. AgeStructuredPopulation.snapshots)
        if hasattr(self, 'snapshots'):
            object.__setattr__(clone, 'snapshots', {})

        return clone

    # ========================================================================
    # Registry and Genotype Initialization
    # ========================================================================

    def _initialize_registry(self) -> None:
        """Template method: Initialize registry and register all genotypes.

        If a registry was already provided (e.g. from Configurator, possibly
        compressed), it is reused.  Otherwise a fresh registry is created
        and populated from the Species.
        """
        # If a registry was already injected (e.g. compressed by Configurator),
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
    def name(self) -> str:  # type: ignore[reportIncompatibleVariableOverride]  # property override from mixin
        """The human-readable name of the population."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:  # type: ignore[reportIncompatibleVariableOverride]  # property override from mixin
        """Set the population name."""
        self._name = value

    @property
    def tick(self) -> int:  # type: ignore[reportIncompatibleVariableOverride]  # property override from mixin
        """The current simulation tick or generation index (read-only)."""
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:  # type: ignore[reportIncompatibleVariableOverride]  # property override from mixin
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
        from natal.frontend.configurator import RuntimeUpdater

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
        self._params_log.append_value(int(self._tick), name, old, new, event, self._deme_id)

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
            self._params_log.append_detail((int(self._tick), name, float(old), float(new)), "update", self._deme_id)

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
                ``Configurator.build()`` (no snapshot exists).
        """
        if self._definition is None:
            raise AttributeError(
                "This population has no declaration snapshot; it was not "
                "built through Configurator.build()."
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
        from natal.frontend.configurator._writers import contract_to_draft_field

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

        Configurator replaces this schema with the final compiled recording
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
    def get_total_count(self) -> float:
        """Return the total number of individuals in the population."""
        pass

    @property
    def observation(self) -> Observation:
        """The immutable :class:`Observation` for this population.

        Configurator installs either an explicit rule or the canonical identity
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
        return obs.project(ic, tick=self._tick)

    @abstractmethod
    def get_female_count(self) -> float:
        """Return the total number of female individuals."""
        pass

    @abstractmethod
    def get_male_count(self) -> float:
        """Return the total number of male individuals."""
        pass

    def __repr__(self) -> str:
        """Return a string summary of the population state."""  # noqa: D400
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tick={self.tick}, "
            f"size={self.get_total_count()})"
        )
