"""Base population model helpers and abstractions.

This module provides the abstract base class and utilities for population
models (discrete-generation and age-structured). The base class defines
common interfaces, evolution methods, history management, and helpers
that are implemented by concrete population classes.

This module provides a common abstraction layer for population models while
keeping internal state representations compatible with the NumPy-based engine.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import (
    TYPE_CHECKING,
    Generic,
    List,
    Optional,
    Self,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray

from natal.frontend.data import (
    DiscretePopulationState,
    ModelDraft,
    PopulationState,
)
from natal.frontend.genetics import Genotype, HaploidGenotype, Species
from natal.frontend.hooks.types import RunProgram, empty_hook_program
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.population._mixins._observation import ObservationMixin
from natal.frontend.population._mixins._output import OutputMixin
from natal.frontend.registry.index import IndexRegistry

T_State = TypeVar("T_State", bound=Union[PopulationState, DiscretePopulationState])

if TYPE_CHECKING:
    from natal.frontend.configurator import Configurator
    from natal.frontend.hooks import (
        CompiledHookDescriptor,
        HookExecutor,
    )
    from natal.frontend.hooks.tick_context import HookRunner
    from natal.frontend.output._recording import RecordingPlan
    from natal.frontend.output.history import History
    from natal.frontend.output.observation import Observation, ObservationResult
    from natal.frontend.population._params_view import ParamsView
    from natal.frontend.presets import GeneticPreset

# A parameter snapshot row: (tick, parameter name, old value, new value).
ParamChange = Tuple[int, str, float, float]

class BasePopulation(OutputMixin, ObservationMixin, ABC, Generic[T_State]):
    """Abstract base class for population models.

    The base class unifies common behavior for different population model
    implementations (for example, discrete-generation and age-structured
    models). It manages the species/genetic architecture,
    indexing, hook registration, and modifier pipelines.

    Attributes:
        ALLOWED_EVENTS (List[str]): Event names supported by the hook system.
        species (Species): Genetic architecture descriptor for this population.
        name (str): Human-readable population name.
        tick (int): Current simulation tick.
        registry (IndexRegistry): Index registry for genotype/haplotype mappings.
        config (ModelDraft): Active static draft/config container.
        state (T_State): Active population state container.
        history (List[Tuple[int, np.ndarray]]): Recorded state snapshots by tick.
        compiled_hook_descriptors (List[CompiledHookDescriptor]): Ordered list
            of compiled hook descriptors (CSR plans and Python callbacks)
            sorted by priority.  Homogeneous demes cloned from the
            same template share this list object via identity.
        hook_executor (Optional[HookExecutor]): Python-side coordinator for
            all hook types.  Lazily built on first use and invalidated whenever
            hook registration changes the compiled descriptor list.
    """

    # Allowed hook events (subclasses may extend this list).
    ALLOWED_EVENTS = [
        "initialization",
        "first",
        "early",
        "late",
        "finish",
    ]

    def __init__(
        self,
        species: Species,
        name: str = "Population",
        hook_items: Optional[List[object]] = None,
    ):
        """Initialize the base population.

        Args:
            species: Genetic architecture specifying chromosomes, loci, and alleles.
            name: Optional population name (default: "Population").
            hook_items: Optional hook registrations (``Op`` objects,
                ``@hook``-decorated functions, or single-parameter
                callables).  Registration is deferred to
                :meth:`_finalize_hooks` so the IndexRegistry is ready
                when declarative ops compile.

        Note:
            Registry and genotypes are initialized lazily via Template Method.
            Subclasses must implement _create_registry() and _get_genotypes().
        """
        self._species = species
        self._name = name
        self._hook_slot = self._derive_hook_slot(name)
        self._tick = 0
        # Deme index this population executes as: 0 for panmictic models,
        # the live deme index when a SpatialPopulation manages this object
        # as one of its demes.  Hooks read it via pop.deme_id, so the Rust
        # per-deme kernel and the Python reference lifecycle must agree.
        self._deme_id: int = 0
        # DELAYED: Registry will be created via _initialize_registry()
        self._index_registry: Optional[IndexRegistry] = None
        self._registry: Optional[IndexRegistry] = None

        # Program-level plans (CSR hooks + frozen recording plan).
        self._run_program: RunProgram = RunProgram(
            hooks=empty_hook_program(), recording=None
        )

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

        # Compiled hook descriptors (CSR plans | Python callbacks).
        self.compiled_hook_descriptors: List[CompiledHookDescriptor] = []

        # Dispatch pair (Python-side coordinator + callback runner).
        self.hook_executor: Optional[HookExecutor] = None
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
        self._params_log: List[ParamChange] = []
        # Frozen per-deme ecology snapshot for spatial python dispatch:
        # homogeneous demes share one draft, so Op.set_param operands must
        # read the deme's OWN pre-tick column value, not the shared draft
        # another deme may already have written this tick.
        self._eco_value_override: Optional[NDArray[np.float64]] = None

        # Hooks queued for deferred registration after subclass
        # initialization (declarative ops need the IndexRegistry).
        self._pending_hook_items: List[object] = list(hook_items or [])

        # Rust dirty-set bridge: contract field names whose draft values
        # changed after the Rust session was built.  The next run() pulls
        # exactly these fields into the session (no rebuild, no RNG reset).
        # The sentinel "__blueprint__" forces a full backend rebuild instead.
        self._rust_dirty: set[str] = set()

    def set_eco_value_override(self, values: Optional[NDArray[np.float64]]) -> None:
        """Freeze or clear the Op.set_param operand snapshot.

        The spatial python-dispatch tick sets each deme's snapshot to the
        deme's own pre-tick ecology column so shared-draft writes by one
        deme cannot feed another deme's same-tick expression.

        Args:
            values: Frozen row (length ``len(ECO_PARAM_NAMES)``) or
                ``None`` to read live values again.
        """
        self._eco_value_override = values

    def _finalize_hooks(self) -> None:
        """Register deferred hook items after subclass initialization.

        Called by subclasses after their __init__ completes.  Declarative
        ops need the IndexRegistry, which may only be ready once the
        subclass has finished its own setup.
        """
        pending = self._pending_hook_items
        self._pending_hook_items = []
        if pending:
            self.register_hooks(*pending)

    @property
    def _recording_plan(self) -> Optional[RecordingPlan]:
        """The frozen recording plan (lives inside the run program)."""
        recording = cast("Optional[RecordingPlan]", self._run_program.recording)
        return recording

    @_recording_plan.setter
    def _recording_plan(self, plan: Optional[RecordingPlan]) -> None:
        """Install the frozen recording plan into the run program."""
        self._run_program = self._run_program._replace(recording=plan)

    def _clone(
        self,
        name: str,
        config: ModelDraft | None = None,
    ) -> Self:
        """Create a lightweight functional copy sharing compiled state and config.

        Used by ``SpatialBuilder`` to efficiently clone template demes without
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
        clone._rust_dirty = set()

        # --- per-clone bookkeeping (deferred hooks already finalized) ---
        clone._pending_hook_items = []
        clone._params_log = []

        # --- shared identity ---
        clone._species = self._species
        clone._name = name
        clone._hook_slot = self._hook_slot
        clone._tick = int(self._tick)
        # Clones are built via __new__ (no __init__), so the deme index
        # must be copied explicitly; a spatial deme's clone keeps its id
        # until SpatialPopulation.__init__ restamps the whole list.
        clone._deme_id = int(self._deme_id)

        # --- shared hooks (compiled, read-only during simulation) ---
        clone.compiled_hook_descriptors = self.compiled_hook_descriptors
        clone.hook_executor = self.hook_executor
        clone._hook_runner = self._hook_runner
        # Clones start from an empty CSR program; registration populates it
        # lazily via _refresh_run_program (the shared descriptor list is
        # re-read every time the program is rebuilt).
        clone._run_program = RunProgram(
            hooks=self._run_program.hooks, recording=self._run_program.recording
        )

        # --- shared registry ---
        clone._index_registry = self._index_registry
        clone._registry = self._registry

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
        state_cls = type(self.state)
        new_state = state_cls.create(
            n_ztypes=resolved_config.n_ztypes,
            n_sexes=resolved_config.n_sexes,
            n_ages=resolved_config.n_ages,
        )
        object.__setattr__(clone, '_state', new_state)
        clone_state_nn = clone.state
        self_state_nn = self.state
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
        clone._recording_plan = self._recording_plan
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
            self._registry = self._index_registry
            return

        self._index_registry = self._create_registry()
        self._registry = self._index_registry

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

    @staticmethod
    def _derive_hook_slot(name: str) -> int:
        """Derive a stable non-negative hook slot from population name."""
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
        # Keep int32-compatible positive range for config scalar stability.
        return int(digest[:8], 16) & 0x7FFFFFFF

    @property
    def hook_slot(self) -> int:
        """Stable unique slot identifier derived from the population name.

        Used by the hook dispatch system to route hooks to the correct
        population instance in multi-deme simulations.
        """
        return int(self._hook_slot)

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
        """The current simulation tick or generation index."""
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:  # type: ignore[reportIncompatibleVariableOverride]  # property override from mixin
        """Set the current simulation tick."""
        self._tick = value

    @property
    def registry(self) -> IndexRegistry:
        """IndexRegistry instance managing genotype, haplotype, and label indices."""
        if self._registry is None:
            raise AttributeError("Index registry has not been initialized.")
        return self._registry

    @property
    def index_registry(self) -> IndexRegistry:
        """Public accessor for the internal IndexRegistry."""
        if self._index_registry is None:
            raise AttributeError("Index registry has not been initialized.")
        return self._index_registry

    @property
    def config(self) -> ModelDraft:
        """Public accessor for compiled population configuration."""
        if self._config is None:
            raise AttributeError("Population config has not been initialized.")
        return self._config

    def set_config(self, config: ModelDraft) -> None:
        """Replace this population's configuration."""
        self._config = config

    def _create_configurator(self) -> Configurator:
        """Create a ``Configurator`` wired back to this population.

        Subclass ``update()`` methods call this helper so concrete return types
        do not need ``cast()``.
        """
        from natal.frontend.configurator import Configurator

        return Configurator.for_population(self)

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
        return tuple(self._params_log)

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
            self._params_log.append((int(self._tick), name, float(old), float(new)))

    def _absorb_rust_eco_journal(
        self, rows: Sequence[Tuple[int, str, float, float]]
    ) -> None:
        """Merge a drained Rust-session journal into the log and the draft.

        On the Rust run path, ``Op.set_param`` writes evolve inside the
        session-owned ecology columns; when ``run()`` returns, this merge
        makes those writes visible on the population side.  Each row is
        appended to ``params_log`` under its own commit tick (not the
        current tick, so multi-tick batches keep their per-tick audit), and
        each parameter's final value is written into the draft 0-d array.
        No dirty-bridge marking happens: the session already holds the same
        value and the values were bounds-validated on the Rust side, so
        re-pushing them would be a redundant round trip.

        Args:
            rows: ``(tick, name, old, new)`` rows from a Rust backend's
                ``drain_eco_journal`` (change-only, commit order).
        """
        if not rows:
            return
        final_values: dict[str, float] = {}
        for tick, name, old, new in rows:
            self._params_log.append((int(tick), name, float(old), float(new)))
            final_values[name] = float(new)
        draft = self.config
        for name, value in final_values.items():
            slot: object = getattr(draft, name)
            # The journal only ever names the five runtime-mutable ecology
            # scalars; they are immutable NamedTuple slots now, so the
            # merge rebuilds the draft once.
            if isinstance(slot, np.ndarray):
                slot[()] = value
            else:
                draft = draft._replace(**{name: value})
        self.set_config(draft)

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
    def update(self) -> Configurator:
        """Return a ``Configurator`` for modifying this population's config.

        All chainable methods (``.competition()``, ``.reproduction()``, …)
        write changes immediately — no ``.apply()`` or ``.freeze()`` needed
        for simple parameter updates.

        Examples:

            >>> pop.update().competition(carrying_capacity=5000)
            >>> pop.update().reproduction(eggs_per_female=100, sex_ratio=0.6)

        .. versionadded:: NEXT
        """
        ...

    @property
    def state(self) -> T_State:
        """Return the current population state container.

        Returns:
            PopulationState: The current state object used by the population.
        """
        if self._state is None:
            raise AttributeError("Population state has not been initialized.")
        return self._state

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

        state = self.state
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
        self._history_obj = History(schema)

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
    def get_total_count(self) -> int:
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

        Returns:
            ObservationResult with projected values and explicit axes.

        Raises:
            RuntimeError: When state is not available yet.
        """
        obs = self.observation
        state = getattr(self, "state", None)
        if state is None:
            raise RuntimeError("Population has no state to observe.")
        ic = getattr(state, "individual_count", None)
        if ic is None:
            raise RuntimeError("Population state has no individual_count.")
        return obs.project(ic, tick=self._tick)

    @abstractmethod
    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        pass

    @abstractmethod
    def get_male_count(self) -> int:
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
