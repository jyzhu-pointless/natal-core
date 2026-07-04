"""Base population model helpers and abstractions.

This module provides the abstract base class and utilities for population
models (discrete-generation and age-structured). The base class defines
common interfaces, evolution methods, history management, and helpers
that are implemented by concrete population classes.

This module provides a common abstraction layer for population models while
keeping internal state representations compatible with NumPy/Numba engine.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np

from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    PopulationConfig,
    PopulationState,
)
from natal.genetics import Genotype, HaploidGenotype, Species
from natal.modifiers.module import GameteModifier, ZygoteModifier
from natal.population._mixins._observation import ObservationMixin
from natal.population._mixins._output import OutputMixin
from natal.registry.index import IndexRegistry

T_State = TypeVar("T_State", bound=Union[PopulationState, DiscretePopulationState])

if TYPE_CHECKING:
    from natal.configurator import Configurator
    from natal.hooks import (
        CompiledHookDescriptor,
        HookExecutor,
    )
    from natal.output.observation import Observation
    from natal.presets import GeneticPreset

HookCallback = Callable[..., object]
HookEntry = Tuple[int, Optional[str], HookCallback]
HookRegistration = Tuple[HookCallback, Optional[str], Optional[int]]
HookRegistrationMap = Dict[str, List[HookRegistration]]
PendingHook = Tuple[str, HookCallback, Optional[str], Optional[int]]

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
        config (PopulationConfig): Active static tensor/config container.
        state (T_State): Active population state container.
        history (List[Tuple[int, np.ndarray]]): Recorded state snapshots by tick.
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
        hooks: Optional[HookRegistrationMap] = None,
    ):
        """Initialize the base population.

        Args:
            species: Genetic architecture specifying chromosomes, loci, and alleles.
            name: Optional population name (default: "Population").
            hooks: Optional mapping of event names to hook registrations. Each
                entry should be a sequence of tuples in the form ``(func, hook_name, hook_id)``. Hooks
                provided here will be registered during initialization.

        Note:
            Registry and genotypes are initialized lazily via Template Method.
            Subclasses must implement _create_registry() and _get_genotypes().
        """
        self._species = species
        self._name = name
        self._hook_slot = self._derive_hook_slot(name)
        self._tick = 0
        # DELAYED: Registry will be created via _initialize_registry()
        self._index_registry: Optional[IndexRegistry] = None
        self._registry: Optional[IndexRegistry] = None

        # Evolution history as (tick, flattened_array) tuples.
        self._history: List[Tuple[int, np.ndarray]] = []

        # History config
        self.record_every: int = 1
        self.max_history: int = 5000  # Default rolling window size

        # Hooks system: event_name -> [(hook_id, hook_name, hook_func), ...]
        self._hooks: Dict[str, List[HookEntry]] = {
            event: [] for event in self.ALLOWED_EVENTS
        }

        # Presets with priority IDs.  Writes go to _presets; derived
        # modifier lists are rebuilt from _presets + _manual_* on demand.
        self._presets: list[GeneticPreset] = []

        # Directly-added modifiers (manual, not from presets).
        self._manual_gamete: list[tuple[int, str | None, GameteModifier]] = []
        self._manual_zygote: list[tuple[int, str | None, ZygoteModifier]] = []

        # Derived modifier lists — rebuilt by refresh_modifiers().
        self._gamete_modifiers: list[tuple[int, str | None, GameteModifier]] = []
        self._zygote_modifiers: list[tuple[int, str | None, ZygoteModifier]] = []

        # Compiled hook descriptors (for Numba-accelerated execution).
        self._compiled_hooks: List[CompiledHookDescriptor] = []

        # Hook executor (Python-side coordinator for all hook types).
        self._hook_executor: Optional[HookExecutor] = None

        # Static data container.
        self._config: Optional[PopulationConfig | DiscretePopulationConfig] = None

        # PopulationState container.
        self._state: Optional[T_State] = None

        # Evolution status flag: whether simulation is finished.
        self._finished = False

        # Re-entrancy guard flag.
        self._running = False

        # Observation-based history recording.
        self._observation: Optional[Observation] = None
        self._observation_mask: Optional[np.ndarray] = None

        # Hooks queued for deferred compilation after subclass initialization.
        # Format: [(event_name, func, hook_name, hook_id), ...]
        self._pending_hooks: List[PendingHook] = []

        # Register hooks.
        # If a hook carries @hook metadata, compilation may fail at this stage
        # because IndexRegistry may not be fully initialized yet.
        # Plain functions can be registered immediately; @hook functions are queued.
        hooks_map: HookRegistrationMap = hooks or {}
        if hooks_map:
            for event_name, hooks_list in hooks_map.items():
                for hook_info in hooks_list:
                    func, hook_name, hook_id = hook_info

                    # Check if function has @hook metadata
                    hook_meta = getattr(func, 'meta', None)
                    if hook_meta is not None:
                        # Defer compilation until _finalize_hooks() is called
                        self._pending_hooks.append((event_name, func, hook_name, hook_id))
                    else:
                        # Plain function, register immediately
                        self.set_hook(event_name, func, hook_id=hook_id, hook_name=hook_name, compile=False)

    def _finalize_hooks(self) -> None:
        """Compile pending hooks after subclass initialization is complete.

        Called by subclasses after their __init__ completes. This allows hooks
        with @hook metadata to be compiled with the now-initialized IndexRegistry.
        """
        # Compile any pending @hook-decorated functions
        for event_name, func, hook_name, hook_id in self._pending_hooks:
            self.set_hook(event_name, func, hook_id=hook_id, hook_name=hook_name, compile=True)
        self._pending_hooks.clear()
        self._hook_executor = None

    def _clone(self, name: str, config: Optional[PopulationConfig] = None) -> Any:
        """Create a lightweight functional copy sharing compiled state and config.

        Used by ``SpatialBuilder`` to efficiently clone template demes without
        re-running hook compilation or preset application. The clone shares
        compiled hooks, index registry, modifier pipelines, and config arrays
        with the template. Only state arrays and history are independent.

        Args:
            name: Unique name for the clone.
            config: Optional ``PopulationConfig`` to use (default: template's config).

        Returns:
            A new population instance of the same type with shared compiled state.
        """
        cls = type(self)
        clone = cls.__new__(cls)

        # --- shared identity ---
        clone._species = self._species
        clone._name = name
        clone._hook_slot = self._hook_slot
        clone._tick = int(self._tick)

        # --- shared hooks (compiled, read-only during simulation) ---
        clone._hooks = self._hooks
        clone._pending_hooks = []
        clone._compiled_hooks = self._compiled_hooks
        clone._hook_executor = self._hook_executor

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

        # --- runtime state (independent per deme) ---
        # --- observation recording (independent per deme) ---
        clone._observation = None
        clone._observation_mask = None

        clone._history = []
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
        return IndexRegistry()

    def _get_genotypes(self) -> List[Genotype]:
        return self.species.get_all_genotypes()
        # return self._registry.index_to_genotype

    def _get_haplogenotypes(self) -> Optional[List[HaploidGenotype]]:
        return self.species.get_all_haploid_genotypes()
        # return self._registry.index_to_haplo

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
    def name(self) -> str:  # type: ignore[reportIncompatibleVariableOverride]
        """The human-readable name of the population."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:  # type: ignore[reportIncompatibleVariableOverride]
        """Set the population name."""
        self._name = value

    @property
    def tick(self) -> int:  # type: ignore[reportIncompatibleVariableOverride]
        """The current simulation tick or generation index."""
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:  # type: ignore[reportIncompatibleVariableOverride]
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
    def config(self) -> PopulationConfig | DiscretePopulationConfig:
        """Public accessor for compiled population configuration."""
        if self._config is None:
            raise AttributeError("Population config has not been initialized.")
        return self._config

    def set_config(self, config: PopulationConfig | DiscretePopulationConfig) -> None:
        """Replace this population's configuration."""
        self._config = config

    def _create_configurator(self) -> Configurator:
        """Create a ``Configurator`` wired back to this population.

        Subclass ``update()`` methods call this helper so concrete return types
        do not need ``cast()``.
        """
        from natal.configurator import Configurator

        return Configurator.for_population(self)

    @property
    def presets(self) -> List[GeneticPreset]:
        """Return the presets applied to this population."""
        return self._presets

    @property
    def gamete_modifiers(self) -> List[tuple[int, str | None, GameteModifier]]:
        """Return the list of registered gamete modifiers."""
        return self._gamete_modifiers

    @property
    def zygote_modifiers(self) -> List[tuple[int, str | None, ZygoteModifier]]:
        """Return the list of registered zygote modifiers."""
        return self._zygote_modifiers

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
    def history(self) -> List[Tuple[int, np.ndarray]]:
        """A list of recorded historical states as ``(tick, flattened_array)`` tuples."""
        return list(self._history)

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

    @abstractmethod
    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        pass

    @abstractmethod
    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tick={self.tick}, "
            f"size={self.get_total_count()})"
        )
