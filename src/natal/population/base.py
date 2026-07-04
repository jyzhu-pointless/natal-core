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
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
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

import natal.data as _population_config
import natal.modifiers.module as _modifiers
from natal.data import (
    DiscretePopulationConfig,
    DiscretePopulationState,
    PopulationConfig,
    PopulationState,
)
from natal.engine.lifecycle_wrappers import (
    LifecycleWrappers,
    compile_lifecycle_wrappers,
)
from natal.engine.simulation.age_structured import compute_offspring_probability_tensor
from natal.genetics import Genotype, HaploidGenotype, Species
from natal.modifiers.module import GameteModifier, ZygoteModifier
from natal.numba.utils import is_numba_enabled
from natal.output.translation import (
    output_current_state as _output_current_state,
)
from natal.output.translation import (
    output_history as _output_history,
)
from natal.presets import CytoplasmicPreset
from natal.registry.index import IndexRegistry

T_State = TypeVar("T_State", bound=Union[PopulationState, DiscretePopulationState])

if TYPE_CHECKING:
    from natal.configurator import Configurator
    from natal.hooks import (
        CompiledHookDescriptor,
        DemeSelector,
        HookExecutor,
        HookProgram,
    )
    from natal.output.observation import GroupsInput, Observation
    from natal.presets import GeneticPreset

HookCallback = Callable[..., object]
HookEntry = Tuple[int, Optional[str], HookCallback]
HookRegistration = Tuple[HookCallback, Optional[str], Optional[int]]
HookRegistrationMap = Dict[str, List[HookRegistration]]
PendingHook = Tuple[str, HookCallback, Optional[str], Optional[int]]
ModifierWrapperBuilder = Callable[..., Tuple[List[HookCallback], List[HookCallback]]]
MapInitializer = Callable[..., np.ndarray]
build_modifier_wrappers = cast(ModifierWrapperBuilder, _modifiers.build_modifier_wrappers)
initialize_gamete_map = cast(MapInitializer, _population_config.initialize_gamete_map)
initialize_zygote_map = cast(MapInitializer, _population_config.initialize_zygote_map)
build_population_config = cast(Callable[..., PopulationConfig], _population_config.build_population_config)

class BasePopulation(ABC, Generic[T_State]):
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
    # Observation-based history recording
    # ========================================================================

    @property
    def record_observation(self) -> Optional[Observation]:
        """The compiled Observation used for observation-mode history."""
        return self._observation

    @record_observation.setter
    def record_observation(self, obs: Optional[Observation]) -> None:
        """Set the observation and rebuild the binary observation mask."""
        self._observation = obs
        if obs is not None:
            self._observation_mask = self._build_observation_mask(obs)

    def set_observations(self, groups: GroupsInput, *, collapse_age: bool = False) -> None:
        """Register observation groups and immediately compile the binary mask.

        Once set, the mask is passed to simulation engine to record
        observation-aggregated snapshots (compressed format) instead of raw
        flattened state.

        Args:
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether to collapse the age axis during projection.
                The stored kernel mask is always 4-D; collapse_age is recorded
                as metadata and respected by export functions.
        """
        from natal.output.observation import ObservationFilter

        obs_filter = ObservationFilter(self.index_registry)
        self._observation = obs_filter.build_filter(
            diploid_genotypes=self.species,
            groups=groups,
            collapse_age=bool(collapse_age),
        )
        self._observation_mask = self._build_observation_mask(self._observation)

    def _build_observation_mask(self, obs: Observation) -> np.ndarray:
        """Build the 4-D binary mask from an Observation and current state dims."""
        state = self.state
        ind = state.individual_count
        return obs.build_mask(
            n_sexes=ind.shape[0],
            n_ages=ind.shape[1] if ind.ndim == 3 else 1,
            n_ztypes=ind.shape[-1],
        )

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
    def name(self) -> str:
        """The human-readable name of the population."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the population name."""
        self._name = value

    @property
    def tick(self) -> int:
        """The current simulation tick or generation index."""
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:
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

    def _enforce_history_limit(self) -> None:
        """Ensure history size does not exceed max_history by dropping oldest entries."""
        if self.max_history > 0:
            excess = len(self._history) - self.max_history
            if excess > 0:
                self._history = self._history[excess:]

    @abstractmethod
    def clear_history(self) -> None:
        """Clear all recorded history states.

        Subclasses must implement this to reset their history storage
        (e.g., ``_history`` list and any subclass-specific history buffers).
        """
        pass

    def _process_kernel_history(
        self,
        history_new: Optional[np.ndarray],
        clear_history_on_start: bool
    ) -> None:
        """Process and append history array returned from simulation engine.

        Handles duplication checking (overlapping start/end ticks) and enforces limit.
        """
        if history_new is None or history_new.shape[0] == 0:
            return

        if clear_history_on_start:
            self.clear_history()

        for row_idx in range(history_new.shape[0]):
            row = history_new[row_idx, :]
            tick = int(row[0])
            # Skip duplicate entry if continuing history (overlap check)
            if not clear_history_on_start and self._history and self._history[-1][0] == tick:
                continue
            self._history.append((tick, row.copy()))

        self._enforce_history_limit()

    # ========================================================================
    # Modifier management
    # ========================================================================
    def _next_modifier_id(self, modifiers: Sequence[Tuple[int, Optional[str], Any]]) -> int:
        """Return the next auto-assigned modifier id."""
        # Keep compatibility with legacy in-memory lists that may contain None ids.
        ids = [mid for mid, _, _ in modifiers]
        return (max(ids) + 1) if ids else 0

    def _resolve_modifier_id(self, modifier_id: Optional[int], modifiers: Sequence[Tuple[int, Optional[str], Any]]) -> int:
        """Normalize optional modifier_id into a concrete integer id."""
        if modifier_id is not None:
            return int(modifier_id)
        return self._next_modifier_id(modifiers)

    def reapply_preset_fitness(self) -> None:
        """Reset fitness tensors to 1.0 and re-apply all preset fitness patches.

        Called after structural changes to presets (addition, removal, or
        reconfiguration).  Only preset-derived fitness is restored — any
        fitness values set directly via ``pop.update().fitness()`` will be
        overwritten, because there is currently no manual-fitness storage
        analogous to ``_manual_gamete`` / ``_manual_zygote``.
        """
        from natal.fitness import apply_preset_fitness_patch

        if self._config is None:
            return
        self._config.viability_fitness.fill(1.0)
        self._config.fecundity_fitness.fill(1.0)
        self._config.sexual_selection_fitness.fill(1.0)
        self._config.zygote_viability_fitness.fill(1.0)
        for preset in sorted(self._presets, key=lambda p: p.priority):
            preset.bind_species(self._species)
            patch = preset.fitness_patch()
            if patch:
                apply_preset_fitness_patch(self, patch)

    def refresh_modifiers(self) -> None:
        """Rebuild derived modifier lists and maps from _presets + _manual_*.

        Presets are applied in priority order, then manual modifiers are
        appended.  Modifier maps (zygotes_to_gametes_map,
        gametes_to_zygotes_map, offspring_tensor) are rebuilt from the
        combined list.

        .. note::

            This method does **not** touch fitness tensors.  Callers that
            need a full fitness rebuild should also call
            :meth:`reapply_preset_fitness`.
        """
        self._gamete_modifiers.clear()
        self._zygote_modifiers.clear()
        for preset in sorted(self._presets, key=lambda p: p.priority):
            preset.bind_species(self._species)
            if gm := preset.gamete_modifier(self):
                self._gamete_modifiers.append((
                    self._next_modifier_id(self._gamete_modifiers),
                    f"{preset.name}/gamete", gm,
                ))
            if zm := preset.zygote_modifier(self):
                self._zygote_modifiers.append((
                    self._next_modifier_id(self._zygote_modifiers),
                    f"{preset.name}/zygote", zm,
                ))
        self._gamete_modifiers.extend(self._manual_gamete)
        self._zygote_modifiers.extend(self._manual_zygote)
        self.refresh_modifier_maps()

    def refresh_modifier_maps(self) -> None:
        """Rebuild the three modifier maps from current modifier lists.

        Recomputes:
        - ``zygotes_to_gametes_map``: mapping from diploid genotype indices
          to haploid gamete probability distributions (one per sex).
        - ``gametes_to_zygotes_map``: mapping from paired haploid gametes back
          to diploid offspring genotype indices.
        - ``offspring_tensor``: precomputed 4-D tensor combining both maps
          for efficient Numba-based reproduction.

        The maps are stored in ``_config`` via ``_replace``, which creates a
        shallow copy of the config with updated fields.

        .. note::

            This method is called automatically by :meth:`refresh_modifiers`
            and by individual ``add_gamete_modifier`` / ``add_zygote_modifier``
            when ``refresh=True``.
        """
        if self._config is None or self._registry is None:
            return

        haploid_genotypes = self._registry.index_to_haplo
        diploid_genotypes = self._registry.index_to_genotype
        if not haploid_genotypes or not diploid_genotypes:
            return

        n_glabs = int(self._config.n_glabs)
        n_slabs = int(self._config.n_slabs)

        # Step 1: Build wrapper callables from the combined modifier
        # lists (preset-derived + manually added).  Each wrapper is a
        # callable that accepts genotype indices and returns modified
        # probability vectors.
        gamete_funcs, zygote_funcs = build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            index_registry=self._index_registry,
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
        )

        # Step 2: Build the gametogenesis map.  For each diploid genotype
        # and sex, produce a probability distribution over the resulting
        # haploid gametes per gamete label.
        zygotes_to_gametes_map = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            gamete_modifiers=gamete_funcs,
            n_slabs=n_slabs,
        )

        # Step 3: Build the fusion map.  For each pair of haploid gametes
        # (one maternal, one paternal), determine the resulting diploid
        # offspring genotype index.
        gametes_to_zygotes_map = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            zygote_modifiers=zygote_funcs,
            n_slabs=n_slabs,
        )

        # Step 4: Apply cytoplasmic preset effects if presets are configured.
        # This must happen BEFORE the offspring tensor is computed, so the
        # tensor reflects the modified maps.
        for preset in self._presets:
            if isinstance(preset, CytoplasmicPreset):
                n_genotypes = len(diploid_genotypes)
                n_gtypes = len(haploid_genotypes)
                species = getattr(self, "_species", None)
                if species is not None:
                    CytoplasmicPreset.tag_maternal_gametes(
                        zygotes_to_gametes_map, species.gamete_labels,
                        species.somatic_labels,
                        n_genotypes, n_gtypes, n_glabs, n_slabs,
                    )
                    CytoplasmicPreset.redirect_zygotes(
                        gametes_to_zygotes_map, species.gamete_labels,
                        species.somatic_labels,
                        n_genotypes, n_gtypes, n_glabs, n_slabs,
                    )

        # Step 5: Compute the full offspring probability tensor by
        # convolving the maternal and paternal gametogenesis maps through
        # the fusion map.  The result is a 4-D array indexed by
        # (maternal_genotype, paternal_genotype, gamete_label, offspring_genotype).
        n_g = int(self._config.n_ztypes)
        n_hg = int(self._config.n_gtypes)
        offspring_tensor = compute_offspring_probability_tensor(
            meiosis_f=zygotes_to_gametes_map[0],
            meiosis_m=zygotes_to_gametes_map[1],
            haplo_to_genotype_map=gametes_to_zygotes_map,
            n_ztypes=n_g,
            n_gtypes=n_hg,
        )

        # Step 6: Persist all three maps into the config via shallow copy.
        self._config = self._config._replace(
            zygotes_to_gametes_map=zygotes_to_gametes_map,
            gametes_to_zygotes_map=gametes_to_zygotes_map,
            offspring_tensor=offspring_tensor,
            n_ztypes=n_g,
            n_gtypes=n_hg,
            n_glabs=n_glabs,
        )

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
        resolved_id = self._resolve_modifier_id(modifier_id, self._manual_gamete)
        self._manual_gamete.append((resolved_id, name, modifier))
        self._manual_gamete.sort(key=lambda x: x[0])
        self._gamete_modifiers.append((resolved_id, name, modifier))
        self._gamete_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

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
        resolved_id = self._resolve_modifier_id(modifier_id, self._manual_zygote)
        self._manual_zygote.append((resolved_id, name, modifier))
        self._manual_zygote.sort(key=lambda x: x[0])
        self._zygote_modifiers.append((resolved_id, name, modifier))
        self._zygote_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self.refresh_modifier_maps()

    def add_preset(self, preset: GeneticPreset) -> None:
        """Add a preset to this population.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).
        """
        self._presets.append(preset)

    def apply_preset(self, preset: GeneticPreset) -> None:
        """Apply a genetic preset to this population.

        This is the preferred API for registering presets. The preset's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).

        Examples:
            >>> from natal.presets import HomingDrive
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     drive_conversion_rate=0.95
            ... )
            >>> population.apply_preset(drive)

        See Also:
:class:`natal.presets.GeneticPreset` - Base class for creating custom presets
:class:`natal.presets.HomingDrive` - Built-in gene drive preset
        """
        self.add_preset(preset)
        self.refresh_modifiers()
        self.reapply_preset_fitness()

    @classmethod
    def builder(cls, species: Species) -> Any:
        """Create a builder for this population type.

        This is the recommended way to construct populations with presets.

        Args:
            species: Genetic architecture for the population.

        Returns:
            A builder instance for this population type.

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

    # ========================================================================
    # Common methods (can be inherited or overridden by subclasses)
    # ========================================================================

    @property
    def total_population_size(self) -> int:
        """Total population size (alias of ``get_total_count``)."""
        return self.get_total_count()

    @property
    def total_females(self) -> int:
        """Total number of females (alias of ``get_female_count``)."""
        return self.get_female_count()

    @property
    def total_males(self) -> int:
        """Total number of males (alias of ``get_male_count``)."""
        return self.get_male_count()

    @property
    def sex_ratio(self) -> float:
        """Return the female-to-male ratio, or ``np.inf`` when male count is zero."""
        males = self.get_male_count()
        return self.get_female_count() / males if males > 0 else np.inf

    def create_observation(
        self,
        *,
        groups: Optional[GroupsInput] = None,
        collapse_age: bool = False,
    ) -> Observation:
        """Create a compiled observation from the current population schema.

        Args:
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether observation collapses the age axis.

        Returns:
            Compiled ``Observation`` object that can be reused across states.
        """
        from natal.output.observation import ObservationFilter

        obs_filter = ObservationFilter(self.index_registry)
        return obs_filter.build_filter(
            diploid_genotypes=self.species,
            groups=groups,
            collapse_age=collapse_age,
        )

    def output_current_state(
        self,
        *,
        observation: Optional[Observation] = None,
        groups: Optional[GroupsInput] = None,
        collapse_age: bool = False,
        include_zero_counts: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """Export the current population state with observation rules applied.

        This method integrates observation with state translation and can
        optionally write the JSON payload to a file.

        Args:
            observation: Optional prebuilt observation object. When provided,
                ``groups`` and ``collapse_age`` are ignored.
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether observation rule generation collapses age axis.
            include_zero_counts: Whether to keep zero-valued entries.
            output_path: Optional JSON file path. When provided, the payload is
                written to this file as UTF-8 JSON.
            indent: Indentation used when writing JSON.

        Returns:
            A dictionary with observation metadata and observed counts.
        """
        return _output_current_state(
            self,
            observation=observation,
            groups=groups,
            collapse_age=collapse_age,
            include_zero_counts=include_zero_counts,
            output_path=output_path,
            indent=indent,
        )

    def output_history(
        self,
        *,
        observation: Optional[Observation] = None,
        groups: Optional[GroupsInput] = None,
        collapse_age: bool = False,
        include_zero_counts: bool = False,
        history: Optional[np.ndarray] = None,
        output_path: Optional[Union[str, Path]] = None,
        indent: int = 2,
    ) -> Dict[str, Any]:
        """Export the observation history for this population.

        Args:
            observation: Optional prebuilt observation object. When provided,
                ``groups`` and ``collapse_age`` are ignored.
            groups: Observation groups passed to ``ObservationFilter``.
                When ``None``, one group per genotype index is used.
            collapse_age: Whether observation rule generation collapses age axis.
            include_zero_counts: Whether to keep zero-valued entries.
            history: Optional flattened history array. When omitted, the
                population history is fetched from ``get_history()``.
            output_path: Optional JSON file path. When provided, the payload is
                written to this file as UTF-8 JSON.
            indent: Indentation used when writing JSON.

        Returns:
            A dictionary containing observation metadata and per-snapshot outputs.
        """
        return _output_history(
            self,
            observation=observation,
            groups=groups,
            collapse_age=collapse_age,
            include_zero_counts=include_zero_counts,
            history=history,
            output_path=output_path,
            indent=indent,
        )

    @property
    def is_finished(self) -> bool:
        """Whether the population is marked as finished (``finish=True``)."""
        return self._finished

    def finish_simulation(self) -> None:
        """
        End simulation, trigger the ``finish`` event, and lock the population.

        This method may be called by hooks for early termination.
        After calling it, ``step()``, ``run_tick()``, and ``run()`` cannot run again.

        Raises:
            RuntimeError: If the population is already finished.

        Examples:
            >>> def check_extinction(pop):
            ...     if pop.get_total_count() == 0:
            ...         print("Population extinct, finishing simulation.")
            ...         pop.finish_simulation()
            >>> pop.set_hook('late', check_extinction)
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has already finished."
            )

        self._finished = True
        self.trigger_event("finish")

    @abstractmethod
    def run(
        self,
        n_steps: int,
        record_every: Optional[int] = None,
        finish: bool = False
    ) -> BasePopulation[Any]:
        """
        Run multi-step evolution.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the population to its initial state."""
        pass

    def compute_allele_frequencies(self) -> Dict[str, float]:
        """
        Compute frequencies of all alleles in the population, normalized per locus.

        Returns:
            Dict[str, float]: Mapping ``{allele_name: frequency}``.
            Frequencies are per-locus proportions in the range ``[0.0, 1.0]``.
        """
        if self._state is None or self._registry is None:
            return {}

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
        genotype_counts = self._state.individual_count.sum(axis=(0, 1))

        registry = self._registry
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

    # ========================================================================
    # Hooks system
    # ========================================================================

    def set_hook(
        self,
        event_name: str,
        func: HookCallback,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None,
        compile: bool = True,
        deme_selector: Optional[DemeSelector] = None,
    ) -> None:
        """
        Register an event hook with optional automatic compilation.

        When ``compile=True`` and the function carries ``@hook`` metadata,
        it enters the DSL compilation pipeline:
        - declarative hook -> CSR plan in HookProgram (kernel executable)
        - selector hook -> ``py_wrapper`` or ``njit_fn`` (mode dependent)
        - numba hook -> ``njit_fn``

        Plain Python functions are still registered in traditional ``_hooks``
        for backward-compatible execution.

        Args:
            event_name: Event name (must exist in ``ALLOWED_EVENTS``).
            func: Callback function, supported forms include:
                  - plain function: ``func(population)``
                  - declarative ``@hook`` function: returns ``[Op.scale(...), ...]``
                  - selector ``@hook(selectors={...})`` function
            hook_id: Numeric execution priority (optional, auto-assigned if omitted).
                     Lower IDs execute first.
            hook_name: Optional human-readable name for debugging.
            compile: Whether to try compiling ``@hook``-decorated functions (default ``True``).
            deme_selector: Optional deme selector.
                - ``None``: keep panmictic default behavior (no explicit selector override)
                - non-``None``: passed into hook registration for spatial filtering

        Raises:
            ValueError: If event does not exist or hook_id is already in use.

        Examples:
            >>> # Plain function (backward compatible)
            >>> pop.set_hook('first', lambda p: print(f'Step {p.tick}'))
            >>>
            >>> # Declarative @hook function (auto-compiled)
            >>> @hook()
            >>> def reduce_juveniles():
            ...     return [Op.scale(genotypes='AA', ages=[0, 1], factor=0.9)]
            >>> pop.set_hook('early', reduce_juveniles)
            >>>
            >>> # Selector @hook function (auto-compiled)
            >>> @hook(selectors={'target': 'AA'})
            >>> def release(pop, target):
            ...     pop.state.individual_count[1, 2, target] += 100
            >>> pop.set_hook('first', release)
        """
        if event_name not in self.ALLOWED_EVENTS:
            raise ValueError(f"Event '{event_name}' not in {self.ALLOWED_EVENTS}")

        # BasePopulation itself is panmictic. Non-wildcard deme selectors are
        # interpreted by SpatialPopulation orchestration and should not be
        # consumed here.
        if deme_selector is not None and deme_selector != "*":
            warnings.warn(
                "BasePopulation ignores non-'*' deme_selector. "
                "Apply deme selection through SpatialPopulation-level logic instead.",
                UserWarning,
                stacklevel=2,
            )
            deme_selector = None

        # Check if function has @hook metadata and should be compiled
        hook_meta = getattr(func, 'meta', None)
        if is_numba_enabled() and hook_meta is None:
            raise TypeError(
                "Python-layer hooks are not allowed when Numba is enabled. "
                "Use @hook(...) with a compilable body or disable Numba."
            )

        if compile and hook_meta is not None:
            # Use the hook's register method with event override
            register_fn = getattr(func, 'register', None)
            if register_fn is not None:
                # Panmictic path: do not force any selector override.
                if deme_selector is None:
                    register_fn(self, event_override=event_name)
                else:
                    register_fn(self, event_override=event_name, deme_selector_override=deme_selector)
                # Compiled hooks are stored in _compiled_hooks.
                # Only selector-mode hooks with py_wrapper are mirrored to _hooks.
                self._hook_executor = None
                return

        # Traditional registration (no compilation)
        actual_name = hook_name or getattr(func, '__name__', None)

        current_ids = [hid for hid, _, _ in self._hooks[event_name]]

        if hook_id is None:
            hook_id = (max(current_ids) + 1) if current_ids else 0

        if hook_id in current_ids:
            raise ValueError(f"hook_id {hook_id} already exists in event '{event_name}'")

        self._hooks[event_name].append((hook_id, actual_name, func))
        # Sort by hook ID to preserve execution order.
        self._hooks[event_name].sort(key=lambda x: x[0])
        self._hook_executor = None

    def trigger_event(self, event_name: str, deme_id: int = -1) -> int:
        """
                Trigger an event and execute all registered hooks.

                Execution order:
                1. CSR operations (Numba fast path)
                2. ``njit_fn`` hooks (user-defined Numba functions)
                3. ``py_wrapper`` hooks (Python wrapper functions)

        Args:
                        event_name: Event name to trigger.
                        deme_id: Deme index. Default -1 for non-spatial populations.

        Returns:
                        int: ``RESULT_CONTINUE`` (0) to continue, ``RESULT_STOP`` (1) to stop.

        Note:
                        - Prefer HookExecutor (unified three-layer coordination).
                        - If executor is not built, fall back to traditional ``_hooks``
                            (Python callbacks only).
                        - In accelerated ``run()``, core events are mostly executed by engine;
                            ``trigger_event`` is used mainly for explicit events (for example ``finish``)
                            and compatibility paths.

        Examples:
                        >>> result = pop.trigger_event('first')  # Executes all 'first' hooks
            >>> if result == RESULT_STOP:
            ...     print("Simulation stopped by hook")
        """
        from natal.hooks import RESULT_CONTINUE

        # Prefer HookExecutor when available.
        if self._hook_executor is not None:
            from natal.hooks import EVENT_ID_MAP
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is not None:
                result = self._hook_executor.execute_event(event_id, self, self.tick, deme_id=deme_id)
                return result

        # Fallback to traditional _hooks for compatibility.
        for _, _, hook in self._hooks.get(event_name, []):
            hook(self)

        return RESULT_CONTINUE


    def get_hooks(self, event_name: str) -> List[HookEntry]:
        """
        Get all registered hooks for a specific event.

        Args:
            event_name: Event name.

        Returns:
            List of tuples ``[(hook_id, hook_name, hook_func), ...]``.
        """
        return list(self._hooks.get(event_name, []))

    def remove_hook(self, event_name: str, hook_id: int) -> bool:
        """
        Remove a specific hook from an event.

        Args:
            event_name: Event name.
            hook_id: Hook ID.

        Returns:
            True if removed successfully, otherwise False.
        """
        if event_name not in self._hooks:
            return False

        original_len = len(self._hooks[event_name])
        self._hooks[event_name] = [(hid, name, func) for hid, name, func in self._hooks[event_name]
                                    if hid != hook_id]
        self._hook_executor = None
        return len(self._hooks[event_name]) < original_len

    # ========================================================================
    # Compiled Hooks (DSL / Numba-friendly)
    # ========================================================================

    def _register_compiled_hook(self, desc: Any) -> None:
        """Register a compiled hook descriptor.

        Args:
            desc: CompiledHookDescriptor from hooks module.

        Note:
            To avoid maintaining two divergent hook sources, this method only
            mirrors compiled hooks into traditional ``_hooks`` when a real
            Python wrapper exists (selector-mode hooks). Pure declarative and
            njit hooks stay in ``_compiled_hooks`` and are executed by engine
            (or by HookExecutor when trigger_event is used).
        """
        self._compiled_hooks.append(desc)
        self._hook_executor = None

        from natal.numba.utils import NUMBA_ENABLED
        if NUMBA_ENABLED and desc.py_wrapper is not None and desc.njit_fn is None:
            raise TypeError(
                f"Python py_wrapper hook '{desc.name}' is not allowed when Numba is enabled. "
                "Please convert it to @njit or use declarative Op hooks."
            )

        # Mirror only real Python wrappers for trigger_event compatibility.
        # Do not inject no-op placeholders for declarative/njit hooks.
        if desc.py_wrapper is None:
            return
        hook_func = desc.py_wrapper

        # Register with traditional system
        event_name = desc.event
        if event_name in self._hooks:
            current_ids = [hid for hid, _, _ in self._hooks[event_name]]
            hook_id = desc.priority
            # Avoid duplicate IDs
            while hook_id in current_ids:
                hook_id += 1
            self._hooks[event_name].append((hook_id, desc.name, hook_func))
            self._hooks[event_name].sort(key=lambda x: x[0])

    def has_python_hooks(self) -> bool:
        """Return whether any Python-layer hooks are currently registered."""
        hooks_map = cast(Dict[str, List[HookEntry]], getattr(self, "_hooks", {}))
        return any(len(entries) > 0 for entries in hooks_map.values())

    def has_mixed_hook_types(self) -> bool:
        """Return whether any event mixes declarative/njit/python hook types."""
        for event_name in self.ALLOWED_EVENTS:
            kinds: set[str] = set()
            for desc in self.get_compiled_hooks(event_name):
                if getattr(desc, "plan", None) is not None:
                    kinds.add("declarative")
                if getattr(desc, "njit_fn", None) is not None:
                    kinds.add("njit")
                if getattr(desc, "py_wrapper", None) is not None and getattr(desc, "njit_fn", None) is None:
                    kinds.add("python")
            if len(kinds) > 1:
                return True
        return False

    def should_use_python_dispatch(self) -> bool:
                """Return whether this population should run with Python event dispatch.

                Policy:
                        - When Numba is disabled, any registered hook type uses Python
                            dispatch so py/declarative/njit hooks share one sequential path.
                        - When Numba is enabled, mixed hook-type timelines are handled by
                            unified njit functions generated in ``CompiledEventHooks.from_compiled_hooks``,
                            so no Python fallback is needed.
                """
                if not is_numba_enabled():
                        return self.has_python_hooks() or len(self.get_compiled_hooks()) > 0
                return False

    def ensure_hook_executor(self) -> None:
        """Build HookExecutor lazily for Python event-dispatch paths."""
        if self._hook_executor is None:
            self._hook_executor = self._build_hook_executor()

    def register_compiled_hook(self, desc: Any) -> None:
        """Public wrapper for registering compiled hooks."""
        self._register_compiled_hook(desc)

    def get_compiled_hooks(self, event: Optional[str] = None) -> List[Any]:
        """Get compiled hook descriptors, optionally filtered by event.

        Args:
            event: Optional event name to filter by.

        Returns:
            List of CompiledHookDescriptor sorted by priority.
        """
        hooks = cast(List[Any], getattr(self, "_compiled_hooks", []))
        if event is not None:
            hooks = [h for h in hooks if h.event == event]
        return sorted(hooks, key=lambda h: h.priority)

    def register_declarative_hook(
        self,
        event: str,
        ops: List[Any],
        priority: int = 0,
        name: str = "declarative_hook"
    ) -> Any:
        """Register a declarative hook from a list of operations.

        This is an alternative to using the @hook decorator.

        Args:
            event: Event name ('first', 'early', 'late', 'finish')
            ops: List of HookOp operations (from Op.scale, Op.add, etc.)
            priority: Execution priority (lower = earlier)
            name: Hook name for debugging

        Returns:
            CompiledHookDescriptor: The compiled descriptor

        Examples:
            >>> from natal.hooks import Op
            >>> pop.register_declarative_hook(
            ...     event='early',
            ...     ops=[
            ...         Op.scale(genotypes='AA', ages=[0, 1], factor=0.9),
            ...         Op.add(genotypes='*', ages=0, delta=50, when='tick % 10 == 0'),
            ...     ],
            ...     name='juvenile_control'
            ... )
        """
        from natal.hooks import compile_declarative_hook
        desc = compile_declarative_hook(
            ops,
            self,
            event,
            priority=priority,
            name=name,
        )
        self._register_compiled_hook(desc)
        return desc

    def _build_hook_program(self) -> HookProgram:
        """Build HookProgram from compiled hooks.

        This packs all compiled hooks into a Numba-compatible jitclass
        for efficient execution during simulation.

        Returns:
            HookProgram: Compiled hook program data
        """
        from natal.hooks import EVENT_NAMES, HookProgram

        events = EVENT_NAMES
        n_events = len(events)

        # 1. Collect all hooks per event
        hook_offsets: List[int] = [0]
        hook_list_by_event: List[List[CompiledHookDescriptor]] = []

        for event_name in events:
            hooks = self.get_compiled_hooks(event_name)
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))

        n_hooks = hook_offsets[-1]

        # 2. Pack all operation data
        all_op_types: List[int] = []
        all_zidx_offsets: List[int] = [0]
        all_zidx_data: List[int] = []
        all_age_offsets: List[int] = [0]
        all_age_data: List[int] = []
        all_sex_masks: List[bool] = []
        all_params: List[float] = []
        all_cond_offsets: List[int] = [0]
        all_cond_types: List[int] = []
        all_cond_params: List[int] = []

        all_deme_sel_types: List[int] = []
        all_deme_sel_offsets: List[int] = [0]
        all_deme_sel_data: List[int] = []

        n_ops_list: List[int] = []
        op_offsets: List[int] = [0]

        for hooks in hook_list_by_event:
            for hook in hooks:
                plan = hook.plan
                if plan is None or plan.n_ops == 0:
                    n_ops_list.append(0)
                    op_offsets.append(op_offsets[-1])
                    continue

                n_ops_list.append(plan.n_ops)

                # Pack operation data
                all_op_types.extend(plan.op_types.tolist())

                # Handle zidx (adjust offsets for concatenation)
                zidx_offset_base = len(all_zidx_data)
                for i in range(plan.n_ops):
                    all_zidx_offsets.append(
                        zidx_offset_base + plan.zidx_offsets[i + 1] - plan.zidx_offsets[0]
                    )
                all_zidx_data.extend(plan.zidx_data.tolist())

                # Handle age
                age_offset_base = len(all_age_data)
                for i in range(plan.n_ops):
                    all_age_offsets.append(
                        age_offset_base + plan.age_offsets[i + 1] - plan.age_offsets[0]
                    )
                all_age_data.extend(plan.age_data.tolist())

                # Handle sex masks (flatten 2D -> 1D)
                all_sex_masks.extend(plan.sex_masks.flatten().tolist())

                # Handle params, conditions
                all_params.extend(plan.params.tolist())
                cond_offset_base = len(all_cond_types)
                for i in range(plan.n_ops):
                    all_cond_offsets.append(
                        cond_offset_base + plan.condition_offsets[i + 1] - plan.condition_offsets[0]
                    )
                all_cond_types.extend(plan.condition_types.tolist())
                all_cond_params.extend(plan.condition_params.tolist())

                op_offsets.append(len(all_op_types))

                # Pack deme selector from CompiledHookDescriptor
                sel = hook.deme_selector
                if sel == "*":
                    all_deme_sel_types.append(0)
                elif isinstance(sel, int):
                    all_deme_sel_types.append(1)
                    all_deme_sel_data.append(int(sel))
                elif isinstance(sel, range):
                    all_deme_sel_types.append(2)
                    all_deme_sel_data.append(int(sel.start))
                    all_deme_sel_data.append(int(sel.stop))
                else:
                    all_deme_sel_types.append(3)
                    all_deme_sel_data.extend([int(x) for x in sel])
                all_deme_sel_offsets.append(len(all_deme_sel_data))

        # 3. Create HookProgram
        return HookProgram(
            n_events=np.int32(n_events),
            n_hooks=np.int32(n_hooks),
            hook_offsets=np.array(hook_offsets, dtype=np.int32),
            n_ops_list=np.array(n_ops_list, dtype=np.int32),
            op_offsets=np.array(op_offsets, dtype=np.int32),
            op_types_data=np.array(all_op_types, dtype=np.int32),
            zidx_offsets_data=np.array(all_zidx_offsets, dtype=np.int32),
            zidx_data=np.array(all_zidx_data, dtype=np.int32),
            age_offsets_data=np.array(all_age_offsets, dtype=np.int32),
            age_data=np.array(all_age_data, dtype=np.int32),
            sex_masks_data=np.array(all_sex_masks, dtype=np.bool_),
            params_data=np.array(all_params, dtype=np.float64),
            condition_offsets_data=np.array(all_cond_offsets, dtype=np.int32),
            condition_types_data=np.array(all_cond_types, dtype=np.int32),
            condition_params_data=np.array(all_cond_params, dtype=np.int32),
            deme_selector_types=np.array(all_deme_sel_types, dtype=np.int32),
            deme_selector_offsets=np.array(all_deme_sel_offsets, dtype=np.int32),
            deme_selector_data=np.array(all_deme_sel_data, dtype=np.int32),
        )

    def _build_hook_executor(self):
        """Build HookExecutor from compiled hooks and HookProgram.

        HookExecutor is a Python-layer coordinator that manages:
        1. CSR operations via execute_csr_event_program()
        2. njit_fn hooks (user Numba functions)
        3. py_wrapper hooks (Python wrappers for selector mode)

        Returns:
            HookExecutor: Executor instance, or None if no hooks compiled
        """
        from natal.hooks import HookExecutor

        # Get or build HookProgram for CSR operations
        program = self._build_hook_program()
        program_available = True

        # Get all compiled hooks
        compiled_hooks = self._compiled_hooks
        if not compiled_hooks:
            return None

        # If no program (no CSR operations), create an empty one
        # so HookExecutor can still manage njit_fn and py_wrapper hooks
        if not program_available:
            program = self._create_empty_hook_program()

        # Create executor
        executor = HookExecutor.from_compiled_hooks(program, compiled_hooks)
        return executor

    def _create_empty_hook_program(self):
        """Create an empty HookProgram for non-CSR operations.

        Used when there are no declarative Op.* operations,
        but there are njit_fn or py_wrapper hooks.
        """
        from natal.hooks import NUM_EVENTS, HookProgram

        n_events = NUM_EVENTS

        # Create empty CSR arrays
        hook_offsets = np.zeros(n_events + 1, dtype=np.int32)
        op_offsets = np.array([0], dtype=np.int32)

        return HookProgram(
            n_events=np.int32(n_events),
            n_hooks=np.int32(0),
            hook_offsets=hook_offsets,
            n_ops_list=np.array([], dtype=np.int32),
            op_offsets=op_offsets,
            op_types_data=np.array([], dtype=np.int32),
            zidx_offsets_data=np.array([0], dtype=np.int32),
            zidx_data=np.array([], dtype=np.int32),
            age_offsets_data=np.array([0], dtype=np.int32),
            age_data=np.array([], dtype=np.int32),
            sex_masks_data=np.array([], dtype=np.bool_),
            params_data=np.array([], dtype=np.float64),
            condition_offsets_data=np.array([0], dtype=np.int32),
            condition_types_data=np.array([], dtype=np.int32),
            condition_params_data=np.array([], dtype=np.int32),
            deme_selector_types=np.array([], dtype=np.int32),
            deme_selector_offsets=np.array([0], dtype=np.int32),
            deme_selector_data=np.array([], dtype=np.int32),
        )

    def get_compiled_event_hooks(self) -> LifecycleWrappers:
        """Get compiled hooks and lifecycle wrappers for kernel-based simulation.

        This method collects all registered hooks, compiles them into
        Numba-friendly combined functions, and wraps them in pre-compiled
        lifecycle loop functions (tick / run).

        Returns:
            LifecycleWrappers: Container with compiled event hooks
                (``.hooks.first`` etc.) plus pre-compiled lifecycle loop
                functions (``.run_fn``, ``.run_discrete_fn``, etc.).

        Examples:
            >>> wrappers = pop.get_compiled_event_hooks()
            >>> wrappers.run_fn is not None
            True
        """
        registry = self._build_hook_program()
        return compile_lifecycle_wrappers(
            self._compiled_hooks,
            registry=registry,
            include_spatial_wrappers=False,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tick={self.tick}, "
            f"size={self.get_total_count()})"
        )
