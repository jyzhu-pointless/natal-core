"""Base population model helpers and abstractions.

This module provides the abstract base class and utilities for population
models (Wright-Fisher, age-structured non-Wright-Fisher, and related
architectures). The base class defines common interfaces, hook management,
modifier registration, and helpers that are implemented by concrete
population classes.

Design goals:
- Provide a user-friendly high-level API using Python objects (e.g. ``Genotype``).
- Store internal state in NumPy arrays for compatibility with Numba acceleration.
- Separate logical indexing from storage via an index mapping layer.

Docstring style: Google style (Args, Returns, Raises, Example).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
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

import natal.modifiers as _modifiers
import natal.population_config as _population_config
from natal.genetic_entities import Genotype, HaploidGenotype
from natal.genetic_structures import Species
from natal.helpers import resolve_sex_label
from natal.hook_dsl import CompiledEventHooks
from natal.index_registry import IndexRegistry
from natal.modifiers import GameteModifier, ZygoteModifier
from natal.population_config import PopulationConfig
from natal.population_state import DiscretePopulationState, PopulationState

T_State = TypeVar("T_State", bound=Union[PopulationState, DiscretePopulationState])

if TYPE_CHECKING:
    from natal.genetic_presets import GeneticPreset
    from natal.hook_dsl import CompiledHookDescriptor, DemeSelector, HookProgram

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
    implementations (for example, Wright-Fisher and age-structured
    non-Wright-Fisher models). It manages the species/genetic architecture,
    indexing, hook registration, and modifier pipelines.

    Core components:
        - ``species``: Genetic architecture descriptor.
        - ``registry``: ``IndexRegistry`` instance for managing genotype/haplotype indices.
        - ``state``: Abstract property implemented by subclasses (``PopulationState`` or
          age-structured variants).
        - ``_hooks``: Event hook registry mapping event names to ordered hook lists.
        - ``_compiled_hooks``: Compiled event hooks for efficient execution.
    """

    # 允许的 Hook 事件（子类可扩展此列表）
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

        # 演化历史：(tick, flattened_array) 对的列表
        self._history: List[Tuple[int, np.ndarray]] = []

        # History config
        self.record_every: int = 1
        self.max_history: int = 5000  # Default rolling window size

        # Hooks 系统：事件名 -> [(hook_id, hook_name, hook_func), ...]
        self._hooks: Dict[str, List[HookEntry]] = {
            event: [] for event in self.ALLOWED_EVENTS
        }

        # 统一的配子修饰器列表
        self._gamete_modifiers: List[Tuple[int, Optional[str], GameteModifier]] = []

        # 统一的合子修饰器列表
        self._zygote_modifiers: List[Tuple[int, Optional[str], ZygoteModifier]] = []

        # 编译后的 Hook Description符列表（用于 numba 加速）
        self._compiled_hooks: List[Any] = []  # List[CompiledHookDescriptor]

        # Hook 执行器（Python 层协调器，管理所有Type的 hooks）
        self._hook_executor: Optional[Any] = None  # HookExecutor

        # 静态数据容器
        self._config: Optional[PopulationConfig] = None

        # PopulationState 容器
        self._state: Optional[T_State] = None

        # 演化状态：是否已完成（finish）
        self._finished = False

        # 防止递归调用的标志
        self._running = False

        # 存储待延迟编译的 hooks（在子类初始化完成后编译）
        # 格式: [(event_name, func, hook_name, hook_id), ...]
        self._pending_hooks: List[PendingHook] = []

        # 注册 hooks
        # 注意：如果 hook 带有 @hook 元数据，此时可能无法编译（IndexRegistry未完全设置）
        # 普通函数可以直接注册，带 @hook 的函数会被添加到 _pending_hooks 延迟编译
        hooks_map: HookRegistrationMap = hooks or {}
        if hooks_map:
            for event_name, hooks_list in hooks_map.items():
                for hook_info in hooks_list:
                    func, hook_name, hook_id = hook_info

                    # Check if function has @hook metadata
                    hook_meta = getattr(func, '_hook_meta', None)
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

    # ========================================================================
    # Registry and Genotype Initialization
    # ========================================================================

    def _initialize_registry(self) -> None:
        """Template method: Initialize registry and register all genotypes.

        This method uses the Template Method Pattern to orchestrate
        initialization in a consistent sequence:
          1. Call _create_registry() to get a registry instance
          2. Register all genotypes from _get_genotypes()
          3. Attempt to get haplogenotypes and register them (if available)

        Subclasses customize behavior via _create_registry() and _get_genotypes().
        This method should be called once during subclass __init__,
        after super().__init__() but before other initialization.
        """
        # Step 1: Create registry
        self._index_registry = self._create_registry()
        self._registry = self._index_registry

        # Step 2: Register genotypes
        genotypes = self._get_genotypes()
        for genotype in genotypes:
            self._index_registry.register_genotype(genotype)

        # Step 3: Try to register haplogenotypes if available
        haplogenotypes = self._get_haplogenotypes()
        if haplogenotypes:
            for hg in haplogenotypes:
                self._index_registry.register_haplogenotype(hg)

        # Step 4: Register gamete labels if provided
        raw_glabs = cast(Optional[List[str]], getattr(self._species, "gamete_labels", None))
        glabs = raw_glabs or ["default"]
        for glab in glabs:
            self._index_registry.register_gamete_label(glab)

    # Helpers
    def _create_registry(self) -> IndexRegistry:
        return IndexRegistry()

    def _get_genotypes(self) -> List[Genotype]:
        return self.species.get_all_genotypes()
        # return self._registry.index_to_genotype

    def _get_haplogenotypes(self) -> Optional[List[HaploidGenotype]]:
        return self.species.get_all_haploid_genotypes()
        # return self._registry.index_to_haplo

    def _resolve_genotype_key(self, genotype_key: Union[Genotype, str]) -> Genotype:
        if isinstance(genotype_key, Genotype):
            return genotype_key
        return self.species.get_genotype_from_str(genotype_key)

    @staticmethod
    def _derive_hook_slot(name: str) -> int:
        """Derive a stable non-negative hook slot from population name."""
        digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
        # Keep int32-compatible positive range for config scalar stability.
        return int(digest[:8], 16) & 0x7FFFFFFF

    @property
    def hook_slot(self) -> int:
        return int(self._hook_slot)

    # ========================================================================
    # 基础属性
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
        self._name = value

    @property
    def tick(self) -> int:
        """The current simulation tick or generation index."""
        return self._tick

    @tick.setter
    def tick(self, value: int) -> None:
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
    def config(self) -> PopulationConfig:
        """Public accessor for compiled population configuration."""
        if self._config is None:
            raise AttributeError("Population config has not been initialized.")
        return self._config

    def _require_registry(self) -> IndexRegistry:
        """Return the initialized registry or raise a clear initialization error."""
        if self._registry is None:
            raise AttributeError("Index registry has not been initialized.")
        return self._registry

    def _require_config(self) -> PopulationConfig:
        """Return the initialized config or raise a clear initialization error."""
        if self._config is None:
            raise AttributeError("Population config has not been initialized.")
        return self._config

    def _require_state(self) -> T_State:
        """Return the initialized state or raise a clear initialization error."""
        if self._state is None:
            raise AttributeError("Population state has not been initialized.")
        return self._state

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
        pass

    def _process_kernel_history(
        self,
        history_new: Optional[np.ndarray],
        clear_history_on_start: bool
    ) -> None:
        """Process and append history array returned from simulation kernels.

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
    # Modifier 管理
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

    def _refresh_modifier_maps(self) -> None:
        if self._config is None or self._registry is None:
            return

        haploid_genotypes = self._registry.index_to_haplo
        diploid_genotypes = self._registry.index_to_genotype
        if not haploid_genotypes or not diploid_genotypes:
            return

        n_glabs = int(self._config.n_glabs)
        gamete_funcs, zygote_funcs = build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            index_registry=self._index_registry,
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
        )

        genotype_to_gametes_map = initialize_gamete_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            gamete_modifiers=gamete_funcs,
        )

        gametes_to_zygote_map = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            zygote_modifiers=zygote_funcs,
        )

        self._config = self._config._replace(
            genotype_to_gametes_map=genotype_to_gametes_map,
            gametes_to_zygote_map=gametes_to_zygote_map,
        )

    def refresh_modifier_maps(self) -> None:
        """Public wrapper that rebuilds modifier maps from registered modifiers."""
        self._refresh_modifier_maps()

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
        """
        resolved_id = self._resolve_modifier_id(modifier_id, self._gamete_modifiers)
        self._gamete_modifiers.append((resolved_id, name, modifier))
        self._gamete_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self._refresh_modifier_maps()

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
        """
        resolved_id = self._resolve_modifier_id(modifier_id, self._zygote_modifiers)
        self._zygote_modifiers.append((resolved_id, name, modifier))
        self._zygote_modifiers.sort(key=lambda x: x[0])
        if refresh:
            self._refresh_modifier_maps()

    # 确保 set_zygote_modifier 方法与 ZygoteModifier 定义一致
    def set_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        modifier_id: Optional[int] = None,
        modifier_name: Optional[str] = None
    ) -> None:
        """Register a zygote modifier with an optional priority.

        Args:
            modifier: A ``ZygoteModifier`` instance or callable.
            modifier_id: Numeric priority (lower values execute earlier). If omitted
                an id will be auto-assigned.
            modifier_name: Optional name for debugging.
        """
        if not callable(modifier):
            raise TypeError("Zygote modifier must be callable")

        resolved_id = self._resolve_modifier_id(modifier_id, self._zygote_modifiers)

        # 添加并排序
        self._zygote_modifiers.append((resolved_id, modifier_name, modifier))
        self._zygote_modifiers.sort(key=lambda x: x[0])

    def set_gamete_modifier(
        self,
        modifier: GameteModifier,
        modifier_id: Optional[int] = None,
        modifier_name: Optional[str] = None
    ) -> None:
        """Register a gamete modifier with optional priority and name."""
        if not callable(modifier):
            raise TypeError("Gamete modifier must be callable")

        resolved_id = self._resolve_modifier_id(modifier_id, self._gamete_modifiers)

        # 添加并排序
        self._gamete_modifiers.append((resolved_id, modifier_name, modifier))
        self._gamete_modifiers.sort(key=lambda x: x[0])

    def apply_preset(self, preset: GeneticPreset) -> None:
        """Apply a genetic preset to this population.

        This is the preferred API for registering presets. The preset's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).

        Example:
            >>> from natal.genetic_presets import HomingDrive
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     drive_conversion_rate=0.95
            ... )
            >>> population.apply_preset(drive)

        See Also:
            :class:`natal.genetic_presets.GeneticPreset` - Base class for creating custom presets
            :class:`natal.genetic_presets.HomingDrive` - Built-in gene drive preset
        """
        from natal.genetic_presets import apply_preset_to_population
        apply_preset_to_population(self, preset)

    @classmethod
    def builder(cls, species: Species) -> Any:
        """Create a builder for this population type.

        This is the recommended way to construct populations with presets.

        Args:
            species: Genetic architecture for the population.

        Returns:
            A builder instance for this population type.

        Example:
            >>> pop = (AgeStructuredPopulation.builder(species)
            ...     .set_age_structure(n_ages=10)
            ...     .add_preset(HomingModificationDrive(...))
            ...     .build())
        """
        raise NotImplementedError(f"{cls.__name__} must implement builder()")

    def initialize_config(self) -> None:
        """Initialize static lookup tensors used by the population model.

        This prepares precomputed maps such as ``gametes_to_zygote_map`` and
        ``genotype_to_gametes_map`` and wraps high-level modifiers so they can
        be applied at tensor-level during simulation steps.

        Note:
            Ensures registry is initialized before proceeding.
        """
        # ✅ Ensure registry is initialized
        if self._index_registry is None:
            self._initialize_registry()

        # 获取所有可能的单倍型和二倍型
        haploid_genotypes: List[HaploidGenotype] = self.species.get_all_haploid_genotypes()
        diploid_genotypes: List[Genotype] = self.species.get_all_genotypes()

        n_hg = len(haploid_genotypes)
        n_genotypes = len(diploid_genotypes)
        n_glabs = 1  # 根据实际情况设置

        # 创建静态数据容器
        self._config = build_population_config(
            n_genotypes=n_genotypes,
            n_haploid_genotypes=n_hg,
            n_sexes=2, # TODO
            n_glabs=n_glabs
        )

        # 使用统一的 wrapper 生成器将高层 modifier 转换为 tensor-level modifier
        gamete_modifier_funcs, zygote_modifier_funcs = self._build_modifier_wrappers(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs
        )

        # 初始化 gametes_to_zygote_map 与 genotype_to_gametes_map
        gametes_to_zygote_map = initialize_zygote_map(
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
            zygote_modifiers=zygote_modifier_funcs,
        )

        genotype_to_gametes_map = initialize_gamete_map(
            diploid_genotypes=diploid_genotypes,
            haploid_genotypes=haploid_genotypes,
            n_glabs=n_glabs,
            gamete_modifiers=gamete_modifier_funcs,
        )

        self._config = self._config._replace(
            gametes_to_zygote_map=gametes_to_zygote_map,
            genotype_to_gametes_map=genotype_to_gametes_map,
        )

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
            if lab not in self._index_registry.glab_to_index:
                self._index_registry.register_gamete_label(lab)

    # ------------------------------------------------------------------
    # Helper routines to simplify modifier key/value parsing. These were
    # extracted from the inline closures in _build_modifier_wrappers to
    # reduce cognitive complexity and improve testability.
    # ------------------------------------------------------------------
    def _resolve_hg_glab(
        self,
        haploid_genotypes: List[HaploidGenotype],
        part: Any,
        n_glabs: int
    ) -> Tuple[int, int]:
        """Resolve a flexible haploid/genotype+glab part into numeric indices.

        Args:
            haploid_genotypes: list of HaploidGenotype objects.
            part: flexible selector (HaploidGenotype, int, str, or tuple).
            n_glabs: number of gamete labels.

        Returns:
            (hg_idx, glab_idx)
        """
        return self.index_registry.resolve_hg_glab_part(haploid_genotypes, part, n_glabs)

    def _parse_zygote_key(self, key: Any, haploid_genotypes: List[HaploidGenotype], n_glabs: int) -> Tuple[int, int]:
        """Parse modifier key for zygote wrappers into compressed coords (c1,c2).

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import parse_zygote_key

        return parse_zygote_key(key, self._index_registry, haploid_genotypes, n_glabs)

    def _normalize_zygote_val(self, val: Any, diploid_genotypes: List[Genotype]) -> Dict[int, float]:
        """Normalize zygote replacement `val` into a mapping idx->prob.

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import normalize_zygote_val

        return normalize_zygote_val(val, self._index_registry, diploid_genotypes)

    def _write_zygote_mapping(self, modified: np.ndarray, c1: int, c2: int, mapping: Dict[int, float]) -> None:
        """Apply mapping (idx->prob) to the compressed zygote slice.

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import write_zygote_mapping

        write_zygote_mapping(modified, c1, c2, mapping)

    def _resolve_sex_name(self, key: str) -> Optional[int]:
        """Normalize string sex names to sex index (0=female,1=male).

        Returns None for unknown keys.
        """
        try:
            return resolve_sex_label(key)
        except ValueError:
            return None

    def _apply_comp_map(self, modified: np.ndarray, sex_idx: int, gidx: int, comp_map: Any, haploid_genotypes: List[HaploidGenotype], n_glabs: int, n_hg_glabs: int) -> None:
        """Apply a comp_map (comp_key->freq) into the provided modified tensor slice.

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import apply_comp_map

        apply_comp_map(
            modified,
            sex_idx,
            gidx,
            comp_map,
            self._index_registry,
            haploid_genotypes,
            n_glabs,
            n_hg_glabs,
        )

    def _build_modifier_wrappers(
        self,
        haploid_genotypes: List[HaploidGenotype],
        diploid_genotypes: List[Genotype],
        n_glabs: int = 1
    ) -> Tuple[List[HookCallback], List[HookCallback]]:
        """Wrap high-level gamete/zygote modifiers into tensor-level callables.

        Delegates to the shared ``build_modifier_wrappers`` in the modifiers module.

        Returns:
            Tuple containing two lists: ``(gamete_modifier_funcs, zygote_modifier_funcs)``.
        """
        return build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            index_registry=self._index_registry,
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
        )

    # ========================================================================
    # 核心方法
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
    # 通用方法（可被子类继承或覆写）
    # ========================================================================

    @property
    def total_population_size(self) -> int:
        """种群总大小（get_total_count 的别名）"""
        return self.get_total_count()

    @property
    def total_females(self) -> int:
        """雌性总数（get_female_count 的别名）"""
        return self.get_female_count()

    @property
    def total_males(self) -> int:
        """雄性总数（get_male_count 的别名）"""
        return self.get_male_count()

    @property
    def sex_ratio(self) -> float:
        """Return the female-to-male ratio, or ``np.inf`` when male count is zero."""
        males = self.get_male_count()
        return self.get_female_count() / males if males > 0 else np.inf

    @property
    def is_finished(self) -> bool:
        """检查种群是否已完成（finish=True）"""
        return self._finished

    def finish_simulation(self) -> None:
        """
        结束模拟，触发 'finish' 事件并锁定种群。

        此方法可以被 hooks 调用以提前结束模拟。
        调用后，种群将无法再运行 step()/run_tick()/run()。

        Raises:
            RuntimeError: 如果种群已经 finished

        Example:
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
        record_every: int = 1,
        finish: bool = False
    ) -> BasePopulation[Any]:
        """
        运行多步演化。
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the population to its initial state."""
        pass

    def compute_allele_frequencies(self) -> Dict[str, float]:
        """
        计算种群中所有等位基因的频率（按位点归一化）。

        Returns:
            Dict[str, float]: 映射 {allele_name: frequency}。
            频率是相对于该位点总等位基因数的比例 (0.0 - 1.0)。
        """
        if self._state is None or self._registry is None:
            return {}

        # 1. 初始化计数器
        allele_counts: Dict[str, float] = {}
        locus_totals: Dict[str, float] = {}  # locus_name -> total_count

        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                locus_totals[locus.name] = 0.0
                for gene in locus.alleles:
                    allele_counts[gene.name] = 0.0

        # 2. 聚合基因型计数
        # individual_count shape: (n_sexes, n_ages, n_genotypes)
        # 对性别和年龄求和，得到每个基因型的总数
        genotype_counts = self._state.individual_count.sum(axis=(0, 1))

        registry = self._registry
        for g_idx, count in enumerate(genotype_counts):
            if count <= 0:
                continue

            genotype = registry.index_to_genotype[g_idx]
            for chrom in self.species.chromosomes:
                for locus in chrom.loci:
                    mat, pat = genotype.get_alleles_at_locus(locus)
                    for allele in (mat, pat):
                        if allele is not None:
                            allele_counts[allele.name] += count
                            locus_totals[locus.name] += count

        # 3. 计算频率
        frequencies: Dict[str, float] = {}
        for allele_name, count in allele_counts.items():
            # 找到该等位基因对应的 locus total
            # (由于我们没有直接的 gene->locus 快速反查，这里稍微低效一点但安全)
            # 实际上我们可以通过 self.species.gene_index 查找
            gene = self.species.gene_index.get(allele_name)
            if gene and locus_totals[gene.locus.name] > 0:
                frequencies[allele_name] = count / locus_totals[gene.locus.name]
            else:
                frequencies[allele_name] = 0.0

        return frequencies

    # ========================================================================
    # Hooks 系统
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
        注册事件 Hook，支持自动编译。

        当 `compile=True` 且函数带有 `@hook` 元数据时，会走 DSL 编译管线：
        - 声明式 hook -> CSR 计划，进入 HookProgram（kernel 可执行）
        - selector hook -> py_wrapper 或 njit_fn（取决于模式）
        - numba hook -> njit_fn

        普通 Python 函数仍可直接注册到传统 `_hooks`（兼容路径）。

        Args:
            event_name: 事件名称（必须在 ALLOWED_EVENTS 中）
            func: 回调函数，支持以下形式：
                  - 普通函数: func(population)
                  - @hook 装饰的声明式函数: Returns [Op.scale(...), ...]
                  - @hook(selectors={...}) 装饰的选择器函数
            hook_id: Hook 的数值优先级（可选，自动分配）
                     较小的 ID 先执行
            hook_name: Hook 的可读名称（可选，用于调试）
            compile: 是否尝试编译 @hook 装饰的函数（Default True）
            deme_selector: 可选的子种群选择器。
                - None: 保持 panmictic Default行为（不显式覆写 selector）
                - 非 None: 传给 hook 编译注册流程用于 spatial 过滤

        Raises:
            ValueError: 如果事件不存在或 hook_id 已被使用

        Example:
            >>> # 普通函数（向后兼容）
            >>> pop.set_hook('first', lambda p: print(f'Step {p.tick}'))
            >>>
            >>> # @hook 装饰的声明式函数（自动编译）
            >>> @hook()
            >>> def reduce_juveniles():
            ...     return [Op.scale(genotypes='AA', ages=[0, 1], factor=0.9)]
            >>> pop.set_hook('early', reduce_juveniles)
            >>>
            >>> # @hook 装饰的选择器函数（自动编译）
            >>> @hook(selectors={'target': 'AA'})
            >>> def release(pop, target):
            ...     pop.state.individual_count[1, 2, target] += 100
            >>> pop.set_hook('first', release)
        """
        if event_name not in self.ALLOWED_EVENTS:
            raise ValueError(f"Event '{event_name}' not in {self.ALLOWED_EVENTS}")

        # Check if function has @hook metadata and should be compiled
        hook_meta = getattr(func, '_hook_meta', None)

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
                return

        # Traditional registration (no compilation)
        actual_name = hook_name or getattr(func, '__name__', None)

        current_ids = [hid for hid, _, _ in self._hooks[event_name]]

        if hook_id is None:
            hook_id = (max(current_ids) + 1) if current_ids else 0

        if hook_id in current_ids:
            raise ValueError(f"hook_id {hook_id} already exists in event '{event_name}'")

        self._hooks[event_name].append((hook_id, actual_name, func))
        # 按 ID 排序保证执行顺序
        self._hooks[event_name].sort(key=lambda x: x[0])

    def trigger_event(self, event_name: str, deme_id: int = 0) -> int:
        """
        触发事件，执行所有已注册的 hooks。

        执行顺序：
        1. CSR操作（Numba快速路径）
        2. njit_fn hooks（用户自定义Numba函数）
        3. py_wrapper hooks（Python包装函数）

        Args:
            event_name: 要触发的事件名称
            deme_id: 子种群 ID（可选，Default为 0）

        Returns:
            int: RESULT_CONTINUE (0) 继续运行，RESULT_STOP (1) 请求停止

        Note:
            - 优先走 HookExecutor（三层统一协调）
            - 若执行器未构建，降级到传统 `_hooks`（仅 Python 回调）
            - 在加速 run() 中，核心事件主要由 kernel 执行；trigger_event
              主要用于显式事件触发（如 finish）与兼容路径

        Example:
            >>> result = pop.trigger_event('first')  # 执行所有 'first' hooks
            >>> if result == RESULT_STOP:
            ...     print("Simulation stopped by hook")
        """
        from natal.hook_dsl import RESULT_CONTINUE

        # 优先使用 HookExecutor（如果已构建）
        if self._hook_executor is not None:
            from natal.hook_dsl import EVENT_ID_MAP
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is not None:
                result = self._hook_executor.execute_event(event_id, self, self.tick, deme_id=deme_id)
                return result

        # 降级到传统 _hooks 系统（兼容性）
        for _, _, hook in self._hooks.get(event_name, []):
            hook(self)

        return RESULT_CONTINUE


    def get_hooks(self, event_name: str) -> List[HookEntry]:
        """
        获取特定事件的所有已注册 hooks。

        Args:
            event_name: 事件名称

        Returns:
            [(hook_id, hook_name, hook_func), ...] 列表
        """
        return list(self._hooks.get(event_name, []))

    def remove_hook(self, event_name: str, hook_id: int) -> bool:
        """
        删除指定事件的指定 hook。

        Args:
            event_name: 事件名称
            hook_id: Hook 的 ID

        Returns:
            删除成功Returns True，否则Returns False
        """
        if event_name not in self._hooks:
            return False

        original_len = len(self._hooks[event_name])
        self._hooks[event_name] = [(hid, name, func) for hid, name, func in self._hooks[event_name]
                                    if hid != hook_id]
        return len(self._hooks[event_name]) < original_len

    # ========================================================================
    # Compiled Hooks (DSL / Numba-friendly)
    # ========================================================================

    def _register_compiled_hook(self, desc: Any) -> None:
        """Register a compiled hook descriptor.

        Args:
            desc: CompiledHookDescriptor from hook_dsl module.

        Note:
            To avoid maintaining two divergent hook sources, this method only
            mirrors compiled hooks into traditional ``_hooks`` when a real
            Python wrapper exists (selector-mode hooks). Pure declarative and
            njit hooks stay in ``_compiled_hooks`` and are executed by kernels
            (or by HookExecutor when trigger_event is used).
        """
        self._compiled_hooks.append(desc)

        from natal.numba_utils import NUMBA_ENABLED
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
        hooks = self._compiled_hooks
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

        Example:
            >>> from natal.hook_dsl import Op
            >>> pop.register_declarative_hook(
            ...     event='early',
            ...     ops=[
            ...         Op.scale(genotypes='AA', ages=[0, 1], factor=0.9),
            ...         Op.add(genotypes='*', ages=0, delta=50, when='tick % 10 == 0'),
            ...     ],
            ...     name='juvenile_control'
            ... )
        """
        from natal.hook_dsl import compile_declarative_hook
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
        from natal.hook_dsl import EVENT_NAMES, HookProgram

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
        all_gidx_offsets: List[int] = [0]
        all_gidx_data: List[int] = []
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

                # Handle gidx (adjust offsets for concatenation)
                gidx_offset_base = len(all_gidx_data)
                for i in range(plan.n_ops):
                    all_gidx_offsets.append(
                        gidx_offset_base + plan.gidx_offsets[i + 1] - plan.gidx_offsets[0]
                    )
                all_gidx_data.extend(plan.gidx_data.tolist())

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
            gidx_offsets_data=np.array(all_gidx_offsets, dtype=np.int32),
            gidx_data=np.array(all_gidx_data, dtype=np.int32),
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
        from natal.hook_dsl import HookExecutor

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
        from natal.hook_dsl import NUM_EVENTS, HookProgram

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
            gidx_offsets_data=np.array([0], dtype=np.int32),
            gidx_data=np.array([], dtype=np.int32),
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

    def get_compiled_event_hooks(self) -> CompiledEventHooks:
        """Get compiled hooks for use with generated kernel wrappers.

        This method collects all registered hooks and compiles them into
        Numba-friendly combined functions, one per event.

        Returns:
            CompiledEventHooks: Container with combined @njit hooks per event.
                                Access via .first, .early, .late, .finish

        Example:
            >>> hooks = pop.get_compiled_event_hooks()
            >>> hooks.run_fn is not None
            True
        """
        from natal.hook_dsl import CompiledEventHooks

        registry = self._build_hook_program()
        return CompiledEventHooks.from_compiled_hooks(
            self._compiled_hooks,
            registry=registry
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"tick={self.tick}, "
            f"size={self.get_total_count()})"
        )
