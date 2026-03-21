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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Set, Callable, Any, FrozenSet, Union, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
import hashlib
import numpy as np
from natal.genetic_structures import *
from natal.genetic_entities import *
from natal.index_core import IndexCore
from natal.type_def import *
from natal.population_state import PopulationState
from natal.population_config import PopulationConfig
from natal.modifiers import GameteModifier, ZygoteModifier, build_modifier_wrappers, _resolve_sex_name
from natal.hook_dsl import CompiledEventHooks

class BasePopulation(ABC):
    """Abstract base class for population models.

    The base class unifies common behavior for different population model
    implementations (for example, Wright-Fisher and age-structured
    non-Wright-Fisher models). It manages the species/genetic architecture,
    indexing, hook registration, and modifier pipelines.

    Core components:
        - ``species``: Genetic architecture descriptor.
        - ``registry``: ``IndexCore`` instance for managing genotype/haplotype indices.
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
        hooks: Optional[Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]]] = None
    ):
        """Initialize the base population.

        Args:
            species: Genetic architecture specifying chromosomes, loci, and alleles.
            name: Optional population name (default: "Population").
            hooks: Optional mapping of event names to hook registrations. Each
                entry should be a sequence of tuples in the form ``(func,)``,
                ``(func, hook_name)``, or ``(func, hook_name, hook_id)``. Hooks
                provided here will be registered during initialization.

        Note:
            Registry and genotypes are initialized lazily via Template Method.
            Subclasses must implement _create_registry() and _get_genotypes().
        """
        if not isinstance(species, Species):
            raise TypeError("species must be a Species instance.")
        
        self._species = species
        self._name = name
        self._hook_slot = self._derive_hook_slot(name)
        self._tick = 0
        # DELAYED: Registry will be created via _initialize_registry()
        self._index_core: Optional[IndexCore] = None
        self._registry: Optional[IndexCore] = None
        
        # 演化历史：(tick, flattened_array) 对的列表
        self._history: List[Tuple[int, np.ndarray]] = []
        
        # Hooks 系统：事件名 -> [(hook_id, hook_name, hook_func), ...]
        self._hooks: Dict[str, List[Tuple[int, Optional[str], Callable]]] = {
            event: [] for event in self.ALLOWED_EVENTS
        }

        # 统一的配子修饰器列表
        self._gamete_modifiers: List[Tuple[int, Optional[str], GameteModifier]] = []

        # 统一的合子修饰器列表
        self._zygote_modifiers: List[Tuple[int, Optional[str], ZygoteModifier]] = []

        # 编译后的 Hook 描述符列表（用于 numba 加速）
        self._compiled_hooks: List[Any] = []  # List[CompiledHookDescriptor]
        
        # Hook 执行器（Python 层协调器，管理所有类型的 hooks）
        self._hook_executor: Optional[Any] = None  # HookExecutor

        # 静态数据容器
        self._config: Optional[PopulationConfig] = None

        # PopulationState 容器
        self._state: Optional[PopulationState] = None

        # 演化状态：是否已完成（finish）
        self._finished = False
        
        # 防止递归调用的标志
        self._running = False
        
        # 存储待延迟编译的 hooks（在子类初始化完成后编译）
        # 格式: [(event_name, func, hook_name, hook_id), ...]
        self._pending_hooks: List[Tuple[str, Callable, Optional[str], Optional[int]]] = []
        
        # 注册 hooks
        # 注意：如果 hook 带有 @hook 元数据，此时可能无法编译（IndexCore未完全设置）
        # 普通函数可以直接注册，带 @hook 的函数会被添加到 _pending_hooks 延迟编译
        if hooks:
            for event_name, hooks_list in hooks.items():
                for hook_info in hooks_list:
                    if len(hook_info) == 1:
                        func = hook_info[0]
                        hook_name = None
                        hook_id = None
                    elif len(hook_info) == 2:
                        func, hook_name = hook_info
                        hook_id = None
                    else:
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
        with @hook metadata to be compiled with the now-initialized IndexCore.
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
        self._index_core = self._create_registry()
        self._registry = self._index_core
        
        # Step 2: Register genotypes
        genotypes = self._get_genotypes()
        for genotype in genotypes:
            self._index_core.register_genotype(genotype)
        
        # Step 3: Try to register haplogenotypes if available
        haplogenotypes = self._get_haplogenotypes()
        if haplogenotypes:
            for hg in haplogenotypes:
                self._index_core.register_haplogenotype(hg)
        
        # Step 4: Register gamete labels if provided
        glabs = self._species.gamete_labels or ["default"]
        for glab in glabs:
            self._index_core.register_gamete_label(glab)
    
    # Helpers
    def _create_registry(self) -> IndexCore:
        return IndexCore()

    def _get_genotypes(self) -> List[Genotype]:
        return self._genotypes_list

    def _get_haplogenotypes(self) -> Optional[List]:
        return self._haploid_genotypes_list

    def _resolve_genotype_key(self, genotype_key: Union[Genotype, str]) -> Genotype:
        if isinstance(genotype_key, Genotype):
            return genotype_key
        if isinstance(genotype_key, str):
            return self.species.get_genotype_from_str(genotype_key)
        raise TypeError(f"Unsupported genotype key type: {type(genotype_key)}")

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
    def registry(self) -> IndexCore:
        """IndexCore instance managing genotype, haplotype, and label indices."""
        return self._registry
    
    @property
    def state(self) -> PopulationState:
        """Return the current population state container.

        Returns:
            PopulationState: The current state object used by the population.
        """
        return self._state
    
    @property
    def history(self) -> List[Tuple[int, np.ndarray]]:
        """A list of recorded historical states as ``(tick, flattened_array)`` tuples."""
        return list(self._history)
    
    # ========================================================================
    # Modifier 管理
    # ========================================================================

    def _refresh_modifier_maps(self) -> None:
        if self._config is None or self._registry is None:
            return

        haploid_genotypes = self._registry.index_to_haplo
        diploid_genotypes = self._registry.index_to_genotype
        if not haploid_genotypes or not diploid_genotypes:
            return

        from natal.population_config import initialize_gamete_map, initialize_zygote_map

        n_glabs = int(self._config.n_glabs)
        gamete_funcs, zygote_funcs = build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            index_core=self._index_core,
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
    
    def add_gamete_modifier(
        self, 
        modifier: GameteModifier, 
        name: Optional[str] = None, 
        hook_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a gamete-level modifier.

        Args:
            modifier: A ``GameteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            hook_id: Optional numeric priority used for ordering.
        """
        self._gamete_modifiers.append((hook_id, name, modifier))
        if refresh:
            self._refresh_modifier_maps()
    
    def add_zygote_modifier(
        self, 
        modifier: ZygoteModifier, 
        name: Optional[str] = None, 
        hook_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a zygote-level modifier.

        Args:
            modifier: A ``ZygoteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            hook_id: Optional numeric priority used for ordering.
        """
        self._zygote_modifiers.append((hook_id, name, modifier))
        if refresh:
            self._refresh_modifier_maps()

    # 确保 set_zygote_modifier 方法与 ZygoteModifier 定义一致
    def set_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None
    ) -> None:
        """Register a zygote modifier with an optional priority.

        Args:
            modifier: A ``ZygoteModifier`` instance or callable.
            hook_id: Numeric priority (lower values execute earlier). If omitted
                an id will be auto-assigned.
            hook_name: Optional name for debugging.
        """
        if not callable(modifier):
            raise TypeError("Zygote modifier must be callable")
        
        # 自动分配 hook_id
        if hook_id is None:
            if self._zygote_modifiers:
                hook_id = max(hid for hid, _, _ in self._zygote_modifiers) + 1
            else:
                hook_id = 0
        
        # 添加并排序
        self._zygote_modifiers.append((hook_id, hook_name, modifier))
        self._zygote_modifiers.sort(key=lambda x: x[0])

    def set_gamete_modifier(
        self,
        modifier: GameteModifier,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None
    ) -> None:
        """Register a gamete modifier with optional priority and name."""
        if not callable(modifier):
            raise TypeError("Gamete modifier must be callable")
        
        # 自动分配hook_id
        if hook_id is None:
            hook_id = max((hid for hid, _, _ in self._gamete_modifiers), default=0) + 1
        
        # 添加并排序
        self._gamete_modifiers.append((hook_id, hook_name, modifier))
        self._gamete_modifiers.sort(key=lambda x: x[0])

    def apply_recipe(self, recipe) -> None:
        """Apply a gene drive recipe to this population.
        
        This is the preferred API for registering recipes. The recipe's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.
        
        Args:
            recipe: A GeneDriveRecipe instance.
        
        Example:
            >>> from natal.recipes import HomingModificationDrive
            >>> drive = HomingModificationDrive(drive_genotype, resistance_genotype)
            >>> population.apply_recipe(drive)
        """
        from natal.recipes import apply_recipe_to_population
        apply_recipe_to_population(self, recipe)
    
    @classmethod
    def builder(cls, species: 'Species'):
        """Create a builder for this population type.
        
        This is the recommended way to construct populations with recipes.
        
        Args:
            species: Genetic architecture for the population.
        
        Returns:
            A builder instance for this population type.
        
        Example:
            >>> pop = (AgeStructuredPopulation.builder(species)
            ...     .set_age_structure(n_ages=10)
            ...     .add_recipe(HomingModificationDrive(...))
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
        if self._index_core is None:
            self._initialize_registry()
        
        from natal.population_config import build_population_config, initialize_gamete_map, initialize_zygote_map
        from natal.genetic_entities import HaploidGenotype, Genotype
        from natal.type_def import Sex
        
        # 获取所有可能的单倍型和二倍型
        haploid_genotypes = self._get_all_possible_haploid_genotypes()
        diploid_genotypes = self._get_all_possible_diploid_genotypes()
        
        n_hg = len(haploid_genotypes)
        n_genotypes = len(diploid_genotypes)
        n_glabs = 1  # 根据实际情况设置
        
        # 创建静态数据容器
        self._config = build_population_config(
            n_genotypes=n_genotypes,
            n_haploid_genotypes=n_hg,
            n_sexes=None,
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
        Register gamete labels in the IndexCore.

        Args:
            labels: Sequence of string labels to register. Labels must be
                unique in the provided sequence. Existing labels are ignored.
        """
        if not hasattr(self, "_index_core") or self._index_core is None:
            raise RuntimeError("IndexCore not initialized; cannot register gamete labels")

        if labels is None:
            return

        # Normalize and validate input
        try:
            seq = list(labels)
        except Exception:
            raise TypeError("labels must be a sequence of strings")

        if not all(isinstance(l, str) for l in seq):
            raise TypeError("all labels must be strings")

        # Ensure provided labels are unique
        if len(set(seq)) != len(seq):
            raise ValueError("labels must be unique")

        # Register each string label if not already present
        for lab in seq:
            if lab not in self._index_core.glab_to_index:
                self._index_core.register_gamete_label(lab)

    # ------------------------------------------------------------------
    # Helper routines to simplify modifier key/value parsing. These were
    # extracted from the inline closures in _build_modifier_wrappers to
    # reduce cognitive complexity and improve testability.
    # ------------------------------------------------------------------
    def _resolve_hg_glab(self, haploid_genotypes: List[HaploidGenotype], part: Any, n_glabs: int, strict: bool = True) -> Tuple[int, int]:
        """Resolve a flexible haploid/genotype+glab part into numeric indices.

        Args:
            haploid_genotypes: list of HaploidGenotype objects.
            part: flexible selector (HaploidGenotype, int, str, or tuple).
            n_glabs: number of gamete labels.
            strict: pass-through to IndexCore resolver.

        Returns:
            (hg_idx, glab_idx)
        """
        return self._index_core.resolve_hg_glab_part(haploid_genotypes, part, n_glabs, strict=strict)

    def _parse_zygote_key(self, key: Any, haploid_genotypes: List[HaploidGenotype], n_glabs: int) -> Tuple[int, int]:
        """Parse modifier key for zygote wrappers into compressed coords (c1,c2).

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import _parse_zygote_key
        return _parse_zygote_key(key, self._index_core, haploid_genotypes, n_glabs)

    def _normalize_zygote_val(self, val: Any, diploid_genotypes: List[Genotype]) -> Dict[int, float]:
        """Normalize zygote replacement `val` into a mapping idx->prob.

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import _normalize_zygote_val
        return _normalize_zygote_val(val, self._index_core, diploid_genotypes)

    def _write_zygote_mapping(self, modified: np.ndarray, c1: int, c2: int, mapping: Dict[int, float]) -> None:
        """Apply mapping (idx->prob) to the compressed zygote slice.

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import _write_zygote_mapping
        _write_zygote_mapping(modified, c1, c2, mapping)

    def _resolve_sex_name(self, key: str) -> Optional[int]:
        """Normalize string sex names to sex index (0=female,1=male).

        Returns None for unknown keys.
        """
        return _resolve_sex_name(key)

    def _apply_comp_map(self, modified: np.ndarray, sex_idx: int, gidx: int, comp_map: Any, haploid_genotypes: List[HaploidGenotype], n_glabs: int, n_hg_glabs: int) -> None:
        """Apply a comp_map (comp_key->freq) into the provided modified tensor slice.

        Delegates to the shared implementation in modifiers module.
        """
        from natal.modifiers import _apply_comp_map
        _apply_comp_map(modified, sex_idx, gidx, comp_map, self._index_core, haploid_genotypes, n_glabs, n_hg_glabs)

    def _build_modifier_wrappers(
        self,
        haploid_genotypes: List[HaploidGenotype],
        diploid_genotypes: List[Genotype],
        n_glabs: int = 1
    ) -> Tuple[List[Callable], List[Callable]]:
        """Wrap high-level gamete/zygote modifiers into tensor-level callables.

        Delegates to the shared ``build_modifier_wrappers`` in the modifiers module.

        Returns:
            Tuple containing two lists: ``(gamete_modifier_funcs, zygote_modifier_funcs)``.
        """
        return build_modifier_wrappers(
            gamete_modifiers=self._gamete_modifiers,
            zygote_modifiers=self._zygote_modifiers,
            population=self,
            index_core=self._index_core,
            haploid_genotypes=haploid_genotypes,
            diploid_genotypes=diploid_genotypes,
            n_glabs=n_glabs,
        )

    def _get_all_possible_haploid_genotypes(self) -> List[HaploidGenotype]:
        """
        获取所有可能的单倍型列表
        这是一个示例实现，需要根据实际情况扩展
        """
        # 简化实现，假设已经有方法可以获取所有可能的单倍型
        # 实际应用中，这可能需要通过枚举所有可能的等位基因组合来生成
        pass

    def _get_all_possible_diploid_genotypes(self) -> List[Genotype]:
        """
        获取所有可能的二倍型列表
        这是一个示例实现，需要根据实际情况扩展
        """
        # 简化实现，假设已经有方法可以获取所有可能的二倍型
        # 实际应用中，这可能需要通过枚举所有可能的单倍型组合来生成
        pass

    # ========================================================================
    # 核心方法
    # ========================================================================
    
    def step(self) -> 'BasePopulation':
        """
        执行一个演化步骤。
        
        标准流程：
        1. 检查是否已 finish
        2. 设置 _running 标志防止递归
        3. 触发 'first' hook
        4. 调用 _step_reproduction()
        5. 触发 'early' hook
        6. 调用 _step_survival()
        7. 触发 'late' hook
        8. 调用 _step_aging()
        9. 更新 tick
        10. 清除 _running 标志
        
        如果任何 hook 返回 RESULT_STOP，会立即停止执行后续步骤，
        并自动设置 is_finished=True。
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish 或正在运行中
        """
        from natal.hook_dsl import RESULT_STOP
        
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot step() after finish=True."
            )
        
        if self._running:
            raise RuntimeError(
                f"Population '{self.name}' is already running. "
                "Cannot call step()/run_tick()/run() recursively (e.g., from within a hook)."
            )
        
        try:
            self._running = True
            
            # first hook
            if self.trigger_event("first") == RESULT_STOP:
                self._finished = True
                return self
            
            # 繁殖阶段
            self._step_reproduction()
            
            # early hook
            if self.trigger_event("early") == RESULT_STOP:
                self._finished = True
                return self
            
            # 生存阶段
            self._step_survival()
            
            # late hook
            if self.trigger_event("late") == RESULT_STOP:
                self._finished = True
                return self

            # update age
            self._step_aging()
            
            # 更新 tick
            self._tick += 1
            
        finally:
            self._running = False
        
        return self
        
    @abstractmethod
    def _step_reproduction(self) -> None:
        """
        繁殖阶段的内部实现。
        
        子类必须实现此方法来定义具体的繁殖逻辑。
        注意：此方法不应更新 tick。
        """
        pass
    
    @abstractmethod
    def _step_survival(self) -> None:
        """
        生存阶段的内部实现。
        
        子类必须实现此方法来定义具体的生存/选择逻辑。
        注意：此方法不应更新 tick。
        """
        pass

    @abstractmethod
    def _step_aging(self) -> None:
        """
        老化阶段的内部实现。
        
        子类必须实现此方法来定义具体的年龄逻辑。
        注意：此方法不应更新 tick。
        """
        pass
    
    def run_tick(self) -> 'BasePopulation':
        """
        run_tick 是 step() 的完全别名。
        
        两个方法严格等价，都执行相同的逻辑。
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish 或正在运行中
        
        Example:
            >>> pop.run_tick()  # 与 pop.step() 完全等价
        """
        return self.step()
    
    @abstractmethod
    def get_total_count(self) -> int:
        """返回种群总个体数"""
        pass
    
    @abstractmethod
    def get_female_count(self) -> int:
        """返回雌性总个体数"""
        pass
    
    @abstractmethod
    def get_male_count(self) -> int:
        """返回雄性总个体数"""
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
        """性比（雌/雄），雄性为0时返回 np.inf"""
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
    
    def run(
        self, 
        n_steps: int, 
        record_every: int = 1,
        finish: bool = False
    ) -> 'BasePopulation':
        """
        运行多步演化。
        
        Args:
            n_steps: 要运行的步数
            record_every: 每隔多少步记录一次快照（0 表示不记录）
            finish: 是否在运行完成后标记为 finished
                如果为 True，运行完成后会触发 'finish' 事件，
                并将种群标记为已完成，之后无法再运行 run_tick()
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish，无法继续运行
        """
        if self._finished:
            raise RuntimeError(
                f"Population '{self.name}' has finished. "
                "Cannot run() again after finish=True."
            )
        
        # Create a snapshot at the beginning if tick is 0
        if self.tick == 0:
            self.create_snapshot()
        
        for i in range(n_steps):
            self.step()
            # 如果 step() 中的 hook 触发了终止条件，提前退出循环
            if self._finished:
                break
            if record_every > 0 and self.tick % record_every == 0:
                self.create_snapshot()
        
        # 只有在没有被 hook 终止的情况下，且用户请求 finish 时才调用
        if finish and not self._finished:
            self.finish_simulation()
        
        return self
    
    def create_snapshot(self) -> None:
        """
        创建当前种群状态的历史记录。
        
        将当前 tick 和 state 的副本保存到历史列表。
        """
        state_copy = (self.state.individual_count.copy(), 
                      self.state.sperm_storage.copy() if self.state.sperm_storage is not None else None)
        self._history.append((self.tick, state_copy))

    def reset(self) -> None:
        """Reset the population to its initial state.

        Behavior:
        - Reset `self._tick` to 0.
        - Clear the history list.
        - Clear the `finished` flag so the population may be run again.
        - If the instance provides an `_initial_population_snapshot` (tuple
          of arrays created by subclasses), restore it. Otherwise reallocate
          an empty `PopulationState` with the same array shapes.
        """
        # reset tick and flags
        self._tick = 0
        self._history = []
        self._finished = False
        # restore initial snapshot if subclass saved one
        if hasattr(self, '_initial_population_snapshot') and self._initial_population_snapshot is not None:
            ind_copy, sperm_copy, _ = self._initial_population_snapshot
            from natal.population_state import PopulationState
            n_genotypes = len(self._index_core.index_to_genotype)
            # infer ages/sexes from saved arrays
            n_ages = None
            n_sexes = None
            if ind_copy is not None:
                if ind_copy.ndim == 3:
                    n_sexes, n_ages, _ = ind_copy.shape
                else:
                    n_sexes, _ = ind_copy.shape
            self._state = PopulationState.create(
                n_genotypes=n_genotypes,
                n_ages=n_ages,
                n_sexes=n_sexes,
                n_tick=0,
                individual_count=ind_copy.copy() if ind_copy is not None else None,
                sperm_storage=sperm_copy.copy() if sperm_copy is not None else None,
            )
    
    def compute_allele_frequencies(self) -> Dict[str, float]:
        """
        计算种群中所有等位基因的频率。
        
        默认实现，子类可覆写以优化性能。
        
        Returns:
            Dict[allele_name, frequency]
        """
        # 初始化所有等位基因频率为 0
        allele_frequencies = {}
        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                for gene in locus.alleles:
                    allele_frequencies[gene.name] = 0.0
        
        # 具体实现依赖子类的数据结构
        # 这里提供一个空壳，子类应覆写
        return allele_frequencies
    
    # ========================================================================
    # Hooks 系统
    # ========================================================================
    
    def set_hook(
        self,
        event_name: str,
        func: Callable,
        hook_id: Optional[int] = None,
        hook_name: Optional[str] = None,
        compile: bool = True
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
                  - @hook 装饰的声明式函数: 返回 [Op.scale(...), ...] 
                  - @hook(selectors={...}) 装饰的选择器函数
            hook_id: Hook 的数值优先级（可选，自动分配）
                     较小的 ID 先执行
            hook_name: Hook 的可读名称（可选，用于调试）
            compile: 是否尝试编译 @hook 装饰的函数（默认 True）
        
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
            if hasattr(func, 'register'):
                func.register(self, event_override=event_name)
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
    
    def trigger_event(self, event_name: str) -> int:
        """
        触发事件，执行所有已注册的 hooks。

        执行顺序：
        1. CSR操作（Numba快速路径）
        2. njit_fn hooks（用户自定义Numba函数）
        3. py_wrapper hooks（Python包装函数）
        
        Args:
            event_name: 要触发的事件名称
        
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
        from natal.hook_dsl import RESULT_CONTINUE, RESULT_STOP
        
        # 优先使用 HookExecutor（如果已构建）
        if self._hook_executor is not None:
            from natal.hook_dsl import EVENT_ID_MAP
            event_id = EVENT_ID_MAP.get(event_name)
            if event_id is not None:
                result = self._hook_executor.execute_event(event_id, self, self.tick)
                return result
        
        # 降级到传统 _hooks 系统（兼容性）
        for _, _, hook in self._hooks.get(event_name, []):
            hook(self)
        
        return RESULT_CONTINUE
    
    
    def get_hooks(self, event_name: str) -> List[Tuple[int, Optional[str], Callable]]:
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
            删除成功返回 True，否则返回 False
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
        desc = compile_declarative_hook(ops, self, event, priority, name)
        self._register_compiled_hook(desc)
        return desc
    
    def _build_hook_program(self):
        """Build HookProgram from compiled hooks.
        
        This packs all compiled hooks into a Numba-compatible jitclass
        for efficient execution during simulation.
        
        Returns:
            HookProgram: Compiled hook program data, or None if no hooks
        """
        from natal.hook_dsl import HookProgram, EVENT_NAMES
        
        events = EVENT_NAMES
        n_events = len(events)
        
        # 1. Collect all hooks per event
        hook_offsets = [0]
        hook_list_by_event = []
        
        for event_name in events:
            hooks = self.get_compiled_hooks(event_name)
            hook_list_by_event.append(hooks)
            hook_offsets.append(hook_offsets[-1] + len(hooks))
        
        n_hooks = hook_offsets[-1]
        
        # 2. Pack all operation data
        all_op_types = []
        all_gidx_offsets = [0]
        all_gidx_data = []
        all_age_offsets = [0]
        all_age_data = []
        all_sex_masks = []
        all_params = []
        all_cond_offsets = [0]
        all_cond_types = []
        all_cond_params = []
        
        n_ops_list = []
        op_offsets = [0]
        
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
        
        # 3. Create HookProgram
        if n_hooks == 0:
            # No hooks, return empty program
            return None
        
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
        if program is None:
            program_available = False
        else:
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
        from natal.hook_dsl import HookProgram, NUM_EVENTS
        
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
        )

    def get_compiled_event_hooks(self) -> 'CompiledEventHooks':
        """Get compiled hooks for use with simulation_kernels.run_tick.
        
        This method collects all registered hooks and compiles them into
        Numba-friendly combined functions, one per event.
        
        Returns:
            CompiledEventHooks: Container with combined @njit hooks per event.
                                Access via .first, .early, .late, .finish
        
        Example:
            >>> hooks = pop.get_compiled_event_hooks()
            >>> state, result = sk.run_tick(
            ...     state, config, hooks.registry
            ... )
        """
        from natal.hook_dsl import CompiledEventHooks
        
        registry = self._build_hook_program()
        if registry is None:
            registry = self._create_empty_hook_program()
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
