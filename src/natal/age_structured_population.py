"""Age-structured population models.

This module implements age-structured (overlapping generation) population
models and utilities for survival, reproduction, juvenile recruitment, and
fitness management.

Primary class:
    ``AgeStructuredPopulation``: An age-structured population model built on
    ``BasePopulation`` and ``PopulationState``.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable, Set, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from natal.base_population import BasePopulation, Species, Genotype, Sex, HaploidGenome
from natal.population_state import PopulationState
from natal.population_config import PopulationConfig, initialize_gamete_map, initialize_zygote_map
from natal.index_registry import IndexRegistry
import natal.simulation_kernels as sk

if TYPE_CHECKING:
    from natal.population_builder import AgeStructuredPopulationBuilder

__all__ = ["AgeStructuredPopulation"]

# =============================================================================
# Age-structured population model (based on BasePopulation)
# =============================================================================

class AgeStructuredPopulation(BasePopulation[PopulationState]):
    """Age-structured population model (overlapping generations).

    An age-structured population built on ``BasePopulation`` and
    ``PopulationState``. Supports age-dependent survival and fecundity,
    juvenile recruitment modes, optional sperm-storage mechanics, and a
    hook/modifier system for user extensions.

    The constructor accepts explicit configuration for ages, survival
    schedules, mating and recruitment behavior, and user-provided
    modifiers/hooks. See ``__init__`` for the full parameter list.
    """
    
    def __init__(
        self,
        species: Species,
        population_config: PopulationConfig,
        name: Optional[str] = None,
        initial_individual_count: Optional[Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]]] = None,
        initial_sperm_storage: Optional[Dict[Union[Genotype, str], Dict[Union[Genotype, str], Union[Dict[int, float], List[float], float]]]] = None,
        hooks: Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]] = {},
    ):
        """Initialize an age-structured population instance using a PopulationConfig.

        This constructor directly accepts a PopulationConfig created by PopulationConfigBuilder.
        For building via the high-level API, use the builder() class method instead.

        Args:
            species (Species): Species object describing genetic architecture.
            population_config (PopulationConfig): Fully initialized PopulationConfig instance.
            name (Optional[str]): Human-readable population name. If None, uses "AgeStructuredPop".
            initial_individual_count (Optional[Dict]): Initial population distribution (required unless pre-initialized).
                Format: {sex: {genotype: counts_by_age}}
            initial_sperm_storage (Optional[Dict]): Initial sperm storage state (if supported).
            hooks (Dict): Event hook registrations to apply.

        Example:
            >>> pop_config = PopulationConfigBuilder.build(species, ...)
            >>> pop = AgeStructuredPopulation(
            ...     species, 
            ...     pop_config,
            ...     name="MyPop",
            ...     initial_individual_count={...}
            ... )

        Raises:
            ValueError: If configuration is invalid (e.g., empty initial_individual_count).
        """
        # Set population name
        if name is None:
            name = "AgeStructuredPop"
        
        # Initialize parent with hooks
        super().__init__(species, name, hooks=hooks)
        
        # Store configuration
        config_hook_slot = int(getattr(population_config, "hook_slot", 0))
        if config_hook_slot <= 0:
            config_hook_slot = self.hook_slot
        self._config = population_config._replace(hook_slot=np.int32(config_hook_slot))

        self._genotypes_list = species.get_all_genotypes()
        self._haploid_genotypes_list = species.get_all_haploid_genotypes()
        
        # Create IndexRegistry for genotype and gamete label mapping
        self._initialize_registry()
        
        # Create PopulationState
        self._state = PopulationState.create(
            n_genotypes=population_config.n_genotypes,
            n_sexes=population_config.n_sexes,
            n_ages=population_config.n_ages,
        )

        # Preferred path: initialize from builder-injected config arrays.
        cfg_init_ind = population_config.get_scaled_initial_individual_count()
        if cfg_init_ind.shape == self._state.individual_count.shape:
            self._state.individual_count[:] = cfg_init_ind
        cfg_init_sperm = population_config.get_scaled_initial_sperm_storage()
        if cfg_init_sperm.shape == self._state.sperm_storage.shape:
            self._state.sperm_storage[:] = cfg_init_sperm
        
        # Initialize snapshot tracking
        self.snapshots = {}
        
        # Backward-compatible override path from constructor args.
        if initial_individual_count is not None:
            self._state.individual_count.fill(0.0)
            self._distribute_initial_population(initial_individual_count)
        
        # Backward-compatible override path from constructor args.
        if initial_sperm_storage is not None:
            # TODO: add population_config.use_sperm_storage
            self._distribute_initial_sperm_storage(species, initial_sperm_storage)

        self._initial_population_snapshot = (
            self._state.individual_count.copy(),
            self._state.sperm_storage.copy() if self._state.sperm_storage is not None else None,
            None,
        )
        
        # Initialize registry using Template Method Pattern
        self._initialize_registry()
    
    @classmethod
    def setup(
        cls,
        species: Species,
        name: str = "AgeStructuredPop",
        stochastic: bool = True,
        use_dirichlet_sampling: bool = False,
        gamete_labels: Optional[List[str]] = None,
        use_fixed_egg_count: bool = False,
    ) -> 'AgeStructuredPopulationBuilder':
        """Create and preconfigure an age-structured population builder.

        This is a convenience forwarding entry point. Parameter semantics and
        defaults are the same as ``AgeStructuredPopulationBuilder.setup``.

        Args:
            species: Species definition used to initialize the builder.
            name: Population name passed through to ``builder.setup``.
            stochastic: Passed through to ``builder.setup``.
            use_dirichlet_sampling: Passed through to ``builder.setup``.
            gamete_labels: Passed through to ``builder.setup``.
            use_fixed_egg_count: Passed through to ``builder.setup``.

        Returns:
            A configured ``AgeStructuredPopulationBuilder`` for fluent chaining.

        Example:
            ``AgeStructuredPopulation.setup(species).age_structure(...).initial_state(...).build()``
        """
        from natal.population_builder import AgeStructuredPopulationBuilder
        builder = AgeStructuredPopulationBuilder(species)
        builder.setup(
            name=name,
            stochastic=stochastic,
            use_dirichlet_sampling=use_dirichlet_sampling,
            use_fixed_egg_count=use_fixed_egg_count
        )
        return builder
    
    def _distribute_initial_population(
        self,
        distribution: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]]
    ) -> None:
        """Distribute initial population from a specification dictionary.
        
        Args:
            distribution: Format {sex: {genotype: age_counts}}
                where age_counts can be a list or dict of age -> count.
        """
        self._state.individual_count.fill(0.0)
        for sex_key, genotype_dist in distribution.items():
            sex_key_norm = sex_key.lower().strip()
            if sex_key_norm == "female":
                sex_idx = int(Sex.FEMALE.value)
            elif sex_key_norm == "male":
                sex_idx = int(Sex.MALE.value)
            else:
                raise ValueError(f"Sex must be 'female' or 'male', got '{sex_key}'")

            for genotype_key, age_data in genotype_dist.items():
                genotype = self._resolve_genotype_key(genotype_key)
                genotype_idx = self._registry.genotype_to_index[genotype]

                if isinstance(age_data, list):
                    for age, count in enumerate(age_data):
                        if age < self._config.n_ages and count > 0:
                            self._state.individual_count[sex_idx, age, genotype_idx] = float(count)
                elif isinstance(age_data, dict):
                    for age, count in age_data.items():
                        if age < self._config.n_ages and count > 0:
                            self._state.individual_count[sex_idx, age, genotype_idx] = float(count)
                else:
                    raise TypeError(f"age_data must be a list or dict, got {type(age_data)}")
    
    def _distribute_initial_sperm_storage(
        self,
        species: Species,
        sperm_storage_dist: Dict[Union[Genotype, str], Dict[Union[Genotype, str], Union[Dict[int, float], List[float], float]]]
    ) -> None:
        """Populate the internal sperm storage from user-provided initial distribution.

        Supported formats for age_data (innermost value):
            - Dict[int, float]: Sparse mapping {age: count, ...}
            - List[float]: Dense list [count_age0, count_age1, ...]
            - float/int: Scalar value applied to all adult ages (>= new_adult_age)

        Args:
            species: Species object for genotype parsing.
            sperm_storage_dist: Mapping of {female_genotype: {male_genotype: age_data}}.
                Genotype keys can be Genotype objects or strings.
        """
        self._state.sperm_storage.fill(0.0)
        for female_key, male_dict in sperm_storage_dist.items():
            # 解析雌性基因型
            if isinstance(female_key, str):
                female_genotype = species.get_genotype_from_str(female_key)
            elif isinstance(female_key, Genotype):
                female_genotype = female_key
            else:
                raise TypeError(f"Female genotype key must be Genotype or str, got {type(female_key)}")
            
            female_idx = self._registry.genotype_to_index[female_genotype]
            
            for male_key, age_data in male_dict.items():
                # 解析雄性基因型
                if isinstance(male_key, str):
                    male_genotype = species.get_genotype_from_str(male_key)
                elif isinstance(male_key, Genotype):
                    male_genotype = male_key
                else:
                    raise TypeError(f"Male genotype key must be Genotype or str, got {type(male_key)}")
                
                male_idx = self._registry.genotype_to_index[male_genotype]
                
                # 解析 age_data：支持多种格式
                if isinstance(age_data, dict):
                    # Dict 格式：{age: count, ...}
                    for age, count in age_data.items():
                        if not isinstance(age, int):
                            raise TypeError(f"Age must be int, got {type(age)}")
                        if age < 0 or age >= self.n_ages:
                            raise ValueError(f"Age {age} out of range [0, {self.n_ages})")
                        if count < 0:
                            raise ValueError(f"Sperm count must be non-negative, got {count}")
                        if count > 0:
                            self._state.sperm_storage[age, female_idx, male_idx] = float(count)
                            
                elif isinstance(age_data, (list, tuple)):
                    # List 格式：[count_age0, count_age1, ...]
                    for age, count in enumerate(age_data):
                        if age >= self.n_ages:
                            break
                        if count < 0:
                            raise ValueError(f"Sperm count must be non-negative, got {count}")
                        if count > 0:
                            self._state.sperm_storage[age, female_idx, male_idx] = float(count)
                            
                elif isinstance(age_data, (int, float)) and not isinstance(age_data, bool):
                    # 标量格式：应用到所有成年年龄
                    if age_data < 0:
                        raise ValueError(f"Sperm count must be non-negative, got {age_data}")
                    if age_data > 0:
                        for age in range(self.new_adult_age, self.n_ages):
                            self._state.sperm_storage[age, female_idx, male_idx] = float(age_data)
                else:
                    raise TypeError(f"Age data must be Dict, List, or numeric scalar, got {type(age_data)}")
    
    @property
    def state(self) -> PopulationState:
        """Population state data container."""
        return self._state
    
    def reset(self) -> None:
        """Reset the population to its initial state."""
        self._tick = 0
        self._history = []
        self._finished = False
        if hasattr(self, '_initial_population_snapshot') and self._initial_population_snapshot is not None:
            ind_copy, sperm_copy, _ = self._initial_population_snapshot
            
            # Recreate state with initial data
            self._state = PopulationState.create(
                n_genotypes=self._config.n_genotypes,
                n_sexes=self._config.n_sexes,
                n_ages=self._config.n_ages,
                n_tick=0,
                individual_count=ind_copy.copy() if ind_copy is not None else None,
                sperm_storage=sperm_copy.copy() if sperm_copy is not None else None,
            )

    @property
    def n_ages(self) -> int:
        """Number of age classes in this population."""
        return self._config.n_ages
    
    @property
    def new_adult_age(self) -> int:
        """Minimum age at which individuals are considered adults."""
        return self._config.new_adult_age
    
    def get_total_count(self) -> int:
        """Return the total number of individuals in the population."""
        return self._state.individual_count.sum()
    
    def get_female_count(self) -> int:
        """Return the total number of female individuals."""
        return self._state.individual_count[Sex.FEMALE.value, :, :].sum()
    
    def get_male_count(self) -> int:
        """Return the total number of male individuals."""
        return self._state.individual_count[Sex.MALE.value, :, :].sum()
    
    def get_adult_count(self, sex: str = 'both') -> int:
        """Return the number of adult individuals for the given sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'`` (aliases accepted).

        Returns:
            int: Total number of adults for the requested sex(es).
        """
        if sex not in ('female', 'male', 'both', 'F', 'M'):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")
        
        total = 0
        
        if sex in ('female', 'F', 'both'):
            total += self._state.individual_count[Sex.FEMALE.value, self.new_adult_age:self.n_ages, :].sum()
        
        if sex in ('male', 'M', 'both'):
            total += self._state.individual_count[Sex.MALE.value, self.new_adult_age:self.n_ages, :].sum()
        
        return int(total)
    

    def _get_fecundity(self, genotype: Genotype, sex: Sex) -> float:
        """Internal helper: return fecundity for a genotype and sex."""
        genotype_idx = self._registry.genotype_to_index[genotype]
        sex_idx = int(sex.value)
        return self._config.fecundity_fitness[sex_idx, genotype_idx]
    
    def _get_sexual_preference(self, female_genotype: Genotype, male_genotype: Genotype) -> float:
        """Internal helper: return sexual preference value for a genotype pair."""
        f_idx = self._registry.genotype_to_index[female_genotype]
        m_idx = self._registry.genotype_to_index[male_genotype]
        return self._config.sexual_selection_fitness[f_idx, m_idx]
    
    # ========================================================================
    # 状态导出/导入（与 simulation_kernels 接口）
    # ========================================================================
    
    def export_config(self) -> 'PopulationConfig':
        """导出种群配置到 Config jitclass。
        
        Returns:
            PopulationStatic: A copy of the population configuration.
        """
        return self._config
    
    def import_config(self, config: 'PopulationConfig') -> None:
        """导入配置到种群。
        
        Args:
            config: Config jitclass instance
        """
        # 配置通常是只读的（通过 run_tick 使用），这里仅为完整性保留
        # 实际上很少需要导入配置，除非在高级用途下
        self._config = config
    
    def create_history_snapshot(self) -> None:
        """记录当前状态到历史记录。
        
        将当前 tick 和 state 的展平副本保存到 _history。
        """
        flattened = self._state.flatten_all()
        self._history.append((self._tick, flattened.copy()))
    
    def get_history(self) -> np.ndarray:
        """获取历史记录为 2D numpy 数组。
        
        Returns:
            np.ndarray: 形状 (n_snapshots, 1 + n_sexes*n_ages*n_genotypes + n_ages*n_genotypes^2)
                的 float64 数组，每行为一个快照的展平状态。
        
        Raises:
            ValueError: 如果没有历史记录。
        """
        if len(self._history) == 0:
            raise ValueError("No history recorded")
        
        # 堆叠所有快照的展平数据
        flat_array = np.array([rec[1] for rec in self._history], dtype=np.float64)
        return flat_array
    
    def clear_history(self) -> None:
        """清空历史记录。"""
        self._history.clear()
    
    def export_state(self) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]]]:
        """导出种群状态为展平数组。
        
        Returns:
            Tuple containing:
            - state_flat: flattened 1D array [n_tick, individual_count.ravel(), sperm_storage.ravel()]
            - history: optional numpy array of shape (n_snapshots, flatten_size) or None
        """
        state_flat = self._state.flatten_all()
        history = self.get_history() if self._history else None
        return state_flat, history
    
    def import_state(self, state: Union['PopulationState', NDArray[np.float64], Dict[str, np.ndarray]], 
                     history: Optional[np.ndarray] = None) -> None:
        """导入状态、可选的历史记录。
        
        Args:
            state: 可以是以下格式之一：
                - NDArray: 展平状态数组 [n_tick, ind_count.ravel(), sperm_storage.ravel()]
                - PopulationState: PopulationState 实例
                - Dict: {'individual_count': ..., 'sperm_storage': ...} 字典
            history: 可选的历史记录 2D 数组 (n_snapshots, flatten_size)
        """
        from natal.population_state import PopulationState, parse_flattened_state
        
        if isinstance(state, np.ndarray):
            # 从展平数组重建状态
            n_sexes = self._state.individual_count.shape[0]
            n_ages = self._state.individual_count.shape[1]
            n_genotypes = self._state.individual_count.shape[2]
            state_obj = parse_flattened_state(state, n_sexes, n_ages, n_genotypes)
            self._state.individual_count[:] = state_obj.individual_count
            self._state.sperm_storage[:] = state_obj.sperm_storage
            self._state = PopulationState(
                n_tick=state_obj.n_tick,
                individual_count=self._state.individual_count,
                sperm_storage=self._state.sperm_storage,
            )
        elif isinstance(state, dict):
            # 从字典重建状态
            self._state.individual_count[:] = state['individual_count']
            self._state.sperm_storage[:] = state['sperm_storage']
        elif isinstance(state, PopulationState):
            # 直接使用 PopulationState
            self._state.individual_count[:] = state.individual_count
            self._state.sperm_storage[:] = state.sperm_storage
            self._state = PopulationState(
                n_tick=state.n_tick,
                individual_count=self._state.individual_count,
                sperm_storage=self._state.sperm_storage,
            )
        else:
            # 兼容旧的元组格式
            self._state.individual_count[:] = state[0]
            self._state.sperm_storage[:] = state[1]
        
        # 导入历史记录（如果提供）
        if history is not None and history.shape[0] > 0:
            self.clear_history()
            for row_idx in range(history.shape[0]):
                flat = history[row_idx, :]
                tick = int(flat[0])
                self._history.append((tick, flat.copy()))
    
    # ========================================================================
    # 历史记录恢复工具
    # ========================================================================
    
    def get_history_as_objects(self, indices: Optional[List[int]] = None) -> List[Tuple[int, PopulationState]]:
        """将选定的展平快照转换回 PopulationState 对象。
        
        Args:
            indices: 要转换的快照索引列表。如果为 None，转换所有快照。
        
        Returns:
            (tick, PopulationState) 元组的列表
        """
        if indices is None:
            indices = list(range(len(self._history)))
        
        from natal.population_state import parse_flattened_state
        result = []
        for idx in indices:
            if idx < 0 or idx >= len(self._history):
                raise IndexError(f"History index {idx} out of range [0, {len(self._history)})")
            
            tick, flattened = self._history[idx]
            # 使用 parse_flattened_state 重构 PopulationState
            state = parse_flattened_state(
                flattened,
                n_sexes=2,
                n_ages=self._config.n_ages,
                n_genotypes=len(self._registry.index_to_genotype)
            )
            result.append((tick, state))
        return result
    
    def restore_checkpoint(self, tick: int) -> None:
        """将种群恢复到特定的历史 tick。
        
        Args:
            tick: 目标 tick 号
        
        Raises:
            ValueError: 如果找不到指定 tick 的历史记录
        """
        from natal.population_state import parse_flattened_state
        
        for t, flattened in self._history:
            if t == tick:
                # 使用 parse_flattened_state 重构状态
                state = parse_flattened_state(
                    flattened,
                    n_sexes=2,
                    n_ages=self._config.n_ages,
                    n_genotypes=len(self._registry.index_to_genotype)
                )
                # 直接复制状态数据
                self._state.individual_count[:] = state.individual_count
                self._state.sperm_storage[:] = state.sperm_storage
                self._tick = tick
                return
        
        raise ValueError(f"No history record found for tick {tick}")
    
    # ========================================================================
    # Hooks 系统
    # ========================================================================

    # 允许的 Hook 事件
    #
    #     Before simulation:  [initialization]
    #                                |
    #                                v
    #     For tick in T:    |-------------------------------------------------------------------------|
    #                       |     [first] -> [reproduction] --> [early] --> [survival] --> [late]     |
    #                       |        |<-------------------------------------------------------|       |
    #                       |-------------------------------------------------------------------------| 
    #                                |
    #                                v
    #     After simulation:      [finish]                       
    #
    
    # ========================================================================
    # 演化逻辑
    # ========================================================================
    
    def _get_kernel_config(self) -> tuple:
        """Build configuration tuple for simulation_kernels.
        
        Returns:
            tuple: Configuration in the format expected by simulation_kernels.
        """
        return sk.export_config(self)
    
    def run(
        self,
        n_steps: int,
        record_every: int = 1,
        finish: bool = False,
        clear_history_on_start: bool = False
    ) -> 'AgeStructuredPopulation':
        """
        运行多步演化（使用 simulation_kernels 加速）。
        
        如果种群定义了 hooks（特别是提前终止条件），hooks 会在演化过程中执行。
        若某个 hook 触发了终止条件，演化会提前结束，并自动设置 is_finished=True。
        
        Args:
            n_steps: 要运行的步数
            record_every: 每隔多少步记录一次快照（0 表示不记录）
            finish: 是否在运行完成后标记为 finished
                如果为 True，运行完成后会触发 'finish' 事件，
                并将种群标记为已完成，之后无法再运行 run_tick()
            clear_history_on_start: 是否在开始时清空现有历史记录
                如果为 True，会清空所有旧的快照，只保留本次 run() 的结果
                如果为 False，本次 run() 的快照会累积到现有历史中
        
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
        
        # 获取核心配置
        config = sk.export_config(self)
        
        # 获取编译后的事件 hooks
        hooks = self.get_compiled_event_hooks()
        
        # run_fn 和 registry 总是由 get_compiled_event_hooks() 初始化的
        # 但类型检查器可能看不到这个保证，所以我们显式地处理
        assert hooks.run_fn is not None, "hooks.run_fn should always be initialized"
        assert hooks.registry is not None, "hooks.registry should always be initialized"
        
        run_fn = hooks.run_fn
        registry = hooks.registry

        # 直接调用固定签名 runner 执行多步演化
        final_state_tuple, history_new, was_stopped = run_fn(
            state=self._state,
            config=config,
            registry=registry,
            n_ticks=n_steps,
            record_interval=record_every,
        )
        
        # 处理最终状态（tuple 格式：ind_count, sperm, tick）
        self._state = PopulationState(
            n_tick=int(final_state_tuple[2]),
            individual_count=final_state_tuple[0],
            sperm_storage=final_state_tuple[1],
        )
        self._tick = int(final_state_tuple[2])
        
        # 处理历史记录
        # history_new 是 2D NDArray (n_snapshots, history_size)，其中每行是 [tick_val, ind_count_flat..., sperm_storage_flat...]
        self._process_kernel_history(history_new, clear_history_on_start)
        
        # 如果因 hooks 提前终止，设置 _finished 标志
        if was_stopped:
            self._finished = True
            self.trigger_event("finish")
        elif finish:
            # 否则，如果 finish 参数为 True，则主动触发完成
            self.finish_simulation()
        
        return self
    
    def run_tick(self) -> 'AgeStructuredPopulation':
        """
        执行单个 tick 的演化（便捷方法）。
        
        等价于 run(n_steps=1, record_every=1)，但更简洁。
        
        Returns:
            self（支持链式调用）
        
        Raises:
            RuntimeError: 如果种群已 finish，无法继续运行
        
        Example:
            >>> pop.run_tick()  # 执行一个演化步骤
            >>> pop.tick  # 查看当前 tick 数
            1
        """
        return self.run(n_steps=1, record_every=self.record_every, clear_history_on_start=False)
    
    def get_age_distribution(self, sex: str = 'both') -> np.ndarray:
        """Return the age distribution for the requested sex.

        Args:
            sex: One of ``'female'``, ``'male'``, or ``'both'``.

        Returns:
            np.ndarray: Age distribution array with shape ``(n_ages,)``.
        """
        if sex not in ('female', 'male', 'both', 'F', 'M'):
            raise ValueError(f"sex must be 'female', 'male', or 'both', got '{sex}'")
        
        # 直接从 PopulationState 访问
        if sex in ('female', 'F'):
            return self._state.individual_count[Sex.FEMALE.value, :, :].sum(axis=1)
        elif sex in ('male', 'M'):
            return self._state.individual_count[Sex.MALE.value, :, :].sum(axis=1)
        else:
            return self._state.individual_count.sum(axis=(0, 2))
    
    def get_genotype_count(self, genotype: Genotype) -> Tuple[int, int]:
        """Return total counts for a genotype as (female_count, male_count).

        Args:
            genotype: Target genotype instance.

        Returns:
            Tuple[int,int]: ``(female_count, male_count)`` across all ages.
        """
        genotype_idx = self._registry.genotype_to_index[genotype]
        female_count = self._state.individual_count[Sex.FEMALE.value, :, genotype_idx].sum()
        male_count = self._state.individual_count[Sex.MALE.value, :, genotype_idx].sum()
        return (female_count, male_count)
    
    @property
    def genotypes_present(self) -> Set[Genotype]:
        """Return the set of genotypes currently present in the population."""
        present = set()
        for genotype_idx, genotype in enumerate(self._registry.index_to_genotype):
            total_count = self._state.individual_count[:, :, genotype_idx].sum()
            if total_count > 0:
                present.add(genotype)
        return present
    
    def __repr__(self) -> str:
        """Return a compact string representation of the population."""
        return (f"AgeStructuredPopulation(name='{self.name}', n_ages={self.n_ages}, "
                f"total_count={self.get_total_count()}, "
                f"adult_females={self.get_adult_count('female')}, "
                f"adult_males={self.get_adult_count('male')})")
