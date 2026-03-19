"""Age-structured population models.

This module implements age-structured (overlapping generation) population
models and utilities for survival, reproduction, juvenile recruitment, and
fitness management.

Primary class:
    ``AgeStructuredPopulation``: An age-structured population model built on
    ``BasePopulation`` and ``PopulationState``.
"""

from typing import Dict, List, Optional, Union, Tuple, Set, Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from natal.base_population import BasePopulation, Species, Genotype, Sex, HaploidGenome
from natal.population_state import PopulationState
from natal.population_config import PopulationConfig, initialize_gamete_map, initialize_zygote_map
from natal.index_core import IndexCore
import natal.simulation_kernels as sk

if TYPE_CHECKING:
    from natal.population_builder import AgeStructuredPopulationBuilder

__all__ = ["AgeStructuredPopulation"]

# =============================================================================
# 新架构年龄结构种群模型（基于 BasePopulation）
# =============================================================================

class AgeStructuredPopulation(BasePopulation):
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
        hooks: Optional[Dict[str, List[Tuple[Callable, Optional[str], Optional[int]]]]] = None,
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
            hooks (Optional[Dict]): Event hook registrations to apply.

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
        super().__init__(species, name, hooks=hooks or {})
        
        # Store configuration
        self._config = population_config
        
        # Create IndexCore for genotype and gamete label mapping
        self._index_core = IndexCore(species, population_config)
        
        # Create PopulationState
        self._state = PopulationState.create(
            n_genotypes=population_config.n_genotypes,
            n_sexes=population_config.n_sexes,
            n_ages=population_config.n_ages,
        )
        
        # Initialize snapshot tracking
        self.snapshots = {}
        
        # Distribute initial population if provided
        if initial_individual_count is not None:
            self._distribute_initial_population(initial_individual_count)
        
        # Initialize sperm storage if provided
        if initial_sperm_storage is not None and population_config.use_sperm_storage:
            self._initialize_sperm_storage(initial_sperm_storage)
        
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
            gamete_labels=gamete_labels,
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
        # Parse and distribute to population state
        parsed_dist = self._parse_population_distribution(
            self.species, distribution, self._config.n_ages
        )
        
        for sex_idx, sex_data in parsed_dist.items():
            for genotype_idx, age_data in sex_data.items():
                for age, count in age_data.items():
                    self._state.individuals[sex_idx, age, genotype_idx] = count
    
    def _initialize_sperm_storage(
        self,
        sperm_storage: Dict[Union[Genotype, str], Dict[Union[Genotype, str], Union[Dict[int, float], List[float], float]]]
    ) -> None:
        """Initialize sperm storage from specification.
        
        Args:
            sperm_storage: Mapping of female genotypes to sperm collections.
        """
        # Would extract this from _setup_from_config logic
        # Placeholder for sperm storage initialization
        pass
    
    # ========================================================================
    # Template Method Pattern Implementation
    # (implements BasePopulation abstract methods)
    # ========================================================================
    
    def _create_registry(self) -> IndexCore:
        """Create a fresh IndexCore for this age-structured population.
        
        Returns:
            A new IndexCore instance.
        """
        return IndexCore()
    
    def _get_genotypes(self) -> List[Genotype]:
        """Return the list of genotypes for this population.
        
        Returns:
            The stored list of genotypes.
        """
        return self._genotypes_list
    
    def _get_haplogenotypes(self) -> Optional[List]:
        """Return haploid genotypes for registration.
        
        Returns:
            The stored list of haploid genotypes.
        """
        return self._haploid_genotypes_list

    def _parse_population_distribution(
        self,
        species: Species,
        dist: Dict[str, Dict[Union[Genotype, str], Union[List[int], Dict[int, int]]]],
        n_ages: int
    ) -> Dict[Sex, Dict[Genotype, Dict[int, int]]]:
        """Validate and parse the initial population distribution.

        Supported genotype keys:
            - ``Genotype`` instances
            - ``str`` values parsable by ``species.get_genotype_from_str``

        Returns:
            Dict[Sex, Dict[Genotype, Dict[int,int]]]: Parsed sparse mapping of
            counts by age for each genotype and sex.
        """
        parsed_dist = {}
        
        for sex_str, genotype_dist in dist.items():
            if sex_str not in ("male", "female"):
                raise ValueError(f"Sex must be 'male' or 'female', got '{sex_str}'")
            
            sex = Sex.MALE if sex_str == "male" else Sex.FEMALE
            parsed_dist[sex] = {}
            
            for genotype_key, age_data in genotype_dist.items():
                # 支持字符串和 Genotype 对象
                if isinstance(genotype_key, str):
                    genotype = species.get_genotype_from_str(genotype_key)
                elif isinstance(genotype_key, Genotype):
                    genotype = genotype_key
                else:
                    raise TypeError(f"Genotype key must be Genotype object or str, got {type(genotype_key)}")
                
                if genotype.species is not species:
                    raise ValueError("Genotype must belong to this species")
                
                # 转换为 Dict[int, int]（稀疏格式）
                if isinstance(age_data, list):
                    # List 格式：[count_age0, count_age1, ...]
                    age_dict = {age: count for age, count in enumerate(age_data) if count > 0}
                elif isinstance(age_data, dict):
                    # Dict 格式：{age: count, ...}
                    age_dict = {}
                    for age, count in age_data.items():
                        if not isinstance(age, int):
                            raise TypeError(f"Age must be int, got {type(age)}")
                        if age < 0 or age >= n_ages:
                            raise ValueError(f"Age {age} out of range [0, {n_ages})")
                        if count < 0:
                            raise ValueError(f"Count must be non-negative, got {count}")
                        if count > 0:
                            age_dict[age] = count
                else:
                    raise TypeError(f"Age data must be List or Dict, got {type(age_data)}")
                
                parsed_dist[sex][genotype] = age_dict
        
        return parsed_dist
    
    def _extract_genotypes_from_distribution(
        self,
        parsed_dist: Dict[Sex, Dict[Genotype, Dict[int, int]]]
    ) -> List[Genotype]:
        """Extract unique genotypes from a parsed distribution mapping.

        Returns:
            List[Genotype]: Sorted list of unique genotypes appearing in the
            provided distribution.
        """
        genotypes_set = set()
        for sex_dict in parsed_dist.values():
            for genotype in sex_dict.keys():
                genotypes_set.add(genotype)
        return sorted(list(genotypes_set), key=lambda gt: str(gt))

    def _distribute_initial_population(
        self,
        parsed_dist: Dict[Sex, Dict[Genotype, Dict[int, int]]]
    ) -> None:
        """Populate the internal ``PopulationState`` from a parsed distribution.

        Args:
            parsed_dist: Parsed mapping returned by ``_parse_population_distribution``.
        """
        for sex, genotype_dict in parsed_dist.items():
            sex_idx = int(sex.value)
            for genotype, age_dict in genotype_dict.items():
                genotype_idx = self._index_core.genotype_to_index[genotype]
                
                # 直接写入 PopulationState
                for age, count in age_dict.items():
                    if 0 <= age < self._n_ages:
                        self._state.individual_count[sex_idx, age, genotype_idx] += float(count)
    
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
        for female_key, male_dict in sperm_storage_dist.items():
            # 解析雌性基因型
            if isinstance(female_key, str):
                female_genotype = species.get_genotype_from_str(female_key)
            elif isinstance(female_key, Genotype):
                female_genotype = female_key
            else:
                raise TypeError(f"Female genotype key must be Genotype or str, got {type(female_key)}")
            
            female_idx = self._index_core.genotype_to_index[female_genotype]
            
            for male_key, age_data in male_dict.items():
                # 解析雄性基因型
                if isinstance(male_key, str):
                    male_genotype = species.get_genotype_from_str(male_key)
                elif isinstance(male_key, Genotype):
                    male_genotype = male_key
                else:
                    raise TypeError(f"Male genotype key must be Genotype or str, got {type(male_key)}")
                
                male_idx = self._index_core.genotype_to_index[male_genotype]
                
                # 解析 age_data：支持多种格式
                if isinstance(age_data, dict):
                    # Dict 格式：{age: count, ...}
                    for age, count in age_data.items():
                        if not isinstance(age, int):
                            raise TypeError(f"Age must be int, got {type(age)}")
                        if age < 0 or age >= self._n_ages:
                            raise ValueError(f"Age {age} out of range [0, {self._n_ages})")
                        if count < 0:
                            raise ValueError(f"Sperm count must be non-negative, got {count}")
                        if count > 0:
                            self._state.sperm_storage[age, female_idx, male_idx] = float(count)
                            
                elif isinstance(age_data, (list, tuple)):
                    # List 格式：[count_age0, count_age1, ...]
                    for age, count in enumerate(age_data):
                        if age >= self._n_ages:
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
                        for age in range(self._new_adult_age, self._n_ages):
                            self._state.sperm_storage[age, female_idx, male_idx] = float(age_data)
                else:
                    raise TypeError(f"Age data must be Dict, List, or numeric scalar, got {type(age_data)}")

    def _get_all_possible_haploid_genotypes_from_genotypes(self, genotypes: List[Genotype]) -> List[HaploidGenome]:
        """Return all unique haploid genomes found in a list of diploid genotypes."""
        haplotypes = set()
        for genotype in genotypes:
            haplotypes.add(genotype.maternal)
            haplotypes.add(genotype.paternal)
        return sorted(haplotypes, key=lambda h: str(h)) # TODO: 可能需要支持两性不同的配子
    
    def _compute_initial_age2_total(self) -> int:
        """Compute the total number of individuals at age 2 in the initial state."""
        if self._n_ages <= 2:
            return 0
        return self._state.individual_count[:, 2, :].sum()

    def _resolve_survival_rates(self, rates, n_ages: int, default: List[float]) -> np.ndarray:
        """Parse a flexible survival-rate specification into a NumPy array.

        Supported input formats (in order):
            - sequence (list/tuple/ndarray) of floats: truncated or padded as needed;
              a sentinel ``None`` at the end indicates fill-with-last-non-None.
            - dict mapping age (int) -> float: unspecified ages default to 1.0.
            - callable(age) -> float: invoked per age index.
            - scalar float: same value for all ages.

        Args:
            rates: User-provided specification (one of the supported formats).
            n_ages: Target length of the returned array.
            default: Default fallback list used when ``rates`` is None.

        Returns:
            np.ndarray: Array of length ``n_ages`` containing parsed survival rates.
        """
        # 默认
        if rates is None:
            return np.array(default[:n_ages], dtype=float)

        # 常数情况
        if isinstance(rates, (int, float)) and not isinstance(rates, bool):
            val = float(rates)
            if val < 0:
                raise ValueError("Survival rates must be non-negative")
            return np.full(n_ages, val, dtype=float)

        # 序列情况（list/tuple/ndarray）
        if isinstance(rates, (list, tuple, np.ndarray)):
            arr = np.array(rates, dtype=object)
            # 支持以 None 结尾表示用最后一个非 None 值填充
            if arr.size > 0 and arr[-1] is None:
                # 找到最后一个非 None
                non_none = None
                for v in arr[::-1]:
                    if v is not None:
                        non_none = float(v)
                        break
                if non_none is None:
                    # 全 None -> 使用默认
                    return np.array(default[:n_ages], dtype=float)
                vals = []
                for v in arr[:-1]:
                    if v is None:
                        raise TypeError("None only allowed as final sentinel in survival list")
                    vals.append(float(v))
                # pad with last non-none
                if len(vals) >= n_ages:
                    return np.array(vals[:n_ages], dtype=float)
                padded = np.empty(n_ages, dtype=float)
                padded[: len(vals)] = vals
                padded[len(vals) :] = non_none
                return padded

            # 普通序列：不足部分使用 0 填充
            arrf = np.array(arr, dtype=float)
            if arrf.size >= n_ages:
                out = arrf[:n_ages].astype(float)
            else:
                out = np.zeros(n_ages, dtype=float)
                out[: arrf.size] = arrf
            if (out < 0).any():
                raise ValueError("Survival rates must be non-negative")
            return out

        # 字典情况：缺省为 1.0
        if isinstance(rates, dict):
            out = np.ones(n_ages, dtype=float)
            for k, v in rates.items():
                if not isinstance(k, int):
                    raise TypeError("Age keys in survival dict must be int")
                if k < 0 or k >= n_ages:
                    raise ValueError(f"Age {k} out of range [0, {n_ages})")
                val = float(v)
                if val < 0:
                    raise ValueError("Survival rates must be non-negative")
                out[k] = val
            return out

        # 可调用情况：逐年龄调用 callable(age)
        if callable(rates):
            vals = []
            for age in range(n_ages):
                try:
                    v = rates(age)
                    vals.append(float(v))
                except Exception as e:
                    raise ValueError(f"Error calling survival rate function at age {age}: {e}")
            arrf = np.array(vals, dtype=float)
            if (arrf < 0).any():
                raise ValueError("Survival rates must be non-negative")
            return arrf

        raise TypeError("female_survival_rates / male_survival_rates must be None, sequence, dict, callable or numeric constant")

    def set_female_survival_rates(self, rates) -> None:
        """Set or update female per-age survival rates using the same formats as initialization."""
        self._female_survival_rates = self._resolve_survival_rates(rates, self._n_ages, self._female_survival_rates.tolist())

    def set_male_survival_rates(self, rates) -> None:
        """Set or update male per-age survival rates using the same formats as initialization."""
        self._male_survival_rates = self._resolve_survival_rates(rates, self._n_ages, self._male_survival_rates.tolist())
    
    @property
    def state(self) -> PopulationState:
        """Population state data container."""
        return self._state
    
    @property
    def n_ages(self) -> int:
        """Number of age classes in this population."""
        return self._n_ages
    
    @property
    def new_adult_age(self) -> int:
        """Minimum age at which individuals are considered adults."""
        return self._new_adult_age
    
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
            total += self._state.individual_count[Sex.FEMALE.value, self._new_adult_age:self._n_ages, :].sum()
        
        if sex in ('male', 'M', 'both'):
            total += self._state.individual_count[Sex.MALE.value, self._new_adult_age:self._n_ages, :].sum()
        
        return int(total)
    

    def _get_fecundity(self, genotype: Genotype, sex: Sex) -> float:
        """Internal helper: return fecundity for a genotype and sex."""
        genotype_idx = self._index_core.genotype_to_index[genotype]
        sex_idx = int(sex.value)
        return self._config.fecundity_fitness[sex_idx, genotype_idx]
    
    def _get_sexual_preference(self, female_genotype: Genotype, male_genotype: Genotype) -> float:
        """Internal helper: return sexual preference value for a genotype pair."""
        f_idx = self._index_core.genotype_to_index[female_genotype]
        m_idx = self._index_core.genotype_to_index[male_genotype]
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
                如果没有历史记录，返回形状为 (0, history_shape) 的空数组。
        """
        if len(self._history) == 0:
            return np.zeros((0, self._history_shape[0]), dtype=np.float64)
        
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
                n_ages=self._n_ages,
                n_genotypes=len(self._index_core.index_to_genotype)
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
                    n_ages=self._n_ages,
                    n_genotypes=len(self._index_core.index_to_genotype)
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
    
    def _step_reproduction(self) -> None:
        """Reproduction step: compute newborns and add them to age 0.

        Adult individuals produce offspring during this step.
        Delegates to simulation_kernels.run_reproduction for core logic.
        """
        
        config = self._get_kernel_config()
        
        new_ind, new_sperm = sk.run_reproduction(
            self._state.individual_count,
            self._state.sperm_storage,
            config,
        )
        
        self._state.individual_count[:] = new_ind
        self._state.sperm_storage[:] = new_sperm
    
    def _step_survival(self) -> None:
        """Survival step: apply survival rates and aging-related updates.
        
        Delegates to simulation_kernels.run_survival for core logic.
        Note: run_survival now handles juvenile_growth_mode internally,
        so _apply_juvenile_growth is no longer called here.
        """
        
        config = self._get_kernel_config()
        
        new_ind, new_sperm = sk.run_survival(
            self._state.individual_count,
            self._state.sperm_storage,
            config,
        )
        
        self._state.individual_count[:] = new_ind
        self._state.sperm_storage[:] = new_sperm
    
    def _step_aging(self) -> None:
        """Aging step: advance ages by one year.
        
        Delegates to simulation_kernels.run_aging for core logic.
        """
        
        config = self._get_kernel_config()
        
        new_ind, new_sperm = sk.run_aging(
            self._state.individual_count,
            self._state.sperm_storage,
            config,
        )
        
        self._state.individual_count[:] = new_ind
        self._state.sperm_storage[:] = new_sperm
    
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
        clear_history_on_start: bool = True
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
                如果为 True（默认），会清空所有旧的快照，只保留本次 run() 的结果
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
        
        # 直接调用 sk.run 执行多步演化
        final_state_tuple, history_new, was_stopped = sk.run(
            state=self._state,
            config=config,
            registry=hooks.registry,
            n_ticks=n_steps,
            first_hook=hooks.first,
            reproduction_hook=hooks.reproduction,
            early_hook=hooks.early,
            survival_hook=hooks.survival,
            late_hook=hooks.late,
            record_history=(record_every > 0)
        )
        
        # 处理最终状态（tuple 格式：ind_count, sperm, tick）
        self._state = PopulationState(
            n_tick=np.int32(final_state_tuple[2]),
            individual_count=final_state_tuple[0],
            sperm_storage=final_state_tuple[1],
        )
        self._tick = int(final_state_tuple[2])
        
        # 处理历史记录
        # history_new 是 2D NDArray (n_snapshots, history_size)，其中每行是 [tick_val, ind_count_flat..., sperm_storage_flat...]
        if history_new is not None and history_new.shape[0] > 0:
            if clear_history_on_start:
                self.clear_history()
            for row_idx in range(history_new.shape[0]):
                row = history_new[row_idx, :]
                tick = int(row[0])
                flattened = row.copy()
                self._history.append((tick, flattened))
        
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
        return self.run(n_steps=1, record_every=1)
    
    def compute_allele_frequencies(self) -> Dict[str, float]:
        """Compute allele frequencies across the entire population.

        Returns:
            Dict[str, float]: Mapping from allele name to frequency in the
            population (based on allele counts aggregated over sexes and ages).
        """
        # 初始化等位基因频率
        allele_freqs = {}
        allele_counts = {}
        
        # 收集所有等位基因
        for chrom in self.species.chromosomes:
            for locus in chrom.loci:
                for gene in locus.alleles:
                    allele_freqs[gene.name] = 0.0
                    allele_counts[gene.name] = 0
        
        total_alleles = 0
        
        # 统计每个基因型的个体数（使用索引访问）
        for genotype_idx, genotype in enumerate(self._index_core.index_to_genotype):
            # 所有性别和年龄的总计数
            total_count = self._state.individual_count[:, :, genotype_idx].sum()
            
            if total_count == 0:
                continue
            
            # 统计等位基因
            for chrom in self.species.chromosomes:
                for locus in chrom.loci:
                    mat_gene, pat_gene = genotype.get_alleles_at_locus(locus)
                    
                    for gene in [mat_gene, pat_gene]:
                        if gene is not None:
                            allele_counts[gene.name] = allele_counts.get(gene.name, 0) + total_count
                            total_alleles += total_count
        
        # 计算频率
        if total_alleles > 0:
            for allele_name in allele_freqs.keys():
                allele_freqs[allele_name] = allele_counts.get(allele_name, 0) / total_alleles
        
        return allele_freqs
    
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
        genotype_idx = self._index_core.genotype_to_index[genotype]
        female_count = self._state.individual_count[Sex.FEMALE.value, :, genotype_idx].sum()
        male_count = self._state.individual_count[Sex.MALE.value, :, genotype_idx].sum()
        return (female_count, male_count)
    
    @property
    def genotypes_present(self) -> Set[Genotype]:
        """Return the set of genotypes currently present in the population."""
        present = set()
        for genotype_idx, genotype in enumerate(self._index_core.index_to_genotype):
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