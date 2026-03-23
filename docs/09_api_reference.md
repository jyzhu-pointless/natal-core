# API 完整参考

本章为所有主要类和方法提供完整的参考文档。

## 目录

- [Species](#species)
- [AgeStructuredPopulation](#agestructuredpopulation)
- [PopulationState](#populationstate)
- [PopulationConfig](#populationconfig)
- [IndexRegistry](#indexcore)
- [Simulation Kernels](#simulation-kernels)
- [Hook 系统](#hook-dsl)

---

## Species

物种和遗传架构的根节点。

### 构造函数

```python
Species(name: str)
```

**参数**：
- `name` (str): 物种名称

### 类方法

#### from_dict

```python
@classmethod
Species.from_dict(
    name: str,
    structure: Dict[str, Dict[str, List[str]]]
) -> Species
```

快速创建物种和遗传架构。

**参数**：
- `name`: 物种名称
- `structure`: 嵌套字典
  - 第一层键：染色体名称
  - 第二层键：位点名称
  - 第二层值：等位基因名称列表

**返回**：Species 对象

**例**：
```python
sp = Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "A": ["A1", "A2"],
            "B": ["B1", "B2", "B3"],
        }
    }
)
```

### 实例方法

#### add

```python
def add(
    name: str,
    sex_type: Optional[str] = None
) -> Chromosome
```

添加染色体。

**参数**：
- `name`: 染色体名称
- `sex_type`: 性染色体类型，可选值：
  - `None` 或 `"autosome"`: 常染色体（默认）
  - `"X"` 或 `"Y"`: XY 系统的 X/Y 染色体
  - `"Z"` 或 `"W"`: ZW 系统的 Z/W 染色体

**返回**：Chromosome 对象

#### get_gene

```python
def get_gene(name: str) -> Gene
```

按名称获取等位基因。

**参数**：
- `name`: 等位基因名称

**返回**：Gene 对象

**异常**：KeyError 如果等位基因不存在

#### get_genotype_from_str

```python
def get_genotype_from_str(s: str) -> Genotype
```

从字符串解析基因型。

**参数**：
- `s`: 基因型字符串，格式 "A1|A2"

**返回**：Genotype 对象

**异常**：KeyError 如果基因型不合法

**例**：
```python
gt = sp.get_genotype_from_str("WT|Drive")
```

#### get_all_genotypes

```python
def get_all_genotypes() -> List[Genotype]
```

获取所有可能的二倍体基因型。

**返回**：Genotype 对象列表

#### get_all_haploid_genotypes

```python
def get_all_haploid_genotypes() -> List[HaploidGenotype]
```

获取所有可能的单倍体基因组。

**返回**：HaploidGenotype 对象列表

### 属性

#### name

```python
@property
name() -> str
```

物种名称

#### chromosomes

```python
@property
chromosomes() -> List[Chromosome]
```

所有染色体列表

---

## AgeStructuredPopulation

年龄结构种群模型。

### 构造函数

```python
AgeStructuredPopulation(
    species: Species,
    name: str = "Population",
    n_ages: int = 8,
    is_stochastic: bool = False,
    initial_individual_count: Optional[Dict] = None,
    female_survival_rates: Union[List, float, Callable, Dict] = None,
    male_survival_rates: Union[List, float, Callable, Dict] = None,
    female_age_based_mating_rates: Optional[List] = None,
    male_age_based_mating_rates: Optional[List] = None,
    expected_eggs_per_female: float = 1.0,
    use_sperm_storage: bool = False,
    sperm_displacement_rate: float = 0.5,
    gamete_labels: Optional[List[str]] = None,
    juvenile_growth_mode: int = 2,  # LOGISTIC
    old_juvenile_carrying_capacity: Optional[float] = None,
    expected_num_adult_females: Optional[float] = None,
    gamete_modifiers: Optional[List[Tuple]] = None,
    zygote_modifiers: Optional[List[Tuple]] = None,
    hooks: Optional[Dict] = None,
    seed: Optional[int] = None,
)
```

**关键参数**：
- `species`: Species 对象
- `n_ages`: 年龄类别数
- `is_stochastic`: 是否为随机模型
- `initial_individual_count`: 初始分布字典
- `female_survival_rates`/`male_survival_rates`: 生存率（多种格式）
- `expected_eggs_per_female`: 每只雌性产卵数
- `use_sperm_storage`: 启用精子存储
- `gamete_labels`: 配子标签列表
- `gamete_modifiers`: 配子修饰器列表
- `zygote_modifiers`: 合子修饰器列表

### 实例方法

#### run

```python
def run(
    n_steps: int,
    record_every: int = 1,
    finish: bool = False
) -> None
```

运行模拟。

**参数**：
- `n_steps`: 运行的时间步数
- `record_every`: 每隔多少步记录一次历史（0 为不记录）
- `finish`: 是否在最后调用 finish hook

**例**：
```python
pop.run(n_steps=100, record_every=10)
```

#### step

```python
def step() -> None
```

执行单个时间步。

#### reset

```python
def reset() -> None
```

重置种群到初始状态。

#### set_viability

```python
def set_viability(
    genotype: Union[Genotype, str],
    value: float,
    sex: Optional[str] = None,
    age: int = -1
) -> None
```

设置存活力适应度。

**参数**：
- `genotype`: Genotype 对象或字符串
- `value`: 适应度值 [0, 1]
- `sex`: 性别（"female", "male", 或 None 表示两性）
- `age`: 年龄（-1 表示所有年龄）

#### set_fecundity

```python
def set_fecundity(
    genotype: Union[Genotype, str],
    value: float,
    sex: Optional[str] = None
) -> None
```

设置生育力适应度（仅对雌性）。

#### set_gamete_modifier

```python
def set_gamete_modifier(
    func: Callable,
    hook_name: str = "",
    priority: int = 0
) -> None
```

注册配子修饰器。

**参数**：
- `func`: 修饰器函数
- `hook_name`: 修饰器名称
- `priority`: 优先级（低值先执行）

#### set_zygote_modifier

```python
def set_zygote_modifier(
    func: Callable,
    hook_name: str = "",
    priority: int = 0
) -> None
```

注册合子修饰器。

#### set_hook

```python
def set_hook(
    event: str,
    func: Callable,
    hook_name: str = ""
) -> None
```

注册 Hook。

**参数**：
- `event`: 事件名称 ("first", "reproduction", "early", "survival", "late", "finish")
- `func`: Hook 函数
- `hook_name`: Hook 名称

### 查询方法

#### get_total_count

```python
def get_total_count() -> float
```

获取总个体数。

#### get_female_count

```python
def get_female_count() -> float
```

获取雌性总数。

#### get_male_count

```python
def get_male_count() -> float
```

获取雄性总数。

#### get_adult_count

```python
def get_adult_count(sex: Optional[str] = None) -> float
```

获取成年个体数。

### 属性

#### state

```python
@property
state() -> PopulationState
```

当前种群状态。

#### species

```python
@property
species() -> Species
```

物种对象。

#### registry

```python
@property
registry() -> IndexRegistry
```

索引注册表。

#### tick

```python
@property
tick() -> int
```

当前时间步。

#### history

```python
@property
history() -> List[Tuple]
```

历史记录列表。

---

## PopulationState

运行时种群状态。

### 属性

#### n_tick

```python
n_tick: int
```

当前时间步。

#### individual_count

```python
individual_count: np.ndarray[np.float64]
```

个体计数数组，shape: (n_sexes, n_ages, n_genotypes)

#### sperm_storage

```python
sperm_storage: np.ndarray[np.float64]
```

精子存储数组，shape: (n_ages, n_female_genotypes, n_male_genotypes)

---

## PopulationConfig

静态配置。

### 关键属性

#### genotype_to_gametes_map

```python
genotype_to_gametes_map: np.ndarray[np.float64]
```

基因型→配子映射矩阵，shape: (n_sexes, n_genotypes, n_haploid_genotypes, n_glabs)

#### gametes_to_zygote_map

```python
gametes_to_zygote_map: np.ndarray[np.float64]
```

配子→合子映射矩阵，shape: (n_haploid*n_glabs, n_haploid*n_glabs, n_genotypes)

#### viability_fitness

```python
viability_fitness: np.ndarray[np.float64]
```

存活力适应度，shape: (n_sexes, n_ages, n_genotypes)

#### fecundity_fitness

```python
fecundity_fitness: np.ndarray[np.float64]
```

生育力适应度，shape: (n_genotypes,)

---

## IndexRegistry

对象↔索引注册表。

### 注册方法

#### register_genotype

```python
def register_genotype(genotype_id: Any) -> int
```

注册基因型并返回索引。

#### register_haplogenotype

```python
def register_haplogenotype(haplo_id: Any) -> int
```

注册单倍基因型并返回索引。

#### register_gamete_label

```python
def register_gamete_label(gamete_label: Any) -> int
```

注册配子标签并返回索引。

### 查询方法

#### genotype_index

```python
def genotype_index(genotype_id: Any) -> int
```

获取基因型的索引。

#### haplo_index

```python
def haplo_index(haplo_id: Any) -> int
```

获取单倍基因型的索引。

#### gamete_label_index

```python
def gamete_label_index(gamete_label: Any) -> int
```

获取配子标签的索引。

### 统计方法

#### num_genotypes

```python
def num_genotypes() -> int
```

基因型总数。

#### num_haplogenotypes

```python
def num_haplogenotypes() -> int
```

单倍基因型总数。

#### num_gamete_labels

```python
def num_gamete_labels() -> int
```

配子标签总数。

### 属性

#### index_to_genotype

```python
index_to_genotype: List[Any]
```

索引→基因型映射列表。

#### index_to_haplo

```python
index_to_haplo: List[Any]
```

索引→单倍基因型映射列表。

#### index_to_glab

```python
index_to_glab: List[Any]
```

索引→配子标签映射列表。

---

## Simulation Kernels

`natal.simulation_kernels` 模块的核心函数。

### 导出/导入

#### export_state

```python
def export_state(pop: AgeStructuredPopulation) -> Tuple[
    PopulationState,
    PopulationConfig,
    Optional[np.ndarray]
]
```

导出种群状态、配置和历史。

#### import_state

```python
def import_state(
    pop: AgeStructuredPopulation,
    state: PopulationState,
    history: Optional[np.ndarray] = None
) -> None
```

导入状态到种群。

### 执行函数

#### run_tick

```python
@njit
def run_tick(
    state: PopulationState,
    config: PopulationConfig,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
) -> Tuple[Tuple[NDArray, NDArray, int], int]
```

执行单个时间步。

**返回**：(state, result_code)
- result_code: 0 (继续) 或 1 (停止)

#### run

```python
@njit
def run(
    state: PopulationState,
    config: PopulationConfig,
    n_ticks: int,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
    record_history: bool = False
) -> Tuple[Tuple[NDArray, NDArray, int], Optional[NDArray], bool]
```

执行多个时间步。

**返回**：(state, history, was_stopped)

#### run_reproduction

```python
@njit
def run_reproduction(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

执行繁殖阶段。

#### run_survival

```python
@njit
def run_survival(
    ind_count: NDArray[np.float64],
    config: PopulationConfig,
) -> NDArray[np.float64]
```

执行生存阶段。

#### run_aging

```python
@njit
def run_aging(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]
```

执行衰老阶段。

#### batch_ticks

```python
def batch_ticks(
    initial_state: PopulationState,
    config: PopulationConfig,
    n_particles: int,
    n_steps_per_particle: int,
    rng: np.random.Generator,
    record_history: bool = False
) -> List[PopulationState]
```

批量运行 Monte Carlo 模拟。

**参数**：
- `n_particles`: 并行模拟数量
- `n_steps_per_particle`: 每个模拟的步数
- `rng`: 随机数生成器

**返回**：最终状态列表

---

## Hook 系统

`natal.hook_dsl` 模块。

### hook 装饰器

```python
@hook(
    event: str,
    selectors: Optional[Dict[str, Any]] = None
)
```

创建声明式 Hook。

**参数**：
- `event`: 事件名称 ("first", "early", "late", "finish")
- `selectors`: 预解析选择器字典

**返回**：Hook 对象，需要调用 `.register(pop)` 注册

**例**：
```python
@hook(event='late')
def my_hook():
    return [
        Op.add(genotypes='A1|A2', ages=[0, 1], delta=100),
    ]

my_hook.register(pop)
```

### Op 操作

所有操作在 `natal.hook_dsl.Op` 命名空间下。

#### Op.add

```python
Op.add(
    genotypes: Union[str, List[str]] = '*',
    ages: Union[int, List[int], range, str] = '*',
    sex: str = 'both',
    delta: float = 0,
    when: Optional[str] = None
) -> HookOp
```

增加个体。

#### Op.subtract

```python
Op.subtract(
    genotypes: Union[str, List[str]] = '*',
    ages: Union[int, List[int], range, str] = '*',
    sex: str = 'both',
    delta: float = 0,
    when: Optional[str] = None
) -> HookOp
```

减少个体。

#### Op.scale

```python
Op.scale(
    genotypes: Union[str, List[str]] = '*',
    ages: Union[int, List[int], range, str] = '*',
    sex: str = 'both',
    factor: float = 1.0,
    when: Optional[str] = None
) -> HookOp
```

乘以因子。

#### Op.set

```python
Op.set(
    genotypes: Union[str, List[str]] = '*',
    ages: Union[int, List[int], range, str] = '*',
    sex: str = 'both',
    value: float = 0,
    when: Optional[str] = None
) -> HookOp
```

设置为特定值。

#### Op.stop_if_extinction

```python
Op.stop_if_extinction() -> HookOp
```

如果种群灭绝则停止。

#### Op.stop_if_zero

```python
Op.stop_if_zero(
    genotypes: Union[str, List[str]] = '*',
    ages: Union[int, List[int], range, str] = '*',
) -> HookOp
```

如果选定基因型为零则停止。

---

## 工具函数

### compress_hg_glab / decompress_hg_glab

```python
def compress_hg_glab(
    haploid_idx: int,
    label_idx: int,
    n_glabs: int
) -> int

def decompress_hg_glab(
    compressed: int,
    n_glabs: int
) -> Tuple[int, int]
```

在配子索引和 (haploid, label) 之间转换。

---

## 使用示例索引

| 功能 | 位置 |
|------|------|
| 创建物种 | [快速开始 - 第一步](01_quickstart.md#1️⃣-第一步定义遗传架构2-分钟) |
| 初始化种群 | [快速开始 - 第二步](01_quickstart.md#2️⃣-第二步初始化种群3-分钟) |
| 设置适应度 | [快速开始 - 第三步](01_quickstart.md#3️⃣-第三步设置适应度可选2-分钟) |
| 编写 Modifier | [Modifier 机制](06_modifiers.md) |
| 编写 Hook | [Hook 系统](07_hooks.md) |
| 导出和批量运行 | [Simulation Kernels](03_simulation_kernels.md#批量-monte-carlo-模拟) |
| 性能优化 | [Numba 优化指南](08_numba_optimization.md) |

---

## 常见问题

### Q: 如何查找某个方法的参数？

本参考中的每个函数都列有完整的签名和参数说明。使用 Ctrl+F 搜索。

### Q: 哪些函数被 Numba 编译？

查看函数上是否有 `@njit` 或 `@jitclass` 装饰器。所有 `simulation_kernels` 中的函数都被编译。

### Q: 可以修改 PopulationConfig 吗？

初始化后，大多数参数可以修改（如 fitness 值），但映射矩阵修改需要通过 Modifier。详见 [Modifier 机制](06_modifiers.md)。

---

**返回到特定章节**: [完整文档索引](README.md)
