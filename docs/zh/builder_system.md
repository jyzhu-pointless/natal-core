# Builder 系统详解（参数全解 + 模拟流程对照）

本章目标是把 Builder 讲透：

1. 每个公开参数是什么、默认值是什么。
2. 参数会影响模拟流程中的哪个阶段。
3. 如何按“先跑通、再精调”的方式配置。

如果你只读一章 Builder，请读这一章。

## 1. Builder 在模拟中的位置

Builder 的职责不是直接跑模拟，而是把高层输入编译成 `PopulationConfig` 与 `PopulationState`，然后交给模拟执行流程。

流程可以简化为：

```text
Builder 链式配置
  -> build()
  -> PopulationConfig / PopulationState
  -> run_tick / run
  -> reproduction -> survival -> aging（以及 hooks）
```

相关章节：

- [PopulationState & PopulationConfig：编译与配置](population_state_config.md)
- [模拟内核深度解析](simulation_kernels.md)

## 2. 两类 Builder

- `AgeStructuredPopulationBuilder`
  - 适合多年龄层模型。
  - 典型特征：`n_ages` 可配置，支持年龄向量输入、可选精子存储。
- `DiscreteGenerationPopulationBuilder`
  - 适合离散世代模型。
  - 典型特征：默认采用两年龄阶段（`n_ages=2`，`new_adult_age=1`）。

## 3. AgeStructuredPopulationBuilder 参数手册

### 3.1 `setup(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `name` | `str` | 种群的标识名称。 | `"AgeStructuredPop"` | 全流程 | 用于日志和标识，不影响动力学；建议实验批次显式命名。 |
| `stochastic` | `bool` | 采用随机或确定性采样。 | `True` | reproduction/survival 等采样阶段 | `True`=随机，`False`=确定性；调参期建议先用 `False`。 |
| `use_continuous_sampling` | `bool` | 采样策略的选择。 | `False` | 概率采样细节 | 控制采样方式；大多数场景保持默认即可。 |
| `use_fixed_egg_count` | `bool` | 产卵数是否固定。 | `False` | reproduction | `True` 固定产卵数，`False` 更接近随机产卵过程。 |

### 3.2 `age_structure(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `n_ages` | `int` | 分年龄阶段的总数。 | `8` | 全流程（数组维度） | 约束初始状态、存活率向量等数组长度；必须与所有年龄相关参数长度一致。 |
| `new_adult_age` | `int` | 个体进入成虫阶段的年龄索引。 | `2` | reproduction/survival | 建议与目标物种的生命史阶段一致；低于此年龄为幼体。 |
| `generation_time` | `Optional[int]` | 代时标记。 | `None` | 编译参数 | 用于建模解释；与 `age_structure` 重名参数，后设值覆盖先设值。 |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | 平衡分布辅助参数。 | `None` | competition/初始化标度 | 与 `age_structure` 重名参数，后设值覆盖先设值；仅明确做稳态标定时使用。 |

### 3.3 `initial_state(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `individual_count` | `Mapping[str, Mapping[Union[Genotype, str], Union[int, float, List, Tuple, NDArray, Dict[int, float]]]]` | 初始个体数量分布，格式 `{sex: {genotype: age_data}}`。 | 必填 | 初始状态 | 不设置会在 `build()` 报错；支持标量、序列、映射等多种格式。 |
| `sperm_storage` | `Optional[Mapping[Union[Genotype, str], Mapping[Union[Genotype, str], Union[int, float, List, Tuple, NDArray, Dict[int, float]]]]]` | 初始精子库存（若启用储精）。 | `None` | reproduction | 仅在 `use_sperm_storage=True` 场景必要；格式为三层映射。 |

`age_data` 支持：标量、列表/元组/数组、`{age: value}` 映射。必须非负。

代码对齐格式（来自 `PopulationConfigBuilder.resolve_age_structured_initial_individual_count` 与相关解析函数）：

```python
# 1) 标量：会分配到 [new_adult_age, n_ages) 的所有年龄
{"female": {"WT|WT": 100.0}}

# 2) 序列：按年龄索引写入，超长截断，不足部分默认为 0
{"female": {"WT|WT": [0, 100, 80, 60]}}

# 3) 映射：显式给出某些年龄
{"female": {"WT|WT": {2: 100, 3: 80}}}
```

`sperm_storage` 的格式是三层映射：

```python
{
  "WT|WT": {                 # female genotype
    "Drive|WT": [0, 0, 20],  # male genotype -> age_data
  }
}
```

校验规则（源码行为）：

- 年龄索引必须在 `[0, n_ages)`。
- 所有计数必须 `>= 0`。
- 基因型字符串必须能被当前 `Species` 解析。

### 3.4 `survival(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `female_age_based_survival_rates` | `Optional[Union[int, float, List[float], NDArray[np.float64], Dict[int, float], Callable]]` | 雌性按年龄的生存率。 | `None` | survival | 支持标量/序列/映射/函数等多种格式；`None` 使用默认曲线；建议限定在 `[0, 1]`。 |
| `male_age_based_survival_rates` | `Optional[Union[int, float, List[float], NDArray[np.float64], Dict[int, float], Callable]]` | 雄性按年龄的生存率。 | `None` | survival | 同 `female_age_based_survival_rates`。 |
| `generation_time` | `Optional[int]` | 代时标记。 | `None` | 编译参数 | 与 `age_structure` 同名参数；后设值覆盖先设值。 |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | 平衡分布辅助参数。 | `None` | 标度辅助 | 与 `age_structure` 同名参数；后设值覆盖先设值。 |

实用建议：

- 生存率建议限定在 `[0, 1]`。
- 先给平滑曲线跑通，再做年龄尖峰。

代码对齐格式（来自 `_resolve_survival_param`）：

```python
# A) None -> 使用默认曲线
.survival(female_age_based_survival_rates=None)

# B) 标量 -> 全年龄同值
.survival(female_age_based_survival_rates=0.85)

# C) 序列 -> 按年龄写入；不足补 0，超长截断
.survival(female_age_based_survival_rates=[1.0, 1.0, 0.9, 0.7])

# D) 稀疏映射 -> 未给出的年龄默认 1.0
.survival(female_age_based_survival_rates={0: 1.0, 1: 0.95, 2: 0.8})

# E) 函数 -> 必须接收一个 age 参数并返回数值
.survival(female_age_based_survival_rates=lambda age: 1.0 if age < 2 else 0.8)

# F) 序列末尾 None 哨兵 -> 末尾用最后一个非 None 值填充
.survival(female_age_based_survival_rates=[1.0, 0.9, None])
```

校验规则（源码行为）：

- 存活率要求非负（源码未强制上限 1.0，但建模上建议不超过 1.0）。
- 字典 key 必须是合法年龄。
- callable 若签名不合法会抛 `TypeError`。

### 3.5 `reproduction(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `female_age_based_mating_rates` | `Optional[Union[List[float], NDArray[np.float64]]]` | 雌性按年龄的交配率。 | `None` | reproduction | 长度需等于 `n_ages`；未设定时使用默认值。 |
| `male_age_based_mating_rates` | `Optional[Union[List[float], NDArray[np.float64]]]` | 雄性按年龄的交配率。 | `None` | reproduction | 长度需等于 `n_ages`；未设定时使用默认值。 |
| `female_age_based_relative_fertility` | `Optional[Union[List[float], NDArray[np.float64]]]` | 雌性按年龄的相对生育力权重。 | `None` | reproduction | 长度需等于 `n_ages`；用于调节不同年龄雌性的产卵財献。 |
| `eggs_per_female` | `float` | 每雌个体的基础产卵数。 | `50.0` | reproduction | 用作种群产卵数的基准；先从中性值开始调参。 |
| `use_fixed_egg_count` | `bool` | 产卵数是否固定。 | `False` | reproduction | `True` 固定产卵数，`False` 随机产卵。 |
| `sex_ratio` | `float` | 后代中雌性的比例。 | `0.5` | reproduction | 范围应在 `[0, 1]`；0.5 表示男女等比。 |
| `use_sperm_storage` | `bool` | 是否启用储精机制。 | `True` | reproduction | `True` 启用储精，`False` 禁用；仅当代交配。 |
| `sperm_displacement_rate` | `float` | 新精子替换旧精子的速率。 | `0.05` | reproduction | 范围通常在 `(0, 1]`；较高值表示新精子替换速度快。 |

格式与长度要求（源码行为）：

- `female_age_based_mating_rates` / `male_age_based_mating_rates` / `female_age_based_relative_fertility`
  - 传入后会被转成 `np.array`。
  - 最终在配置编译时要求长度必须等于 `n_ages`，否则报 `ValueError`。

### 3.6 `competition(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `competition_strength` | `float` | 幼体竞争的强度因子。 | `5.0` | 幼体密度调节 | 影响密度制约效应的强弱；越大则调控效应越强。 |
| `juvenile_growth_mode` | `Union[int, str]` | 幼体生长的密度调节模式。 | `"logistic"` | 幼体密度调节 | 支持 `"logistic"`, `"beverton_holt"` 等模式；通常用 `logistic`。 |
| `low_density_growth_rate` | `float` | 低密度下的内秱增长率。 | `6.0` | 幼体密度调节 | 表示无竞争时的增长倍数；过大易导致種群振荡。 |
| `old_juvenile_carrying_capacity` | `Optional[int]` | 幼体的承载容量。 | `None` | 幼体密度调节 | 若指定则优先使用；否则从 `expected_num_adult_females` 推导。 |
| `expected_num_adult_females` | `Optional[int]` | 期望的成体雌性数量。 | `None` | 容量推导 | 用于推导承载容量；若已指定 `old_juvenile_carrying_capacity` 则无须提供。 |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | 平衡分布辅助参数。 | `None` | 标度辅助 | 与 `age_structure` 同名参数；后设值覆盖先设值。 |

### 3.7 `presets(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `*preset_list` | `Any` 变长参数 | 预设配置对象列表。 | 空 | build 后处理 | 预设会先应用，建立基线配置；其后 `fitness`/`modifiers`/`hooks` 可覆盖预设值。 |

### 3.8 `fitness(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `viability` | `Optional[Dict[GenotypeSelector, Union[float, Dict[Union[str, Sex, int], Union[float, Dict[int, float]]]]]]` | 生存适应度系数。 | `None` | survival | 支持多层级：按基因型、按性别、按年龄、按性别+年龄；默认 `None` 不改动。 |
| `fecundity` | `Optional[Dict[GenotypeSelector, Union[float, Dict[str, float]]]]` | 生殖力适应度系数。 | `None` | reproduction | 按基因型和/或性别指定；默认 `None` 不改动。 |
| `sexual_selection` | `Optional[Dict[GenotypeSelector, Union[float, Dict[GenotypeSelector, float]]]]` | 配对偏好权重。 | `None` | reproduction | 支持平铺映射 `{male: value}` 或嵌套映射 `{female: {male: value}}`。 |
| `mode` | `str` | 適应度值的写入模式。 | `"replace"` | fitness 写入策略 | `"replace"` 覆盖原值，`"multiply"` 按倍数缩放。 |

代码对齐格式（来自 `fitness()` 注释与 `_iter_sexual_selection_entries`）：

```python
# viability: genotype -> float
.fitness(viability={"WT|WT": 1.0, "Drive|Drive": 0.6})

# viability: genotype -> {sex: float}
.fitness(viability={"Drive|WT": {"female": 0.9, "male": 0.8}})

# viability: genotype -> {age: float}，会同时作用到 female/male
.fitness(viability={"Drive|WT": {0: 0.95, 1: 0.85}})

# viability: genotype -> {sex: {age: float}}，可做性别+年龄细分
.fitness(viability={"Drive|WT": {"female": {1: 0.9}, "male": {2: 0.8}}})

# fecundity: genotype -> float 或 {sex: float}
.fitness(fecundity={"Drive|Drive": 0.7})

# sexual_selection 扁平格式: {male_selector: value}，会自动当作 female='*'
.fitness(sexual_selection={"Drive|WT": 1.2, "WT|WT": 1.0})

# sexual_selection 嵌套格式: {female_selector: {male_selector: value}}
.fitness(sexual_selection={"WT|WT": {"Drive|WT": 0.8, "WT|WT": 1.0}})
```

`GenotypeSelector` 支持：

- 单个选择器：`"Drive|WT"` 或 `Genotype` 对象
- 选择器并集：`("Drive|WT", "Drive|Drive")`

`viability` 的年龄键约束（与代码一致）：

- 年龄键必须是整数，且范围在 `[0, n_ages)`。
- 若用 `{sex: ...}` 形式，可继续给 `{age: float}`。
- 若直接用 `{age: float}`，表示两性同值。
- 未显式给年龄时，默认写入 `new_adult_age - 1`（离散世代默认年龄为 `0`）。

### 3.9 `modifiers(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `gamete_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | 配子阶段的转换函数列表。 | `None` | gamete->zygote 映射编译 | 格式为 `(优先级, 名称, 函数)` 元组；按优先级排序后应用。 |
| `zygote_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | 合子阶段的转换函数列表。 | `None` | zygote 映射编译 | 格式为 `(优先级, 名称, 函数)` 元组；按优先级排序后应用。 |

代码对齐格式：

```python
.modifiers(
  gamete_modifiers=[(10, "drive_gamete", my_gamete_modifier_fn)],
  zygote_modifiers=[(20, "drive_zygote", my_zygote_modifier_fn)],
)
```

说明：

- 元组结构固定为 `(priority_or_hook_id, optional_name, callable)`。
- 在配置编译阶段会按第一个字段排序后应用。

### 3.10 `hooks(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `*hook_items` | `Callable` 或 `HookMap` | 钩子函数或钩子注册映射。 | 空 | 事件点（first/early/late/finish 等） | 支持两种形式：直接传函数（需 @hook 元数据）或传事件映射字典；未声明事件会报错。 |

两种合法输入：

```python
# 1) 直接传函数（函数需带 @hook(event='...') 元数据）
.hooks(my_hook_fn)

# 2) 传映射
.hooks({
  "late": [(my_hook_fn, "my_hook", 10)],
  "finish": [(finish_hook, "finish", 0)],
})
```

常见报错：

- 传入普通函数但没有 `event` 元数据 -> `ValueError`。
- 传入非 `callable` 且非 `dict` -> `TypeError`。

### 3.11 `build()`

`build()` 无参数，但有强约束：

- 必须先设置 `initial_state(...)`。
- 执行顺序是：
  - 先构建 `PopulationConfig`
  - 再创建种群对象
  - 再应用 presets
  - 再应用 fitness/modifiers/hooks

这也是为什么建议把 `build()` 放在链式调用最后。

## 4. DiscreteGenerationPopulationBuilder 参数手册

离散世代 Builder 的关键差异：

- 默认采用 `n_ages=2`、`new_adult_age=1`。
- 你不需要 `age_structure(...)`。

### 4.1 `setup(...)`

与年龄结构模型一致：

- `name`
- `stochastic`
- `use_continuous_sampling`
- `use_fixed_egg_count`

### 4.2 `initial_state(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `individual_count` | `dict` | 初始个体数量分布，格式 `{sex: {genotype: age_data}}`。 | 必填 | 初始状态 | 离散世代模型仅支持 age 0 和 age 1；各输入格式同年龄模型。 |

离散模型 `age_data` 代码对齐格式（来自 `_resolve_discrete_age_distribution`）：

```python
# 标量 -> (age0=0, age1=value)
{"female": {"WT|WT": 1000}}

# 长度 1 序列 -> (0, value)
{"female": {"WT|WT": [1000]}}

# 长度 2 序列 -> (age0, age1)
{"female": {"WT|WT": [200, 800]}}

# 映射 -> 只允许 key 为 0/1
{"female": {"WT|WT": {0: 200, 1: 800}}}
```

校验规则：

- 列表长度必须 `<= 2`。
- 字典只允许年龄键 `0` 和 `1`。
- 所有计数必须非负。

### 4.3 `reproduction(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `eggs_per_female` | `float` | 每雌个体每代产卵数。 | `50.0` | reproduction | 用作产卵的基准值；调参时从中性值开始。 |
| `sex_ratio` | `float` | 后代中雌性的比例。 | `0.5` | reproduction | 范围应在 `[0, 1]`；0.5 表示男女等比。 |
| `female_adult_mating_rate` | `float` | 成体雌性的交配率。 | `1.0` | reproduction | 表示雌性参与交配的比例；范围 `[0, 1]`。 |
| `male_adult_mating_rate` | `float` | 成体雄性的交配率。 | `1.0` | reproduction | 表示雄性参与交配的比例；范围 `[0, 1]`。 |

### 4.4 `survival(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `female_age0_survival` | `float` | 雌性幼体（age 0）的存活率。 | `1.0` | survival | 范围 `[0, 1]`；1.0 表示完全存活。 |
| `male_age0_survival` | `float` | 雄性幼体（age 0）的存活率。 | `1.0` | survival | 范围 `[0, 1]`；1.0 表示完全存活。 |
| `adult_survival` | `float` | 成体代际间存活率。 | `0.0` | survival/aging 边界 | 范围 `[0, 1]`；设 0 可近似严格非重叠世代；较高值允许成体跨代存活。 |

建模约束建议：

- 这三个概率在实践中都建议位于 `[0, 1]`。
- `adult_survival=0.0` 常用于严格离散世代。

### 4.5 `competition(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `juvenile_growth_mode` | `Union[int, str]` | 幼体生长的密度调节模式。 | `"logistic"` | 幼体密度调节 | 常用 `"logistic"`；也可用 `"beverton_holt"` 等其他模式。 |
| `low_density_growth_rate` | `float` | 低密度下的内秱增长倍数。 | `1.0` | 幼体密度调节 | 表示无竞争条件下的增长倍数；过大易导致振荡。 |
| `carrying_capacity` | `Optional[int]` | 幼体的承载容量。 | `None` | 密度上限 | 未设定时自动推导；优先使用显式指定的值。 |

### 4.6 `presets(...)` / `fitness(...)` / `modifiers(...)` / `hooks(...)` / `build()`

语义与年龄结构模型一致，区别仅在离散世代内核与固定年龄结构。

## 5. 参数如何作用到模拟过程

按一个 tick 来看：

1. reproduction（繁殖）
- 核心参数：交配率、相对生育力、`eggs_per_female`、`sex_ratio`、`fecundity`、`sexual_selection`、储精参数。

2. survival（生存）
- 核心参数：生存率向量/标量、`viability`。

3. aging（衰老）
- 核心参数：`n_ages`、`new_adult_age`（年龄结构模型）或 `adult_survival`（离散世代）。

4. 密度调节（幼体竞争）
- 核心参数：`juvenile_growth_mode`、`low_density_growth_rate`、`carrying_capacity` 相关参数。

5. hook 事件点
- `hooks(...)` 注册逻辑在固定事件点触发，用于阶段前后干预。

## 6. 推荐配置顺序（可直接照抄）

### 6.1 年龄结构模型

1. `setup(...)`
2. `age_structure(...)`
3. `initial_state(...)`
4. `survival(...)`
5. `reproduction(...)`
6. `competition(...)`
7. `presets(...)` -> `fitness(...)` -> `modifiers(...)` -> `hooks(...)`
8. `build()`

### 6.2 离散世代模型

1. `setup(...)`
2. `initial_state(...)`
3. `reproduction(...)`
4. `survival(...)`
5. `competition(...)`
6. `presets(...)` -> `fitness(...)` -> `modifiers(...)` -> `hooks(...)`
7. `build()`

## 7. 常见错误与排查

1. 忘记 `initial_state(...)`
  - 现象：`build()` 直接报错。
2. 年龄向量长度与 `n_ages` 不一致
  - 现象：初始化或编译阶段报错。
3. `sex_ratio` 或概率参数越界
  - 现象：结果异常或运行时错误。
4. 同名参数多次设置导致覆盖
  - 现象：行为与预期不一致。比如 `generation_time`、`equilibrium_distribution` 在多个方法都可设置，后调用覆盖先调用。

## 8. 本章小结

Builder 不是“语法糖”，而是把模型参数组织成可审查、可追溯、可调参的配置流程。

你可以把它当成三层：

1. 参数层：每个参数都映射到明确阶段。
2. 编译层：`build()` 固化为配置与状态。
3. 执行层：模拟流程按阶段运行并消费这些参数。

**下一章**：[IndexRegistry 索引机制](index_registry.md)
