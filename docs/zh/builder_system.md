# Builder 系统详解

Builder 是 NATAL Core 的核心配置系统，用于构建和管理种群模拟的配置参数。它提供链式 API 来分类设置各种模拟参数，并将高层配置编译成底层可执行的数值配置。

## Builder 功能概览

Builder 本身不直接执行模拟，其主要作用是将高层输入编译成 `PopulationConfig` 和初始的 `PopulationState`（`DiscretePopulationState`），然后交由模拟引擎运行。

整个流程可简化为：

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

## 两类 Builder

- **`AgeStructuredPopulationBuilder`**：适用于多年龄层模型。主要特征包括可配置 `n_ages`、支持按年龄向量输入、可选精子存储机制。
- **`DiscreteGenerationPopulationBuilder`**：适用于离散世代模型。典型特征是默认使用两个年龄阶段（`n_ages=2`，`new_adult_age=1`）。

## AgeStructuredPopulationBuilder 参数类型

### `setup(...)` – 基本设置

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `name` | `str` | 种群标识名称 | `"AgeStructuredPop"` | 全流程 | 仅用于日志和标识，不影响动力学；实验时建议显式命名以便区分 |
| `stochastic` | `bool` | 是否采用随机采样 | `True` | reproduction / survival 等采样阶段 | `True` 表示随机，`False` 表示确定性；调参阶段建议先使用 `False` |
| `use_continuous_sampling` | `bool` | 采样策略选择 | `False` | 概率采样细节 | 控制采样方式，大多数场景保持默认即可 |
| `use_fixed_egg_count` | `bool` | 产卵数是否固定 | `False` | reproduction | `True` 表示固定产卵数，`False` 更接近随机产卵过程 |

### `age_structure(...)` – 年龄结构

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `n_ages` | `int` | 年龄阶段总数 | `8` | 全流程（数组维度） | 约束初始状态、存活率向量等数组长度；必须与所有年龄相关参数长度一致 |
| `new_adult_age` | `int` | 个体进入成虫阶段的年龄索引 | `2` | reproduction / survival | 建议与目标物种的生命史阶段对齐；低于此年龄的个体视为幼体 |
| `generation_time` | `Optional[int]` | 代时标记 | `None` | 编译参数 | 仅用于建模解释；与 `age_structure` 中的同名参数互斥，后设置的会覆盖先设置的 |
| `equilibrium_distribution` | `Optional[Union[List[float], NDArray[np.float64]]]` | 平衡分布辅助参数 | `None` | competition / 初始化标度 | 与 `age_structure` 中的同名参数互斥，后设置的会覆盖先设置的；仅在需要显式稳态标定时使用 |

### `initial_state(...)` – 初始状态

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `individual_count` | `Mapping` | 初始个体数量分布，格式为 `{性别: {基因型: 年龄数据}}` | 必填 | 初始状态 | 如未设置，`build()` 会报错；支持标量、序列、映射等多种格式 |
| `sperm_storage` | `Optional[Mapping]` | 初始精子库存（仅在启用储精时需要） | `None` | reproduction | 仅在 `use_sperm_storage=True` 时必需；格式为三层映射 |

**年龄数据（`age_data`）的格式**（所有计数必须为非负数）：

- **标量**：分配到 `[new_adult_age, n_ages)` 范围内的所有年龄
- **列表 / 元组 / 数组**：按年龄索引依次写入，超长部分截断，不足部分补 `0`
- **字典**：显式指定特定年龄的数值，例如 `{2: 100, 3: 80}`

示例：

```python
# 标量：所有成年年龄（>= new_adult_age）都分配 100
{"female": {"WT|WT": 100.0}}

# 序列：按年龄顺序写入
{"female": {"WT|WT": [0, 100, 80, 60]}}

# 映射：只给部分年龄赋值
{"female": {"WT|WT": {2: 100, 3: 80}}}
```

**精子存储（`sperm_storage`）的格式**（三层映射）：

```python
{
  "WT|WT": {                 # 雌性基因型
    "Drive|WT": [0, 0, 20],  # 雄性基因型 -> 按年龄的精子数量
  }
}
```

校验规则：

- 年龄索引必须在 `[0, n_ages)` 范围内
- 所有计数必须 `>= 0`
- 基因型字符串必须能被当前的 `Species` 正确解析

### `survival(...)` – 存活率

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `female_age_based_survival_rates` | `Optional` | 雌性按年龄的存活率。 | `None` | survival | 支持标量、序列、映射、函数等；`None` 表示使用默认曲线；取值 `[0, 1]`。 |
| `male_age_based_survival_rates` | `Optional` | 雄性按年龄的存活率。 | `None` | survival | 同上。 |
| `generation_time` | `Optional[int]` | 代时标记。 | `None` | 编译参数 | 与 `age_structure` 中的同名参数互斥，后设置的会覆盖先设置的。 |
| `equilibrium_distribution` | `Optional` | 平衡分布的辅助参数。 | `None` | 标度辅助 | 与 `age_structure` 中的同名参数互斥，后设置的会覆盖先设置的。 |

**代码示例**（来自 `_resolve_survival_param`）：

```python
# A) None -> 使用默认曲线
.survival(female_age_based_survival_rates=None)

# B) 标量 -> 所有年龄使用相同值
.survival(female_age_based_survival_rates=0.85)

# C) 序列 -> 按年龄依次写入，不足补 0，超长截断
.survival(female_age_based_survival_rates=[1.0, 1.0, 0.9, 0.7])

# D) 稀疏映射 -> 未指定的年龄默认为 1.0
.survival(female_age_based_survival_rates={0: 1.0, 1: 0.95, 2: 0.8})

# E) 函数 -> 必须接收一个 age 参数并返回数值
.survival(female_age_based_survival_rates=lambda age: 1.0 if age < 2 else 0.8)

# F) 序列末尾用 None 哨兵 -> 末尾用最后一个非 None 值填充
.survival(female_age_based_survival_rates=[1.0, 0.9, None])
```

### `reproduction(...)` – 繁殖参数

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `female_age_based_mating_rates` | `Optional` | 雌性按年龄的交配率。 | `None` | reproduction | 长度必须等于 `n_ages`；未设置时使用默认值。 |
| `male_age_based_mating_rates` | `Optional` | 雄性按年龄的交配率。 | `None` | reproduction | 长度必须等于 `n_ages`；未设置时使用默认值。 |
| `female_age_based_relative_fertility` | `Optional` | 雌性按年龄的相对生育力权重。 | `None` | reproduction | 长度必须等于 `n_ages`；用于调节不同年龄雌性的产卵贡献。 |
| `eggs_per_female` | `float` | 每只雌性的基础产卵数。 | `50.0` | reproduction | 作为种群产卵数的基准；调参时可从中性值开始。 |
| `use_fixed_egg_count` | `bool` | 产卵数是否固定。 | `False` | reproduction | `True` 表示固定产卵数，`False` 表示随机产卵。 |
| `sex_ratio` | `float` | 后代中雌性的比例。 | `0.5` | reproduction | 取值范围 `[0, 1]`；`0.5` 表示雌雄各半。当性染色体约束可以确定后代性别时（例如 XX/ZW 为雌、XY/ZZ 为雄），该参数会被忽略。 |
| `use_sperm_storage` | `bool` | 是否启用精子存储机制。 | `True` | reproduction | `True` 启用，`False` 禁用（此时仅考虑当次交配）。 |
| `sperm_displacement_rate` | `float` | 新精子替换旧精子的速率。 | `0.05` | reproduction | 取值范围通常为 `(0, 1]`；值越大表示新精子替换速度越快。 |

### `competition(...)` – 竞争与密度调节

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `competition_strength` | `float` | 老幼体（age=1）的相对竞争因子。 | `5.0` | 幼体密度调节 | 竞争权重按年龄区分：age=0 固定为 `1.0`，age=1 使用 `competition_strength`。 |
| `juvenile_growth_mode` | `Union[int, str]` | 幼体生长的密度调节模式。 | `"logistic"` | 幼体密度调节 | 支持 `"logistic"`、`"beverton_holt"` 等模式，通常使用 `"logistic"`。 |
| `low_density_growth_rate` | `float` | 低密度下的内禀增长率。 | `6.0` | 幼体密度调节 | 表示无竞争时的增长倍数；取值过大容易导致种群振荡。 |
| `age_1_carrying_capacity` | `Optional[int]` | age=1 阶段的种群承载容量。 | `None` | 幼体密度调节 | 如果显式指定，会优先使用该值（优先级最高）。 |
| `old_juvenile_carrying_capacity` | `Optional[int]` | 与 `age_1_carrying_capacity` 功能相同的遗留参数名（已弃用）。 | `None` | 幼体密度调节 | 推荐使用 `age_1_carrying_capacity`，两者同时设置时以 `age_1_carrying_capacity` 为准。 |
| `expected_num_adult_females` | `Optional[int]` | 预期的成体雌性数量。 | `None` | 容量推导 | 用于通过平衡分布分析来反推承载容量（详见下文）。 |
| `equilibrium_distribution` | `Optional` | 平衡分布的辅助参数。 | `None` | 标度辅助 | 与 `age_structure` 中的同名参数互斥，后设置的会覆盖先设置的。 |

**承载容量的解析逻辑**：

当没有显式指定 `age_1_carrying_capacity` 或 `old_juvenile_carrying_capacity` 时，系统会尝试通过平衡分布分析，利用 `expected_num_adult_females` 来推导承载容量：

1. 如果提供了 `age_1_carrying_capacity` 或 `old_juvenile_carrying_capacity`（遗留别名），直接使用该值（最高优先级）。
2. 否则，如果提供了 `expected_num_adult_females`，系统会根据年龄相关的存活率，把这个数量分配到各个年龄阶段。
3. 基于平衡的年龄分布，计算来自成体雌性的期望 age=0 卵子产量（考虑交配率和生育力权重）。
4. 根据 age=0 的产量以及从 age=0 到 age=1 的基础存活率，反推出承载容量（即 age=1 时的 $K$ 值）。
5. 如果没有任何可用的承载容量来源，系统会尝试从 `initial_state()` 中推导（前提是初始状态已提供）。

这套方法可以确保承载容量与平衡种群分布保持一致，而不是简单地使用一个幼稚的缩放因子。

### `presets(...)` – 预设配置

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `*preset_list` | `Any` 变长参数 | 预设配置对象列表。 | 空 | `build` 后的后处理 | 预设会先被应用，建立基线配置；之后通过 `fitness`、`modifiers`、`hooks` 等设置可以覆盖预设值。 |

### `fitness(...)` – 适应度系数

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `viability` | `Optional[Dict]` | 生存适应度系数。 | `None` | survival | 支持多层嵌套：按基因型、按性别、按年龄、按性别+年龄。默认 `None` 表示不做修改。 |
| `fecundity` | `Optional[Dict]` | 生殖力适应度系数。 | `None` | reproduction | 按基因型和/或性别指定。默认 `None` 表示不做修改。 |
| `sexual_selection` | `Optional[Dict]` | 配对偏好权重。 | `None` | reproduction | 支持扁平映射 `{雄性: 值}` 或嵌套映射 `{雌性: {雄性: 值}}`。 |
| `zygote_viability` | `Optional[Dict]` | 合子存活适应度系数。 | `None` | reproduction | 在 survival 阶段之前应用，表示合子存活成为个体的概率。 |
| `mode` | `str` | 适应度值的写入模式。 | `"replace"` | fitness 写入策略 | `"replace"` 表示覆盖原有值，`"multiply"` 表示按倍数缩放。 |

**代码示例**：

```python
# viability: 基因型 -> 浮点数
.fitness(viability={"WT|WT": 1.0, "Drive|Drive": 0.6})

# viability: 基因型 -> {性别: 浮点数}
.fitness(viability={"Drive|WT": {"female": 0.9, "male": 0.8}})

# viability: 基因型 -> {年龄: 浮点数}，雌雄共用
.fitness(viability={"Drive|WT": {0: 0.95, 1: 0.85}})

# viability: 基因型 -> {性别: {年龄: 浮点数}}，可细分到性别+年龄
.fitness(viability={"Drive|WT": {"female": {1: 0.9}, "male": {2: 0.8}}})

# fecundity: 基因型 -> 浮点数 或 {性别: 浮点数}
.fitness(fecundity={"Drive|Drive": 0.7})

# sexual_selection 扁平格式: {雄性选择器: 值}，自动认为雌性为 '*'
.fitness(sexual_selection={"Drive|WT": 1.2, "WT|WT": 1.0})

# sexual_selection 嵌套格式: {雌性选择器: {雄性选择器: 值}}
.fitness(sexual_selection={"WT|WT": {"Drive|WT": 0.8, "WT|WT": 1.0}})

# zygote_viability fitness: 基因型 -> 浮点数（两性通用）
.fitness(zygote_viability={"A|A": 0.5, "a|a": 0.8})

# zygote_viability fitness: 基因型 -> {性别: 浮点数}（性别特异）
.fitness(zygote_viability={"a|a": {"female": 0.3, "male": 0.4}})
```

**`GenotypeSelector` 支持**：

- 单个选择器：`"Drive|WT"` 或 `Genotype` 对象
- 选择器并集：`("Drive|WT", "Drive|Drive")`

**`viability` 中年龄键的约束**（与代码一致）：

- 年龄键必须是整数，且范围在 `[0, n_ages)` 内
- 如果使用 `{性别: ...}` 形式，可以继续嵌套 `{年龄: 浮点数}`
- 如果直接使用 `{年龄: 浮点数}`，表示雌雄使用相同的值
- 未显式指定年龄时，默认写入 `new_adult_age - 1`（在离散世代模型中默认年龄为 `0`）

### `modifiers(...)` – 修饰器（配子/合子转换）

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `gamete_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | 配子阶段转换函数列表 | `None` | gamete -> zygote 映射编译 | 每个元素为 `(优先级, 名称, 函数)` 的元组；按优先级排序后依次应用 |
| `zygote_modifiers` | `Optional[List[Tuple[int, Optional[str], Callable]]]` | 合子阶段转换函数列表 | `None` | zygote 映射编译 | 每个元素为 `(优先级, 名称, 函数)` 的元组；按优先级排序后依次应用 |

**示例**：

```python
.modifiers(
  gamete_modifiers=[(10, "drive_gamete", my_gamete_modifier_fn)],
  zygote_modifiers=[(20, "drive_zygote", my_zygote_modifier_fn)],
)
```

说明：

- 元组结构固定为 `(优先级或钩子 ID, 可选名称, 可调用对象)`
- 在配置编译阶段，会按照元组的第一个字段（优先级）排序后再应用

### `hooks(...)` – 钩子函数

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `*hook_items` | `Callable` 或 `HookMap` | 钩子函数或钩子注册映射。 | 空 | 事件点（first / early / late / finish 等） | 支持两种形式：直接传入函数（需要带有 `@hook` 元数据），或者传入事件映射字典。如果函数未声明事件，会报错。 |

**两种合法的输入方式**：

```python
# 1) 直接传入函数（函数必须带有 @hook(event='...') 元数据）
.hooks(my_hook_fn)

# 2) 传入映射字典
.hooks({
  "late": [(my_hook_fn, "my_hook", 10)],
  "finish": [(finish_hook, "finish", 0)],
})
```

常见错误：

- 传入普通函数但没有 `event` 元数据 → 抛出 `ValueError`。
- 传入既不是可调用对象也不是字典的值 → 抛出 `TypeError`。

### `build()` – 编译构建

`build()` 方法没有参数，但有强约束：

- 必须先调用 `initial_state(...)` 设置初始状态。
- 执行顺序为：
  1. 构建 `PopulationConfig`
  2. 创建种群对象
  3. 应用 presets
  4. 应用 fitness / modifiers / hooks

因此，建议把 `build()` 放在链式调用的最后。

## DiscreteGenerationPopulationBuilder 参数手册

离散世代 Builder 与年龄结构模型的关键差异：

- 默认使用 `n_ages=2` 和 `new_adult_age=1`。
- 不需要调用 `age_structure(...)`。

### `setup(...)`

参数与年龄结构模型一致：`name`、`stochastic`、`use_continuous_sampling`、`use_fixed_egg_count`。

### `initial_state(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `individual_count` | `dict` | 初始个体数量的分布，格式为 `{性别: {基因型: 年龄数据}}`。 | 必填 | 初始状态 | 离散世代模型只支持 age 0 和 age 1；各种输入格式的含义与年龄结构模型相同。 |

**离散模型中 `age_data` 的解析规则**（来自 `_resolve_discrete_age_distribution`）：

```python
# 标量 -> (age0=0, age1=value)
{"female": {"WT|WT": 1000}}

# 长度 1 的序列 -> (0, value)
{"female": {"WT|WT": [1000]}}

# 长度 2 的序列 -> (age0, age1)
{"female": {"WT|WT": [200, 800]}}

# 映射 -> 只允许键为 0 或 1
{"female": {"WT|WT": {0: 200, 1: 800}}}
```

校验规则：

- 列表长度必须 `<= 2`。
- 字典的键只允许 `0` 和 `1`。
- 所有计数必须非负。

### `reproduction(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `eggs_per_female` | `float` | 每只雌性每代产卵数。 | `50.0` | reproduction | 产卵的基准值；调参时可从中性值开始。 |
| `sex_ratio` | `float` | 后代中雌性的比例。 | `0.5` | reproduction | 取值范围 `[0, 1]`；`0.5` 表示雌雄各半。当性染色体约束可以确定后代性别时（例如 XX/ZW 为雌、XY/ZZ 为雄），该参数会被忽略。 |
| `female_adult_mating_rate` | `float` | 成体雌性的交配率。 | `1.0` | reproduction | 表示雌性参与交配的比例；取值范围 `[0, 1]`。 |
| `male_adult_mating_rate` | `float` | 成体雄性的交配率。 | `1.0` | reproduction | 表示雄性参与交配的比例；取值范围 `[0, 1]`。 |

### `survival(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `female_age0_survival` | `float` | 雌性幼体（age 0）的存活率。 | `1.0` | survival | 取值范围 `[0, 1]`；`1.0` 表示全部存活。 |
| `male_age0_survival` | `float` | 雄性幼体（age 0）的存活率。 | `1.0` | survival | 取值范围 `[0, 1]`；`1.0` 表示全部存活。 |
| `adult_survival` | `float` | 成体在代际之间的存活率。 | `0.0` | survival / aging 边界 | 取值范围 `[0, 1]`；设为 `0` 可以近似严格的非重叠世代，较高的值允许成体跨代存活。 |

建模建议：

- 这三个概率最好都限制在 `[0, 1]` 之间。
- `adult_survival=0.0` 常用于严格离散世代的模型。

### `competition(...)`

| 参数 | 类型 | 说明 | 默认值 | 影响阶段 | 备注 |
|---|---|---|---|---|---|
| `juvenile_growth_mode` | `Union[int, str]` | 幼体生长的密度调节模式。 | `"logistic"` | 幼体密度调节 | 常用 `"logistic"`，也可以使用 `"beverton_holt"` 等其他模式。 |
| `low_density_growth_rate` | `float` | 低密度下的内禀增长倍数。 | `1.0` | 幼体密度调节 | 表示无竞争条件下的增长倍数；取值过大容易导致振荡。 |
| `carrying_capacity` | `Optional[int]` | 幼体的承载容量。 | `None` | 密度上限 | 如果未设置，系统会尝试自动推导；显式指定的值优先级最高。 |

### `presets(...)` / `fitness(...)` / `modifiers(...)` / `hooks(...)` / `build()`

这些方法的语义与年龄结构模型完全一致，区别仅在于离散世代的内核使用固定的年龄结构。

## 参数如何影响模拟过程（一个 tick 的视角）

下面以一个 tick 为例，说明各参数在哪个阶段发挥作用：

1. **reproduction（繁殖）**
   核心参数：交配率、相对生育力、`eggs_per_female`、`sex_ratio`、`fecundity`、`sexual_selection`、精子存储相关参数。

2. **survival（生存）**
   核心参数：按年龄的存活率向量 / 标量、`viability`。

3. **aging（衰老）**
   核心参数：年龄结构模型中的 `n_ages` 和 `new_adult_age`，或者离散世代模型中的 `adult_survival`。

4. **密度调节（幼体竞争）**
   发生在生存阶段的早期。核心参数：`juvenile_growth_mode`、`low_density_growth_rate`、承载容量相关参数。

5. **hook 事件点**
   通过 `hooks(...)` 注册的逻辑会在固定的阶段事件点触发，用于在各阶段前后进行干预。

## 推荐的配置顺序（可直接照抄）

### 年龄结构模型

1. `setup(...)`
2. `age_structure(...)`
3. `initial_state(...)`
4. `survival(...)`
5. `reproduction(...)`
6. `competition(...)`
7. `presets(...)` → `fitness(...)` → `modifiers(...)` → `hooks(...)`
8. `build()`

### 离散世代模型

1. `setup(...)`
2. `initial_state(...)`
3. `reproduction(...)`
4. `survival(...)`
5. `competition(...)`
6. `presets(...)` → `fitness(...)` → `modifiers(...)` → `hooks(...)`
7. `build()`

## 常见错误与排查

| 错误现象 | 可能原因 | 解决方法 |
|---|---|---|
| `build()` 直接报错 | 忘记设置 `initial_state(...)` | 在 `build()` 之前调用 `initial_state(...)` |
| 初始化或编译阶段报错 | 年龄向量长度与 `n_ages` 不一致 | 确保所有年龄相关参数的长度等于 `n_ages` |
| 结果异常或运行时错误 | `sex_ratio` 或其他概率参数越界 | 检查参数是否在合法范围内（如 `[0, 1]`） |
| 行为与预期不符 | 同名参数多次设置导致覆盖 | 注意 `generation_time`、`equilibrium_distribution` 等参数在多个方法中都可设置，后调用会覆盖先调用 |

## 本章小结

Builder 将种群的参数组织成可分类、可链式配置的流程，并在构建时注册到底层 `PopulationConfig` 中，以此实现高层易用性和底层高性能的统一。

***

**下一章**：[IndexRegistry 索引机制](index_registry.md)
