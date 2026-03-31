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
- [Simulation Kernels 深度解析](simulation_kernels.md)

## 2. 两类 Builder

- `AgeStructuredPopulationBuilder`
  - 适合多年龄层模型。
  - 典型特征：`n_ages` 可配置，支持年龄向量输入、可选精子存储。
- `DiscreteGenerationPopulationBuilder`
  - 适合离散世代模型。
  - 典型特征：默认采用两年龄阶段（`n_ages=2`，`new_adult_age=1`）。

## 3. AgeStructuredPopulationBuilder 参数手册

### 3.1 `setup(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `name` | `str` | `"AgeStructuredPop"` | 全流程 | 仅用于标识与日志，不改变动力学。建议实验批次显式命名。 |
| `stochastic` | `bool` | `True` | reproduction/survival 等采样阶段 | `True` 为随机采样，`False` 为确定性。调参期建议先 `False`。 |
| `use_dirichlet_sampling` | `bool` | `False` | 概率采样细节 | 控制采样策略。大多数场景保持默认即可。 |
| `use_fixed_egg_count` | `bool` | `False` | reproduction | `True` 产卵数固定，`False` 更接近随机产卵过程。 |

### 3.2 `age_structure(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `n_ages` | `int` | `8` | 全流程（数组维度） | 年龄层数量。会约束初始状态、存活率向量等长度。 |
| `new_adult_age` | `int` | `2` | reproduction/survival | 成虫起始年龄索引。建议与物种生命史一致。 |
| `generation_time` | `Optional[int]` | `None` | 编译参数 | 代时标记，可用于建模解释；不建议与年龄定义冲突。 |
| `equilibrium_distribution` | `Optional[list/ndarray]` | `None` | competition/初始化标度 | 平衡分布辅助参数。仅在你明确要做稳态标定时使用。 |

### 3.3 `initial_state(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `individual_count` | `dict` | 必填 | 初始状态 | 格式 `{sex: {genotype: age_data}}`。不设置会在 `build()` 报错。 |
| `sperm_storage` | `Optional[dict]` | `None` | reproduction（若启用储精） | 初始精子库存。仅在 `use_sperm_storage=True` 场景必要。 |

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

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `female_age_based_survival_rates` | `Optional[Any]` | `None` | survival | 支持标量/序列/映射/函数。`None` 使用默认曲线。 |
| `male_age_based_survival_rates` | `Optional[Any]` | `None` | survival | 同上。 |
| `generation_time` | `Optional[int]` | `None` | 编译参数 | 与 `age_structure` 同名参数，后设会覆盖先设值。 |
| `equilibrium_distribution` | `Optional[list/ndarray]` | `None` | 标度辅助 | 与 `age_structure` 同名参数，后设覆盖先设值。 |

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

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `female_age_based_mating_rates` | `Optional[list/ndarray]` | `None` | reproduction | 雌性按年龄交配率。 |
| `male_age_based_mating_rates` | `Optional[list/ndarray]` | `None` | reproduction | 雄性按年龄交配率。 |
| `female_age_based_relative_fertility` | `Optional[list/ndarray]` | `None` | reproduction | 雌性按年龄相对生育力权重。 |
| `eggs_per_female` | `float` | `50.0` | reproduction | 每雌基础产卵规模。先从中性值开始。 |
| `use_fixed_egg_count` | `bool` | `False` | reproduction | 固定/随机产卵开关。 |
| `sex_ratio` | `float` | `0.5` | reproduction | 后代雌性比例，通常设为 0.5。 |
| `use_sperm_storage` | `bool` | `True` | reproduction | 是否启用储精机制。 |
| `sperm_displacement_rate` | `float` | `0.05` | reproduction | 新精子替换旧精子的强度。 |

格式与长度要求（源码行为）：

- `female_age_based_mating_rates` / `male_age_based_mating_rates` / `female_age_based_relative_fertility`
  - 传入后会被转成 `np.array`。
  - 最终在配置编译时要求长度必须等于 `n_ages`，否则报 `ValueError`。

### 3.6 `competition(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `competition_strength` | `float` | `5.0` | 幼体密度调节 | 竞争强度因子。 |
| `juvenile_growth_mode` | `int\|str` | `"logistic"` | 幼体密度调节 | 支持字符串或常量。常用 `logistic`。 |
| `low_density_growth_rate` | `float` | `6.0` | 幼体密度调节 | 低密度增长速率，过大易振荡。 |
| `old_juvenile_carrying_capacity` | `Optional[int]` | `None` | 幼体密度调节 | 优先于 `expected_num_adult_females` 推导。 |
| `expected_num_adult_females` | `Optional[int]` | `None` | 容量推导 | 若未给 carrying capacity，可与产卵数一起推导容量。 |
| `equilibrium_distribution` | `Optional[list/ndarray]` | `None` | 标度辅助 | 与前述同名参数一致，后设覆盖先设。 |

### 3.7 `presets(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `*preset_list` | `Any` 变长参数 | 空 | build 后处理 | 预设会先应用，随后 `fitness/modifiers/hooks` 可继续覆盖。 |

### 3.8 `fitness(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `viability` | `Optional[dict]` | `None` | survival | 生存适应度映射。支持按基因型、按性别、按年龄、按性别+年龄。 |
| `fecundity` | `Optional[dict]` | `None` | reproduction | 生殖力适应度映射。 |
| `sexual_selection` | `Optional[dict]` | `None` | reproduction（配对偏好） | 支持平铺或嵌套映射。 |
| `mode` | `str` | `"replace"` | fitness 写入策略 | `replace` 覆盖，`multiply` 乘法缩放。 |

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

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `gamete_modifiers` | `Optional[list[(hook_id,name,fn)]]` | `None` | gamete->zygote 映射编译 | 通常用于配子阶段修饰。 |
| `zygote_modifiers` | `Optional[list[(hook_id,name,fn)]]` | `None` | zygote 映射编译 | 通常用于合子阶段修饰。 |

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

| 参数 | 类型 | 默认值 | 影响阶段 | 说明与建议 |
|---|---|---|---|---|
| `*hook_items` | `Callable` 或 `HookMap` | 空 | 事件点（first/early/late/finish 等） | 可混用函数与字典注册形式。未声明事件会报错。 |

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
- `use_dirichlet_sampling`
- `use_fixed_egg_count`

### 4.2 `initial_state(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明 |
|---|---|---|---|---|
| `individual_count` | `dict` | 必填 | 初始状态 | 仍是 `{sex: {genotype: age_data}}`，但仅支持 age 0/1。 |

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

| 参数 | 类型 | 默认值 | 影响阶段 | 说明 |
|---|---|---|---|---|
| `eggs_per_female` | `float` | `50.0` | reproduction | 每雌产卵规模。 |
| `sex_ratio` | `float` | `0.5` | reproduction | 后代雌性比例。 |
| `female_adult_mating_rate` | `float` | `1.0` | reproduction | 成体雌性交配率。 |
| `male_adult_mating_rate` | `float` | `1.0` | reproduction | 成体雄性交配率。 |

### 4.4 `survival(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明 |
|---|---|---|---|---|
| `female_age0_survival` | `float` | `1.0` | survival | 雌性幼体阶段存活率。 |
| `male_age0_survival` | `float` | `1.0` | survival | 雄性幼体阶段存活率。 |
| `adult_survival` | `float` | `0.0` | survival/aging 边界 | 成体跨步存活率。设 0 可近似非重叠世代。 |

建模约束建议：

- 这三个概率在实践中都建议位于 `[0, 1]`。
- `adult_survival=0.0` 常用于严格离散世代。

### 4.5 `competition(...)`

| 参数 | 类型 | 默认值 | 影响阶段 | 说明 |
|---|---|---|---|---|
| `juvenile_growth_mode` | `int\|str` | `"logistic"` | 幼体密度调节 | 增长模式。 |
| `low_density_growth_rate` | `float` | `1.0` | 幼体密度调节 | 低密度增长速率。 |
| `carrying_capacity` | `Optional[int]` | `None` | 密度上限 | 承载上限。 |

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
