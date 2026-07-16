# 提取种群模拟数据

从 NATAL Core 模拟中提取和分析数据，包括观察规则、历史记录和输出格式。这些功能是进行数据分析、可视化和统计推断的关键组件。

## 数据提取概览

NATAL Core 提供三种主要的数据提取方式：

### 观察规则
观察规则用于从完整种群状态中提取特定子群体，支持灵活的分组和聚合，适用于实时监控和统计分析。

### 历史记录
历史记录功能记录模拟过程中的状态快照，支持时间序列分析，可配置记录频率和存储格式。

### 输出格式
输出格式系统提供多种数据导出格式，支持与外部工具集成，便于后续分析和可视化。

## 观察规则系统

### 核心对象

| 对象 | 作用 |
|------|------|
| **pop.observation** | 构建时生成的不可变观测规则；每个 Population 都有 |
| **pop.observe()** | 使用 canonical observation 投影当前状态 |
| **pop.history** | 带不可变 schema 的类型化 `History` 容器 |

### 推荐工作流

在构建阶段用 `with_observation()` 定义分组。其参数必须是非空的有序映射：
键是非空字符串，值是 `IndividualSelector`。普通 `dict` 会保留插入顺序，
这个顺序就是结果中的 group 轴顺序。

```python
import natal as nt

pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(...)
    .with_observation(
        {
            "adult_wt": nt.IndividualSelector(ztype="WT|WT", age=[1]),
            "drive_carriers": (
                nt.IndividualSelector(ztype="WT|Drive")
                | nt.IndividualSelector(ztype="Drive|Drive")
            ),
        },
        collapse_age=False,
    )
    .record_history(mode="raw")
    .build()
)

# 投影当前状态；group 始终是第一条轴
current = pop.observe()
print(current.axes)                 # ("group", "sex", "age")
print(current.labels["group"])
print(current.values)

# raw History 可在运行后使用同一规则做事后投影
observed_history = pop.history.observe(pop.observation)
print(observed_history.values.shape)  # (record, group, sex, age)
```

如果没有调用 `with_observation()`，构建过程会自动生成恒等观测：每个
ZType 对应一个分组，不丢失任何计数。也就是说，`pop.observation` 和
`pop.observe()` 对 panmictic、年龄结构、离散世代及 spatial Population
始终可用，不会返回 `None`。

## 基于 Observation 的历史记录（压缩模式）

大模拟中（大量 genotype、大量 deme），全量原始历史记录的存储开销极高——每个快照包含所有 genotype 的计数。Observation 系统能将 genotype 维度投影到用户关心的分组上，在 recording 阶段直接做聚合，只需记录聚合后的结果，大幅减少内存占用。

### 两种模式对比

| 模式 | `History` 的类型化数据 | 典型形状（panmictic） | 能否事后更换观测 |
|------|----------------------|-------------------------|--------------------|
| 原始（默认） | `individual_count`、可选 `sperm_storage` | `(record, sex, age, ztype)` | 可以 |
| 观测 | `values` | `(record, group, sex, age)` | 不可以；原始 ZType 已丢弃 |

当 `n_groups << n_genotypes` 时（常见场景），压缩比约为 `n_genotypes / n_groups` 倍。

### 配置方式

观测规则和历史记录模式只在构建阶段配置：

**方式一：构建时 `with_observation()` + `record_history(mode="observation")`（推荐）**

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(...)
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(groups={
        "wt": nt.IndividualSelector(ztype="WT|WT"),
        "het": nt.IndividualSelector(ztype="WT|Dr"),
        "hom": nt.IndividualSelector(ztype="Dr|Dr"),
    }, collapse_age=True)
    .record_history(mode="observation", max_rows=5000)
    .build()
)
pop.run(n_steps=100, record_every=10)
```

`record_history()` 和 `with_observation()` **相互独立**——链式调用顺序无关紧要。
当设置 `mode="observation"` 但没有显式调用 `with_observation()` 时，会自动生成恒等观测。

**方式二：构建时自动恒等观测（无需显式分组）**

```python
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(...)
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .record_history(mode="observation")  # 自动生成恒等观测
    .build()
)
```
无需 `with_observation()`——每个 ZType 自动对应一个分组，提供无损投影。

### Panmictic 示例

```python
import natal as nt

species = nt.Species.from_dict(
    name="demo",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="obs_demo", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500, "Dr|WT": 50},
        "male": {"WT|WT": 500, "Dr|WT": 50},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(
        {
            "wildtype": nt.IndividualSelector(ztype="WT|WT"),
            "drive": (
                nt.IndividualSelector(ztype="WT|Dr")
                | nt.IndividualSelector(ztype="Dr|Dr")
            ),
        },
        collapse_age=True,
    )
    .record_history(mode="observation")
    .build()
)

pop.run(n_steps=100, record_every=10)

# 当前观测与历史使用同一套 canonical observation
print(pop.observe().axes)        # ("group", "sex")
print(pop.history.ticks)
print(pop.history.values.shape)  # (record, group, sex)
```

### Spatial 示例

```python
from natal import SpatialPopulation, HexGrid
import numpy as np

species = nt.Species.from_dict(
    name="spatial_obs",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

kernel = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

spatial = (
    SpatialPopulation.builder(species, n_demes=9, topology=HexGrid(3, 3))
    .setup(name="spatial_obs_demo", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500}, "male": {"WT|WT": 500},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(kernel=kernel, migration_rate=0.2)
    .with_observation(
        {
            "wt": nt.IndividualSelector(ztype="WT|WT"),
            "dr": (
                nt.IndividualSelector(ztype="WT|Dr")
                + nt.IndividualSelector(ztype="Dr|Dr")
            ),
        },
        collapse_age=True,
        demes=[2, 0],
        deme_mode="preserve",
    )
    .record_history(mode="observation")
    .build()
)

spatial.run(n_steps=50, record_every=5)

# preserve 按 demes=[2, 0] 的顺序保留共享 deme 轴
print(spatial.observe().axes)        # ("group", "deme", "sex")
print(spatial.history.values.shape)  # (record, group, 2, sex)
```

### Post-hoc 观测（不修改 recording 模式）

原始模式保留完整状态，因此可以在模拟完成后应用 canonical observation，
而不改变原始历史：

```python
# 对已经记录的原始历史做 post-hoc 观测
observed_history = pop.history.observe(pop.observation)

# 返回新的 observation-mode History；pop.history 仍是原始 History
print(observed_history.values.shape)
print(pop.history.individual_count.shape)
```

`History.observe()` 会比较 observation 与 History 的 Population 布局指纹。
即使两个种群的数组形状相同，只要物种/ZType 布局不同，也会抛出
`ValueError`，避免把一套分组规则错误地套到另一个种群上。观测模式
History 已经丢弃原始 ZType 数据，不能再次事后投影。

### Spatial History 的轴

Spatial `with_observation()` 用 `demes` 为所有 group 指定同一个有序 deme
集合。`deme_mode="preserve"` 保留这个共享轴；`"aggregate"` 对它求和并
移除该轴。`demes=None` 默认选择全部 deme。各模式的公开数组形状如下：

| 数据/模式 | `collapse_age=False` | `collapse_age=True` |
|-----------|----------------------|---------------------|
| `spatial.observe().values`，preserve | `(group, selected_deme, sex, age)` | `(group, selected_deme, sex)` |
| `spatial.observe().values`，aggregate | `(group, sex, age)` | `(group, sex)` |
| raw `spatial.history.individual_count` | `(record, deme, sex, age, ztype)` | 不适用；raw 不折叠年龄 |
| observation `spatial.history.values`，preserve | `(record, group, selected_deme, sex, age)` | `(record, group, selected_deme, sex)` |
| observation `spatial.history.values`，aggregate | `(record, group, sex, age)` | `(record, group, sex)` |

Raw History 始终保存所有 deme，不受 Observation 的选择或聚合模式影响。
即使空间 Population 只有一个 deme，raw 数组仍保留长度为 1 的 deme 轴；
只有非空间 Population 才省略该轴。

### 何时使用观测模式 vs post-hoc

| 场景 | 推荐方式 |
|------|---------|
| 需要全量 genotype 数据的精细分析 | 原始历史（默认） |
| 只关心几个分组的时间序列 | `record_history(mode="observation")` |
| 需要保留完整状态并在事后投影 | 原始历史 + `history.observe(pop.observation)` |
| 大规模 spatial（数千 deme） | `record_history(mode="observation")` |
| 内存受限环境 | `record_history(mode="observation")` |

## 历史记录系统

### 记录模式与容量

Configurator 提供 `record_history()` 方法，在构建阶段设置记录模式和容量。该方法**独立于** `with_observation()`——链式调用的顺序无关紧要。

```python
# 构建时：配置记录模式和容量
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .initial_state(individual_count={"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .record_history(mode="observation", max_rows=5000)  # 观测模式，FIFO 上限
    .build()
)
```

当设置 `mode="observation"` 但没有显式调用 `with_observation()` 时，会自动生成**恒等观测（identity observation）**——每个 ZType 一个分组，提供无损投影，无需手动定义分组规则。

| 参数 | 默认值 | 说明 |
|-----------|---------|------|
| `mode` | `"raw"` | `"raw"` 记录完整状态；`"observation"` 记录压缩后的观测聚合 |
| `max_rows` | `None` | 最多保存的快照数（FIFO 淘汰）。`None` = 无限制 |

### 运行时空录配置

种群对象也提供运行时记录控制（向后兼容）：

```python
pop.record_every = 10  # 每10步记录一次
pop.max_history = 1000  # 最多保存1000个快照（旧版）
```

录制 schema（模式、行大小、布局）在**构建时冻结**，记录首行后无法更改。一旦通过 Configurator 配置完成，`pop.record_every` 和 `pop.max_history` 只控制记录**频率**和**旧版上限**，不影响 schema。

```python
# 运行模拟并记录历史
results = pop.run(n_steps=500, record_every=5)

# 获取类型化历史数据
history = pop.history
print("历史记录数量:", history.n_records)
print("记录模式:", history.schema.mode)
```

### 历史数据访问

```python
# 每个 Population（含 Spatial）始终拥有同一个 History 容器
history = pop.history
ticks = history.ticks
print("记录的时间步:", ticks)

# raw 模式读取完整个体计数
if history.schema.mode == "raw":
    print(history.individual_count.shape)
    print(history.sperm_storage)  # 离散世代种群为 None
else:
    print(history.values.shape)

# 清空历史记录以节省内存
pop.clear_history()
```

### pop.observation — 只读观测属性

每个种群都通过只读属性暴露其构建时冻结的 canonical observation：

```python
obs = pop.observation
print(f"分组: {obs.labels}")
print(f"是否折叠年龄: {obs.collapse_age}")
```

没有显式配置时，它是每个 ZType 一个分组的恒等观测。观测在构建时冻结，
之后无法更改。

### pop.observe() — 投影当前状态

通过配置的观测投影当前种群状态，返回结构化的 `ObservationResult`：

```python
result = pop.observe()
print(f"Tick: {result.tick}")
print(f"轴: {result.axes}")          # ("group", "sex", "age")
print(f"值形状: {result.values.shape}")
print(f"分组标签: {result.labels['group']}")
```

结果始终以 group 为第一条轴。`collapse_age=True` 时，年龄求和后移除 age
轴；spatial 的 `deme_mode="preserve"` 在 group 后保留 deme 轴，
`"aggregate"` 则对选中 deme 求和并移除该轴。种群尚无状态时抛出
`RuntimeError`。

### pop.record_snapshot() — 手动记录

在 `run()` 之外手动将当前稳定状态记录到历史中：

```python
pop.run_tick()
pop.record_snapshot()  # 在单个 tick 后手动记录
```

应在两次 `run()` 调用之间的稳定边界调用。当前 tick 已有记录时抛出
`ValueError`；在已结束的种群上调用会抛出 `RuntimeError`。

### pop.restore_checkpoint(tick) — 状态恢复

从原始模式的历史记录中恢复种群状态到指定 tick。该 tick 之后的所有记录将被删除：

```python
# 在模拟过程中记录原始历史
pop.run(n_steps=100, record_every=1)

# 恢复到 tick 50
pop.restore_checkpoint(tick=50)
# 种群状态现在与 tick 50 时完全相同
# tick 50 之后的所有历史记录均被丢弃
```

仅对原始模式历史（`mode="raw"`）有效。当模式为 `"observation"` 时抛出 `ValueError`——观测模式的历史不保留恢复所需的完整状态数据。因此如有检查点需求，请使用原始模式记录。

### 历史数据分析

```python
# 分析等位基因频率随时间变化
allele_freq_history = []
for snapshot in full_history["snapshots"]:
    # 重新计算每个时间步的等位基因频率
    # 这里需要根据实际数据结构进行调整
    freq = calculate_allele_frequency(snapshot)
    allele_freq_history.append(freq)

# 绘制时间序列图
import matplotlib.pyplot as plt
plt.plot(ticks, allele_freq_history)
plt.xlabel("时间步")
plt.ylabel("等位基因频率")
plt.show()
```

## 输出格式系统

### 当前状态输出

```python
# Project current population state through the canonical observation
result = pop.observe()
print("Tick:", result.tick)
print("Axes:", result.axes)
print("Labels:", result.labels)
print("Values shape:", result.values.shape)
print("Values:", result.values)
```

### 数据导出

```python
import json

# Get current projected state as a structured result
result = pop.observe()

# Convert to a JSON-serializable dictionary
data_dict = {
    "tick": result.tick,
    "axes": list(result.axes),
    "labels": {k: list(v) for k, v in result.labels.items()},
    "values": result.values.tolist(),
}

# Save to JSON file
with open("population_state.json", "w") as f:
    json.dump(data_dict, f, indent=2)
```

### 与外部工具集成

```python
import pandas as pd

# Convert observation-mode history to pandas DataFrame
def history_to_dataframe(observed_history):
    """Convert observed history records to DataFrame"""
    data = []
    group_labels = observed_history.labels["group"]
    for i, tick in enumerate(observed_history.ticks):
        row = {
            "tick": tick,
            "total_population": observed_history.values[i].sum(),
        }
        for j, group in enumerate(group_labels):
            row[group] = observed_history.values[i, j].sum()
        data.append(row)
    return pd.DataFrame(data)

# Usage example
observed = pop.history.observe(pop.observation)
history_df = history_to_dataframe(observed)
print(history_df.head())
```

## 观察规则详解

### Group Format

Groups must be defined using `IndividualSelector` instances passed to
`.with_observation()` at build time. Each key in the groups mapping becomes
a group label, and its value is an `IndividualSelector` that selects the
individuals belonging to that group.

`IndividualSelector` accepts the following keyword-only arguments:

| Argument | Type | Description |
|----------|------|-------------|
| `ztype` | `str` | Diploid genotype string (e.g. `"WT|Dr"`) |
| `gtype` | `str` | Haploid genotype string |
| `sex` | `str` or `int` | `"female"`, `"male"`, or `0` / `1` |
| `age` | `range`, `int`, or sequence of `int` | Age or age interval |

Selectors can be combined with `|` (union) and `+` (intersection) operators:

```python
# Union — individuals matching either selector
combined = nt.IndividualSelector(ztype="WT|Dr") | nt.IndividualSelector(ztype="Dr|Dr")

# Intersection — individuals matching both selectors
both = nt.IndividualSelector(sex="female") + nt.IndividualSelector(age=range(2, 5))
```

### 分组示例

```python
# Single genotype group
{"wt": nt.IndividualSelector(ztype="WT|WT")}

# Age range group
{"adults": nt.IndividualSelector(age=range(2, 8))}

# Combined criteria
{"juvenile_female": nt.IndividualSelector(sex="female", age=range(0, 2))}

# Union of genotypes
{"drive_carriers": (
    nt.IndividualSelector(ztype="WT|Drive")
    | nt.IndividualSelector(ztype="Drive|Drive")
)}

# Wildcard — all genotypes (identity group)
{"all": nt.IndividualSelector()}
```

## 实用示例

### 监控基因驱动传播

```python
import natal as nt

species = nt.Species.from_dict(
    name="drive_monitor",
    structure={"chr1": {"loc": ["WT", "Drive"]}},
)

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="drive_monitor", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500, "Drive|WT": 50},
        "male": {"WT|WT": 500, "Drive|WT": 50},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(
        {
            "wild_type": nt.IndividualSelector(ztype="WT|WT"),
            "heterozygotes": nt.IndividualSelector(ztype="WT|Drive"),
            "homozygotes": nt.IndividualSelector(ztype="Drive|Drive"),
            "total_drive": (
                nt.IndividualSelector(ztype="WT|Drive")
                | nt.IndividualSelector(ztype="Drive|Drive")
            ),
        },
        collapse_age=True,
    )
    .build()
)

for step in range(100):
    pop.run_tick()
    if step % 10 == 0:
        result = pop.observe()
        group_map = {
            name: i for i, name in enumerate(result.labels["group"])
        }
        values = result.values
        print(f"Step {step}: "
              f"WT={values[group_map['wild_type']].sum():.0f}, "
              f"Het={values[group_map['heterozygotes']].sum():.0f}, "
              f"Hom={values[group_map['homozygotes']].sum():.0f}")
```

### 年龄结构分析

```python
import natal as nt

species = nt.Species.from_dict(
    name="age_analysis",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="age_demo", stochastic=False)
    .initial_state(individual_count={
        "female": {"WT|WT": 500, "Dr|WT": 50},
        "male": {"WT|WT": 500, "Dr|WT": 50},
    })
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .with_observation(
        {
            "juveniles": nt.IndividualSelector(age=range(0, 2)),
            "young_adults": nt.IndividualSelector(age=range(2, 4)),
            "mature_adults": nt.IndividualSelector(age=range(4, 6)),
            "old_adults": nt.IndividualSelector(age=range(6, 8)),
        },
        collapse_age=True,
    )
    .record_history(mode="raw")
    .build()
)

pop.run(n_steps=100, record_every=1)

# Project raw history through the canonical observation
observed = pop.history.observe(pop.observation)
for i, tick in enumerate(observed.ticks):
    values = observed.values[i]  # (group, sex)
    total = float(values.sum())
    if total > 0:
        group_labels = observed.labels["group"]
        juv_idx = group_labels.index("juveniles")
        juvenile_ratio = values[juv_idx].sum() / total
        print(f"Tick {tick}: juvenile ratio = {juvenile_ratio:.3f}")
```

## 最佳实践

### 观察规则设计
- 使用有意义的组名便于后续分析
- 保持组间互斥性避免重复计数
- 优先使用模式匹配而非硬编码基因型列表

### 历史记录管理
- 设置合适的 `record_every` 参数平衡精度和性能
- 使用 `clear_history()` 管理内存使用
- 定期导出历史数据避免数据丢失

### 数据导出
- 使用标准格式（JSON、字典）便于工具集成
- 包含足够的元数据（时间步、参数设置等）
- 考虑数据压缩和存储效率

## 常见问题

### 观察规则和历史记录有什么区别？
观察规则定义如何从状态中提取数据，历史记录保存状态的时间序列。观察规则可应用于当前状态或历史记录。

### 如何优化大数据量的历史记录？
增加 `record_every` 间隔，使用 `clear_history()` 定期清理，或导出到外部存储。注意 `clear_history()` 会保留录制 schema——清空数据后无需重新配置即可继续记录。

### 观察规则会影响模拟性能吗？
观察规则本身不影响模拟性能，但频繁的数据提取和存储可能影响整体性能。

### 构建 Population 后还能修改录制规则吗？
不能。canonical observation 和 History schema 都在 `build()` 时冻结。
`pop.update().with_observation(...)` 与 `pop.update().record_history(...)` 会抛出
`RuntimeError`。运行时只读取 `pop.observation`、调用 `pop.observe()`，或在
raw History 上调用 `pop.history.observe(pop.observation)`。

### `record_history()` 和 `with_observation()` 有什么区别？
`with_observation()` 定义*观测哪些分组*（观测投影规则）。`record_history()` 设置*如何记录*——原始完整状态还是压缩后的观测聚合。两者相互独立：可以有观测分组但不启用压缩记录，也可以启用压缩记录但无需显式定义分组（自动恒等观测）。

### 能否将种群恢复到之前的状态？
可以，如果使用了原始模式记录（`mode="raw"`），通过 `pop.restore_checkpoint(tick)` 即可恢复。它会将个体计数（以及适用时的精子存储）恢复到该 tick 的精确状态。观测模式的历史不支持检查点恢复，因为它不保留逐基因型的数据。

---

介绍了如何从 NATAL Core 模拟中提取和分析数据。在实际项目中，建议先设计合适的观察规则，再根据需求选择合适的数据提取方式。
