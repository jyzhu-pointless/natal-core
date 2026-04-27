# 提取种群模拟数据

本章节介绍如何从 NATAL Core 模拟中提取和分析数据，包括观察规则、历史记录和输出格式。这些功能是进行数据分析、可视化和统计推断的关键组件。

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
| **Population.create_observation(...)** | 推荐的公开入口，用于构建可复用观测对象 |
| **output_current_state / output_history** | 推荐的公开输出 API，内置 observation 集成 |
| **ObservationFilter** | 高级编译辅助类（仅低层定制流程需要） |

### 推荐工作流

不建议在业务代码中直接实例化 `Observation`，优先使用 population 层公开方法。

```python
# 创建观察规则
observation = pop.create_observation(
    groups={
        "adult_wt": {"genotype": ["WT|WT"], "age": [1]},
        "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]}
    },
    collapse_age=False,
)

# 获取当前状态
current = pop.output_current_state(observation=observation)
print("当前观察数据:", current["observed"])

# 获取历史数据
history = pop.output_history(observation=observation)
print("历史观察数据:", history["observed"])
```

这种写法可复用观测定义，并将维度有效性校验延迟到应用/输出阶段。

## 基于 Observation 的历史记录（压缩模式）

大模拟中（大量 genotype、大量 deme），全量原始历史记录的存储开销极高——每个快照包含所有 genotype 的计数。Observation 系统能将 genotype 维度投影到用户关心的分组上，在 recording 阶段直接做聚合，只需记录聚合后的结果，大幅减少内存占用。

### 两种模式对比

| 模式 | 记录内容 | 每行大小 | 导出时是否需重新解析 |
|------|---------|---------|-----------------|
| 原始 | `[tick, ind.ravel(), sperm.ravel()]` | `1 + n_sexes×n_ages×n_geno + …` | 需要，按 genotype 展开 |
| 观测 | `[tick, observed.ravel()]` | `1 + n_groups×n_sexes×n_ages` | 不需要，直接按分组名展开 |

当 `n_groups << n_genotypes` 时（常见场景），压缩比约为 `n_genotypes / n_groups` 倍。

### 配置方式

两种方式均可激活观测模式：

**方式一：先创建 Observation，再设置 `record_observation`**

```python
obs = pop.create_observation(
    groups={
        "wt": {"genotype": ["WT|WT"]},
        "het": {"genotype": ["WT|Dr"]},
        "hom": {"genotype": ["Dr|Dr"]},
    },
    collapse_age=True,
)
pop.record_observation = obs  # 激活观测模式
pop.run(n_steps=100, record_every=10)
```

**方式二：直接使用 `set_observations()`**

```python
pop.set_observations(
    groups={
        "wt": {"genotype": ["WT|WT"]},
        "het": {"genotype": ["WT|Dr"]},
        "hom": {"genotype": ["Dr|Dr"]},
    },
    collapse_age=True,
)
pop.run(n_steps=100, record_every=10)
```

`record_observation` 被设置后，内核在 recording 时自动使用观测聚合模式。`output_history()` 自动检测并选择正确的导出路径：

```python
history = pop.output_history()
# 自动按观测模式导出，每行包含：
# - tick
# - labels: ["wt", "het", "hom"]
# - observed: { "wt": { "female": ..., "male": ... }, ... }
```

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
    .build()
)

# 激活观测模式
pop.set_observations(
    groups={
        "wildtype": {"genotype": ["WT|WT"]},
        "drive_het": {"genotype": ["WT|Dr"]},
        "drive_hom": {"genotype": ["Dr|Dr"]},
    },
    collapse_age=True,
)
pop.run(n_steps=100, record_every=10)

# 导出——自动使用观测模式
history = pop.output_history()
for snap in history["snapshots"]:
    print(f"tick {snap['tick']}: {snap['observed']}")

# 可以随时切回原始模式查看
pop.record_observation = None  # 关闭观测模式
# 后续 run() 将恢复原始 recording
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
    .build()
)

# 激活观测模式
spatial.set_observations(
    groups={
        "wt": {"genotype": ["WT|WT"]},
        "dr": {"genotype": ["WT|Dr", "Dr|Dr"]},
    },
    collapse_age=True,
)
spatial.run(n_steps=50, record_every=5)

# 导出——按 deme 逐个展开，附带跨 deme 汇总
history = spatial.output_history()
for snap in history["snapshots"]:
    print(f"tick {snap['tick']}")
    for deme_key, deme_obs in snap["demes"].items():
        print(f"  {deme_key}: {deme_obs}")
    print(f"  aggregate: {snap['aggregate']}")
```

### Post-hoc 观测（不修改 recording 模式）

如果不想改变 recording 模式，但需要以观测格式查看历史，可以传入 `observation` 参数：

```python
obs = pop.create_observation(groups={
    "females": {"sex": "female"},
    "males": {"sex": "male"},
})

# 对已经记录的原始历史做 post-hoc 观测
history = pop.output_history(observation=obs)
# 注意：如果历史是原始模式（未设置 record_observation），
# 观测会在导出时按每个快照重新计算（慢但无需重跑模拟）。
# 如果历史是观测模式，则直接读取压缩后的数据。
```

### Spatial 的 Deme 选择器

Spatial 模式下，group spec 支持 `"deme"` 键来控制哪些 deme 纳入该分组：

```python
spatial.set_observations(
    groups={
        "center_release": {
            "genotype": ["Dr|Dr"],
            "deme": [(1, 1)],          # 只观察中心 deme
        },
        "all_wt": {
            "genotype": ["WT|WT"],
            "deme": "all",             # 所有 deme（默认）
        },
    },
)
```

`"deme"` 支持：
- `"all"` 或省略：所有 deme
- 整数列表：扁平索引，如 `[0, 2, 4]`
- 坐标列表：`(row, col)` 元组，自动通过 topology 解析

### 何时使用观测模式 vs post-hoc

| 场景 | 推荐方式 |
|------|---------|
| 需要全量 genotype 数据的精细分析 | 原始历史（默认） |
| 只关心几个分组的时间序列 | `record_observation` 观测模式 |
| 需要事后按不同分组反复查看 | 原始历史 + post-hoc `output_history(observation=obs)` |
| 大规模 spatial（数千 deme） | `record_observation` 观测模式 |
| 内存受限环境 | `record_observation` 观测模式 |

## 历史记录系统

### 历史记录配置

种群对象内置历史记录功能，可配置记录频率和存储格式：

```python
# 配置历史记录
pop.record_every = 10  # 每10步记录一次
pop.max_history = 1000  # 最多保存1000个快照

# 运行模拟并记录历史
results = pop.run(n_steps=500, record_every=5)

# 获取历史数据
history_data = pop.output_history()
print("历史记录数量:", len(history_data["snapshots"]))
print("最后一步数据:", history_data["snapshots"][-1])
```

### 历史数据访问

```python
# 获取完整历史记录
full_history = pop.output_history()

# 获取特定时间步的历史
history_at_tick_100 = pop.output_history(tick=100)

# 获取历史记录的时间步列表
ticks = [snapshot["tick"] for snapshot in full_history["snapshots"]]
print("记录的时间步:", ticks)

# 清空历史记录以节省内存
pop.clear_history()
```

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
# 获取当前状态的详细快照
current_state = pop.output_current_state()
print("当前状态:", current_state)

# 获取可读的字典格式
readable_state = pop.output_current_state(as_dict=True)
print("可读状态:", readable_state)

# 获取JSON格式（便于传输和存储）
json_state = pop.output_current_state(as_json=True)
print("JSON状态:", json_state[:200])  # 显示前200个字符
```

### 数据导出

```python
# 导出为字典格式
data_dict = pop.output_current_state(as_dict=True)

# 导出为JSON格式
json_data = pop.output_current_state(as_json=True, indent=2)

# 保存到文件
import json
with open("population_state.json", "w") as f:
    json.dump(data_dict, f, indent=2)

# 导出观察数据
observation_data = pop.output_current_state(
    observation=observation,
    as_dict=True
)
```

### 与外部工具集成

```python
import pandas as pd
import numpy as np

# 将历史数据转换为pandas DataFrame
def history_to_dataframe(history_data):
    """将历史记录转换为DataFrame"""
    data = []
    for snapshot in history_data["snapshots"]:
        row = {
            "tick": snapshot["tick"],
            "total_population": snapshot["total_count"],
            "females": snapshot["female_count"],
            "males": snapshot["male_count"]
        }
        # 添加观察数据
        if "observed" in snapshot:
            for group_name, count in snapshot["observed"].items():
                row[f"observed_{group_name}"] = count
        data.append(row)

    return pd.DataFrame(data)

# 使用示例
history_df = history_to_dataframe(full_history)
print(history_df.head())
```

## 观察规则详解

### 分组格式

观察规则支持多种分组格式：

#### 1. 字典格式（有名分组）

```python
groups = {
    "all_females": {"sex": "female"},
    "adults": {"age": [2, 3, 4, 5, 6, 7]},
    "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]},
    "juvenile_drive": {
        "genotype": ["WT|Drive"],
        "age": [0, 1],
        "sex": "female"
    },
}
```

#### 2. 模式匹配（推荐）

```python
groups = {
    "target_female": {
        # 有序匹配 Maternal|Paternal
        "genotype": "A1/B1|A2/B2; C1/D1|C2/D2",
        "sex": "female",
    },
    "target_female_unordered": {
        # 无序匹配（同源染色体两条拷贝可交换）
        "genotype": "A1/B1::A2/B2; C1/D1::C2/D2",
        "sex": "female",
    }
}
```

### 选择器格式

#### 基因型选择器

```python
# 字符串（逗号分隔）
{"genotype": "WT|WT"}

# pattern 字符串（推荐）
{"genotype": "A1/B1|A2/B2; C1/D1::C2/D2"}

# 字符串列表
{"genotype": ["WT|WT", "WT|Drive", "Drive|Drive"]}

# 通配符（所有基因型）
{"genotype": "*"}
```

#### 性别选择器

```python
# 字符串
{"sex": "female"}  # 或 {"sex": "male"}

# 整数
{"sex": 0}  # 雌性，{"sex": 1} 雄性

# 列表
{"sex": ["female", "male"]}  # 两性
```

#### 年龄选择器

```python
# 显式列表
{"age": [2, 3, 4]}

# 闭区间元组
{"age": [2, 7]}  # ages 2,3,4,5,6,7

# 区间列表（并集）
{"age": [[0, 1], [4, 6]]}  # ages 0,1,4,5,6

# 可调用对象
{"age": lambda a: a >= 2}  # 年龄大于等于2
```

## 实用示例

### 监控基因驱动传播

```python
# 创建专门监控基因驱动的观察规则
drive_observation = pop.create_observation(
    groups={
        "wild_type": {"genotype": ["WT|WT"]},
        "heterozygotes": {"genotype": ["WT|Drive"]},
        "homozygotes": {"genotype": ["Drive|Drive"]},
        "total_drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]}
    }
)

# 运行模拟并实时监控
for step in range(100):
    pop.run_tick()

    if step % 10 == 0:
        current = pop.output_current_state(observation=drive_observation)
        observed = current["observed"]
        print(f"Step {step}: WT={observed['wild_type']}, "
              f"Het={observed['heterozygotes']}, "
              f"Hom={observed['homozygotes']}")
```

### 年龄结构分析

```python
# 分析不同年龄段的种群动态
age_observation = pop.create_observation(
    groups={
        "juveniles": {"age": [0, 1]},
        "young_adults": {"age": [2, 3]},
        "mature_adults": {"age": [4, 5]},
        "old_adults": {"age": [6, 7]}
    }
)

# 获取历史数据并分析
history = pop.output_history(observation=age_observation)

# 分析年龄结构变化
for snapshot in history["snapshots"]:
    total = sum(snapshot["observed"].values())
    if total > 0:
        juvenile_ratio = snapshot["observed"]["juveniles"] / total
        print(f"幼体比例: {juvenile_ratio:.3f}")
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
增加 `record_every` 间隔，使用 `clear_history()` 定期清理，或导出到外部存储。

### 观察规则会影响模拟性能吗？
观察规则本身不影响模拟性能，但频繁的数据提取和存储可能影响整体性能。

---

本章节介绍了如何从 NATAL Core 模拟中提取和分析数据。在实际项目中，建议先设计合适的观察规则，再根据需求选择合适的数据提取方式。
