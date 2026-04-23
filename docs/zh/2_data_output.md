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
