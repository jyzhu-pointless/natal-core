# 种群模型

`Population` 类是 NATAL Core 的核心组件，负责管理种群的遗传状态和模拟过程。

## 种群类型

NATAL Core 提供两种主要的种群类型：

### 离散世代种群
`DiscreteGenerationPopulation` 适用于世代不重叠的物种，每代完全替换，模拟过程简单高效。

### 年龄结构化种群
`AgeStructuredPopulation` 适用于世代重叠的物种，支持年龄依赖的生存和繁殖力，可配置精子储存机制。

> 这两种种群均为 `BasePopulation` 的子类，共享大多数方法。

## 创建种群

推荐通过链式 API 创建种群，具体方法参见 [种群初始化](2_population_initialization.md)。

```python
import natal as nt

# 创建年龄结构化种群
pop = (
    nt.AgeStructuredPopulation.setup(species)
    .name("MyExperiment")
    .age_structure(n_ages=8)
    .initial_state({"WT|WT": 1000})
    .build()
)

# 创建离散世代种群
pop = (
    nt.DiscreteGenerationPopulation.setup(species)
    .name("DiscreteExp")
    .initial_state({"WT|WT": 500})
    .build()
)
```

## 启动模拟

### 单步模拟

```python
# 模拟一步（一个时间单位）
pop.run_tick()

# 模拟多步，打印每步状态
for _ in range(100):
    pop.run_tick()
    print(pop.output_current_state())
```

### 批量模拟

```python
# 模拟100步
pop.run(100)
# 或
pop.run(n_steps=100)
```

## 访问种群状态

### 当前状态信息

```python
# 种群大小
current_size = pop.total_population_size
print(f"当前种群大小: {current_size}")

# 雌性数量
female_count = pop.total_females
print(f"雌性数量: {female_count}")

# 雄性数量
male_count = pop.total_males
print(f"雄性数量: {male_count}")

# 性别比例
ratio = pop.sex_ratio
print(f"性别比例（雌性/雄性）: {ratio}")

# 当前时间步
current_tick = pop.tick
print(f"当前时间步: {current_tick}")
```

### 等位基因频率

```python
# 计算等位基因频率
allele_freqs = pop.compute_allele_frequencies()
print("等位基因频率:", allele_freqs)

# 获取特定等位基因频率
drive_freq = allele_freqs.get("D", 0.0)
print(f"驱动等位基因频率: {drive_freq}")
```

## 历史记录系统

### 历史记录配置

种群对象内置历史记录功能，可配置记录频率和存储格式：

```python
# 配置历史记录
pop.record_every = 10  # 每10步记录一次
pop.max_history = 1000  # 最多保存1000个快照

# 运行模拟并记录历史
results = pop.run(n_steps=500, record_every=5)
```

### 历史数据访问

```python
# 获取完整历史记录
full_history = pop.output_history()
print("历史记录数量:", len(full_history["snapshots"]))
print("最后一步数据:", full_history["snapshots"][-1])

# 获取特定时间步的历史
history_at_tick_100 = pop.output_history(tick=100)
print("第100步的状态:", history_at_tick_100)

# 获取历史记录的时间步列表
ticks = [snapshot["tick"] for snapshot in full_history["snapshots"]]
print("记录的时间步:", ticks)
```

### 历史记录管理

```python
# 清空历史记录以节省内存
pop.clear_history()

# 重新开始记录
results = pop.run(n_steps=100, record_every=5)
```

## 输出功能

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

### 与观察规则集成

结合观察规则可从种群状态中提取特定子群体的数据，详细方法参见 [提取种群模拟数据](2_data_output.md)。

```python
# 创建观察规则
observation = pop.create_observation(
    groups={
        "adult_wt": {"genotype": ["WT|WT"], "age": [1]},
        "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]}
    },
    collapse_age=False,
)

# 使用观察规则获取当前状态
current = pop.output_current_state(observation=observation)
print("当前观察数据:", current["observed"])

# 使用观察规则获取历史数据
history = pop.output_history(observation=observation)
print("历史观察数据:", history["observed"])
```

## 重置和重新开始

```python
# 重置到初始状态
pop.reset()

# 重置后重新模拟
pop.reset()
results = pop.run(n_steps=50)
```

## 模拟控制

### 检查模拟状态

```python
# 检查模拟是否完成
if pop.is_finished:
    print("模拟已完成")
else:
    print("模拟仍在进行中")

# 手动结束模拟
pop.finish_simulation()
```
