# Spatial 模拟指南

本章介绍 SpatialPopulation 的实际用法：用 SpatialBuilder 快速构建多 deme 种群，配置拓扑与迁移核，控制 deme 间流动。

阅读完成后，可以写出下面这类代码：

```python
spatial = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name=”demo”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 100}, “male”: {“A|A”: 100}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(kernel=my_kernel, migration_rate=0.15)
    .build()
)
```

> **提示**：`SpatialBuilder` 是同构/异构空间种群的首选构造方式。2601 个同构 deme 的构造时间从 ~2.6s 降至 ~16ms。详见 [SpatialBuilder 文档](spatial_builder.md)。

## 两种构造路径

### 推荐：SpatialBuilder（链式 API）

```python
from natal import Species, HexGrid, SpatialPopulation
from natal.spatial_builder import batch_setting

species = Species.from_dict(name=”demo”, structure={“chr1”: {“loc”: [“A”, “B”]}})

# 同构：所有 deme 参数一致
pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10))
    .setup(name=”homo_demo”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 5000}, “male”: {“A|A”: 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(migration_rate=0.1)
    .build()
)

# 异构：通过 batch_setting 为不同 deme 指定不同参数
pop_het = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name=”het_demo”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 5000}, “male”: {“A|A”: 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=batch_setting([10000, 5000, 5000, 8000]))
    .migration(migration_rate=0.1)
    .build()
)
```

### 手动构造（兼容路径）

如果已有独立构建好的 deme 列表，可以直接传给 `SpatialPopulation` 构造函数。所有 deme 必须共享同一个 Species 对象：

```python
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid

shared_config = demes[0].export_config()
for deme in demes[1:]:
    deme.import_config(shared_config)

spatial = SpatialPopulation(
    demes=demes,
    topology=SquareGrid(rows=2, cols=2),
    migration_rate=0.15,
)
```

## SpatialPopulation 的核心参数

`SpatialPopulation` 的构造函数支持这几个最常用参数：

- `demes`：已经构建好的 deme 列表。
- `topology`：可选的网格拓扑，常见是 `SquareGrid` 或 `HexGrid`。
- `adjacency`：显式邻接矩阵；如果不传，通常会从 `topology` 推导。
- `migration_kernel`：迁移核，走 kernel 路径时使用。
- `kernel_bank`：可选的 kernel 集合，用于不同 source deme 使用不同 kernel。
- `deme_kernel_ids`：可选的 per-deme kernel id，索引到 `kernel_bank`。
- `migration_rate`：每步参与迁移的比例。
- `migration_strategy`：`auto`、`adjacency`、`kernel`、`hybrid`，默认 `auto`。
- `kernel_include_center`：kernel 路径下是否把中心格也算进迁移目标。
- `adjust_migration_on_edge`：是否在边界调整迁移量（见「migration_rate 与边界效应」一节），默认 `False`。

最重要的规则：

1. 传 `adjacency` 就走邻接矩阵路径。
2. 传 `migration_kernel` 就走 kernel 路径，而且 topology 必须同时存在。
3. 传 `kernel_bank` + `deme_kernel_ids` 也走 kernel 路径（异构 kernel）。
4. `hybrid` 预留给 adjacency+kernel 混合分流策略，不是异构 kernel 的必选项。

## 链式 API

`SpatialBuilder` 的链式调用流程与 panmictic builder 一致，以下按推荐顺序列出各方法。带 `→` 标记的是空间特有方法，`[B]` 标记的参数接受 `batch_setting`（跨 deme 异构配置）。

```python
pop = (
    SpatialPopulation.builder(species, n_demes=9, topology=SquareGrid(3, 3))
    →                   # 入口：指定 deme 数量和拓扑
    .setup(name=”demo”, stochastic=False, use_continuous_sampling=False)
                        # 基本设定：名称、随机性、采样模式
    .age_structure(n_ages=8, new_adult_age=2)
                        # [仅 age_structured] 年龄分组数、成体起始年龄
    .initial_state(individual_count={“female”: {“A|A”: 500}, “male”: {“A|A”: 500}})
                        # [B] 初始基因型分布
    .survival(female_age_based_survival_rates=[...], ...)
                        # 存活率（age_structured 为年龄向量，discrete 为标量）
    .reproduction(eggs_per_female=50.0, sex_ratio=0.5)
                        # [B] 繁殖参数
    .competition(carrying_capacity=10000, juvenile_growth_mode=”logistic”)
                        # [B] 密度制约
    .presets(HomingDrive(name=”Drive”, ...))
                        # [B] 基因驱动预设
    .fitness(viability={“R2|R2”: 0.0}, mode=”replace”)
                        # [B] 适应性
    .hooks(my_hook)
                        # 生命周期钩子（不接受 batch_setting）
    .migration(kernel=kernel, migration_rate=0.2)
    →                   # [B] 空间特有：迁移核、迁移率
    .build()            # → SpatialPopulation
)
```

各方法的详细参数说明见 [种群初始化](2_population_initialization.md)（setup、initial_state、survival、reproduction、competition）、[Hook 系统](2_hooks.md)、[基因驱动预设](2_genetic_presets.md)。

### 空间特有：`.migration()`

```python
.migration(
    kernel=None,                     # [B] NDArray: 奇数维迁移核
    migration_rate=0.0,             # float: 迁移比例
    strategy=”auto”,                # “auto” | “adjacency” | “kernel” | “hybrid”
    adjacency=None,                 # 显式邻接矩阵
    kernel_bank=None,               # 异构 kernel 集合
    deme_kernel_ids=None,           # per-deme kernel 索引
    kernel_include_center=False,    # 是否包含中心格
    adjust_migration_on_edge=False, # 是否调整边界迁移量
)
```

`kernel` 接受 `batch_setting`，传入 per-deme kernel 列表后自动转换为 `kernel_bank` + `deme_kernel_ids`，等价于手动指定异构 kernel。`kernel_bank` / `deme_kernel_ids` 与 `batch_setting` 互斥。

详见「迁移路径」及「migration_rate 与边界效应」两节。

### 支持 `[B]` 的参数一览

| 方法 | 参数 | 类型 |
|------|------|------|
| `initial_state` | `individual_count` | dict（基因型→数量） |
| `initial_state` | `sperm_storage` | dict |
| `reproduction` | `eggs_per_female` | float |
| `reproduction` | `sex_ratio` | float |
| `competition` | `carrying_capacity` / `age_1_carrying_capacity` | float |
| `competition` | `low_density_growth_rate` | float |
| `competition` | `juvenile_growth_mode` | str |
| `competition` | `expected_num_adult_females` | float |
| `age_structure` | `equilibrium_distribution` | list[float] |
| `presets` | 位置参数 | preset 对象 |
| `fitness` | `viability` / `fecundity` / `sexual_selection` / `zygote_viability` | dict |
| `migration` | `kernel` | NDArray |

以下参数**不接受** `batch_setting`：
- **hooks**：通过 `@hook(deme=...)` 实现 per-deme 选择性执行。
- **空间函数需要 topology**：`(row, col)` 形式要求 builder 传入了 `topology`。`(flat_idx)` 形式不依赖 topology。

## batch_setting 异构配置

`batch_setting` 是 `SpatialBuilder` 的核心机制，允许不同 deme 在同一链式调用中指定不同的参数值。内部通过 config 等价性分组自动优化——相同参数的 deme 共享编译产物，仅 state 数组独立。

### 四种输入形式

```python
from natal.spatial_builder import batch_setting
import numpy as np

# 1. 标量列表（一一对应 n_demes 个 deme）
batch_setting([10000, 5000, 5000, 8000])

# 2. 1D NumPy 数组
batch_setting(np.array([10000, 5000, 5000, 8000]))

# 3. 2D NumPy 数组（形状 = (rows, cols)，按行主序展平）
batch_setting(np.array([[10000, 5000],
                         [5000, 8000]]))

# 4. 空间函数：(flat_idx) -> float 或 (row, col) -> float
batch_setting(lambda i: 10000 if i < 4 else 5000)
batch_setting(lambda r, c: 10000 if r == 0 else 5000)
```

空间函数根据参数个数自动检测：1 个参数传入 `(flat_idx)`，2 个参数传入 `(row, col)`。要求 builder 传入了 `topology` 参数，在 `build()` 时展开求值。

### 模式 1：异构容纳量

```python
pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name=”het_K”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 5000}, “male”: {“A|A”: 5000}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=batch_setting([10000, 5000, 5000, 8000]))
    .migration(migration_rate=0.1)
    .build()
)
# deme 0: K=10000, deme 1: K=5000, deme 2: K=5000, deme 3: K=8000
# builder 自动分组: {10000: [0], 5000: [1,2], 8000: [3]} → 3 组模板
```

### 模式 2：异构初始状态

为每个 deme 指定不同的初始基因型分布，常用于空间驱动释放场景：

```python
from natal.spatial_builder import batch_setting

# 默认所有 deme 只有 WT
n_demes = 100
default_state = {“female”: {“WT|WT”: 500}, “male”: {“WT|WT”: 500}}

# 中心 deme 释放 drive 杂合子
release_state = {“female”: {“WT|WT”: 450, “Dr|WT”: 50},
                 “male”:   {“WT|WT”: 450, “Dr|WT”: 50}}

states = [default_state] * n_demes
states[n_demes // 2] = release_state

pop = (
    SpatialPopulation.builder(species, n_demes=n_demes, topology=HexGrid(10, 10))
    .setup(name=”drive_release”, stochastic=True, use_continuous_sampling=True)
    .initial_state(individual_count=batch_setting(states))
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000, low_density_growth_rate=6,
                 juvenile_growth_mode=”concave”)
    .presets(HomingDrive(name=”Drive”, drive_allele=”Dr”, target_allele=”WT”,
                         resistance_allele=”R2”, functional_resistance_allele=”R1”,
                         drive_conversion_rate=0.95))
    .fitness(fecundity={“R2::!Dr”: 1.0, “R2|R2”: {“female”: 0.0}})
    .migration(kernel=kernel, migration_rate=0.2)
    .build()
)
```

### 模式 3：多个 batch 参数组合

多个 `batch_setting` 同时使用时，builder 按参数值元组计算签名并分组：

```python
pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name=”multi_het”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 500}, “male”: {“A|A”: 500}})
    .reproduction(eggs_per_female=batch_setting([50, 50, 30, 30]))
    .competition(
        carrying_capacity=batch_setting([10000, 5000, 10000, 5000]),
        low_density_growth_rate=batch_setting([6, 6, 4, 4]),
    )
    .migration(migration_rate=0.1)
    .build()
)
# 签名分组:
#   deme 0: (eggs=50, K=10000, r=6)
#   deme 1: (eggs=50, K=5000,  r=6)
#   deme 2: (eggs=30, K=10000, r=4)
#   deme 3: (eggs=30, K=5000,  r=4)
# → 4 个独立组，每组构建一个模板
```

### 模式 4：空间梯度函数

利用 `lambda` 创建平滑的空间梯度（如南北梯度、中心-边缘梯度）：

```python
# 中心高、边缘低的容纳量梯度 —— 使用 (row, col) 两参数签名
def capacity_gradient(r, c):
    center_r, center_c = 4.5, 4.5  # 10x10 网格的中心
    dist = ((r - center_r)**2 + (c - center_c)**2) ** 0.5
    max_dist = (center_r**2 + center_c**2) ** 0.5
    return 10000 * (1 - 0.8 * dist / max_dist)  # 边缘降至 2000

pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10))
    .setup(name=”gradient”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 500}, “male”: {“A|A”: 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=batch_setting(capacity_gradient))
    .migration(migration_rate=0.1)
    .build()
)
```

### 模式 5：异构适应性

不同 deme 可以有不同的 fitness 配置，常用于空间分化的选择压力场景：

```python
pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2))
    .setup(name="het_fitness", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .fitness(viability=batch_setting([
        {"A|A": 1.0},   # deme 0: 正常
        {"A|A": 0.5},   # deme 1: A|A 半致死
        {"A|A": 0.0},   # deme 2: A|A 完全致死
        {"A|A": 1.0},   # deme 3: 正常
    ]))
    .migration(migration_rate=0.1)
    .build()
)
# deme 0 和 3 签名相同 → 共享一组 config
# deme 1 和 2 各自独立重建
```

### 模式 6：异构迁移核

通过 `batch_setting` 为不同 deme 指定不同的迁移核，自动转换为 `kernel_bank` + `deme_kernel_ids`：

```python
import numpy as np

# 两个不对称核：右向偏移和左向偏移
right_kernel = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
left_kernel  = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

pop = (
    SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(1, 4))
    .setup(name="het_kernel", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(kernel=batch_setting([right_kernel, left_kernel, right_kernel, left_kernel]),
               migration_rate=0.5)
    .build()
)
# 等价于：
#   .migration(kernel_bank=(right_kernel, left_kernel),
#              deme_kernel_ids=np.array([0, 1, 0, 1]),
#              migration_rate=0.5)
```

## 迁移路径

### Kernel 路径

```python
import numpy as np
from natal import Species, SpatialPopulation, HexGrid

species = Species.from_dict(name=”hex_demo”, structure={“chr1”: {“loc”: [“A”, “B”]}})

spatial = (
    SpatialPopulation.builder(species, n_demes=10000, topology=HexGrid(100, 100))
    .setup(name=”SpatialHexDemo”, stochastic=False)
    .initial_state(individual_count={“female”: {“A|A”: 100}, “male”: {“A|A”: 100}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=10000)
    .migration(
        kernel=np.array(
            [[0.00, 0.10, 0.05],
             [0.10, 0.00, 0.10],
             [0.05, 0.10, 0.00]],
            dtype=np.float64,
        ),
        kernel_include_center=False,
        migration_rate=0.2,
        adjust_migration_on_edge=False,
    )
    .build()
)
```

### 异构 Kernel（Kernel Bank）

不同 source deme 可以使用不同的迁移核。通过 `kernel_bank` + `deme_kernel_ids` 实现：

```python
right_only = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
left_only  = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

spatial = SpatialPopulation(
    demes=demes,
    topology=SquareGrid(rows=1, cols=3),
    kernel_bank=(right_only, left_only),
    deme_kernel_ids=np.array([0, 1, 0], dtype=np.int64),
    migration_rate=1.0,
)
```

每个 source deme 通过 `deme_kernel_ids[src]` 选择自己的 kernel。内部按 kernel 分组构建 offset table，迁移时在 `prange` 内按 `deme_kernel_ids[src]` 查表——不会预构建 `O(n_demes²)` 的稠密邻接矩阵。

## 运行模拟

`SpatialPopulation` 继承 `BasePopulation` 的全部运行接口，语义与 panmictic 种群一致，但操作作用于所有 deme。

### 单步与批量运行

```python
# 单步推进
pop.run_tick()

# 批量运行 100 步
pop.run(100)

# 带记录的批量运行
pop.run(500, record_every=5)
```

`SpatialPopulation.run()` 的 `record_every` 参数控制 Numba 编译内核中的历史采样间隔。设为 0 表示不记录历史。

### 访问聚合状态

```python
# 跨 deme 汇总
pop.total_population_size   # 总个体数
pop.total_females           # 总雌性数
pop.total_males             # 总雄性数
pop.sex_ratio               # 性别比例（雌/雄）
pop.tick                    # 当前时间步

# 等位基因频率（全空间汇总）
freqs = pop.compute_allele_frequencies()

# 汇总个体数张量（所有 deme 求和）
aggregate = pop.aggregate_individual_count()
```

### 访问单个 deme

```python
# 按索引获取 deme
deme_0 = pop.deme(0)
print(deme_0.total_population_size)
print(deme_0.compute_allele_frequencies())

# 遍历所有 deme
for i in range(pop.n_demes):
    d = pop.deme(i)
    print(f"deme {i}: {d.total_population_size}")
```

每个 deme 是 `AgeStructuredPopulation` 或 `DiscreteGenerationPopulation` 实例，支持 `output_current_state()`、`compute_allele_frequencies()` 等全部 panmictic 接口。

### 重置与控制

```python
# 重置所有 deme 到初始状态
pop.reset()

# 检查是否已终止
if pop.is_finished:
    print("模拟已终止")

# 手动终止
pop.finish_simulation()
```

### 数据输出

`output_current_state()` 和 `output_history()` 的用法与 panmictic 种群一致，支持观察规则筛选：

```python
# 当前状态快照
state = pop.output_current_state()

# 带观察规则的历史导出
observation = pop.create_observation(
    groups={"adult_wt": {"genotype": ["WT|WT"], "age": [2]}},
    collapse_age=True,
)
history = pop.output_history(observation=observation)
```

详细用法见 [提取种群模拟数据](2_data_output.md)。

### 运行时内部流程

每次 `run_tick()` 的内部执行顺序：

1. 检查每个 deme 是否已经 `is_finished`。
2. 把所有 deme 的 state 拼成统一数组，构建 config bank。
3. 运行 Numba 编译的空间生命周期包装器：`prange` 并行执行各 deme 生命周期 → 统一迁移。
4. 将更新后的 state 写回每个 deme。

如果一个 deme 先触发终止条件（如种群灭绝），整个 `SpatialPopulation` 也会停止推进。详细执行流程见 [空间生命周期包装器](spatial_lifecycle_wrapper.md)。

## migration_rate 与边界效应

### migration_rate

`migration_rate` 控制每一步中参与跨 deme 流动的质量比例：

- `0.0`：不迁移。
- `0.1`：每步有 10% 的质量参与迁移。
- `1.0`：每步全部按邻接/迁移核重新分配。

### 边界效应与 adjust_migration_on_edge

当 topology 的 `wrap=False` 时，边界 deme 的有效邻居数少于内部 deme。`adjust_migration_on_edge` 控制如何处理这种差异：

| `adjust_migration_on_edge` | 行为 |
|---|---|
| `False`（默认） | 边界 deme 自然迁出更少。每个邻居的迁移概率 = `weight / kernel_total_sum`，总迁移量正比于有效邻居数 |
| `True` | 所有 deme 迁出相同总量。每个邻居的迁移概率 = `weight / effective_sum`（归一化到 1.0） |

其中 `kernel_total_sum` 是 kernel 中所有正向权重的总和，作为统一的缩放参考基准。

**实际影响**：

```python
# 3x3 kernel，中心权重 0，周围权重 1.0
# kernel_total_sum = 8.0

# 默认行为（adjust_migration_on_edge=False）：
#   内部 deme（8 个邻居）：每个邻居概率 = 1.0/8.0 = 0.125，总迁移 = rate * 1.0
#   角落 deme（3 个邻居）：每个邻居概率 = 1.0/8.0 = 0.125，总迁移 = rate * 0.375
#   → 边界迁出更少，更符合生物直觉

# 调整行为（adjust_migration_on_edge=True）：
#   内部 deme（8 个邻居）：每个邻居概率 = 1.0/8.0 = 0.125，总迁移 = rate * 1.0
#   角落 deme（3 个邻居）：每个邻居概率 = 1.0/3.0 ≈ 0.333，总迁移 = rate * 1.0
#   → 所有 deme 迁出相同总量，边界效应被人为抹平
```

**特殊情况**：当 `topology.wrap=True` 时，所有 deme 都有相同数量的有效邻居，两种模式行为一致。

### 非均匀权重 Kernel

当 kernel 中的权重不全是 1 时（如高斯核），`kernel_total_sum` 保留了 kernel 的相对权重结构：

```python
# 5x5 高斯核：中心权重高，边缘权重低
# kernel_total_sum 是所有权重的总和
#
# 内部 deme（25 个邻居全有效）：
#   每个邻居概率 = weight / kernel_total_sum
#   总迁移率 = rate * (effective_sum / kernel_total_sum) = rate * 1.0
#
# 边界 deme（如 15 个有效邻居）：
#   每个邻居概率 = weight / kernel_total_sum  (相对权重不变)
#   总迁移率 = rate * (effective_sum / kernel_total_sum) ≈ rate * 0.6
```

### 内核实现

关于 kernel offset table、`kernel_total_sum` 的计算、以及 `adjust_on_edge` 在 `prange` 中的具体实现，见 [Migration Kernel 底层实现](migration_kernel_impl.md)。

## 迁移核的数学形式

一个迁移核 $K$ 是奇数维矩阵，中心位于 $(\lfloor R/2 \rfloor, \lfloor C/2 \rfloor)$。对 source deme 坐标 $(r_s, c_s)$，每个非零核权重 $K_{i,j} > 0$ 对应一个潜在目标坐标：

$$(r_d, c_d) = (r_s + (i - i_c),\; c_s + (j - j_c))$$

其中 $(i_c, j_c)$ 是核中心的矩阵坐标。落入网格内的坐标成为有效邻居；超出网格的坐标在 `wrap=False` 时被丢弃，在 `wrap=True` 时取模折回。

源 deme 向邻居 $n$ 迁出的概率由 `adjust_migration_on_edge` 决定：

$$p_n = \frac{w_n}{S_{\text{ref}}}, \quad S_{\text{ref}} = \begin{cases} \sum_{m} w_m & \text{(adjust=True，按有效邻居归一化)} \\ \sum_{i,j} K_{i,j} & \text{(adjust=False，按核总和缩放)} \end{cases}$$

其中 $\sum_{i,j} K_{i,j}$ 是核所有权重之和（记为 `kernel_total_sum`），$\sum_m w_m$ 是当前 deme 实际有效邻居的权重之和。在 `adjust=False` 下，边界 deme 的总迁出量为 $r \cdot \frac{\sum_m w_m}{\sum_{i,j} K_{i,j}}$，自然小于内部 deme。

### 构造常用 Kernel

NATAL 提供 `build_gaussian_kernel()` 工厂函数，自动根据拓扑类型使用正确的距离度量：

```python
from natal.spatial_topology import build_gaussian_kernel, HexGrid, SquareGrid

# 六边形网格高斯核 —— 自动使用余弦定理距离公式
hex_kernel = build_gaussian_kernel(HexGrid, size=11, sigma=1.5)

# 方形网格高斯核 —— 使用 Cartesian 距离
square_kernel = build_gaussian_kernel(SquareGrid, size=7, sigma=2.0)

# 字符串简写
hex_kernel = build_gaussian_kernel("hex", size=11, sigma=1.5)

# 通过 mean_dispersal 指定平均扩散距离（更直观）
# sigma = mean_dispersal / sqrt(π/2)
hex_kernel = build_gaussian_kernel("hex", size=11, mean_dispersal=2.0)
```

`sigma` 与 `mean_dispersal` 互斥。2D 各向同性高斯分布中，平均位移遵循 Rayleigh 分布：
$\bar{d} = \sigma\sqrt{\pi/2}$。

也可以手动构造 kernel（兼容旧代码）：

```python
import numpy as np

# von Neumann 3x3（4 邻居，不含中心）
von_neumann = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

# Moore 3x3（8 邻居，不含中心）
moore = np.ones((3, 3), dtype=np.float64)
moore[1, 1] = 0.0
```

## 拓扑结构

NATAL 提供两种网格拓扑：`SquareGrid` 和 `HexGrid`。两者共享相同的坐标系统——deme 按行主序排列，扁平索引与网格坐标的转换关系为：

$$i_{\text{flat}} = r \cdot \text{cols} + c, \qquad (r, c) = (i_{\text{flat}} \mathbin{//} \text{cols},\; i_{\text{flat}} \bmod \text{cols})$$

边界行为由 `wrap` 参数统一控制，对所有邻居偏移生效：

$$\text{normalize}(r, c) = \begin{cases} (r \bmod R,\; c \bmod C) & \text{wrap=True} \\ \text{None（丢弃）} & \text{wrap=False 且坐标越界} \end{cases}$$

### SquareGrid

```python
SquareGrid(rows=R, cols=C, neighborhood="moore", wrap=False)
```

**Von Neumann 邻域**（`neighborhood="von_neumann"`）：4 方向偏移

$$\Delta = \{(-1,0),\;(1,0),\;(0,-1),\;(0,1)\}$$

**Moore 邻域**（`neighborhood="moore"`，默认）：8 方向偏移

$$\Delta = \{(-1,-1),(-1,0),(-1,1),\;(0,-1),(0,1),\;(1,-1),(1,0),(1,1)\}$$

### HexGrid

```python
HexGrid(rows=R, cols=C, wrap=False)
```

HexGrid 使用平行四边形坐标 $(i, j)$，6 个邻居的偏移固定为：

$$\Delta = \{(1,0),\;(0,1),\;(-1,1),\;(-1,0),\;(0,-1),\;(1,-1)\}$$

平面嵌入采用 pointy-top 六边形：

$$x = i + 0.5j, \qquad y = \frac{\sqrt{3}}{2}\,j$$

六个邻居在嵌入空间中与源 deme 等距，扩散各向同性优于 SquareGrid。

### 边界条件下邻居数量

令 $N_{\text{max}}$ 为网格内部 deme 的最大邻居数（SquareGrid 为 4 或 8，HexGrid 为 6），$(r, c)$ 为网格坐标。

**wrap=False** 时，坐标越界的邻居被丢弃，边界 deme 的邻居数 $N_{\text{eff}}(r, c) < N_{\text{max}}$。角落位置的邻居数最少：

| 拓扑 | 邻域 | 内部 | 边 | 角 |
|------|------|------|-----|-----|
| SquareGrid | von_neumann | 4 | 3 | 2 |
| SquareGrid | moore | 8 | 5 | 3 |
| HexGrid | — | 6 | 4 或 5 | 3 或 4 |

**wrap=True** 时，坐标取模折回，$N_{\text{eff}}(r, c) = N_{\text{max}}$ 对所有位置成立。

### 选择指南

| 场景 | 推荐拓扑 |
|------|---------|
| 快速原型、与邻接矩阵模式混用 | `SquareGrid` + `von_neumann` |
| 更丰富的局部连接 | `SquareGrid` + `moore` |
| 各向同性扩散、大规模空间模拟 | `HexGrid` |
| 消除边界伪影 | 任一拓扑 + `wrap=True` |
| 保留边界自然效应 + 边界感知迁移 | 任一拓扑 + `wrap=False` + `adjust_migration_on_edge=False` |

### 完整示例：SquareGrid

```python
import numpy as np
from natal import Species, SpatialPopulation, SquareGrid

species = Species.from_dict(name="sq", structure={"chr1": {"loc": ["A", "B"]}})

kernel = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)

pop = (
    SpatialPopulation.builder(species, n_demes=9, topology=SquareGrid(3, 3,
        neighborhood="von_neumann", wrap=False))
    .setup(name="square_demo", stochastic=False)
    .initial_state(individual_count={"female": {"A|A": 500}, "male": {"A|A": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000)
    .migration(kernel=kernel, migration_rate=0.2, adjust_migration_on_edge=False)
    .build()
)

pop.run(10)
```

### 完整示例：HexGrid

```python
from natal import Species, SpatialPopulation, HexGrid
from natal.spatial_topology import build_gaussian_kernel

species = Species.from_dict(name="hex", structure={"chr1": {"loc": ["WT", "Dr"]}})

# 使用 build_gaussian_kernel 自动处理 hex 坐标的距离度量
kernel = build_gaussian_kernel(HexGrid, size=11, sigma=1.5)

pop = (
    SpatialPopulation.builder(species, n_demes=100, topology=HexGrid(10, 10, wrap=False))
    .setup(name="hex_demo", stochastic=True, use_continuous_sampling=True)
    .initial_state(individual_count={"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50)
    .competition(carrying_capacity=1000, low_density_growth_rate=6, juvenile_growth_mode="concave")
    .migration(kernel=kernel, migration_rate=0.5)
    .build()
)

pop.run(10)
```

## WebUI 调试

Spatial 模型可以直接接到 `natal.ui.launch(...)`。

```python
from natal.ui import launch

launch(spatial, port=8080, title="Spatial Debug Dashboard")
```

## 常见错误与排查

### 错误 1：deme 不是同一物种

如果 demes 不是同一个 Species，`SpatialPopulation` 会直接报错。

### 错误 2：deme 间迁移采样模式不一致

支持异构 deme config。但当迁移开启时，所有 deme 的
`is_stochastic` 与 `use_continuous_sampling` 必须保持一致；
否则 `run_tick()` / `run(...)` 会报错。

### 错误 3：kernel 维度不对

传入的 `migration_kernel` 如果不是奇数维二维数组，构造时会报错。

### 错误 4：邻接矩阵尺寸不对

`adjacency.shape` 必须等于 `(n_demes, n_demes)`。

### 错误 5：kernel_bank 与 topology 不匹配

异构 kernel（`kernel_bank` + `deme_kernel_ids`）走 kernel 路径，要求 `topology` 必须存在。如果只传了 `kernel_bank` 但没有 `topology`，构造时会报错。

## 本章小结

SpatialPopulation 的实际使用顺序可以记成四步：

1. 用 `SpatialPopulation.builder(...)` 开始链式构造。
2. 可以使用异构 deme config（`batch_setting`），但迁移采样模式要在各 deme 之间保持一致。
3. 选择 adjacency 或 migration_kernel；需要边界感知时用 `adjust_migration_on_edge`。
4. 用 `run_tick()` 调试，用 `run(...)` 跑批量实验。

---

## 相关章节

- [SpatialBuilder：批量构造](spatial_builder.md)
- [空间生命周期包装器](spatial_lifecycle_wrapper.md)
- [Migration Kernel 底层实现](migration_kernel_impl.md)
- [模拟内核深度解析](4_simulation_kernels.md)
- [Hook 系统](2_hooks.md)
