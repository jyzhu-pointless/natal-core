# Spatial 模拟指南

本章介绍 SpatialPopulation 的实际用法：先构建一组共享物种和共享配置的 deme，再把它们包装成一个空间容器，最后用拓扑或迁移核控制 deme 间流动。

阅读完成后，你应当能够直接写出下面这类代码：

```python
spatial = SpatialPopulation(
  demes=demes,
  topology=SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
  adjacency=adjacency,
  migration_rate=0.15,
)
```

## 1. 先记住一个真实约束

SpatialPopulation 不是单独创建“新种群”的工厂，它是把已经构建好的 deme 组织起来。

实现里有两个硬要求：

1. 所有 deme 必须共享同一个 Species 对象。
2. 所有 deme 必须共享同一个 config 对象。

所以在 demo 里会先把每个 deme 建好，再做一次 `share_config(demes)`：

```python
shared_config = demes[0].export_config()
for deme in demes[1:]:
  deme.import_config(shared_config)
```

## 2. SpatialPopulation 的核心参数

`SpatialPopulation` 的构造函数支持这几个最常用参数：

- `demes`：已经构建好的 deme 列表。
- `topology`：可选的网格拓扑，常见是 `SquareGrid` 或 `HexGrid`。
- `adjacency`：显式邻接矩阵；如果不传，通常会从 `topology` 推导。
- `migration_kernel`：迁移核，走 kernel 路径时使用。
- `migration_rate`：每步参与迁移的比例。
- `migration_strategy`：`auto`、`adjacency`、`kernel`、`hybrid`，默认 `auto`。
- `kernel_include_center`：kernel 路径下是否把中心格也算进迁移目标。

如果你只想记最重要的规则，可以记成一句：

1. 传 `adjacency` 就走邻接矩阵路径。
2. 传 `migration_kernel` 就走 kernel 路径，而且 topology 必须同时存在。

## 3. 最小可用流程

### 3.1 先构建 deme

每个 deme 都是一个普通 population 实例。最常见的做法是先构建年龄结构种群或离散世代种群，再把它们放进空间容器。

### 3.2 共享 config

```python
shared_config = demes[0].export_config()
for deme in demes[1:]:
  deme.import_config(shared_config)
```

### 3.3 选择迁移路径

你有两条常用路径：

1. adjacency 路径：适合“先定义拓扑，再生成邻接矩阵”的场景。
2. kernel 路径：适合“用一个 3x3 或 5x5 核描述局部迁移权重”的场景。

### 3.4 组装空间容器

```python
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid, build_adjacency_matrix

demes = [deme0, deme1, deme2, deme3]
shared_config = demes[0].export_config()
for deme in demes[1:]:
  deme.import_config(shared_config)

adjacency = build_adjacency_matrix(
  SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False),
  row_normalize=True,
)

spatial = SpatialPopulation(
  demes=demes,
  adjacency=adjacency,
  migration_rate=0.15,
  name="SpatialDemo",
)

spatial.run_tick()
spatial.run(n_steps=5, record_every=1)
```

### 3.5 kernel 路径示例

```python
import numpy as np
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import HexGrid

spatial = SpatialPopulation(
  demes=demes,
  topology=HexGrid(rows=100, cols=100, wrap=False),
  migration_kernel=np.array(
    [
      [0.00, 0.10, 0.05],
      [0.10, 0.00, 0.10],
      [0.05, 0.10, 0.00],
    ],
    dtype=np.float64,
  ),
  migration_rate=0.2,
  name="SpatialHexDemo",
)
```

## 4. 运行时到底做了什么

`run_tick()` 和 `run(...)` 的语义和普通 population 一样，但它们会把所有 deme 一起推进：

1. 先检查每个 deme 是否已经 finished。
2. 读取第一个 deme 的 compiled hooks。
3. 把所有 deme 的 state 拼成一个大数组。
4. 运行空间版内核。
5. 把更新后的 state 再拆回每个 deme。

因此，Spatial 模型的调试顺序也很明确：

- 先用 `run_tick()` 看一步是否正确。
- 再用 `run(...)` 跑多步。
- 如果某个 deme 先 finished，整体 SpatialPopulation 也会停止继续推进。

## 5. migration_rate 和迁移方式

`migration_rate` 是每一步中参与跨 deme 流动的比例。

你可以直接把它理解成：

- 0.0：不迁移。
- 0.1：每步有 10% 的质量参与迁移。
- 1.0：每步全部按邻接/迁移核重新分配。

在实现里，kernel 路径会要求：

- `topology` 存在。
- `migration_kernel` 是二维数组。
- `migration_kernel` 的行和列都必须是奇数。

这就是为什么 demo 里常见 `3x3` 或 `5x5` 的 kernel。

## 6. 该怎么选拓扑

### 6.1 SquareGrid

如果你只想做最容易理解的空间模型，先用 `SquareGrid`。

它适合这两种邻域：

1. `von_neumann`：只连上下左右。
2. `moore`：连上下左右和四个对角。

一个 2x2 的常用例子是：

```python
SquareGrid(rows=2, cols=2, neighborhood="von_neumann", wrap=False)
```

这就是 demo 里最直接的空间迁移写法。

### 6.2 HexGrid

如果你希望每个 deme 有 6 个近邻，且扩散方向更均匀，用 `HexGrid`。

它的特点很简单：

- `wrap=False` 时，边界 deme 的邻居数会少于 6。
- `wrap=True` 时，边界会折回到对侧。

所以在大规模 hex 空间里，常见写法是：

```python
HexGrid(rows=100, cols=100, wrap=False)
```

## 7. WebUI 调试

Spatial 模型可以直接接到 `natal.ui.launch(...)`。

```python
from natal.ui import launch

launch(spatial, port=8080, title="Spatial Debug Dashboard")
```

这个 dashboard 适合做三件事：

1. 看全局总量有没有异常。
2. 看某个 deme 是否出现局部爆发或枯竭。
3. 看迁移后空间梯度是否符合预期。

## 8. 常见错误与排查

### 错误 1：deme 不是同一物种

如果 demes 不是同一个 Species，`SpatialPopulation` 会直接报错。

### 错误 2：config 没有共享

如果每个 deme 的 config 不是同一个对象，`run_tick()` / `run(...)` 会在共享配置检查时失败。

### 错误 3：kernel 维度不对

如果你传了 `migration_kernel`，但它不是奇数维二维数组，构造时就会报错。

### 错误 4：邻接矩阵尺寸不对

`adjacency.shape` 必须等于 `(n_demes, n_demes)`。

## 9. 实际建议

1. 先从 2x2 或 3x3 的小拓扑开始。
2. 先跑 `run_tick()`，确认单步迁移没有问题。
3. 再切到 `run(n_steps=...)` 做批量实验。
4. 如果你要对比空间扩散速度，就只改 `migration_rate`，不要同时改拓扑和 kernel。

## 10. 本章小结

SpatialPopulation 的实际使用顺序可以记成四步：

1. 构建一组共享 species 的 deme。
2. 让这些 deme 共享同一个 config。
3. 选择 adjacency 或 migration_kernel。
4. 用 `run_tick()` 调试，用 `run(...)` 跑批量实验。

---

## 相关章节

- [模拟内核深度解析](simulation_kernels.md)
- [Hook 系统](hooks.md)
- [Modifier 机制](modifiers.md)
- [PopulationState 与 PopulationConfig](population_state_config.md)
