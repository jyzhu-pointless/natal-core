# Spatial 模拟指南

<!-- TODO: 写得太抽象 -->

本章面向建模用户，介绍如何在 NATAL 中组织空间结构模拟。

阅读完成后，你应当能够：

1. 理解 Spatial 模型在业务层面的含义。
2. 正确构建 demes、拓扑和迁移参数。
3. 使用 `run_tick()` 与 `run(...)` 完成单步与批量模拟。
4. 处理常见输入错误和运行错误。

## 1. Spatial 模型是什么

Spatial 模型可以理解为“多个局部种群（deme）组成的网络系统”。

每个 deme 都有自己的状态；在每个 tick 中：

1. 先在 deme 内部完成本地演化。
2. 再根据拓扑关系在 deme 之间迁移。

因此，Spatial 模型同时包含两类动力学：

- 局部动力学：每个 deme 的繁殖、生存、衰老。
- 空间动力学：deme 之间的人口流动。

## 2. 核心对象

空间模拟通常围绕以下对象组织：

- `SpatialPopulation`：空间容器，持有全部 demes。
- `topology` 或 `adjacency`：定义 deme 之间的连接关系。
- `migration_rate`：控制每步迁移强度。

你可以把 `SpatialPopulation` 理解为“把多个普通 population 协同推进”的调度层。

## 3. 最小构建流程

### 3.1 准备 demes

每个 deme 应该是可独立运行的 population 对象（例如年龄结构模型实例）。

### 3.2 构建拓扑

可使用网格拓扑工具生成邻接矩阵。

### 3.3 组装空间容器并运行

```python
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid, build_adjacency_matrix

# 1) 准备多个 deme（示意）
demes = [deme0, deme1, deme2, deme3]

# 2) 生成拓扑邻接
grid = SquareGrid(rows=2, cols=2, neighborhood="moore", wrap=False)
adjacency = build_adjacency_matrix(grid, row_normalize=True)

# 3) 创建 SpatialPopulation
spatial_pop = SpatialPopulation(
    demes=demes,
    adjacency=adjacency,
    migration_rate=0.1,
)

# 4) 运行
spatial_pop.run_tick()
spatial_pop.run(n_steps=50, record_every=10)
```

如果希望使用基于卷积核的迁移，而不是显式构造整个邻接矩阵，也可以这样写：

```python
import numpy as np
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid

grid = SquareGrid(rows=3, cols=3, neighborhood="von_neumann", wrap=False)

spatial_pop = SpatialPopulation(
    demes=demes,
    topology=grid,
    migration_kernel=np.array(
        [
            [0.05, 0.10, 0.05],
            [0.10, 0.00, 0.10],
            [0.05, 0.10, 0.05],
        ],
        dtype=float,
    ),
    migration_rate=0.2,
)
```

## 4. 运行语义

### 4.1 `run_tick()`

用于单步推进，适合调试和过程观察。

### 4.2 `run(n_steps=...)`

用于批量推进，适合正式实验与参数扫描。

实践建议：

- 开发期先用 `run_tick()` 验证规则。
- 实验期改用 `run(...)` 并设置合适 `record_every`。

## 4.3 WebUI 调试

Spatial 模型现在可以直接复用 `natal.ui.launch(...)` 启动专用 dashboard：

```python
from natal.ui import launch

launch(spatial_pop, port=8080, title="Spatial Debug Dashboard")
```

当前 spatial dashboard 适合开发期调试，重点能力包括：

- 查看 square / hex 几何 landscape（颜色深浅映射 deme 人口规模）
- 查看全局 total population 与 allele frequency
- 点击单个 deme，下钻观察该 deme 的状态（总量、性别、年龄分层、基因型与 fitness）
- 查看当前选中 deme 的迁移权重

当 grid 很大时，dashboard 会自动切换到按 `(row, col)` 展示的 heatmap 模式，而不是逐个 polygon 绘制 landscape。这样可以显著降低 Plotly trace 数量，避免例如 `50x50` 网格在浏览器端失去响应。

## 5. migration_rate 如何理解

`migration_rate` 表示每步中参与跨 deme 流动的人口比例。

直观上：

- `migration_rate = 0`：无迁移，仅局部演化。
- `migration_rate = 1`：完全按邻接权重重新分配。
- `0 < migration_rate < 1`：本地保留与跨 deme 迁移同时发生。

这通常是空间扩散速度与混合程度的关键参数。

### 5.1 migration 内核复杂度（实现细节）

当前 migration 实现不是对每个 source 扫描全部 destination，而是先把每个 source 的路由行压缩为“稀疏 destination 列表”，再只遍历有效 destination：

1. 先构建每个 source 的有效 destination 索引与概率。
2. 对每个 source 的每个状态桶（individual/sperm）只在有效 destination 上分配 outbound 质量。
3. stay 质量直接留在 source，不再经过 destination 全扫描。

这意味着：

- kernel 模式（邻域大小固定、`kernel_nonzero` 近似常数）下，复杂度接近 `O(n_demes * kernel_nonzero)`，可近似线性于 `n_demes`。
- adjacency 模式下，复杂度接近 `O(total_nonzero_edges)`，取决于邻接矩阵稀疏度。

换言之，只要迁移图是稀疏的，migration 阶段就不会再出现“每步全量 destination 扫描”的二次复杂度瓶颈。

## 6. 拓扑如何选

NATAL 目前提供两类规则网格拓扑：

1. `SquareGrid`
2. `HexGrid`

两者都可以配合 `wrap=True/False` 使用，但它们的几何含义不同。

如果只想快速选型，可以先看下面这张表：

| 拓扑 | 每个内部 deme 的典型邻居数 | 坐标复杂度 | 适合场景 |
|---|---:|---|---|
| `SquareGrid(von_neumann)` | 4 | 低 | 只关心上下左右扩散 |
| `SquareGrid(moore)` | 8 | 低 | 希望把对角扩散也算作近邻 |
| `HexGrid` | 6 | 中 | 希望邻居方向更均匀、各向同性更强 |

可以先按一个简单经验选择：

1. 只想快速建模，用 `SquareGrid`。
2. 更在意六方向均匀扩散，用 `HexGrid`。

### 6.1 SquareGrid：方格拓扑

`SquareGrid` 比较直接，因为它的存储坐标 `(row, col)` 和几何直觉基本一致。

常见邻域有两种：

1. `von_neumann`
   - 只连上下左右四个方向
2. `moore`
   - 连上下左右以及四个对角，总共八个方向

可以粗略理解为：

```text
von_neumann           moore

  ●                    ● ● ●
● ○ ●                  ● ○ ●
  ●                    ● ● ●
```

这里：

- `○` 是当前 deme
- `●` 是邻居

对 `SquareGrid` 来说，几何并不复杂：

- `(row, col)` 本身就足够表达邻接
- 不需要像 HexGrid 那样再引入另一套计算坐标

因此在阅读和调试上，`SquareGrid` 通常是更直接的起点。

### 6.2 HexGrid：六边形拓扑

`HexGrid` 的邻接更适合模拟“每个位置有六个近邻”的扩散系统。

和 `SquareGrid` 相比，它的难点不在于邻居数量，而在于：

- 六边形很难直接无失真地塞进规则矩形数组
- 因此实现上需要区分“存储坐标”和“计算坐标”

如果你觉得 `HexGrid` 看起来更复杂，原因基本都来自这一点：同一个 deme 需要同时用“好存储”的坐标和“好计算”的坐标来表达。

### 6.3 HexGrid 的三层表示

可以把 `HexGrid` 理解成三层表示协同工作。

#### 6.3.1 Axial `(q, r)`：计算层

- 这是六边形网格最自然的计算坐标之一。
- 它不是普通直角网格，而是适配六边形邻接关系的斜坐标表示。
- 在这个坐标系里，六个邻居方向始终固定为：
  - `(1, 0)`
  - `(1, -1)`
  - `(0, -1)`
  - `(-1, 0)`
  - `(-1, 1)`
  - `(0, 1)`

因此：

- 邻居查找适合在 axial 中做。
- 路径、距离、扩散方向等“运动逻辑”也适合在 axial 中做。

#### 6.3.2 Offset `(row, col)`：存储层

- 这是把 hex grid 放进二维数组时最方便的表示。
- 好处是每一行长度相同，可以直接用 `grid[row][col]` 这样的方式存放。
- 缺点是它不是几何上最自然的坐标系，需要额外处理行错位。

因此：

- offset 更适合存储
- axial 更适合计算

这是理解后面所有实现细节的关键分工。

#### 6.3.3 Odd-r：当前项目使用的 offset 布局

- odd-r 是 offset 坐标的一种具体布局。
- `r` 表示按“行”组织。
- `odd` 表示奇数行在几何上相对偶数行右移半格。

示意如下：

```text
● ● ● ●
  ● ● ● ●
● ● ● ●
  ● ● ● ●
```

更具体一点，可以想成：

```text
row=0     (0,0)   (0,1)   (0,2)   (0,3)
row=1       (1,0)   (1,1)   (1,2)   (1,3)
row=2     (2,0)   (2,1)   (2,2)   (2,3)
row=3       (3,0)   (3,1)   (3,2)   (3,3)
```

可以看到：

- 偶数行（0, 2, ...）不偏移
- 奇数行（1, 3, ...）在几何上向右错半格

### 6.4 HexGrid 的核心计算流程

`HexGrid` 最核心的一句话是：

```text
offset -> axial -> 计算 -> offset
```

也就是：

1. 先把 `(row, col)` 转成 axial `(q, r)`
2. 在 axial 坐标里做六方向计算
3. 再把结果转回 `(row, col)`
4. 最后再处理边界条件

这条路线比“直接在 `(row, col)` 上分别处理奇偶行邻居”更稳定，也更容易验证。

axial 坐标里的 6 个标准方向可以画成：

```text
            (-1, 1)   (0, 1)
        (-1, 0)   [center]   (1, 0)
            (0, -1)   (1, -1)
```

### 6.5 为什么 HexGrid 不会“走偏”

一个常见疑问是：

- 如果不断沿某个方向移动，会不会因为 odd-r 的错位而逐渐漂移？

答案是：当前实现不会，因为真正的方向推进发生在 axial 层，而不是 offset 层。

例如，在 axial 中连续沿 `(0, 1)` 移动：

```text
(q, r)
-> (q, r + 1)
-> (q, r + 2)
```

这在计算语义里是一致的，不会因为 odd-r 的行错位而累积误差。

所以更准确地说：

- odd-r 的“错半格”只是一种存储和显示效果
- 实际邻居关系与运动方向由 axial 层保证

### 6.6 `sqrt(3) / 2` 是做什么的

实现里会出现：

```text
y = (sqrt(3) / 2) * r
```

它只用于几何映射，不参与邻居计算。

它的作用是：

- 把 axial 坐标映射到二维平面
- 保持相邻 hex center 距离为 1
- 保持正六边形的几何比例

它主要用于：

- `to_xy()`
- `neighbor_direction_vectors()`

因此可以把三层结构总结为：

```text
几何层: (x, y)
  - 用于几何可视化和方向向量

计算层: axial (q, r)
  - 用于邻居、路径、方向、迁移

存储层: offset (row, col)
  - 用于矩形数组存储
```

如果只记一句话，可以记：

`SquareGrid` 基本直接在 `(row, col)` 上理解；`HexGrid` 则要分清“存储层”和“计算层”。

## 7. wrap 如何理解

`wrap` 用来控制网格边界是否采用周期边界条件。

可以把它理解为：当一个 deme 的邻居越过网格边界时，是否从对侧重新进入。

### 7.1 `wrap=False`

- 越界邻居会被直接丢弃
- 边界和角落的 deme 邻居数量会少于内部 deme
- 这种设置适合表示真实有限区域，有明显边界效应

以 `SquareGrid(rows=3, cols=3, neighborhood="von_neumann")` 为例，左上角 `(0, 0)`：

- 不 wrapping 时，它只有两个邻居：`(1, 0)` 和 `(0, 1)`

### 7.2 `wrap=True`

- 越界邻居会按取模规则折回另一侧
- 左右边界相连，上下边界也相连
- 这种设置适合弱化边界效应，把有限网格近似成“重复平铺”的空间

同样以上面的 `SquareGrid` 为例，左上角 `(0, 0)`：

- wrapping 后，向上越界会连接到最后一行
- 向左越界会连接到最后一列
- 因此它会得到四个邻居：`(2, 0)`、`(1, 0)`、`(0, 2)`、`(0, 1)`

### 7.3 HexGrid 中的 `wrap`

对 `HexGrid` 来说，`wrap` 的语义与 `SquareGrid` 相同，区别只在于邻居不是直接在 `(row, col)` 上硬算，而是：

1. 先按 axial 的六个方向生成候选邻居
2. 再把结果转回 `(row, col)`
3. 最后对越界坐标做 wrap

因此：

- `wrap=False` 时，边界 hex deme 的邻居数可能少于 6
- `wrap=True` 时，边界 hex deme 仍然可以保持 6 个邻居

举例说，若某个候选邻居最终得到 `(-1, 0)` 或 `(0, -1)`：

- `wrap=False` 时会被丢弃
- `wrap=True` 时会被折回，如：
  - `(-1, 0) -> (rows - 1, 0)`
  - `(0, -1) -> (0, cols - 1)`

## 8. Hook 在 Spatial 中的使用

用户使用方式与普通 population 一致：定义 Hook、注册 Hook、运行模拟。

建议：

1. 先在单 deme 场景验证 Hook 行为。
2. 再迁移到 Spatial 场景观察整体效应。
3. 对关键事件（release、suppression、stop 条件）保留可读命名。

## 9. 常见错误与排查

### 错误 1：deme 类型错误

现象：构造 `SpatialPopulation` 时提示 deme 不是合法 population 对象。

排查：确认 demes 列表中的每个元素都是标准 population 实例。

### 错误 2：邻接矩阵维度不匹配

现象：`adjacency` 形状与 deme 数量不一致。

排查：确认 `adjacency.shape == (n_demes, n_demes)`。

### 错误 3：运行时拒绝继续

现象：某些 deme 已结束，导致空间容器无法继续推进。

排查：检查每个 deme 的运行状态；必要时重新初始化。

## 10. 参数设置建议

1. 从小规模拓扑开始（2x2 或 3x3），先确认动力学方向正确。
2. 再增加 deme 数量，观察迁移是否造成预期的空间梯度。
3. 把 `migration_rate` 作为独立灵敏度分析参数，不要与其他参数同时大幅波动。
4. 使用固定随机种子进行重复实验，避免把随机波动误判为机制差异。

## 11. 结果分析建议

Spatial 模型建议至少输出三类结果：

1. 全局总量轨迹（用于总体趋势）。
2. 每个 deme 的局部轨迹（用于空间异质性）。
3. 关键基因型在空间上的分布快照（用于扩散前沿判断）。

如果仅看全局总量，容易忽略“局部爆发 + 全局平衡”的空间特征。

## 12. 本章小结

Spatial 模拟的核心是“局部演化 + 空间迁移”的耦合。

对用户而言，最重要的工作流是：

1. 准备可独立运行的 demes。
2. 用拓扑定义连接关系。
3. 用 `migration_rate` 控制跨 deme 流动。
4. 使用 `run_tick()` 调试，使用 `run(...)` 批量实验。

---

## 相关章节

- [Simulation Kernels 深度解析](simulation_kernels.md)
- [Hook 系统](hooks.md)
- [Modifier 机制](modifiers.md)
- [PopulationState 与 PopulationConfig](population_state_config.md)
