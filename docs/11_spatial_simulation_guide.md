# Spatial 模拟指南

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

## 4. 运行语义

### 4.1 `run_tick()`

用于单步推进，适合调试和过程观察。

### 4.2 `run(n_steps=...)`

用于批量推进，适合正式实验与参数扫描。

实践建议：

- 开发期先用 `run_tick()` 验证规则。
- 实验期改用 `run(...)` 并设置合适 `record_every`。

## 5. migration_rate 如何理解

`migration_rate` 表示每步中参与跨 deme 流动的人口比例。

直观上：

- `migration_rate = 0`：无迁移，仅局部演化。
- `migration_rate = 1`：完全按邻接权重重新分配。
- `0 < migration_rate < 1`：本地保留与跨 deme 迁移同时发生。

这通常是空间扩散速度与混合程度的关键参数。

## 6. Hook 在 Spatial 中的使用

用户使用方式与普通 population 一致：定义 Hook、注册 Hook、运行模拟。

建议：

1. 先在单 deme 场景验证 Hook 行为。
2. 再迁移到 Spatial 场景观察整体效应。
3. 对关键事件（release、suppression、stop 条件）保留可读命名。

## 7. 常见错误与排查

### 错误 1：deme 类型错误

现象：构造 `SpatialPopulation` 时提示 deme 不是合法 population 对象。

排查：确认 demes 列表中的每个元素都是标准 population 实例。

### 错误 2：邻接矩阵维度不匹配

现象：`adjacency` 形状与 deme 数量不一致。

排查：确认 `adjacency.shape == (n_demes, n_demes)`。

### 错误 3：运行时拒绝继续

现象：某些 deme 已结束，导致空间容器无法继续推进。

排查：检查每个 deme 的运行状态；必要时重新初始化。

## 8. 参数设置建议

1. 从小规模拓扑开始（2x2 或 3x3），先确认动力学方向正确。
2. 再增加 deme 数量，观察迁移是否造成预期的空间梯度。
3. 把 `migration_rate` 作为独立灵敏度分析参数，不要与其他参数同时大幅波动。
4. 使用固定随机种子进行重复实验，避免把随机波动误判为机制差异。

## 9. 结果分析建议

Spatial 模型建议至少输出三类结果：

1. 全局总量轨迹（用于总体趋势）。
2. 每个 deme 的局部轨迹（用于空间异质性）。
3. 关键基因型在空间上的分布快照（用于扩散前沿判断）。

如果仅看全局总量，容易忽略“局部爆发 + 全局平衡”的空间特征。

## 10. 本章小结

Spatial 模拟的核心是“局部演化 + 空间迁移”的耦合。

对用户而言，最重要的工作流是：

1. 准备可独立运行的 demes。
2. 用拓扑定义连接关系。
3. 用 `migration_rate` 控制跨 deme 流动。
4. 使用 `run_tick()` 调试，使用 `run(...)` 批量实验。

---

## 相关章节

- [Simulation Kernels 深度解析](06_simulation_kernels.md)
- [Hook 系统](09_hooks.md)
- [Modifier 机制](08_modifiers.md)
- [PopulationState 与 PopulationConfig](05_population_state_config.md)
