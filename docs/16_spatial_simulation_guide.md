# Spatial 模拟实现指南（从零看懂版）

这份文档专门写给“第一次看 Spatial 代码的人”。
目标不是讲全，而是让你先看懂：

1. Spatial 在这个项目里到底是什么
2. 一步 tick 到底发生了什么
3. `run_tick` 和 `run` 是怎么接上 kernels + hooks + codegen 的
4. 出错时怎么快速定位

如果你之前看不懂，建议按本文从上往下读，不要跳章节。

---

## 1. 先用一句话理解 Spatial

在 NATAL 里，Spatial 不是“一个超级种群类继承 BasePopulation”，而是：

- 外层：`SpatialPopulation`（空间容器）
- 内层：很多个 deme（每个 deme 都是普通 `BasePopulation` 子类实例）

也就是“组合”而不是“继承”。

实现位置：

- [src/natal/spatial_population.py](src/natal/spatial_population.py)

---

## 2. 你在模拟什么（直觉模型）

想象你有 4 个格子（deme），每个格子里都有一套完整种群状态：

- 个体计数 `individual_count`
- 精子存储 `sperm_storage`
- 当前 tick

Spatial 模拟每一步做两件事：

1. 每个格子先自己按遗传规则演化（繁殖、生存、衰老）
2. 演化后，按邻接关系进行格子之间迁移

所以 Spatial 的核心就是“**每 deme 先局部演化，再做跨 deme 迁移**”。

---

## 3. 相关模块各负责什么

### 3.1 容器层（你直接调用）

- [src/natal/spatial_population.py](src/natal/spatial_population.py)

职责：

- 持有 demes
- 持有 adjacency 与 migration_rate
- 对外提供 `run_tick()` / `run(...)`
- 把结果回写到每个 deme

### 3.2 拓扑层（只负责“谁和谁相邻”）

- [src/natal/spatial_topology.py](src/natal/spatial_topology.py)

职责：

- `SquareGrid` / `HexGrid`
- 生成 adjacency
- 提供卷积式迁移工具

### 3.3 数值内核层（纯计算）

- [src/natal/kernels/spatial_simulation_kernels.py](src/natal/kernels/spatial_simulation_kernels.py)

职责：

- `run_spatial_reproduction`
- `run_spatial_survival`
- `run_spatial_aging`
- `run_spatial_migration`

不负责高层调度，不负责对象管理。

### 3.4 codegen 层（把 hook 和内核拼成可执行 runner）

- 模板：[src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl](src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl)
- 编译器：[src/natal/kernels/codegen.py](src/natal/kernels/codegen.py)
- hook 合并：[src/natal/hooks/compiler.py](src/natal/hooks/compiler.py)

职责：

- 把 `first/early/late` hook 合并
- 生成 spatial 专用 runner：
  - `run_spatial_tick_fn`
  - `run_spatial_fn`

---

## 4. 一步 tick 到底怎么跑（最重要）

Spatial tick 的顺序固定为：

1. `first`（逐 deme）
2. reproduction（全 deme 批量）
3. `early`（逐 deme）
4. survival（全 deme 批量）
5. `late`（逐 deme）
6. aging（全 deme 批量）
7. migration（全 deme 批量）

这条顺序在模板里固定，位置：

- [src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl](src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl)

你可以把它理解成“在大阶段之间插入 hook 检查点”。

---

## 5. `SpatialPopulation.run_tick()` 做了什么

实现位置：

- [src/natal/spatial_population.py](src/natal/spatial_population.py)

按代码实际行为，`run_tick()` 的流程是：

1. 检查是否有 deme 已 finished（有则拒绝继续）
2. 从第一个 deme 取 `CompiledEventHooks`
3. 取 `run_spatial_tick_fn` 与 `registry`
4. 校验并获取共享 config（当前要求所有 deme 共用同一 config 对象）
5. 把所有 deme 的状态堆叠成 4D/5D 数组
6. 调用 codegen 生成的 runner
7. 把 runner 返回结果回写到每个 deme 的 state 与 tick
8. 若返回 stop，则标记所有 deme finished 并触发 `finish` 事件

一句话：`run_tick()` 是“对象层封装”，真正计算由 codegen runner 完成。

---

## 6. `SpatialPopulation.run(n_steps)` 做了什么

`run(...)` 与 `run_tick()` 一样，只是调用 `run_spatial_fn` 做多步循环。

它会：

- 透传 `n_steps`、`record_every`、`adjacency`、`migration_rate`
- 在返回后统一回写状态
- 如果 `was_stopped=True`，统一 finish
- 如果 `finish=True` 且未提前停止，调用每个 deme 的 `finish_simulation()`

---

## 7. 一个“能理解结构”的最小示意

下面示意只强调结构，不保证可直接复制运行：

```python
from natal.spatial_population import SpatialPopulation
from natal.spatial_topology import SquareGrid, build_adjacency_matrix

# 1) 先准备每个 deme（它们是普通人口对象）
deme0 = ...
deme1 = ...
deme2 = ...
deme3 = ...

# 2) 拓扑 -> 邻接矩阵
grid = SquareGrid(rows=2, cols=2, neighborhood="moore", wrap=False)
adj = build_adjacency_matrix(grid, row_normalize=True)

# 3) 组装空间容器
sp = SpatialPopulation(
    demes=[deme0, deme1, deme2, deme3],
    adjacency=adj,
    migration_rate=0.1,
)

# 4) 运行
sp.run_tick()          # 单步
sp.run(n_steps=50)     # 多步
```

如果你只记一个点：

- deme 是“普通 population”
- SpatialPopulation 只是“把多个 deme 一起推进”

---

## 8. Hook 在 Spatial 中怎么理解

你可以把 hook 分成两层理解：

1. **定义层**：你像以前一样写 hook（声明式 / selector / numba）
2. **执行层**：编译后合并进 spatial runner 的事件点

所以不是 Python for 循环里到处调 hook，而是：

- 先合并
- 再由 codegen 产物在固定阶段调用

这样能保证顺序稳定，也减少运行时分发开销。

---

## 9. 当前限制（你最可能踩坑的地方）

### 9.1 所有 deme 必须共享同一个 config 对象

这是当前实现约束，不是文档约束。

触发位置：

- [src/natal/spatial_population.py](src/natal/spatial_population.py)

如果某个 deme 的 `export_config()` 返回了不同对象，会报错。

### 9.2 deme 类型必须是 `BasePopulation` 子类实例

否则构造 `SpatialPopulation` 时直接报错。

### 9.3 adjacency 形状必须是 `(n_demes, n_demes)`

不是这个形状就会报 shape mismatch。

### 9.4 所有 deme 初始 tick 必须一致

`SpatialPopulation` 构造时会检查，不一致就拒绝创建。

---

## 10. 常见报错速查

### 报错：`deme[i] must be a BasePopulation subclass instance`

原因：你传入了普通对象，不是 population 实例。

修复：确保每个 deme 都是 `AgeStructuredPopulation` / `DiscreteGenerationPopulation` 这类对象。

### 报错：`adjacency shape mismatch`

原因：deme 数量和邻接矩阵维度不匹配。

修复：确认 `adj.shape == (len(demes), len(demes))`。

### 报错：`uses a different config object`

原因：各 deme 使用了不同 config 对象。

修复：当前版本下要复用同一个 config 对象。

### 报错：`has finished; cannot run ...`

原因：某个 deme 已 finish。

修复：重建或 reset 对应 deme 后再运行。

---

## 11. 推荐阅读顺序（看不懂就按这个走）

1. 先读容器实现：
- [src/natal/spatial_population.py](src/natal/spatial_population.py)

2. 再看阶段顺序模板：
- [src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl](src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl)

3. 再看数值 kernel：
- [src/natal/kernels/spatial_simulation_kernels.py](src/natal/kernels/spatial_simulation_kernels.py)

4. 最后看 hooks 如何接上：
- [src/natal/hooks/compiler.py](src/natal/hooks/compiler.py)

如果你愿意，我下一版可以继续加一个“逐行注释版 run_tick 流程图”，把 [src/natal/spatial_population.py](src/natal/spatial_population.py) 的每一步映射到具体代码行。

---

## 12. 关键问答（你提的 5 个问题）

### 12.1 migration_rate 如何发挥作用？

短答案：它控制每一步里“有多少比例的人口会参与跨 deme 流动”。

当前实现（邻接矩阵迁移）可以理解为：

- 先保留本地一部分：比例是 `1 - migration_rate`
- 再把迁移部分：比例是 `migration_rate`，按 adjacency 权重分发到其它 deme

在核函数里对应公式是：

$$
X' = (1-r)X + r\cdot\text{Inflow}(A, X)
$$

其中：

- $r$ 就是 migration_rate
- $A$ 是 adjacency
- $X$ 是当前各 deme 状态

直觉上：

- `migration_rate = 0`：完全不迁移
- `migration_rate = 1`：完全按邻接分发
- 中间值：本地保留与外流混合

实现位置：

- [src/natal/kernels/spatial_simulation_kernels.py](src/natal/kernels/spatial_simulation_kernels.py)

### 12.2 hooks 怎么用？hooks 的 deme selector 有没有用？

先说结论：

1. hooks 在 Spatial 主路径是“有用”的：
: `first/early/late` 都会在 spatial wrapper 固定阶段执行。

2. deme selector 在“当前 Spatial codegen 主路径”里还没有完整生效为按 deme 精确过滤：
: 因为 spatial wrapper 调用的是事件级合并结果，不带 HookExecutor 的 `deme_id` 过滤步骤。

更具体地说：

- 在 HookExecutor 路径里，`deme_selector` 是生效的（会根据 `deme_id` 过滤）
- 在 Spatial codegen 路径里，当前主要依赖合并后函数与 CSR program 执行，尚未走 `HookExecutor.execute_event(..., deme_id=...)` 这条过滤逻辑

所以你现在应当这样理解：

- “事件时序控制”可用
- “按 deme 精细过滤”在 Spatial codegen 路径中仍是待完善项

相关实现：

- [src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl](src/natal/kernels/templates/spatial_kernel_wrappers.py.tmpl)
- [src/natal/hooks/compiler.py](src/natal/hooks/compiler.py)
- [src/natal/hooks/executor.py](src/natal/hooks/executor.py)

### 12.3 如何传入卷积核？

先说现状：

- `SpatialPopulation` 目前接收的是 `adjacency`（或 `topology` 自动生成 adjacency）
- 还没有直接参数让你传卷积核并在每 tick 自动应用

那现在怎么做：

1. 如果你的卷积核是固定并且能预先转成等价 adjacency：
: 先离线构造 adjacency，再传给 `SpatialPopulation`。

2. 如果你的卷积核是动态或依赖边界局部行为：
: 在外层循环中取出 stacked state，调用 `apply_migration_convolution(...)`，再回写。

工具函数位置：

- [src/natal/spatial_topology.py](src/natal/spatial_topology.py)

后续可演进方向（建议）：

- 在 `SpatialPopulation.run/run_tick` 增加 `migration_mode` 与 `convolution_kernel` 参数
- 在 spatial wrapper 模板加入卷积迁移分支

### 12.4 如何快速初始化（不一个个传种群）？

现状：

- 当前 `SpatialPopulation` 构造函数要求传入 demes 列表
- 还没有官方批量工厂方法

现在最实用做法：

1. 写一个 deme 工厂函数（传 index 返回一个 BasePopulation 子类实例）
2. 用列表推导快速生成所有 demes
3. 再传给 `SpatialPopulation`

建议后续新增 API（可直接实现）：

- `SpatialPopulation.from_factory(topology, deme_factory, ...)`

其中 `deme_factory(deme_id, coord)` 负责创建每个 deme，框架负责批量拼装。

### 12.5 后续如何支持 config 的差异化？

现状约束：

- 当前 `SpatialPopulation.run/run_tick` 要求所有 deme 共享同一个 config 对象

这是由当前 spatial runner 签名决定的（单 config 输入）。

推荐的演进路线：

1. 阶段一（最小改动）
: 在 `SpatialPopulation` 增加 Python 调度分支：按 deme 循环调用单 deme kernel（允许不同 config），最后统一迁移。

2. 阶段二（保持 codegen 统一）
: 扩展 spatial wrapper 签名，从单 `config` 改为 `config_list`，在模板内按 `deme_id` 取对应 config。

3. 阶段三（性能优化）
: 对常见“少量 config 模板 + 多个 deme”场景做分组批处理（按 config 分 bucket），减少分支开销。

建议最终 API 形态：

- `SpatialPopulation(..., config_mode="shared"|"per_deme")`
- `SpatialPopulation(..., deme_configs=[...])`

这样可向后兼容当前 shared config，又能逐步过渡到异构配置。