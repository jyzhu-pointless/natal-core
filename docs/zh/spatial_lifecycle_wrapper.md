# 空间生命周期执行

本文描述 `SpatialPopulation` 的运行时执行架构（slice-5 数据面落地后的形态）。
此前基于 njit codegen 的 spatial wrapper 管线（`compile_spatial_lifecycle_wrapper`、
`NUMBA_ENABLED`、`numba`/`prange` 导入、`natal.numba` 工具层）已全部移除——
Rust 原生扩展是唯一的执行引擎，所有路径共享同一套 hook 计划
与迁移数据面。

## 执行模型

一次空间 tick 分两个阶段：

```
各 deme 生命周期（per-deme 粒度并行/顺序：first hook → 繁殖 → early hook
  → 密度调节 + 存活 → late hook → 年龄推进）
      ↓
统一迁移（runtime 迁移率列 × 冻结的 CSR 折叠）
```

- **引擎会话**：`build()` 自动创建 Rust 会话；per-deme 生命周期与迁移核都在会话内
  执行。deme `d` 的随机流以 `seed ^ d` 派生，生命周期与迁移持续使用同一条流。

## 迁移数据面（slice-5）

构建时 `fold_migration_csr()` 把迁移配置折叠为 CSR 三件套
（`indptr` / `dest_idx` / `weights`），运行时只做 `outbound * weight`：

- **adjacency 模式**：每个源行按目标升序存原始邻接值。
- **kernel 模式**：按 kernel row-major 访问顺序复现历史 per-source 构建器；
  无效（越界）偏移丢弃或回绕；条目按 `1/kernel_total` 缩放——当
  `adjust_on_edge=True` 时按 `1/valid_row_total` 缩放。
- 两种模式最终都会把行归一化（折叠时对已缩放条目再做一次 emitted-row 求和除法），
  因此**边界 deme 与内部 deme 一样把全部迁出配额送往有效目标**；
  `adjust_on_edge` 开关的意义是保持与旧管线对位的历史位级运算顺序，而不是改变
  目的地分布。
- 迁移率与 CSR 分离：运行时 `migration_rate` 是 `(n_demes, S, A)` 列（写保护
  视图），实际出流量 = 率 × 权重。
- **换拓扑 = 重建**：CSR 在构建时折叠；修改拓扑/邻接/核参数后必须重建种群
  （`pop.params.tensor_write("migration_rate", ...)` 只改率列，不改拓扑）。

## Hook 执行

- 声明式 hook 编译为 CSR 计划，在每条路径上按事件边界执行；
- 回调 hook（`TickContext`）跨桥进入引擎会话执行；带外入口（`trigger_event`、finish 事件）直接调用；
- deme 级 `priority` 只在 deme 内部生效，跨 deme 无全局顺序；
- `@hook(..., deme=[0, 2])` 按 deme 选择器限定目标 deme（默认 `"*"` 全部）。

## 用户 API

```python
from natal.frontend.spatial import batch_setting
from natal.frontend.spatial.population import SpatialPopulation
from natal.frontend.spatial.topology import build_adjacency_matrix

pop = (
    SpatialPopulation.builder(species=sp, n_demes=4, pop_type="age_structured")
    .setup(name="demo", stochastic=False)
    ...
    .migration(adjacency=..., migration_rate=0.1)
    .build()
)
pop.run(n_steps=10, record_every=1)
pop.params.tensor_write("migration_rate", {"F": 0.2, "M": 0.05})  # 运行时改迁移率
```

构建后运行时参数写入只有两个入口：`pop.params.tensor_write(...)`（批量、
推荐）与 `deme(i).write_ecology(...)` / `write_genetics(...)`（单 deme）。
**`SpatialPopulation.update()` 链已删除**。
