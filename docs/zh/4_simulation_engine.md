# 模拟内核深度解析

<!--TODO: 改为数学模型介绍，需要大量公式-->

NATAL 的模拟执行链路：

- 你在用户层调用什么；
- 框架内部如何完成一次 tick；
- 历史记录、状态导入导出与 Hook 在流程中的位置。

阅读本章后，你应当能清楚回答两个问题：

1. 一次 `pop.run(...)` 在内部做了哪些阶段计算。
2. 何时使用 `run(...)`、`run_tick()`、`pop.history`、`export_state()`。

## 1. 用户入口与执行路径

日常使用时，你只需要面向 population 对象编程：

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

在框架内部，执行路径可概括为：

```text
population.run(...) / population.run_tick()
  → 获取已编译的事件 hooks
  → 在原生引擎会话内执行
  → 依次执行阶段内核（reproduction/survival/aging）
  → 更新 state 与 history
```

这意味着你无需手动组织底层内核调用；专注于参数、Hook 与结果分析即可。

## 2. 两类 population 的一致用法

### 2.1 `AgeStructuredPopulation`

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

### 2.2 `DiscreteGenerationPopulation`

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

两者在调用方式上保持一致，差异主要体现在内部状态结构与阶段内核。

## 3. 一次 tick 的阶段顺序

以一个标准 tick 为例，执行顺序是：

1. `first` 用户 Hook
2. `reproduction` 阶段
3. `early` 用户 Hook
4. `survival` 阶段
5. `late` 用户 Hook
6. `aging` 阶段
7. `n_tick` 增加

这一顺序对年龄结构模型与离散世代模型都成立；不同模型会调用对应的内核实现。

### 3.1 `AgeStructuredPopulation` 每步算法细化

以 `AgeStructuredPopulation` 的一个 tick 为例，三大阶段可进一步展开为：

1. reproduction
  - 按年龄加权计算有效雄性数量：`male_count[age, g] * male_mating_rate[age]`。
  - 基于性选择适应度与有效雄性数量，构建交配概率矩阵 `P(g_f -> g_m)`。
  - 调用 `sample_mating(...)` 更新 `sperm_store`（包含精子置换逻辑）。
  - 调用受精函数生成 age-0 新个体（雌/雄分别写入 `ind_count[:, 0, :]`）。
  - 对新生成的 age-0 个体应用合子适应度（zygote fitness）
2. survival
  - 先对 age-0（幼体）做密度调节：`NO_COMPETITION / FIXED / LOGISTIC / BEVERTON_HOLT`。
  - 再计算“年龄生存率 × 生存力（viability）”的联合生存率。
  - 用联合生存率同时更新 `individual_count` 与 `sperm_store`，保证两者一致。
3. aging
  - 所有年龄层向后推进一格。
  - 清空新的 age-0 槽位，等待下一个 tick 的 reproduction 写入。

要点：AgeStructured 是“有长期精子存储”的路径，`sperm_store` 在 reproduction/survival/aging 三个阶段都会被同步更新。

### 3.2 `DiscreteGenerationPopulation` 每步算法细化

`DiscreteGenerationPopulation` 固定 `n_ages=2`（age0=幼体，age1=成体），每个 tick 的算法更紧凑：

1. reproduction
  - 仅使用 age1 成体进行交配与受精。
  - 当步用临时配对/受精缓冲完成受精，不跨 tick 保留精子库（离散模型没有 `sperm_storage` 状态）。
  - 产出的后代写入 age0。
2. survival
  - 先对 age0 做密度调节（同样支持四种 growth mode）。
  - 仅对 age0 应用联合生存率（年龄生存率 × viability）。
3. aging
  - 代际更替：`age0 -> age1`。
  - 原 age1 被覆盖（即离散世代中的“旧成体退出”）。

要点：Discrete 强调“非重叠世代”，没有 AgeStructured 那种跨年龄、跨 tick 的长期精子存储状态。

### 3.3 随机性与确定性：同一流程下的两种执行语义

阶段顺序不变，但数值更新方式由配置决定：

1. `stochastic=False`（确定性）
  - 使用期望值/比例缩放，结果通常是连续值（float）。
  - 不进行 Binomial/Poisson 抽样。
2. `stochastic=True`（随机）
  - 使用抽样更新（如 Binomial/Poisson/Multinomial 等），轨迹会有随机波动。
  - 若 `continuous_sampling=True`，会采用连续近似抽样（如 Beta/Dirichlet/Gamma 近似）以提高可微/连续性和部分场景下的数值稳定性。

此外，reproduction 阶段还受 `fixed_egg_count` 影响：

- `True`：按固定期望卵数产卵。
- `False`：按 Poisson 机制产卵（在随机模式下体现为随机卵数）。

## 4. 引擎实现布局

Rust 原生扩展 `natal._engine_rs` 是唯一的执行引擎。它在引擎会话内拥有
运行状态（泛交配模型每种群一个会话；空间容器一个堆叠会话），并实现
两类模型的阶段内核：

- 年龄结构模型：reproduction、survival、aging（长期精子存储）。
- 离散世代模型：两年龄段紧凑生命周期，精子仅当 tick 有效。

### 4.1 Spatial migration 布局

迁移在空间引擎会话内、各 deme 生命周期之后作为 CSR 阶段执行。前端在
构建期把所有迁移声明（拓扑、邻接矩阵或迁移核）折叠为一份冻结 CSR 加
速率列（`src/natal/frontend/spatial/migration.py`）；会话每个 tick 用
速率列乘以该 CSR。

## 5. 与 `state`/`config` 的关系

运行状态与生态参数由 Rust 会话持有；Python 侧的 `pop.state` 与 `pop.config` 只是**查询快照**（每次访问重新生成，修改它们不影响引擎），详见 [PopulationState 与 ModelDraft](4_population_state_config.md)。

- `state`：当前时刻的数量分布与时间步（快照读）。
- `config`：生存率、交配率、适应度、映射矩阵等规则参数的投影（快照读）。

运行期写入走受控通道：`pop.params.<name>` / `pop.update()`（标量）、`pop.params.tensor_write(...)`（向量与张量）。空间容器不暴露 `state`/`config` 属性，其参数写入见[空间生命周期执行](spatial_lifecycle_wrapper.md)。

如果你已经阅读上一章，可以将本章理解为“这些快照如何在每个 tick 中被消费与更新”。

## 6. 历史记录机制

`run(...)` 可以按间隔写入历史数据：

```python
pop.run(n_steps=200, record_every=10)
history = pop.history.individual_count
```

实践建议：

- `record_every` 越小，历史越密集，便于诊断细节。
- `record_every` 越大，历史越精简，更适合长期模拟。
- 如不需要中间轨迹，可设为 `0` 以减少内存占用。

## 7. 状态导出与恢复

当你需要保存快照、跨脚本传递状态或做分叉实验时，可以使用：

```python
state_flat = pop.export_state()
# ... 保存或外部处理 ...
pop.import_state(state_flat)
# import_state() 同时清空该种群的历史，时间线从头开始
```

典型场景：

1. 运行到某个关键时间点后保存快照。
2. 从同一快照派生多个参数分支。
3. 比较不同策略下的轨迹差异。

### 7.1 随机流（RNG）与 bit-reproducible 承诺范围

- `build()` 自动创建 Rust 会话。会话持有 `SessionRng`，多次 `run()` 持续推进同一随机流；
  不需要选择或手动启用后端。
- 空间模型中 deme `d` 从会话的基础种子按 `seed ^ d` 派生随机流；该 deme 的
  生命周期阶段与迁移共用这条流。
- Python Hook 内的 `ctx.rng` 是当前事件的受控 Rust 采样器。重复访问返回同一个采样器，
  连续取样会推进随机流。回调返回后采样器失效；恢复检查点会恢复当时记录的 RNG 状态。
- **承诺范围**：同一输入（构建参数 + seed + hook 组合）下，确定性
  （`stochastic=False`）轨迹逐位可复现；随机轨迹在固定 seed 下跨进程可复现。
  不承诺跨版本位级稳定（未来算法修复可能改变数值）。

### 7.2 检查点与历史查询

Rust 会话拥有历史数值、检查点和参数日志。`mode="raw"` 的每个保留记录都包含对应的完整检查点：个体与精子状态、tick、执行阶段与状态、RNG、生态参数（含迁移和 custom）及日志位置。遗传表不参与回滚。`mode="observation"` 只保留投影值，不隐藏完整原始历史，也不支持恢复。

`pop.restore_checkpoint(tick)` 只接受仍保留的精确 tick；未记录或已淘汰的 tick 会在修改状态前报错。恢复保留该 tick 及以前的历史，并按检查点中的位置截断未来参数日志，包括同一 tick 上后来发生的更新。普通稳定边界恢复为 `Ready`；手动记录的停止或失败边界保留对应执行状态。

`record_snapshot()` 可在两次运行之间记录当前边界，包括已停止的种群；重复记录同一 tick 会报错。`pop.history.boundary_metadata` 返回不可变的 `(tick, phase_cursor, status)` 元组序列。阶段游标标识生命周期中的位置，例如 `0` 是正常 tick 边界，`2` 是 early 阶段。结合 status 区分完整边界和中途停止。

`pop.params_log_details` 返回 `(tick, event, deme, parameter, old, new)` 元组序列；普通种群的 deme 为 `0`，空间种群可从对应 deme 的查询接口读取日志。值保留布尔、整数、浮点或数组类型，新增与删除分别用 `None` 表示旧值与新值。查询中的数组是独立副本。`params_log` 保留原有四列标量投影，不包含数组和新增／删除记录；需要完整审计时使用 `params_log_details`。已取得的查询结果不会被后续更新或恢复改写。

`max_rows` 同时限制保留的记录与检查点；无 Python 回调的批量运行在 Rust 内逐步记录并淘汰。`clear_history()` 清除记录和对应检查点，保留当前状态、RNG、参数和参数日志。

下面的完整示例演示有界历史及日志回滚：

```python
import natal as nt

species = nt.Species.from_dict(
    "HistoryExample", {"Chr1": {"L1": ["W"]}}, gamete_labels=["default"]
)
pop = (
    nt.DiscreteGenerationPopulation.setup(species=species, stochastic=False)
    .initial_state(individual_count={"female": {"W|W": 10}, "male": {"W|W": 10}})
    .survival(female_age0_survival=1.0, male_age0_survival=1.0)
    .reproduction(eggs_per_female=2.0, sex_ratio=0.5)
    .competition(carrying_capacity=1000.0, low_density_growth_rate=2.0)
    .record_history(mode="raw", max_rows=3)
    .build()
)
pop.run(3, record_every=1)
assert pop.history.ticks == (1, 2, 3)
pop.update().competition(carrying_capacity=500.0)
assert pop.params_log_details[-1][3:] == ("carrying_capacity", 1000.0, 500.0)
pop.restore_checkpoint(1)
assert pop.params.carrying_capacity == 1000.0
assert pop.history.ticks == (1,)
assert pop.params_log_details == ()
assert pop.history.boundary_metadata == ((1, 0, "Ready"),)
```

## 8. Hook 如何嵌入执行链路

用户定义的 Hook（如 `first`/`early`/`late`）会被编译并合并到执行流程中，然后由 runner 在对应阶段触发。

这带来两个好处：

- 使用上保持高层 API 简洁。
- 执行上仍保持统一阶段顺序，结果更可解释。

## 9. 推荐使用模式

1. 批量模拟：优先使用 `pop.run(...)`。
2. 单步观察：使用 `pop.run_tick()`。
3. 分析轨迹：搭配 `record_every` 与 `pop.history`。
4. 快照实验：使用 `export_state()` / `import_state()`。
5. 行为扩展：使用 Hook，而不是自行拼接内核调用。

## 10. 最简示例

```python
# 1) 构建 population
pop = ...

# 2) 连续运行
pop.run(n_steps=100, record_every=10)

# 3) 单步推进
pop.run_tick()

# 4) 获取历史
history = pop.history.individual_count

# 5) 导出与恢复
state_flat = pop.export_state()
pop.import_state(state_flat)
```

## 11. 小结

可以把 NATAL 的执行机制理解为三层分工：

- population 层：提供稳定的用户 API 与生命周期管理。
- runner/hook 层：将阶段流程与事件逻辑组织成统一执行链。
- kernel 层：完成每个阶段的数值计算。

在实际建模中，你通常只需稳定使用 population API，并在需要时通过 Hook 与 history 提升可控性与可解释性。

---

## 相关章节

- [PopulationState 与 ModelDraft](4_population_state_config.md)
- [Modifier 机制](3_modifiers.md)
- [Hook 系统](2_hooks.md)
