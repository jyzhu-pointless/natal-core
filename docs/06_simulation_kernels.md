# Simulation Kernels 深度解析

本章面向使用者说明 NATAL 的模拟执行链路：

- 你在用户层调用什么；
- 框架内部如何完成一次 tick；
- 历史记录、状态导入导出与 Hook 在流程中的位置。

阅读本章后，你应当能清楚回答两个问题：

1. 一次 `pop.run(...)` 在内部做了哪些阶段计算。
2. 何时使用 `run(...)`、`run_tick()`、`get_history()`、`export_state()`。

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
  → 绑定 codegen runner
  → 依次调用阶段内核（reproduction/survival/aging）
  → 更新 state 与 history
```

这意味着你无需手动组织底层内核调用；专注于参数、Hook 与结果分析即可。

## 2. 两类 population 的一致用法

### 2.1 AgeStructuredPopulation

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

### 2.2 DiscreteGenerationPopulation

```python
pop.run(n_steps=100, record_every=10)
pop.run_tick()
```

两者在调用方式上保持一致，差异主要体现在内部状态结构与阶段内核。

## 3. 一次 tick 的阶段顺序

以一个标准 tick 为例，执行顺序是：

1. `first` 事件阶段（框架事件与用户 Hook）
2. `reproduction` 阶段
3. `early` 事件阶段
4. `survival` 阶段
5. `late` 事件阶段
6. `aging` 阶段
7. `n_tick` 增加

这一顺序对年龄结构模型与离散世代模型都成立；不同模型会调用对应的内核实现。

## 4. simulation_kernels 模块的职责

`src/natal/kernels/simulation_kernels.py` 主要提供“阶段级内核函数”，包括：

- 年龄结构模型：`run_reproduction`、`run_survival`、`run_aging`
- 离散世代模型：`run_discrete_reproduction`、`run_discrete_survival`、`run_discrete_aging`

此外，该模块还提供状态/配置导入导出的轻量包装函数，便于与上层对象方法配合使用。

## 5. 与 state/config 的关系

模拟运行时，内核读写的是两个核心对象：

- `state`：当前时刻的数量分布与时间步。
- `config`：生存率、交配率、适应度、映射矩阵等规则参数。

如果你已经阅读上一章，可以将本章理解为“state/config 如何在每个 tick 中被消费与更新”。

## 6. 历史记录机制

`run(...)` 可以按间隔写入历史数据：

```python
pop.run(n_steps=200, record_every=10)
history = pop.get_history()
```

实践建议：

- `record_every` 越小，历史越密集，便于诊断细节。
- `record_every` 越大，历史越精简，更适合长期模拟。
- 如不需要中间轨迹，可设为 `0` 以减少内存占用。

## 7. 状态导出与恢复

当你需要保存快照、跨脚本传递状态或做分叉实验时，可以使用：

```python
state_flat, history = pop.export_state()
# ... 保存或外部处理 ...
pop.import_state(state_flat, history=history)
```

典型场景：

1. 运行到某个关键时间点后保存快照。
2. 从同一快照派生多个参数分支。
3. 比较不同策略下的轨迹差异。

## 8. Hook 如何嵌入执行链路

用户定义的 Hook（如 `first`/`early`/`late`）会被编译并合并到执行流程中，然后由 runner 在对应阶段触发。

这带来两个好处：

- 使用上保持高层 API 简洁。
- 执行上仍保持统一阶段顺序，结果更可解释。

## 9. 推荐使用模式

1. 批量模拟：优先使用 `pop.run(...)`。
2. 单步观察：使用 `pop.run_tick()`。
3. 分析轨迹：搭配 `record_every` 与 `get_history()`。
4. 快照实验：使用 `export_state()` / `import_state()`。
5. 行为扩展：使用 Hook，而不是自行拼接内核调用。

## 10. 最小示例

```python
# 1) 构建 population
pop = ...

# 2) 连续运行
pop.run(n_steps=100, record_every=10)

# 3) 单步推进
pop.run_tick()

# 4) 获取历史
history = pop.get_history()

# 5) 导出与恢复
state_flat, hist = pop.export_state()
pop.import_state(state_flat, history=hist)
```

## 11. 本章小结

可以把 NATAL 的执行机制理解为三层分工：

- population 层：提供稳定的用户 API 与生命周期管理。
- runner/hook 层：将阶段流程与事件逻辑组织成统一执行链。
- kernel 层：完成每个阶段的数值计算。

在实际建模中，你通常只需稳定使用 population API，并在需要时通过 Hook 与 history 提升可控性与可解释性。

---

## 相关章节

- [PopulationState 与 PopulationConfig](05_population_state_config.md)
- [Numba 优化指南](07_numba_optimization.md)
- [Modifier 机制](08_modifiers.md)
- [Hook 系统](09_hooks.md)
