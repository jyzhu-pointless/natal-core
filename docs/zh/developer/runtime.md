# 会话与一次 tick

本章追踪普通离散世代路径。Python 负责入口检查和适配，Rust 会话持有状态，内核执行阶段计算；批量运行不会在每一阶段把完整状态送回 Python。

## 入口与调用关系

[DiscreteGenerationPopulation.run()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/population/discrete_generation.py) 先检查独立执行所有权、重入、失败和停止状态，解析记录间隔，再进入 `_run_rust_lifecycle()`。`run_tick()` 委托给一次步数的 `run()`，并使用种群的记录间隔。

[RustDiscreteLifecycleBackend](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/backends/rust/rust_backend.py) 在构造时物化合同并创建原生会话；`run()` 适配原生批量执行。Builder 路径在构建时建立会话，直接构造的部分路径可以延迟到首次运行。不要把“每次 run 新建会话”作为执行模型，否则无法解释连续随机流和历史记录。

Rust [DiscreteGenerationSession](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/sessions/discrete_generation.rs) 的 `run_inner()` 组织批量循环、日志、记录和检查点；[内核 run_tick()](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs) 组织单步阶段。

## 普通分阶段路径

| 阶段 | 实现行为 | 此时可观察的状态 |
| --- | --- | --- |
| `first` | `HookProgram.execute_event()`，提交事件写入 | 繁殖前状态 |
| reproduction | `reproduction()`：交配、受精、合子适应度 | age-0 已产生后代 |
| `early` | 执行 Hook 并提交 | 后代尚未经过 survival |
| survival | `survival()`：幼体密度调节和生存 | age-0 为存活后代 |
| `late` | 执行 Hook 并提交 | 年龄推进前 |
| aging | `aging()`：age-0 覆盖 age-1，清空 age-0 | 新一代成体 |

`EcoCtx.phase` 随这些边界变化，`stage_sources()` 在阶段执行前选择当前生态参数和遗传数据，因此较早事件提交的修改可以影响同一个 tick 的后续计算。

正常完成阶段后由会话推进 tick。Hook 返回停止结果会跳过剩余阶段，不能假定“进入 run_tick 就一定增加 tick”。失败也不等于整个 tick 回滚；已经完成的阶段和较早回调的提交可能保留，详见[Hook 事务](hooks.md)。

## 执行状态不是独立 Python 标志

[ExecutionStatus](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/sessions/status.rs) 定义 `Ready`、`Running`、`Stopped`、`Failed`。`begin()` 只允许从 Ready 进入 Running，拒绝时不修改状态。正常结束回到可继续运行的边界；停止和失败阻止直接续跑。

Python 的运行保护还防止回调递归启动同一种群。排查异常时，应同时查看原生 execution state、phase 和 tick；只有数量快照不足以说明执行停在什么位置。恢复检查点会恢复其记录的状态，恢复一个 Stopped 检查点不会自动把它变成 Ready。

## 两条需要单独阅读的路径

年龄结构内核同样有 reproduction、survival、aging，但精子存储贯穿阶段；年龄推进移动多个年龄层，而不是简单替换两代。

融合 Wright–Fisher 路径由 `run_wf_tick()` 执行，在会话中另行调度 `first`，不经过普通路径的 `early`、`late` 和三阶段组合。它有独立的模式验证和更新算法；不能只把它解释为普通循环的性能优化，也不能将上表无条件应用于所有执行模式。

## 验证入口

- [test_rust_discrete_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_rust_discrete_lifecycle.py)：离散生命周期的原生执行。
- [test_run_state_and_recording_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py)：停止、失败、恢复、连续记录和原生 tick。
- [test_frozen_lifecycle_rules.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_frozen_lifecycle_rules.py)：生命周期规则的限制。

调整阶段顺序时，应同时核对 Hook 可见状态、参数生效时间、停止边界、记录时机与 RNG 顺序。总数量相同不能证明这些合同保持一致。
