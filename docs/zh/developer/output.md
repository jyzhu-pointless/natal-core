# 观测、历史与检查点

查询结果、历史记录和可恢复的检查点用途不同。理解它们的关系，才能判断内存占用、记录间隔或恢复行为是否符合预期。

## 构建时固定记录布局

[compile_recording_plan()](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/_recording.py) 从最终注册表、种群状态和规范化 `Observation` 构造 `RecordingPlan`。其中 `HistorySchema` 保存布局与名称；observation 模式还生成 `(group, sex, age, ZType)` 的四维选择掩码。

group 是用户定义的观测组。掩码必须使用发布后的 ZType 顺序；否则数值投影合法但组标签会指向错误类型。`collapse_age` 和空间 deme 选择属于观测元数据，不应在展示层随意猜测。

[output/observation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/observation.rs) 的 `project()` 在原生层执行投影。可以把组计数理解为被掩码选中的状态项之和；具体保留哪些年龄和 deme 轴由观测定义控制。这使记录投影无需每个 tick 都拉取完整 Python 状态。

## 两种存储模式

| 模式 | 原生记录内容 | 恢复能力 |
| --- | --- | --- |
| raw | 原始状态行与对应完整检查点 | 可恢复仍保留的精确 tick |
| observation | 预先定义的投影值 | 不保留隐藏原始状态，不支持检查点恢复 |

[Rust HistoryData](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/history.rs) 存储行、边界元数据和共享日志关系，Python [history.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/history.py) 提供布局解释和查询包装。当前状态的 `observe()` 与历史读取不是同一操作：前者可以在当前会话上投影，后者只能使用保留的数据。

raw 历史可支持事后观测，observation 历史则已丢弃未记录的信息。不能从几个组的总数一般性地反推出每个基因型数量。

## 记录的是执行边界

会话批量循环在边界上考虑记录，记录间隔决定哪些 tick 被保留。连续 run 共享历史，精确的续跑边界由记录层去重；手动重复记录同一 tick 的行为要与自动续跑区分。

`record_every=0` 禁用该次运行的自动记录。保留行数限制会淘汰旧行及相应检查点，因此“模拟曾经到过 tick 20”不等于“现在可以恢复 tick 20”。读取 `boundary_metadata` 可区分普通边界与停止、失败的执行位置。

## 恢复不只是复制数量

种群的 `restore_checkpoint()` 经后端进入会话的 `restore_from_checkpoint()`。检查点恢复涉及个体与精子状态、tick、phase、执行状态、RNG、生态参数（包括 custom 和迁移）以及参数日志位置。遗传表不参与回滚。

恢复只接受仍保留的精确 tick。成功后丢弃未来历史，并按检查点记录的日志位置截断参数日志；同一 tick 后来发生的参数更新也可能位于需要删除的未来部分。只用 `log.tick <= restored_tick` 过滤不能表达这个语义。

例如 tick 10 的检查点之后、尚未推进 tick 时修改了参数，再恢复 tick 10，就必须使用检查点中的参数与日志游标。恢复状态仍应保留当时的执行状态，不能无条件设为 Ready。

`export_state()` / `import_state()` 是另一个状态传输入口，不应与完整检查点混淆。分析重放是否可复现时，必须确认恢复的还包括随机流与运行参数，而不只是数量。

## 验证入口与修改约束

- [test_history_observation_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_history_observation_contract.py)：存储模式与查询合同。
- [test_observation_age_axis_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_observation_age_axis_contract.py)：年龄轴语义。
- [test_restore_checkpoint_semantics.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_restore_checkpoint_semantics.py)：精确恢复与参数时间线。
- [test_run_state_and_recording_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py)：记录连续性、淘汰与执行状态。

修改历史保留或恢复时，检查行、检查点、边界元数据和日志是否一起变化，并验证恢复被拒绝时当前会话保持不变。
