# 观测、历史与检查点（导航）

查询结果、历史记录与可恢复的检查点用途不同。相关细节已经拆入三个专题章节，本页保留为导航入口：它给出三者的关系、选择记录模式时的取舍，以及旧的代码与测试链接。

## 三个入口

| 主题 | 现在的入口 | 读完能够回答 |
| --- | --- | --- |
| 当前状态如何投影成分组结果 | [观测如何从状态生成结果](observation.md) | 分组、轴顺序、名称与信息损失 |
| 历史存什么、怎么淘汰、参数时间线如何对齐 | [历史记录与参数时间线](history.md) | 两种模式、记录间隔、标签格式与日志 |
| 回到某个 tick 需要什么、遗传表是否会回滚 | [检查点恢复与实验重放](checkpoints.md) | 恢复包含什么，与导入状态有何区别 |

## 三者的关系

| | 当前观测 | 历史 | 检查点 |
| --- | --- | --- | --- |
| 数据来源 | 当前会话状态的投影 | 已记录的行 | 与历史行对齐的检查点 |
| 时间性 | 点时刻 | 时间序列 | 可回到的过去 |
| 是否可逆 | 无状态、可重复 | 只读 | 可恢复并截断后续 |
| 信息损失 | 取决于分组 | raw 无损、observation 已聚合 | 只覆盖运行状态，不含遗传表 |

三句话概括：**观测是投影，历史是记录，检查点是回退**。它们共享同一套分组规则和 tick 语义，所以"查询看到的"、"记录下来的"和"能恢复的"三件事可以逐一对齐。

## 选择记录模式时的取舍

- 需要事后换分组观察或做分支实验 → 用 `raw`。
- 只关心固定的几个分组、想减小存储 → 用 `observation`，但要接受无法恢复检查点。
- 需要控制存储上限 → 用 `max_rows`，同时记得旧行对应的检查点会一起失效。
- 需要追踪"什么时候改了参数" → 读 `pop.params_log`，它按 tick 与历史对齐。

## 代码与测试入口

- [output/observation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/observation.py)：观测组、掩码与投影。
- [output/_recording.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/_recording.py)：记录计划与掩码的编译。
- [output/history.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/output/history.py)：布局、模式、行访问与截断。
- [rust/src/output/history.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/history.rs)、[parameter_log.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/output/parameter_log.rs)：原生行与日志存储。
- [test_history_observation_contract.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_history_observation_contract.py)、[test_native_history_contracts.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_native_history_contracts.py)、[test_run_state_and_recording_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_run_state_and_recording_lifecycle.py)：记录与恢复合同。

从[观测如何从状态生成结果](observation.md)开始读这条线，或回到[项目架构与职责边界](architecture.md)重新建立整体认识。
