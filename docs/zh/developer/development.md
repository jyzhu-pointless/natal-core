# 修改与验证路线

理解一个函数后，下一步是确定改动穿过哪些边界。本章提供定位路线；质量门禁和风险分类仍以仓库 [AGENTS.en.md](https://github.com/jyzhu-pointless/natal-core/blob/main/AGENTS.en.md) 与 [quality_checks_spec.md](https://github.com/jyzhu-pointless/natal-core/blob/main/quality_checks_spec.md) 为准，避免在指南里复制一份会过期的政策。

## 按行为找到修改入口

| 目标 | 首先阅读 | 随后核对 |
| --- | --- | --- |
| 增加生态参数 | `parameters.jsonc`、`contracts/params.py`、`rust/src/model/ecology.rs` | 默认值、生成字段、物化、运行写入、日志与检查点 |
| 修改遗传转换 | `genetics/matrices/`、`definition_compiler.py` | 编译基线、发布投影、后代派生与运行期重编译 |
| 修改繁殖或生存 | `rust/src/kernels/` 中对应模型 | 阶段次序、随机采样、年龄/精子约束与其他模型差异 |
| 增加 Hook 操作 | `hooks/_compile.py`、Rust `hooks/interpreter.rs` | priority、选择器、事务、停止与失败 |
| 修改观测输出 | `_recording.py`、Python/Rust `output/` | 布局、名称、年龄和 deme 轴、两种历史模式 |
| 修改空间共享 | `spatial/builder.py`、`genetics_variant_bank()` | 共同索引、变体分叉、局部写入、容器所有权 |

参数生成产物位于 [rust/src/generated/ecology_parameters.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/generated/ecology_parameters.rs)。先追溯生成来源和构建流程，再决定改哪些文件；直接手改生成文件可能在下次构建被覆盖。

## 一个具体的调查顺序

以“新增一个会影响 survival 的生态参数”为例，先说明它的单位、默认值、合法范围、是否按 deme 变化、在哪个阶段生效，以及是否随检查点恢复。这些是行为问题，尚未决定字段放在哪里。

接着从声明入口追到 `ModelDraft` 和合同字段，确认 Rust 接收后存在哪个生态列。再沿运行期更新路径追到阶段读取位置，检查 first 与 early 更新是否会按预期影响本步 survival。最后核对快照、参数日志和恢复路径，保证用户查询到的值与内核实际读取的一致。

验证至少应包含一个能区分新旧行为的小模型，以及非法候选不产生部分写入的场景。若参数可在空间容器中使用，再检查只更新目标 deme 时其他 deme 的值和遗传共享关系不受意外影响。不要用重复实现整段算法的测试替代这些边界检查。

## 用已有测试学习合同

各专题的验证入口是调查起点。先读测试的初始化、事件位置和断言，确认它保护的具体行为，再决定是否需要补强。测试名称只能提供线索；测试通过不代表覆盖了未经断言的行为。

数值修改需要有依据的预期值或约束，状态与跨语言修改需要覆盖所有权和失败路径。公开 API 变化还要同步用户指南、双语开发者指南、相关示例和 stub；具体命令以质量规范为准。

## 维护这套指南

修改文档时同时维护中英对应页和两份 MkDocs 导航。具体符号重命名或移动后，检查本指南中的源码链接；增加带索引轴的字段时更新布局表；阶段或恢复行为变化时更新执行边界说明。

将当前行为与设计历史分开。若动机没有代码、测试或设计记录支撑，应表述为推断。若文档与实现冲突，先确认当前合同和测试依据，再修正说明或提出产品问题，不要为了让文字成立而悄悄改变模型。

浏览器界面的独立代码位于 [根目录 frontend/](https://github.com/jyzhu-pointless/natal-core/tree/main/frontend)，Python 服务入口位于 [frontend/webui/](https://github.com/jyzhu-pointless/natal-core/tree/main/src/natal/frontend/webui)。核心模拟指南到这里结束；界面改动还应继续追踪 session、serialization、REST/WebSocket 协议与显示组件之间的数据链路。
