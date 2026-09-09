# docs/ 系统性过时内容修订：实施移交方案

## 1. 任务起点与授权

- 审查基线：`c001027ceaeff11476ab3893a0f94b9047460bff`。
- 目标：使当前用户指南、API 参考、内部实现说明与 Rust 唯一后端的实际行为一致，并将历史方案与当前教程明确分开。
- 本文件是待实施方案，不表示文档修订已完成。当前请求仅授权编写方案，未授权执行本方案或创建 commit。
- 实施时先读取 [AGENTS.md](./AGENTS.md)、[quality_checks_spec.md](./quality_checks_spec.md)；后续 HEAD 如已变化，重新记录实际基线，保留用户修改。
- 背景参考：[Rust 唯一后端方案](./RUST_ONLY_REFACTOR_PLAN.md)、[Rust 模块整理方案](./RUST_MODULE_ORGANIZATION_PLAN.md)。方案表达意图，当前代码与已验证行为决定哪些意图已经实现，不能把计划直接当作事实。

本轮文档审查已确认系统性问题，但没有逐一执行 docs/ 中全部代码块。实施者不得把未检查页面视为已通过，也不得依据关键词命中数量直接计算缺陷数。

## 2. 目标与不变范围

完成后，用户应能清楚区分：

1. 构建时的模型声明、独立配置草稿与 Rust 持有的运行数据。
2. 用于查询的快照与真正提交更新的入口。
3. 普通代码中的即时更新与 callback 内的事务候选、提交和失败回滚。
4. raw 历史、observation 历史、运行检查点和手动状态导入的不同保证。
5. 空间整体的状态所有权与单个 deme 的受控查询、更新能力。
6. 当前受支持功能、历史设计以及尚未实现的设想。

保持链式 API、Species 注册语法、preset 科学规则、声明式 hook 格式不变。本轮不修改模型算法、RNG、生产 API 或运行时行为，不为使旧例子通过而恢复旧后端、旧可变共享入口。

若文档例子暴露真实代码缺陷，记录复现和影响，作为独立修复项处理；不能将行为修复混入文档提交。只有得到对应授权后才扩展代码修复范围。

允许修改范围：`docs/zh/`、`docs/en/`、`docs/api/`、三个 MkDocs 配置和确有必要的文档验证测试。优先修改既有文件；新增页面或测试文件必须服务明确的主题归属或行为验收，不建立通用文档执行平台。

## 3. 已确认问题清单

行号会随修改漂移，下面用路径、章节和关键语句定位。

| 编号 | 页面或主题 | 已确认问题 | 应修成的含义 |
|---|---|---|---|
| D1 | 中英文 `3_runtime_modification.md` 的 `set_param()` 章节 | 推荐 `set_param(pop.config, ...)` 修改运行种群 | `pop.config` 为独立快照；运行更新使用实际受支持的 `pop.update()`、`pop.params` 等入口。底层草稿操作必须标明不会自动提交到种群 |
| D2 | 中英文 `4_population_state_config.md`、`2_population.md` | 将运行生态参数描述为可原地修改的 0-d 数组；暗示直接改遗传数组即可生效 | 区分草稿内部表示与运行合同；参数/遗传更新走受控通道，查询快照不承担写入功能 |
| D3 | 中英文 `2_hooks.md` 的执行路径 | 仍介绍 Reference/Rust 双后端及两路径对拍保证 | Rust 是唯一执行后端；说明声明式执行与 Python callback 的交互，不再承诺已删除路径的行为 |
| D4 | 中英文 `spatial_builder.md`、`spatial_configurator.md` | 旧 Builder 组合、共享 `_config`、直接修改 deme 数组的建议仍作为当前机制 | 使用现有 SpatialConfigurator 和 Rust 原生所有权、配置变体与受控更新的实际说明 |
| D5 | `docs/api/configurator.md` 的 Low-Level API | 仍称 set_param 是所有高层 API 基础、引用旧 registry 名称和自动同步；后文又说快照不会更新 session | 底层草稿写入与绑定运行会话的更新分开解释，API 页内部保持一致 |
| D6 | `spatial_config_replace.md`、`spatial_initialization_plan.md` 及导航 | 旧设计与旧性能数据没有充分区分历史和现状 | 标记历史状态、停止作为当前实现操作指南；现状链接指向权威主题页 |
| D7 | 中英文导航 | 英文有 Spatial Configurator 导航项，中文没有对应项；多个空间页面职责重复 | 按主题同步两种语言的导航和交叉链接，保留有意差异的明确理由 |

D1 已实测：初始 carrying_capacity=10000；`set_param(pop.config, "carrying_capacity", 5000)` 后仍为 10000；`pop.update().competition(carrying_capacity=5000)` 后变为 5000。这是静默无效的用户操作，优先修复。

`2_population_initialization.md`、`migration_builder_to_configurator.md` 中“链式方法立即写入数组”等实现描述列为待逐项核实内容：部分方法当前确实即时编译或更新草稿，不能把所有“立即生效”一律改成“build 时才执行”。

同样不能机械删除所有 Reference、Numba、fallback、zero-copy：历史来源、物种延迟绑定、算法术语中的合法用法与现行行为错误必须区分。

## 4. 文档职责与重复页面处理

### 4.1 每个主题的权威位置

| 主题 | 主要页面 | 其他页面如何处理 |
|---|---|---|
| 种群构建和链式配置 | `2_population_initialization.md` | 入门仅保留最短示例，内部页不重复完整操作教程 |
| 运行参数与遗传更新 | `3_runtime_modification.md` | `2_population.md` 和 API 页提供简述及链接 |
| 状态、配置快照与所有权 | `4_population_state_config.md` | 运行指南解释操作效果，不重复底层结构清单 |
| 声明式 hook 入门 | `2_hooks.md` | 保持用户 DSL 的清晰示例 |
| callback、事务和 RNG | `3_advanced_hooks.md` | 其他 hook 页链接事务规则，不各自维护不同版本 |
| 历史、observation、恢复 | `2_data_output.md` | 内部输出页说明实现，不另定义用户合同 |
| 空间使用与生命周期 | `3_spatial_simulation.md` | 参数更新页引用该页的 deme 限制 |
| 空间构建内部实现 | `spatial_configurator.md` | `spatial_builder.md` 合并后保留简短迁移指引 |
| Rust 执行架构 | `4_simulation_engine.md` | 旧执行引擎、wrapper 页仅保留仍有价值的现状说明或历史提示 |
| 函数签名和局部 API 限制 | `docs/api/` | 不复制长教程，签名与实际导出核对 |

权威位置表示同一主题只维护一份完整行为说明，不表示每个例子都要跳转多次才能使用。入门页应保留独立可理解的关键限制。

### 4.2 历史页面策略

- `spatial_initialization_plan.md` 保留为历史设计，首段明确“历史方案，不是当前行为保证”，指出当前实现入口。不删除历史推理，不声称所有设想都已实现。
- `spatial_config_replace.md` 核对其中仍适用的构建优化，将现行内容合并到空间内部实现页；原页保留历史定位和链接，不能继续宣称旧 `_replace` 快速路径就是当前全部机制。
- `spatial_builder.md` 与 `spatial_configurator.md` 重复内容合并到后者；原路径保留简短说明和替代链接，避免破坏现有引用。不增加需要新依赖的重定向插件。
- `migration_builder_to_configurator.md` 保留迁移意义，但写清起点、目标版本或提交；过渡实现细节不能描述成当前常规使用要求。
- `caching_and_codegen.md`、`spatial_lifecycle_wrapper.md` 等先核对。已经正确标记移除的页面不因文件名旧就重写。
- 历史文件保持现有 URL，移出当前教程/实现导航，归入明确的“历史设计 / Historical Design”栏目或历史索引。中英文对应同步。
- 未注明版本和测量条件的性能表不得继续当作当前性能保证：有来源则注明历史条件，无来源则从当前指南移除并记录原因，不用本轮一次运行的新数字随意替代。

## 5. 内容核对方法

S0 建立页面清单，对 docs/ 的 Markdown 逐页标为“当前指南 / 当前内部说明 / API / 历史设计”，记录中文、英文对应关系及是否在导航。不要只修本文件点名的七项问题。

重点搜索：后端选择、直接写 config/state、旧 Builder 名称、旧导入路径、手动同步、共享数组、恢复和清空历史、RNG、hook 事件、空间子群生命周期、旧参数注册表和性能表。将命中与当前调用路径或行为测试对应，不把全文替换当审查。

查证时重点参考：

- `src/natal/frontend/population/` 的 config/state/update/import/restore 入口。
- `src/natal/frontend/configurator/` 的草稿写入与绑定会话更新。
- `src/natal/frontend/hooks/tick_context.py`、事务实现和现有合同测试。
- `src/natal/frontend/output/` 和当前 Rust output/session 实现。
- `src/natal/frontend/spatial/` 的初始化、deme 访问和生命周期限制。
- `src/natal/_engine_rs.pyi`、顶层导出和参数单一来源 `src/natal/parameters.jsonc`。

Rust 已完成目录整理，接手者应以实际文件路径为准，不将旧计划中的 `rust/src/session.rs` 当作必然存在的路径。源代码注释也可能过时，最终以真实调用关系和可验证行为为依据。

## 6. 需要说明清楚的行为合同

### 参数与快照

- `pop.config` 和普通状态查询的快照性质，修改快照不提交；不要泛称所有返回数组都可写或都零拷贝。
- `set_param`、`Configurator.for_config` 操作草稿的范围，与 `pop.update()` 操作运行种群的范围分开。
- 不把不同更新方法都写成可修改“所有参数”。维度/模式、生态、遗传和自定义字段分别核对合法入口与限制。
- retained updater 的读取和写入应基于当前运行数据；例子不要暗示它是永久持有的旧配置副本。

### Hook 与恢复

- callback 使用事务候选；本次 callback 成功提交、失败丢弃的范围，以及此前成功 callback 的结果是否保留，应按合同准确说明。
- 普通快照与 `TickContext.state` 的可写候选不是同一概念。避免用“所有 state 都只读”掩盖这一区别。
- RNG 由会话持有；NumPy seed 不等于 native RNG seed。文档只能使用实际受支持的种子设置入口，不能为了例子引入私有初始化方法作为推荐 API。
- failed/stopped 会话的继续条件、reset 与 checkpoint restore、raw/observation 的可恢复性、被淘汰检查点的行为应互相一致。
- `pop.tick` 为只读。完整状态导入与 checkpoint restore 的恢复内容分别查证，不承诺二者都恢复 RNG、配置或程序。

### 空间与科学说明

- 空间对象是运行状态所有者，managed deme 的查询和更新能力不等于独立 run/reset/import 权限。
- 解释配置共享时区分构建期数据去重与运行期可变状态共享，不再建议通过 `_config` 数组写入更新整个空间。
- 不把示例退出码为零、热图生成或 GUI 启动写成科学结果已复现。Drive-RIDL 与论文/SLiM 对齐需要独立数值证据，本轮不作此承诺。

## 7. 分步实施与建议 commit

建议按以下边界实施。实际 commit 需要用户另外授权；不要将已发生的历史 commit 授权视为本轮自动授权。

| 步骤 | 建议 commit 标题 | 范围与退出条件 |
|---|---|---|
| S0 | 无须提交 | 固定基线，建立页面/语言/导航与问题清单，确认扩展来自当前源码；记录两种语言文档构建基线和已存在的警告 |
| S1 | `docs: correct runtime updates and snapshot ownership` | 修运行修改、种群概述、状态配置与 API configurator；中英文同步，验证 D1 实际写入与快照隔离 |
| S2 | `docs: align hooks and recovery with the Rust runtime` | 清双后端说明；逐页核对 callback、RNG、停止/失败、历史与恢复，修正与已更新章节的冲突 |
| S3 | `docs: consolidate spatial configuration guidance` | 修空间构建、deme 所有权与合法修改方式；合并重复内容并保留旧路径链接 |
| S4 | `docs: distinguish historical designs from current guides` | 历史标记、性能表、迁移页、导航和索引同步；默认 mkdocs.yml 与英文配置保持一致的站点意图 |
| S5 | `docs: verify bilingual examples and API references` | 全文复查、API 自动引用解析、执行受影响示例、最终双语构建和差异审查；必要且非冗余的例子回归测试随所属阶段加入 |

不要先改完中文、把英文积压到最后。每阶段按页面对同步，S5 做全局一致性验收。API 公共页当前主要为共享英语内容，不凭空增加第二套重复 API 页面；先核实其在两站的实际映射方式。

## 8. 验证方案与风险分类

### 8.1 文档检查

- 检查 Markdown、本地链接、锚点、导航目标、语言切换和历史页标记。
- 分别构建 `mkdocs.en.yml`、`mkdocs.zh.yml`；同时核对默认 `mkdocs.yml`。可以使用 `python -m mkdocs build --strict -f <配置> --site-dir <临时输出目录>`，输出目录不得覆盖用户文件。
- 先调查仓库已有构建入口和 API 页面映射；`docs_dir` 与共享 `docs/api/` 的关系不能靠猜测。报告无法解析的 mkdocstrings 对象、导航缺页和失效链接，不用关闭 strict 或屏蔽警告制造通过。
- 构建基线若已失败，记录相同环境下的旧结果、具体原因与本轮影响。构建失败不能自动归咎于当前文档，也不能因“以前就这样”免除证据。
- 历史段落允许保留旧符号；当前教程的旧入口命中必须解释或修复。检查结果按 PASS/FAIL/NOT CHECKED/N/A 记录。

### 8.2 可执行示例验收

把代码块分为完整示例、延续前文的片段、伪代码/历史代码。完整示例直接执行；片段在明确且记录的上下文中执行，不凭空修补示例遗漏的核心操作；历史代码标明不可作为当前执行教程。

验收至少涵盖：

| 场景 | 行为断言 |
|---|---|
| 草稿操作与 runtime update | 快照修改不影响 live 参数；官方写入入口确实改变参数及后续行为 |
| callback 成功/失败 | 成功更新按事件生效；失败更新按承诺回滚，异常与会话状态正确 |
| raw restore 与 observation | raw 可恢复精确保留 tick；observation 和过期 tick 按合同拒绝 |
| RNG 使用说明 | 相同 native seed 的同场景可复现；checkpoint replay 与文档保证一致，不借 NumPy seed 冒充 native seed |
| 空间局部更新 | 目标 deme 改变、非目标不变；禁止的独立生命周期入口不被教程推荐 |
| 四类冻结用户表达 | 原有 setup 链、Species、preset 和 Op 示例保持语法与科学含义 |

优先复用已有合同测试；若例子修订引入尚未覆盖的真实场景，再加入小型针对性测试。不能只断言不抛异常，也不能用测试复制同一实现公式作为数值预期。

验证前确认 Python 导入路径和 native 扩展身份与当前源码匹配。示例产生 CSV/图像/HTML 时使用隔离输出目录；启动 GUI 后验证必要交互并关闭本轮临时进程，不干扰用户已有服务。

### 8.3 适用门禁与角色

纯正文、注释、导航和历史标记按文档检查，不要求为这些改动单独运行代码全量门禁。

一旦修改调用项目 API 的可执行示例或新增测试，按 AGENTS 的代码风险规则判断并运行受影响示例。涉及本方案中的状态所有权、恢复、RNG 或跨语言合同的示例按高风险安排 tester 和独立 evaluator；最终执行规范要求的 `pytest`、`pyright`、`ruff check src demos`、`python scripts/check_rust.py`。不以“文件后缀是 .md”绕过适用验证，也不为纯历史文字变动制造额外门禁。

只有新增生产可执行代码时才按质量规范测其新增行覆盖率；本计划不预设需要改生产代码。不为维持文档同步机械重写 stub；发现签名不符先区分文档错误与真实导出错误。

## 9. 完成标准与移交内容

- docs/ 页面盘点完成，已确认 D1–D7 均关闭，后续发现的问题有明确处理或披露。
- 当前教程没有仍推荐的已删除后端或静默无效写入路径；同一合同在入门、进阶、API 与内部说明之间没有冲突。
- 历史设计有清晰身份，旧链接可达并指向当前说明；未知来源的旧性能数字不作为现状保证。
- 中英文内容、索引和导航同步，必要的共享 API 页面在两站均能解析。
- 所有适用构建、示例和代码验证有实际结果。未执行项目明确标出，不能宣布文档站已全面通过。
- 最终交付列出页面改动、示例行为断言、构建命令及警告、独立审查结论（如适用）、剩余事项和 commit 列表（如获授权）。

接手 agent 可直接使用以下指令：

> 按 DOCS_CONSISTENCY_REPAIR_PLAN.md 实施 docs/ 修订。先核实 HEAD、工作区和构建基线，完整盘点页面，再按 S1–S5 分步处理。以当前实现和行为验证为依据，中英文成对修改，保留四类用户表达。优先纠正快照写入、hook、恢复和空间所有权的错误指导；合并重复内容并明确历史页身份。不要修改生产行为来迁就旧文档，不 commit/push，除非用户授权。按改动实际风险执行示例与必要独立验证，最后报告证据和未完成项。
