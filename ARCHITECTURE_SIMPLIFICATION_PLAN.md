# Rust-only 架构简化计划

日期：2026-09-10。调查基线：`6bd5731`。同日修订：并入“查询路径轻量化”与“Configurator 职责三拆”两个方向，并按“前期只做不需要重新设计、破坏面无或相对小的内容”重新分期。

状态：设计提案，尚未实施。本文记录目标、实施顺序与验收条件，不表示目标接口已存在，也不自动授权执行所有公开接口变更。本次仅修订计划文档；后续代码实施属于高风险重构。

## 目标与设计原则

让维护者能够直接回答四个问题：

1. 一个功能由哪个模块负责？
2. 数据处于声明、构建、运行还是读取阶段？
3. 一次修改在哪里真正生效，失败时保留什么？
4. 兼容行为在哪里，核心流程是否仍依赖它？

具体目标：

- PopulationBuilder（原 Configurator）负责准备种群，Population 负责控制运行，Rust session 拥有运行数据。
- 删除没有独立职责的包装、Population Mixin 继承链和重复存储位置。
- 计算逻辑用明确输入的普通函数复用，不用 Manager、Service 或新的 Mixin 重建旧依赖。
- Hook 只通过 PopulationBuilder 声明，构建时编译并注入；Population 不再提供注册入口。
- 运行更新不再依赖完整 Configurator，也不临时替换 Population 的内部字段来模拟构建环境；构建侧统一为 PopulationBuilder（原 Configurator，P5 改名），参数更新的推荐路径是构建时声明 Hook、在明确仿真时刻由事件更新，暂停期直接更新仅作辅助能力。
- 空间 deme 是父 session 的访问窗口，不是附带一套独立生命周期的隐藏种群。
- 标量与统计查询按字段、按需读取原生数据，不再为个别字段搬运完整配置或状态数组；Python 侧最终保留用户表达、构建编译、Python callback 执行和展示交互四类职责，运行数据协调留在 Rust。
- 保留已有数值规则、随机过程、执行顺序和明确的所有权合同；公开接口变化单列迁移。

不以删除行数或类的数量作为验收标准。普通数据返回包、只读视图和真实事件事务不因体积小就必须删除；也不把所有函数并入一个巨大 Population 类。

## 现状与证据

| 现状 | 代码入口 | 维护成本 |
|---|---|---|
| Population 按文件组织成多层继承 | [base.py](src/natal/frontend/population/base.py)、[_mixins](src/natal/frontend/population/_mixins/) | Output → Modifiers → Hooks，共享隐含宿主状态；ObservationMixin 已无方法 |
| 同一构建状态由 Configurator 和 CompiledModel 平行持有 | [_base.py](src/natal/frontend/configurator/_base.py)、[definition_compiler.py](src/natal/frontend/genetics/definition_compiler.py) | 草稿、registry、modifiers、来源标识需要分别接纳和同步 |
| 模型声明分成两层，内部又包含完整草稿 | [definition.py](src/natal/frontend/data/definition.py) | ModelDefinition 包含 NormalizedModel，后者的 settings 同时含输入与派生结果 |
| 编译器反向创建 Configurator | [compile_definition](src/natal/frontend/genetics/definition_compiler.py) | 配置器调用编译器，编译器调用配置器私有方法 |
| Hook 临时替换 Population 配置、日志和元数据 | [tick_context.py](src/natal/frontend/hooks/tick_context.py) | 需要准备、恢复、异常回滚和动态属性注入 |
| writer 名称与实际使用不一致 | [_writers.py](src/natal/frontend/configurator/_writers.py) | HookConfigWriter 只有测试构造调用；生产 Hook 更新使用 CoreConfigWriter 加事件事务 |
| 空间更新维护多个副本，deme 动态代理隐藏种群 | [spatial/population.py](src/natal/frontend/spatial/population.py) | Python 参数列、deme 草稿、原生数据之间存在同步代码 |
| 参数描述逐字段复制为另一种类型 | [parameters.py](src/natal/frontend/utils/parameters.py)、[_routes.py](src/natal/frontend/configurator/_routes.py) | ParamDescriptor 与 RouteEntry 保存相同字段 |
| 记录与计划存在薄包装及重复维度 | [hooks/types.py](src/natal/frontend/hooks/types.py)、[history.py](src/natal/frontend/output/history.py) | RunProgram 没有整体提交职责；SpatialHistoryLayout 重复可派生数据 |
| 遗传构建分散于 data、genetics、configurator | [data/_engine.py](src/natal/frontend/data/_engine.py)、[genetics/compile.py](src/natal/frontend/genetics/compile.py) | 遗传矩阵、生态计算、压缩和草稿装配混杂 |
| Configurator 双重身份：构建草稿与运行时句柄 | [_base.py](src/natal/frontend/configurator/_base.py) | 约 13 处构建/运行分支；提交目标选择在 `_make_writer`、`custom()` 内联、`_commit_genetic_candidate` 重复三份 |
| 运行时句柄由 `for_population` 制造并兼作内部编译种子 | [_base.py](src/natal/frontend/configurator/_base.py)、[_modifiers.py](src/natal/frontend/population/_mixins/_modifiers.py) | 除 `pop.update()` 外另有 8 处内部调用（6 个 preset/modifier 刷新方法、`fitness()`、`_genetic_candidate`）；`.build()` 与 `.initial_state()` 在句柄上无守卫，会静默构建第二个种群或静默偏离活动种群 |
| 标量参数读取走完整配置快照 | [_params_view.py](src/natal/frontend/population/_params_view.py)、[rust_backend.py](src/natal/backends/rust/rust_backend.py) | `pop.params.x` 拉取含全部遗传张量的整份快照且无缓存；原生 `get_scalar`/`get_tensor` 已存在但读取路径未使用 |
| 统计量在 Python 侧求和 | [discrete_generation.py](src/natal/frontend/population/discrete_generation.py)、[age_structured.py](src/natal/frontend/population/age_structured.py) | 计数查询对 Python 状态缓存求和，缓存过期时拉取整个状态数组；原生无专门计数接口 |
| 观测查询每次重建 mask，run 边界重装 | [base.py](src/natal/frontend/population/base.py) | `observe()` 每次调用重新 `build_mask`；每次 `run()` 重新配置观测并重拷 mask，构建时已有的冻结 mask 未被查询复用 |
| 运行状态两侧各持一份 | [base.py](src/natal/frontend/population/base.py)、[status.rs](rust/src/sessions/status.rs) | Python `_tick`/`_finished`/`_failed` 与 Rust `ExecutionStatus` 在 run、callback、restore 多点同步并对账 |
| 历史保留由 Python 每次 run 绑定 | [age_structured.py](src/natal/frontend/population/age_structured.py) | Rust 已有 FIFO 淘汰与 checkpoint retain/truncate 能力，但淘汰与历史的配对由 Python 在每次 run 时重新组织 |

使用面量化（2026-09-10 核查，作为迁移规模基线）：后注册 `register_hooks` 约 150 处（149 测试 + 1 demo）；`pop.update(` 约 260 处（测试 160、demo 15、文档 85）、`ctx.update(` 约 44 处，其中句柄上链式调用 `.hooks` 约 55 处；`.normalized` 外部使用 9 处；`CompiledModel` 外部使用 0 处；`HookConfigWriter` 构造 4 处且全在测试。恢复与克隆路径不经过 `for_population`，直接注入 Rust session，不受三拆影响。

上述为静态调用关系与实现事实，不等同于已经复现运行错误。此前提交的测试结果不作为本轮未来实现的验证结果。

## 目标结构与数据归属

### 主要对象

| 对象 | 拥有的内容 | 不负责的内容 |
|---|---|---|
| Species 及遗传实体 | 遗传结构、对象身份、物种级遗传基线 | 运行中的种群参数、状态 |
| ModelDefinition | 完整声明、规则顺序、输入来源、空间设置 | 完整派生矩阵缓存、session |
| PopulationBuilder（原 Configurator，P5 改名） | 当前构建草稿、registry、必要 modifier 产物、编译有效性信息 | 已运行种群的权威参数和状态；`for_population`、`_pop_ref`、`_hook_context` 等运行句柄机制 |
| RuntimeUpdater（`pop.update()` 与 `ctx.update()` 共用） | 提交目标（空闲 session 或当前事件事务，事件绑定实例随回调过期）与必要模型元数据；恰 8 个域方法 | `.build()`、`.initial_state()`、`.hooks()` 等构建专属能力，以及构建缓存与构建日志 |
| ModelDraft | 维度、执行设置、数值参数、遗传/fitness 数组、初始状态、custom | registry、recipe 对象、日志、当前 tick/RNG/History |
| IndexRegistry | 遗传对象与活动数组索引的对应关系 | 运行状态、规则执行 |
| Population | session、必要声明与布局元数据、明确的运行和查询入口 | 第二份权威运行参数、构建编译缓存 |
| SpatialPopulation | 父空间 session、拓扑和布局元数据、deme 访问入口 | 每个 deme 的独立生命周期 |
| TickContext | 当前事件事务、受限读写、RNG 入口，以及挂在其中的事件专用更新入口 | 临时改装 Population 的配置环境；任何构建能力 |
| History、Observation、HookProgram | 各自的记录、投影和事件执行表示 | 彼此的总包装或模型编译状态 |

声明与数值结果分开：例如用户的适合度规则属于 definition，展开后的 fitness 数组属于 draft。原始 fitness 基线、显式数组输入和规则顺序不能因去重而丢失。用户 callable、锁等外部资源保留身份，不强行深拷贝或序列化。

### 模块组织

- `builder/`（原 `configurator/`，P5 随类改名一次迁移）：链式声明、构建流程和候选接纳。
- `genetics/`：结构、实体、遗传基线、modifier 应用与遗传矩阵编译。
- `fitness/`：patch 类型、构造、解析和应用；presets 产生这些规则。
- `registry/`：索引及一致性；遗传编译不得为索引构造反向依赖 Configurator 私有模块。
- `model/`：建议新增的少量普通文件，容纳 definition、draft、初始状态解析及合同装配。不得成为新的总控类体系。
- `population/`：共同运行入口、模型特有状态处理和更新提交；复杂计算放普通函数。
- `spatial/`：拓扑、迁移、分组构建、共享遗传变体及父 session 操作。
- `hooks/`：声明编译、回调执行和事件事务上下文。
- `output/`：观测编译、历史、结果转换。
- `contracts/`：原生交换格式；将认识 ModelDraft 的 materialize 实现移至模型装配侧。
- 原生适配代码：保留类型转换和错误转换，后端目录压平低优先级，不作为本轮收益核心。

文件移动应跟随职责迁移，避免先批量改路径、再维持原有往返依赖。遗传结构内部的 Mixin 本轮不机械删除；先保留其缓存、枚举与对象身份行为。

## 构建流程与类型简化

目标流程：

```text
PopulationBuilder 收集并解析声明
  → 按既有时机准备独立候选、调用领域计算函数
  → 成功后接纳 draft、registry、modifier 产物
  → build 完成活动类型压缩与最终索引
  → 编译 Hook、Observation、记录布局
  → 装配 Blueprint / Params
  → 创建并初始化种群与 session，完整注入计划
```

具体决策：

- 合并 NormalizedModel 与 ModelDefinition，取消 `.normalized` 作为必经中间层。
- 删除 CompiledModel 长期包装。PopulationBuilder 是构建工作状态的唯一持有者，编译函数不再创建构建器或写其私有字段。
- Configurator 职责三拆（P5 随改名落地为 PopulationBuilder 与 RuntimeUpdater）：构建专属 PopulationBuilder（声明、草稿、registry、编译准备数据）；`pop.update()`/`ctx.update()` 共用的 `RuntimeUpdater`（构造注入提交目标：空闲 session 或事件事务）；参数解析、校验、选择器解析和遗传矩阵计算提取为不长期持有状态的普通函数。共享的是函数，不是配置器对象。
- `for_population`、`_pop_ref`、`_hook_context` 与三份重复的提交目标选择（`_make_writer`、`custom()` 内联、`_commit_genetic_candidate`）从 Configurator 删除；8 处内部调用（6 个 preset/modifier 刷新方法、`fitness()`、`_genetic_candidate`）改走共享函数。`.build()`、`.initial_state()` 等构建方法在更新入口上不存在，顺带消除当前句柄上无守卫的静默错误。
- 候选需要隔离，但不因此新增完整 CandidateModel 类；可以由 PopulationBuilder 创建受控副本，领域函数只接收需要的数据。
- 候选接纳集中在一个内部操作，draft、registry 和必要产物一起发布。不能浅拷贝后继续共享将被写入的数组或列表。
- `_ComputedMaps`（位于 `data/_config.py`）随统一草稿装配取消，公共构建直接生成 ModelDraft；保留年龄结构与离散模型的真实差异。
- 原始声明日志服务于来源和重建，完整声明服务于计算；不再通过反复重放日志恢复已有编译结果。
- `.build()` 不重新执行已有效的 recipe；冷重建从声明执行，遵守现有身份与顺序合同。

编译有效性不能只用一个布尔值概括。先列出哪些操作改变声明、哪些产物受影响，再使用最少的来源标识和显式状态表达。尤其保留当前延迟 modifier 注册和“生成 modifier 但未提交矩阵”的合同，除非单独批准移除该公开行为。不引入全局哈希缓存，不默认用户函数是纯函数。

空间构建继续按共享遗传配置分组，保留生态变体复用和压缩所需的全局可达类型集合。不能因删除 CompiledModel 变成每个 deme 都执行一遍相同 recipe 或复制完整张量。

## Hook：仅通过 PopulationBuilder 注入

本轮目标明确取消种群构建后的 Hook 注册和替换。注册和触发是不同能力，手动触发已有事件继续保留。参数更新以 Hook 为推荐路径：修改发生的 tick、事件阶段和执行顺序由此明确，并自然纳入事件事务与日志（立场已定稿，见“已定稿的取舍”第 5 条）。

### 构建职责

1. PopulationBuilder 与 SpatialPopulationBuilder 是唯一声明入口。
2. 在压缩前收集 Hook 引用的遗传类型，保护只由 Hook 引入的类型；不能为了晚编译而把它们剪掉。
3. 最终 registry 确定后解析选择器并生成执行计划。
4. 声明式操作与 Python callback 使用同一优先级和稳定顺序，保持事件、STOP、条件和 deme 选择语义。
5. 空间计划在父空间构建流程中生成，保留共享描述的去重和逐 deme 适用范围，不再由运行种群遍历内部 deme 重新注册。
6. 创建 Population 时一次注入 CSR 计划及其 callback 绑定信息。不新增类似 CompiledEventHooks 或 RunProgram 的总包装。

### 删除与保留

- 删除 Population、SpatialPopulation 的公开 `register_hooks`，并阻止经 deme、更新入口或 Hook 上下文绕回注册。
- 移除 raw `hook_items` 从种群构造后再注册的流程、`_pending_hook_items`、`_finalize_hooks` 和运行时重绑链。
- `build(..., hook_items=...)` 若保留，只作为 PopulationBuilder 内的声明便利写法，不在 Population 上重放。
- 保留手动事件触发、finish 事件以及原生 callback 桥接；HookRunner 若仍负责执行顺序和事务上下文，可保留其执行职责。
- 构建后计划结构固定；外部对原 HookOp、列表、数组的修改不能改变已注入计划。用户 callback 的闭包内容不承诺深度冻结。
- 查询接口如保留，应返回不允许修改内部计划的结果；不能通过取回内部数组实现事实上的热替换。
- 克隆、reset、checkpoint 恢复复用或重新绑定已有计划，不能重新注册、重排或重复执行声明。
- Rust 的低层安装方法可服务于初始化和恢复，不因 Python 注册入口删除而机械删除原生必要功能；不将其包装成新的公开热替换入口。

## Population：取消 Mixin 继承链

保留 BasePopulation、AgeStructuredPopulation、DiscreteGenerationPopulation；SpatialPopulation 独立组织空间生命周期。

- 生命周期入口 run、reset、restore、finish 归种群类。
- state、params、history、observe 是种群的明确访问入口。
- Hook 编译、遗传计算、观测编译、频率统计和格式转换使用普通函数；状态敏感操作由拥有者组织。
- 删除四个 Population Mixin、重复抽象方法、仅为宿主属性声明而保留的 Any 和类型忽略。
- 合并 `_registry` 与 `_index_registry` 存储；需要兼容两个公开属性时，它们读取同一字段。
- 不以代码行数为由新增委托对象。确实拥有独立状态或生命周期的现有对象可以保留。

## 更新：准备、校验、提交

目标是共享参数解析和计算，提交目标明确区分：构建草稿、空闲 session、当前事件事务、父空间 session 的 deme。职责按三拆组织，推荐的更新方式是构建时声明 Hook、在事件内更新；暂停期间直接修改参数仅作辅助入口：

```text
构建：PopulationBuilder → 解析／计算函数 → 草稿 → 创建种群
更新：RuntimeUpdater（事件绑定或空闲 session）→ 解析／计算函数 → 候选 → 提交
```

- 普通运行更新不创建完整构建器。`pop.update()` 与 `ctx.update()` 返回同一个 `RuntimeUpdater`，原名原语法保留；构造时注入提交目标（空闲 session 或当前事件事务），事件绑定实例随回调过期。方法面恰为 8 个域方法：competition、reproduction、survival、custom、presets、modifiers、fitness、reconfigure_preset。
- `.setup()`、`.initial_state()`、`.hooks()`、观测与记录配置、`.build()` 只存在于 PopulationBuilder；更新入口根本不提供这些方法，而不是调用到深处按模式报错。
- 更新句柄保存目标，不保存一份长期参数快照；句柄跨 run 保留后仍按操作时的原生值计算。
- 参数描述合一；保留 ResolvedWrite 或同等明确的验证结果，不为减少类型把已验证和未验证输入混在一起。
- 删除仅测试使用的 HookConfigWriter 生产导出候选；将其他 writer 中实际共用的解析、校验和候选计算提为函数，再取消无必要的类层次。
- 标量更新只读取必要标量，不为一个参数复制完整 genetics。张量更新读取实际依赖；meiosis 与派生 offspring 在同一提交中一致更新。
- 遗传规则更新仍需独立候选，但可使用共有构建计算函数，不通过 runtime Configurator → candidate Configurator → compiler Configurator 绕行。
- 原生提交成功后才发布当前声明、modifier 元数据及日志；失败不泄漏候选。初始 definition 与后续有效声明含义保持区分。
- 构建时的初始状态不能随运行参数更新再次覆盖当前个体数量。

Hook 更新直接从 TickContext 取得事件事务。候选配置、待发布元数据和日志属于本次事件操作，不再临时替换 Population 的 `_config`、`_params_log` 或动态挂载 `_event_*` 字段。保留惰性读取，不让只读 tick 查询复制完整参数。

事件作用域和一次方法调用的原子性沿用现有合同，不扩展为整条链、整批 tick 或整个 run 回滚。NATAL 不承诺撤销用户函数对外部文件、闭包或其他资源的副作用。

## 查询路径轻量化

方向：取消 Python 对原生数据的全量搬运、重建和重复管理。Python 最终保留四类职责——用户表达（Species、Configurator、preset、selector）、构建编译、Python callback 执行、展示与交互；运行参数、状态、RNG、执行与记录的协调尽量留在 Rust。不把遗传对象、参数语法和 UI 搬到 Rust，也不用可写原生 NumPy 视图代替复制（避免重新引入共享可变数据问题）。

| 方向 | 当前做法 | 目标做法 | 前置条件 |
|---|---|---|---|
| 按字段读参数 | `pop.params.x` 经 `pop.config` 拉整份配置快照（含遗传张量），无缓存 | 标量走 `get_scalar`，数组只取目标 tensor；完整配置仅在显式导出时组装 | 原生 `get_scalar`/`get_tensor` 已存在（含事件事务侧）；空间需补会话级标量读取 |
| 直接查询统计量 | 计数在 Python 侧对状态缓存求和，缓存过期拉全量状态数组 | Rust 对已有状态求和，只返回结果 | 需新增少量原生聚合/计数接口，或复用 `observe_current` 聚合能力 |
| 观测计划只安装一次 | 查询每次重建 mask；每次 `run()` 重配观测并重拷 mask | 构建时安装固定方案，查询只返回投影结果，mask 变化时才重装 | 无：构建时已有冻结 mask，仅需复用 |
| 消除空间状态往返 | 拉堆叠状态 → 分发到 deme → 查询时再 stack | 单 deme 读切片，整体查询直接用原生堆叠状态 | 归属空间所有权阶段 |
| 统一运行状态来源 | Python 管 `_tick`/`_finished`/`_failed`，Rust 有 `ExecutionStatus` | 可统一部分归 session；Python 保留必要重入保护 | 先列 STOP、finish、失败恢复的语义清单 |
| 原生管理记录保留 | Python 每次 run 绑定历史淘汰与 checkpoint 配对 | 原生统一管理保留与淘汰，Python 提供操作入口 | Rust 已有 FIFO 淘汰与 retain/truncate 能力；绑定时机与生命周期单独设计 |

前三项为前期低破坏面工作（见分期 P1）；空间往返归空间所有权阶段；运行状态与记录保留单列后期阶段，避免误改 STOP、finish、失败恢复行为。本轮性能收益尚未测量，验证以行为等价与调用/分配证据为准，不以任意耗时阈值代替。

## 空间所有权简化

- 父空间 session 持有堆叠状态、参数列、遗传变体与 RNG。Python 只保留必要声明、布局和受控读取缓存。
- DemeSlice 显式提供与 Population 对齐的访问面（读、查询、`update()` 返回同一 `RuntimeUpdater`）及 `write_ecology`/`write_genetics`，目标为父 session 加 deme 索引；内部 `typing.Protocol` 固定两侧对齐。
- 逐步取消内部完整 Population 被动态代理的设计，移除任意 `__getattr__`、`__setattr__`、`__delattr__` 转发；这是公开行为变化，须先确认迁移范围。
- 不再以刷新全部内部 deme 快照再重新 stack 的方式完成原生已有状态的聚合；按需读取原生堆叠状态或使用已有原生观测能力。
- 空间会话级按字段读取（`get_scalar` 等）作为增量原生接口在前期补齐，供本阶段 deme 读取与提交复用；彻底消除“拉堆叠 → 分发 → 再 stack”的往返在本阶段完成。
- 删除参数列、deme 草稿、原生数据三处同步的权威状态关系；兼容快照由当前 session 派生。
- 保留共享遗传变体及写时隔离。核对 fork 与 tensor refresh 的原子边界，必要时提供一次完成的原生更新；不得在未复现前宣称当前存在原子性 bug。
- 保留实际分组、压缩、迁移算术顺序、标签语义及 callback 作用域。不得以简化为由改成逐 deme 完整复制。
- 测试替身应满足明确合同，不应让生产代码长期维护 `_minimal_contract` 等专为不完整替身设计的降级模型；迁移测试构造后再判断哪些 fallback 可删。

## 其他结构清理

| 项目 | 处理 |
|---|---|
| data 包的混合职责 | 遗传计算归 genetics，草稿/声明/初始输入归 model，生态桥接明确归属；状态快照保留为真实结果类型 |
| fitness 与 presets 往返导入 | patch 类型、构造和应用归 fitness；presets 只产生规则；内部不绕兼容导出 |
| RunProgram | 删除，Hook 与记录计划分别持有 |
| SpatialHistoryLayout、ObservationMetadata.n_groups 等 | 当前布局下可派生的数据改为统一派生；核对已有 schema 导入合同 |
| HistoryBatch | 先确定历史导入能力；若保留，校验移到明确导入入口，不因只剩测试使用而删除有效验证 |
| SpatialMigration | 随合同装配迁移简化；不得与含 stay_after_send 的 MigrationCSR 当作完全等价 |
| BlueprintView | 保留受限只读含义，简化手写 getter，避免引入不必要的生命周期变化 |
| 旧 observation 字典与 selector | 入口转换为统一选择表示；核对年龄、标签、空匹配、压缩索引；UI 和 translation 一起迁移 |
| 无效兼容层 | 删除内部纯转发、重复编号函数和失效说明；公开别名与删除政策单列 |

## 公开接口迁移清单

以下为目标提案，不是当前行为。Hook 注册收敛是用户明确提出的方向；其余破坏性变更需在对应实施前一次性确认具体清单，无需逐文件询问。

| 接口 | 目标 | 迁移与验证 |
|---|---|---|
| Configurator / SpatialConfigurator 类名与包目录 | 改名 PopulationBuilder / SpatialPopulationBuilder，`configurator/` → `builder/`，不留别名 | P5 一次换净：`__all__`、`.pyi`、文档、demos（src 204 / tests 274 处引用机械替换） |
| PopulationBuilder.hooks / SpatialPopulationBuilder.hooks | 保留，唯一 Hook 声明入口 | 迁移 demo、文档和测试中的后注册写法 |
| Population / SpatialPopulation.register_hooks | 删除 | 负合同覆盖所有导出和 deme 入口；构建时声明替代 |
| raw Population 构造器 | 收为内部机制（build/克隆/恢复内部共用）；文档撤低层路径章节 | `extreme_speed_mode` 等回 setup 链式 API；11 处测试直接构造迁移；`hook_items` 参数删除 |
| update().hooks / ctx.update().hooks | 不支持 | 清晰拒绝或无该属性；测试不能误用 build 接口 |
| trigger_event、finish、既有 Hook 执行语义 | 保留 | 不把触发与注册混为一谈 |
| ModelDefinition.normalized、NormalizedModel | 删除 | 读取统一声明；同步序列化/重建用法及 stub |
| CompiledModel | 删除 | 内部构建缓存迁移；搜索外部文档及测试导入 |
| pop.update 返回完整 Configurator | 改为返回 `RuntimeUpdater`（恰 8 个域方法，无任何构建能力） | 原名原语法保留，约 260 处调用零改写；`.hooks()` 链约 55 处随 P4 迁移 |
| ctx.update() 返回类型 | 同一 `RuntimeUpdater`，注入事件事务目标并带寿命守卫 | 迁移约 44 处用法；链式域方法语法不变 |
| 原生查询接口（字段读取、计数聚合） | 新增，纯增量 | 字段值、统计量与现有快照路径行为等价；无公开签名破坏 |
| PopulationConfigBuilder、旧 writer（ConfigWriter/CoreConfigWriter/DraftWriter）与 RouteEntry 导出 | 移出顶层导出，不留别名；PopulationConfigBuilder 类删除、静态方法降为普通函数 | writers 随 P5；RouteEntry 随 P2 参数描述合一；均无文档与 demos 使用，仅测试引用迁移 |
| deme 任意动态属性代理 | 完全撤销；DemeSlice 显式面与 Population 对齐（见定稿第 3 条） | 对齐面约 290 处使用零迁移；`_config`、`custom_marker` 等少量测试迁移；内部 Protocol 固定对齐 |
| HistoryBatch、SpatialHistoryLayout | HistoryBatch 保持内部不动；SpatialHistoryLayout 随 P9 移出导出并统一派生 | 派生清理核对存量 schema 导入兼容 |
| 延迟 modifier 注册 / 刷新方法 | 默认保留行为 | 若要撤销延迟状态，需单独批准；本轮不擅自改为立即执行 |
| pop.config、state 快照与显式导入 | 保留行为 | 返回表示可兼容，运行所有权不回退到 Python 草稿 |

不新增生产依赖，不在本轮改科学模型公式、RNG 算法或迁移舍入次序，不自动 commit/push。

## 分阶段实施与交付物

阶段是逻辑工作单元，不自动要求创建 Git 提交。每一阶段结束后应可运行并有针对性验证，不能直到最后才得到一个可运行工作树。分期原则：前期阶段只包含不需要重新设计、公开行为不变或破坏面相对小的内容；重设计与公开破坏性变更集中在后期并保持串行。

| 阶段 | 破坏面 | 工作内容 | 交付与退出条件 |
|---|---|---|---|
| P0 合同与基线 | — | 记录实际基线；枚举构建/更新/Hook/空间/查询入口及测试并附使用量（150 处后注册、260 处 update 等，见现状量化）；确认公开迁移清单；前置核对：webui 工作落盘、当前分支合并、验证命令在本地环境可用 | 每项行为标记保留、迁移或待确认；已有失败有实际基线证据；前置满足或明确记录未满足原因 |
| P1 查询路径轻量化 | 低 | 按字段参数读取（复用已有 `get_scalar`/`get_tensor`，补空间会话级读取）；观测查询复用冻结 mask、run 边界去重重装；统计量原生查询接口 | 数值等价（事件内外路径一致）；调用/分配证据证明不再全量搬运；无公开签名变化 |
| P2 机械冗余清理 | 低 | 合并参数描述、registry 存储，删除空 Mixin、私有纯转发、`HookConfigWriter`（迁移 4 处测试构造）与 `PopulationConfigBuilder`（静态方法降为普通函数）；初始状态解析提取为函数 | 主流程不借旧 builder 解析；关键类型检查与针对性测试通过 |
| P3 构建状态统一 | 中 | 合并声明，删除 CompiledModel、`_ComputedMaps`（位于 `data/_config.py`）包装，领域编译脱离 Configurator 私有实现 | recipe 次数、顺序、候选隔离、冷构建、空间分组复用保持 |
| P4 Hook 构建注入 | 高 | 在最终索引上编译计划并注入，迁移空间 Hook 编译，删除所有公开后注册入口、`hook_items` 参数与 `RunProgram`，迁移 `update().hooks` 约 55 处链式用法 | 优先级、稳定次序、条件、STOP、保护类型、deme 选择和负合同通过 |
| P5 运行更新三拆与更名 | 高 | 删除 `for_population`/`_pop_ref`/`_hook_context` 与三份重复目标选择；8 处内部调用改共享函数；`pop.update()`/`ctx.update()` 共用 `RuntimeUpdater`；改名 `PopulationBuilder`/`SpatialPopulationBuilder` 并迁移包目录 `builder/`；writers 移出顶层导出 | 句柄有效、失败隔离、日志正确、Hook 生命周期及 RNG 合同通过；更名后 `__all__`、stub、文档、demos 一致，无 Configurator 残留 |
| P6 Population 去 Mixin | 中 | 将剩余共同生命周期归 BasePopulation，算法函数化，删除 Modifier/Output/Hook Mixin | 无 Mixin 串联、重复抽象声明；年龄/离散生命周期测试通过 |
| P7 运行状态与记录生命周期 | 中 | 先列 STOP、finish、失败、恢复语义清单；`_tick`/`_finished`/`_failed` 可统一部分归 session，Python 保留重入保护；历史保留与 checkpoint 淘汰一次性绑定、原生管理 | 语义清单评审通过；恢复、回滚、`max_rows`、`clear_history` 行为与现状一致 |
| P8 空间所有权 | 高 | 显式 DemeSlice（访问面与 Population 对齐，内部 Protocol 校验）、父 session 读取和提交、消除三处同步、隐藏生命周期与空间状态往返 | deme 隔离、变体共享、恢复、聚合、空间 callbacks 通过；未列属性一律 AttributeError |
| P9 输出与遗留入口 | 中 | 统一 selector，迁移 UI/translation，处理计划与历史冗余类型，完成真实职责目录整理 | 核心无旧入口依赖；文档/stub/示例与目标一致 |
| P10 统一独立审查 | — | 对最终整体变更审查并补强测试，修复与复核，执行完整门禁 | evaluator APPROVED，全部适用验证有实际记录 |

主依赖为 P0 → P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10。P1 与 P2 相互独立、均为低破坏面，可先后或并行推进；其余共同核心保持串行，不并行修改 Configurator、Hook bridge 与空间所有权。P7 放在更新与去 Mixin 收敛之后、空间所有权改造之前，避免空间改造重复处理旧状态模型。P6 仍留在 Hook 与更新路径收敛后完成，避免先把绕行逻辑整体搬进 BasePopulation。原计划 P0–P8 依次对应现 P0、P2、P3、P4、P5、P6、P8、P9、P10；P1 与 P7 为本次新增。

## 验证计划

下表列已有证据入口，不要求固定保留测试文件划分。测试应保留行为，去掉对已批准废弃结构的依赖；不得弱化有效断言。

| 合同 | 关键场景 | 已有测试入口 |
|---|---|---|
| 编译 | preset/modifier 顺序、执行次数、fitness 覆盖、延迟刷新、失败候选隔离 | [compile_unification](tests/test_compile_unification.py)、[review_compilation_boundaries](tests/test_review_compilation_boundaries.py) |
| 普通更新 | retained updater 跨 run，标量不拉取遗传张量，错误不发布、派生矩阵一致 | [routes_slice3](tests/test_routes_slice3.py)、[session_state_ownership](tests/test_session_state_ownership.py) |
| Hook | CSR/callback 统一顺序、条件、STOP、真实 tick 日志、后续 callback 失败、上下文过期 | [native_hook_surface](tests/test_native_hook_surface.py)、[rust_session_bridge](tests/test_rust_session_bridge.py) |
| 空间声明 | batch 一次展开、冷构建、资源身份、压缩与共享 | [spatial_normalized_inputs](tests/test_spatial_normalized_inputs.py) |
| 空间运行 | 单 deme 更新隔离、共享变体、所有更新入口一致、快照与父 session 所有权 | [spatial_update](tests/test_spatial_update.py)、[spatial_session_ownership](tests/test_spatial_session_ownership.py) |
| 历史与恢复 | raw/observation、保留界限、记录/checkpoint 间隔、状态导入与完整恢复区别 | [single_history_contract](tests/test_single_history_contract.py)、[session_state_ownership](tests/test_session_state_ownership.py) |
| 查询等价 | 字段读取事件内外一致、统计量与 observe 结果同现有快照路径一致、不再全量搬运 | 新增针对性测试；复用 [native_hook_surface](tests/test_native_hook_surface.py) 的事务内读取场景 |
| 运行状态与保留 | STOP/finish/失败/恢复后两侧状态一致；`max_rows` 与 `clear_history` 淘汰配对不变 | [session_state_ownership](tests/test_session_state_ownership.py)、[single_history_contract](tests/test_single_history_contract.py) |
| 接口删除 | 导出、stub、构造器、deme、update、Hook 所有访问路径 | 迁移已有负合同并补充新的明确删除合同 |

必须新增或核对的跨阶段场景：

- Hook 引用的类型只在事件中出现，压缩仍保留并正确映射。
- 构建后修改原声明列表/HookOp/NumPy 数组，不改变已安装计划；callback 外部资源身份保持。
- reset、clone、restore 不重复注册或额外调用 recipe，session/RNG 行为符合原合同。
- Hook 内遗传更新失败后，候选参数、声明、日志不泄漏；不撤销此前已完成的合法 tick。
- 两个 deme 共享遗传矩阵，更新其中一个后另一个数值和随机轨迹仍符合既有隔离合同。
- 空间总体读取不依赖逐 deme 全量快照再拼回；标量写入不触发完整矩阵复制。使用调用观察或分配证据验证，不以任意耗时阈值代替。
- 简化声明与缓存后，不把已经应用的 fitness 或 modifier 再叠加一次。
- 字段级读取在事件事务内返回候选值、事务外返回已提交值，与现有全量快照路径逐字段一致。
- 运行状态统一后，restore、回滚与失败路径的 tick 与标志同现状一致；重入保护仍在 Python。
- 历史保留绑定一次化后，运行中调整 `max_rows`、`clear_history` 与 checkpoint 淘汰配对行为不变。

最终质量政策以 [quality_checks_spec.md](quality_checks_spec.md) 和 [AGENTS.en.md](AGENTS.en.md) 为准，不在此新增覆盖率或审批规则。当前环境使用 `.venv/bin/python -m pytest`、`-m pyright`、`-m ruff check src demos`，以及 `.venv/bin/python scripts/check_rust.py`；涉及扩展重建时确保 CPython 环境与 PYO3_PYTHON 一致。

公开接口迁移同步 `docs/zh/`、`docs/en/`、相关 demo 与 `.pyi`，按规范生成顶层 stub。大范围迁移执行全部适用 demos 和可运行文档示例；超时、缩小规模、抽样参数和未执行部分明确披露，不能称全部原规模通过。

本计划只有流程示意，没有声称可运行的目标 API 示例。本次文档交付只检查事实、链接、格式和范围，不运行全量代码门禁。

## 协作与范围控制

- 主 agent 负责最终设计、问题定位、任务切分、跨模块关系核对和集成。
- 按用户此前要求，具体实现可交给子 agent，也可以主 agent 自己做。任务按上述阶段明确到文件及合同；产品修复仍由实现者完成。
- 所有实现、基本自测、文档与 stub 完成后进行一次统一 evaluator 审查。独立 evaluator 使用 adversarial-review，按需补强数值测试并执行最终门禁；缺陷修复后由其复核，不以主 agent 自审代替。
- 不把多个实现 agent 的局部“完成”直接等同于整体完成。共同核心顺序推进，独立的测试调查或文档准备才并行。
- 每阶段说明实际删除了哪些状态/路径、保留了哪些合同、已执行哪些验证；不使用 suppress、skip 或删除有效测试制造通过。

## 已定稿的取舍（2026-09-10 grilling 定案）

方向与接口清单均已确认，实施按此执行，不再逐方法审批。编号沿用原待定稿清单。

1. **raw Population 构造器**：收为内部机制，`build()`、克隆、恢复内部共用同一构造器；公开文档撤掉低层构造章节（`2_population.md` 中英）。`extreme_speed_mode` 等低层专属参数回到 setup 链式 API，其他同类参数如出现同样只回链上、不给低层路径开口子；测试直接构造的处理在 P10 审计中澄清：以构造器解析/校验/自定义张量为测试对象、或刻意演练内部物化路径（build/克隆/恢复共用）的测试保留该路径并逐处注明"internal materialization path"；以直接构造作为普通构建便利的站点已在 P4 迁移至构建链。`hook_items` 参数与 `_pending_hook_items`/`_finalize_hooks` 延迟注册机制随 P4 删除。
2. **`pop.update()` 新返回类型**：放弃 Configurator 概念。构建侧更名 `PopulationBuilder`（新名，非复活 `PopulationConfigBuilder`），`SpatialConfigurator` 更名 `SpatialPopulationBuilder`；P5 一次换净类名、`__all__`、`.pyi`、文档、demos 与包目录（`configurator/` → `builder/`），不留任何别名。`pop.update()` 原名原语法保留，与 `ctx.update()` 返回**同一个** `RuntimeUpdater`：构造时注入提交目标（空闲 session 或事件事务 + 寿命守卫），方法面恰为 8 个域方法（competition、reproduction、survival、custom、presets、modifiers、fitness、reconfigure_preset），repr 带目标、过期报错明确；不为两边预分两个类型，将来方法面真分化时再拆子类（加子类不破坏用户）。"运行时 Configurator"无别名、无兼容层。
3. **deme 动态代理**：完全撤销 `__getattr__`/`__setattr__`/`__delattr__` 转发。DemeSlice 显式面与 Population 对齐——读：`name`、`species`、`config`、`state`、`params`、`params_log`、`index_registry`、`presets`、`definition`；查询：`get_total_count`、`get_female_count`、`get_male_count`、`export_config`、`export_state`；更新：`update()`（返回同一 `RuntimeUpdater`）。deme 独有：`index`、`write_ecology`、`write_genetics`；容器独有：`run`、`reset`、`restore_checkpoint`、`finish`、`clone`、`trigger_event`、`history`、`observe`。未列出属性一律 AttributeError。内部 `typing.Protocol`（不导出）固定对齐面，两侧漂移由类型检查拦截。迁移：约 290 处对齐成员使用零迁移；`_config` 2 处、`custom_marker` 1 处测试迁移；`compiled_hook_descriptors` 21 处归 P4 查询接口去留决策。
4. **旧 builder/writer 与公开 schema 类型**：无文档、无使用者的顶层导出全部收编、不留别名——`ConfigWriter`/`CoreConfigWriter`/`DraftWriter` 随 P5 移出导出；`RouteEntry` 随 P2 参数描述合一消失；`PopulationConfigBuilder` 类整体删除、静态方法降为普通函数（P2）。`SpatialHistoryLayout` 随 P9 移出导出，可派生字段统一派生并核对存量 schema 兼容。`HistoryBatch` 保持内部校验边界原样不动，不升公开、不删验证。
5. **更新推荐路径**：采用“Hook 为主、双场景并列”。Hook 更新是默认推荐（时机、事务、顺序明确）；tick 间直接更新（`pop.update()`/`pop.params`）保留为一等场景。文档开头加“怎么选”指引，修正 `pop.params` 上“preferred”措辞。行为不变：hook 内裸 `pop.update()`/`pop.params` 写入本就被 run 守卫拒绝，`ctx.update()`/`ctx.params` 走事件事务。

默认保留延迟 modifier 行为、手动事件触发、数值与 RNG 语义，不把这些纳入简化范围。Hook 构建后固定是本计划的明确目标，不计划另建热替换机制。

## 最终完成标准

- 四个问题均能从明确的拥有者和调用入口回答。
- 无 Population Mixin 链、无 NormalizedModel/CompiledModel 平行模型状态、无 RunProgram 空包装。
- Hook 只由 Configurator 注入，无种群后注册及隐蔽修改旁路。
- 运行更新不创建完整 Configurator，不临时替换 Population 以模拟事件配置环境。
- 查询路径不再为个别字段搬运完整原生配置或状态数组，并有调用或分配证据。
- `RuntimeUpdater` 无任何构建能力；构建侧统一名为 PopulationBuilder，Configurator 概念及其运行句柄机制全部消失。
- 运行状态以 Rust session 为单一权威，Python 仅保留必要重入保护。
- Rust 是运行数值唯一权威来源，deme 明确从属于父 session。
- 删除的实现没有被同等数量的新包装、反射路由或通用框架替代。
- 公开迁移、文档、stub 和测试一致；最终独立审查与完整验证按仓库规范完成。
