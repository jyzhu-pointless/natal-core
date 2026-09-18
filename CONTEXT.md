# NATAL Core — 领域与架构上下文

> 本文件记录当前领域术语、模块职责和架构边界，不维护重构进度或临时任务清单。
>
> **核对基线**：2026-09-14，提交 `71b18c4`。下述路径均相对于仓库根目录。
> 授权与协作遵循 [AGENTS.md](./AGENTS.md)，质量要求遵循
> [quality_checks_spec.md](./quality_checks_spec.md)，文档与类型格式遵循
> [docstring_spec.md](./docstring_spec.md)。本文件不另设文件行数或质量门禁。

## 领域术语

| 术语 | 英文 | 定义 |
|---|---|---|
| 物种 | Species | 种群遗传学中的物种定义，包含染色体组、等位基因集合和标签体系 |
| 染色体 | Chromosome | 遗传物质的载体，包含多个位点（Locus）。可以是常染色体或性染色体 |
| 位点 | Locus | 染色体上的特定位置，每个位点含有一组等位基因 |
| 等位基因 | Allele / Gene | 位点上的遗传变体。本项目中 `Gene` = `Allele`，二者等同 |
| 基因型 | Genotype | 二倍体个体的遗传组成（母系 + 父系单倍型） |
| 单倍型 | Haplotype | 单个染色体上的等位基因组合 |
| 单倍体基因组 | HaploidGenotype / HaploidGenome | 个体一半的遗传物质（配子层面） |
| 合子型 | ZygoteType | 基因型 × 体细胞标签的组合，是引擎中个体的最小识别单位 |
| 配子型 | GameteType | 单倍体基因组 × 配子标签的组合 |
| 标签 | Label | 附加在基因型/配子上的元数据标记（体细胞标签 slab、配子标签 glab） |
| 遗传预设 | GeneticPreset | 预定义的遗传修饰规则组合（如基因驱动 HomingDrive） |
| 修饰器 | Modifier | 改变配子或合子生成频率的规则（GameteModifier / ZygoteModifier） |
| 适应度 | Fitness | 基因型的生存/繁殖优势（viability、fecundity、sexual_selection、zygote_viability） |
| 模型声明 | ModelDefinition | 保存声明顺序和规范化编译输入的冻结快照，用于重新编译；派生遗传矩阵不是声明状态 |
| 模型草稿 | ModelDraft | 构建期中间数据；采用 NamedTuple 容器，但内部数组不因此自动不可变，不是运行时权威状态 |
| 种群状态 | PopulationState | 状态快照（数组容器）；Rust 会话拥有权威运行状态，读取返回独立拷贝 |
| 种群构建器 | PopulationBuilder / SpatialPopulationBuilder | 构建期链式声明入口，将模型声明编译并构建为种群 |
| 运行时更新器 | RuntimeUpdater | 由种群或 Hook 上下文提供的更新句柄，将修改提交到会话或事件事务，不负责构建种群 |
| 结构蓝图 | Blueprint | 固定维度、名称目录、掩码、初始状态及迁移结构等契约数据 |
| 参数载荷 | Params | 承载可更新的生态参数和遗传张量；物化时拥有独立数组，用于向运行时传递数据 |
| Rust 会话 | AgeStructuredSession / DiscreteGenerationSession / SpatialSession | 唯一运行时权威：拥有计数、tick、生态、遗传、随机流、历史与检查点 |
| 索引注册表 | IndexRegistry | 将基因型/单倍型映射为引擎使用的整数索引 |
| Hook | Hook | 模拟过程中的事件干预点（first / early / late / finish） |
| 种群 | Population | 具体的种群模型实例（年龄结构型 / 离散代型 / 空间型） |
| 空间拓扑 | GridTopology / SquareGrid / HexGrid | 空间种群的区域布局（六边形网格、方形网格等） |
| 迁移 | Migration | 空间种群中个体在区域间的移动 |
| 观测 | Observation | 将种群状态按个体选择条件分组、汇总为观测结果的规则 |
| 历史 | History / HistoryStore | Python 提供带 schema 的只读查询和独立数组导出，Rust 存储并维护运行历史 |
| 区域 | Deme | 空间种群中的一个局部子种群 |


## 当前目录与职责

Python 包位于 `src/natal/`，顶层实体包树为 `frontend/`、`backends/` 和 `contracts/`。
仓库根目录的 `ui/` 是 Vue 应用源码，与 Python 的 `src/natal/frontend/` 不同。

| 路径 | 职责 |
|---|---|
| `src/natal/__init__.py` | `_PUBLIC_EXPORTS` 显式声明顶层公开符号，首次访问时惰性导入所属模块 |
| `src/natal/__init__.pyi` | 由 `scripts/generate_init_pyi.py` 根据显式导出生成的类型 stub |
| `src/natal/_engine_rs.pyi` | Rust 原生扩展的 Python 类型接口 |
| `src/natal/parameters.jsonc` | 参数注册表数据源 |
| `src/natal/frontend/genetics/` | 物种、染色体、位点和基因型结构；遗传编译、矩阵构建及频率提取 |
| `src/natal/frontend/patterns/` | 基因型、配子型、合子型模式解析和个体选择器 |
| `src/natal/frontend/registry/` | 遗传实体到整数索引的映射 |
| `src/natal/frontend/model/` | 模型声明、草稿装配、声明编译、初始状态解析及编译产物发布 |
| `src/natal/frontend/builder/` | 构建期链式 API、参数路由和写入，以及独立的 RuntimeUpdater |
| `src/natal/frontend/data/` | 状态快照容器及扁平状态解析，不存放 ModelDraft |
| `src/natal/frontend/population/` | 年龄结构与离散代种群的 Python 入口、运行调度和会话访问 |
| `src/natal/frontend/spatial/` | 空间构建器、种群、网格拓扑、迁移 CSR 生成和区域访问 |
| `src/natal/frontend/modifiers/` | 配子与合子的转换规则 |
| `src/natal/frontend/presets/` | 基因驱动、细胞质等预设规则组合 |
| `src/natal/frontend/fitness/` | 适应度补丁和模式解析写入 |
| `src/natal/frontend/hooks/` | Hook 声明、编译、TickContext 及事件事务接口 |
| `src/natal/frontend/output/` | 记录计划、历史查询、观测和结果转换 |
| `src/natal/frontend/utils/` | 共享类型、参数描述和辅助函数 |
| `src/natal/frontend/webui/` | FastAPI 服务、REST/WebSocket 通道及面向 Vue 的结果序列化 |
| `src/natal/backends/rust/` | Python 与 Rust 会话之间的适配、检查点和错误转换 |
| `src/natal/contracts/` | Blueprint、Params 及从草稿物化契约数据的边界 |
| `ui/` | Vue/Vite 仪表盘；构建产物由 Python Web 服务提供 |

Rust 执行层位于 `rust/src/`：

| 路径 | 职责 |
|---|---|
| `rust/src/lib.rs`、`rust/src/python.rs` | 原生扩展注册与 Python 可调用函数 |
| `rust/src/model/` | 蓝图、生态参数、遗传张量、自定义字段及输入校验 |
| `rust/src/sessions/` | 年龄结构、离散代和空间会话，以及执行状态转换 |
| `rust/src/kernels/` | 生命周期阶段、后代生成、密度调节、平衡态、随机采样、迁移和状态归约 |
| `rust/src/hooks/` | 声明式 Hook 解释和回调事务 |
| `rust/src/output/` | 原生历史存储、参数日志和观测投影 |
| `rust/src/generated/` | 生成的生态参数定义 |

## 构建与执行关系

下面描述数据流和职责分工，不代表逐模块 import 依赖图。

1. **声明与编译**：构建器接收物种、参数、规则和记录要求；`ModelDefinition` 保存声明，`model/definition_compiler.py` 编译派生数据，遗传实体和 `IndexRegistry` 提供类型目录与索引。
2. **发布与物化**：`model/publication.py` 将编译产物转换为运行时产物，处理完整类型目录到运行时索引的投影；`contracts/materialize.py` 生成 Blueprint 和 Params，并隔离草稿数组。
3. **会话执行**：Python 种群通过 `backends/rust/rust_backend.py` 连接 Rust 会话。会话维护运行状态并调用 kernels 完成生命周期计算。
4. **运行时干预**：RuntimeUpdater 将更新送往会话或 Hook 事件事务。声明式 Hook 由 Rust 解释执行；Python 回调通过有生命周期约束的 TickContext 和事务候选数据交互。
5. **结果访问**：Rust 维护历史存储，Python 提供状态快照、History 和 Observation 查询；Web UI 通过服务接口消费这些结果。

空间构建另外负责拓扑解析、迁移权重归一化和 CSR（压缩稀疏行格式）生成，再把结构及区域参数交给空间会话。迁移的实际数值计算在 Rust 内核中执行。

## 关键架构边界

- **唯一执行引擎是 Rust**：年龄结构、离散代（含 Wright–Fisher）和空间模拟使用对应 Rust 会话。Python 负责声明、编译、适配和结果访问。
- **构建与运行时更新分开**：PopulationBuilder 和 SpatialPopulationBuilder 构建种群；RuntimeUpdater 更新既有运行时。当前接口不再以 Configurator 为统一入口。
- **声明、草稿与运行状态分开**：ModelDefinition 保存可重新编译的输入，ModelDraft 承载构建期中间数据，Rust 会话拥有权威运行状态。不能用修改声明或草稿代替会话更新。
- **数据隔离具有明确边界**：契约物化会复制数组；状态和历史的公开数组读取提供独立数据。ModelDefinition 隔离其拥有的容器和数组，但用户提供的 recipe、回调等不透明资源保留身份，不能把“冻结声明”理解为递归复制所有对象。
- **类型目录与运行时索引分开**：`IndexProjection` 记录完整目录到运行时顺序的映射。压缩后的合子型、配子型索引不能直接当作完整目录索引使用。
- **Hook 更新通过事件事务进行**：回调操作候选状态、参数和随机流，并接受校验和生命周期约束；它不是长期持有会话可变数组的入口。
- **历史存储归 Rust 所有**：Python History 保留维度、标签及 schema，提供查询与导出；Python 中存在 HistoryBatch 类型不意味着 Python 拥有运行历史的写入权。
- **顶层导出是显式契约**：仅向子模块 `__all__` 添加符号不会自动发布到 `natal` 顶层；需同步 `_PUBLIC_EXPORTS` 和生成的 stub。`tests/test_phase0_shims.py` 检查相关一致性。
- **Web UI 使用 Vue 与 FastAPI**：发布 wheel 包含包内静态资源，源码环境也可使用 `ui/dist` 或 Vite 开发服务；具体资源选择逻辑位于 `src/natal/frontend/webui/app.py`。

## 历史路径与维护范围

旧的 `frontend/configurator/`、`frontend/ui/`、`population/_mixins/` 和 Python `engine/simulation/` 不属于当前目录结构。`natal.backends.reference`、`natal.engine`、`natal.numba` 及旧顶层转发包也不应作为新增代码的依赖。

此前记录的拆分完成情况、文件行数、UI 类型检查待办和覆盖率判断已移除：它们不能作为当前实现或质量状况的证据。项目另有 [TODO.md](./TODO.md)，其中任务状态仍需结合当前代码和验证结果核对。

本文件只维护稳定的领域与架构事实。模块迁移、公开入口或数据所有权发生变化时，应同步对应描述；科学公式与关键计算分支的详细解释应放在相关实现附近，较长推导放在对应科学文档中。
