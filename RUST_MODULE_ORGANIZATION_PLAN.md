# Rust 模块命名与目录整理：实施移交方案

## 1. 起点与任务边界

- 基线：`5d97f973c12417967db32901773040ec3cb19ef5`，分支 `feat/rust-engine-backend-clean`。
- 本文件是待实施方案，不表示其中的代码修改已经完成或提交。
- 背景：[Rust 唯一后端方案](./RUST_ONLY_REFACTOR_PLAN.md)的实现及审查修复已经落地。本轮整理现有实现，不重新实现该方案，不重新引入 Reference/Numba。
- 用户诉求：Rust 文件命名清楚、同层模块对称、生产源码与大段测试分开，方便后续维护和交给其他 agent 实施。
- 当前仅授权编写方案。实施时遵守 [AGENTS.md](./AGENTS.md)；未经后续明确授权，不 commit、push 或修改 `.gitignore`。
- 质量以 [quality_checks_spec.md](./quality_checks_spec.md)为准；本文件不降低或替代它。

当前基线的独立验收记录为 Python 3301 项、Rust 59 项测试通过，Pyright/Ruff/Rust 门禁通过。它们是历史证据，不能代替重构后的验证。

## 2. 不变量与范围

必须保留以下用户合同：

1. 面向用户的链式 API 语法。
2. Species 遗传结构注册语法。
3. preset 科学规则、优先级及组合语义。
4. 声明式 hook 格式、事件顺序及操作含义。

本轮另外采用更窄的实现约束：保持 Python 扩展 `_engine_rs` 的类名、函数名、签名、异常类别与数据布局；保持 RNG 算法、消费顺序、浮点运算顺序、状态所有权、事务提交与回滚、检查点和历史淘汰语义。内部 Rust 模块路径和类型名可以修改。

不增加依赖，不修改 Cargo package/lib 名称，不修改构建 profile 或性能阈值，不增加 `rlib` 产物，不为了测试把内部实现公开。不顺带修改科学模型、合并两类生命周期算法、引入通用 session trait 或泛型框架。

此任务按高风险代码重构落实 tester 和独立 evaluator：虽然目标是保持行为，但移动范围覆盖科学内核、状态恢复与 Python/Rust 边界。如果只能自审，明确标记独立审查未完成。

## 3. 当前问题与命名规则

当前 `session.rs`/`lifecycle.rs` 实际专属于年龄结构模型，却用了通用名称；离散模型和空间模型则带前缀。同一目录还混合计算内核、运行会话、跨语言解析、测试和生成文件。

统一规则：

- 按职责设目录；同层文件按模型或具体能力命名。
- 生命周期模型使用 `age_structured`、`discrete_generation`；空间多子群编排使用 `spatial`，它是运行维度，不是第三种生物学生命周期。
- 类型表达具体职责，例如 `AgeStructuredSession`、`AgeStructuredConfig`、`ExecutionStatus`。
- 使用 `equilibrium`、`density_regulation`、`ecology` 等完整领域词；不机械删除行业通用且明确的 `rng`、CSR 缩写。
- 不创建新的 `common.rs`、`utils.rs`、`manager.rs` 来承接拆分后的杂项。
- `mod.rs` 只负责模块声明和必要可见性安排，不累积计算逻辑，也不长期保留旧路径别名。

## 4. 目标结构

```text
rust/
  Cargo.toml
  src/
    lib.rs                       # 模块声明、PyO3 注册、测试挂载
    python.rs                    # 当前 lib.rs 的独立 Python 函数适配器
    model/
      mod.rs
      blueprint.rs               # Blueprint
      ecology.rs                 # Params → EcologyParams
      genetics.rs                # TensorSet → GeneticsTensors
      custom_fields.rs           # CustomSlot
      validation.rs              # 跨类型的共享输入校验
      python.rs                  # model 共享的 Python 提取/转换辅助函数
    kernels/
      mod.rs
      age_structured.rs
      discrete_generation.rs
      spatial.rs                 # 现有空间调度和迁移；本轮不再拆分算法
      config.rs                  # AgeStructuredConfig；不强行合并 DiscreteConfig
      density_regulation.rs
      equilibrium.rs
      offspring.rs
      rng.rs
    sessions/
      mod.rs
      age_structured.rs
      discrete_generation.rs
      spatial.rs
      status.rs
      ecology_snapshot.rs        # 原 session.rs 中跨 session 使用的快照/恢复辅助
    hooks/
      mod.rs
      interpreter.rs
      transaction.rs
    output/
      mod.rs
      history.rs
      parameter_log.rs
      observation.rs
    generated/
      mod.rs
      ecology_parameters.rs
  tests/
    unit/
      runtime_boundaries.rs
      ...                        # 按所属模块命名的现有内部测试
```

该树是目标布局，不要求一次创建空文件。只有确有对应职责和调用者时才创建模块。`model/python.rs` 是边界解析辅助，不是第二份运行数据模型；`python.rs` 不复制 session 的方法实现。

### 文件映射

| 原文件 | 目标位置或拆分 |
|---|---|
| `session.rs` | `sessions/age_structured.rs`，共享生态快照辅助移至 `sessions/ecology_snapshot.rs` |
| `discrete_session.rs` | `sessions/discrete_generation.rs` |
| `spatial_session.rs` | `sessions/spatial.rs` |
| `execution.rs` | `sessions/status.rs` |
| `lifecycle.rs` | `kernels/age_structured.rs` |
| `discrete.rs` | `kernels/discrete_generation.rs` |
| `spatial.rs` | `kernels/spatial.rs` |
| `config.rs` | `kernels/config.rs` |
| `curves.rs` | `kernels/density_regulation.rs` |
| `equilibrate.rs` | `kernels/equilibrium.rs` |
| `offspring.rs`、`rng.rs` | `kernels/` 下同名文件 |
| `hooks.rs`、`hook_transaction.rs` | `hooks/interpreter.rs`、`hooks/transaction.rs` |
| `eco_param_wire.rs` | `generated/ecology_parameters.rs`，必须同步生成器 |
| `contract.rs` | 拆入 `model/`，按拥有的数据和对应实现分组 |
| `history.rs` | 拆入 `output/` 的历史存储、参数日志和观测投影 |
| `runtime_boundary_tests.rs` | `tests/unit/runtime_boundaries.rs` |

### 类型和导出名称

| 当前 Rust 类型 | 目标 Rust 类型 | Python 名称 |
|---|---|---|
| `EngineSession` | `AgeStructuredSession` | 保留 `EngineSession` |
| `DiscreteEngineSession` | `DiscreteGenerationSession` | 保留 `DiscreteEngineSession` |
| `HeterogeneousSpatialEngineSession` | `SpatialSession` | 保留 `HeterogeneousSpatialEngineSession` |
| `Execution` | `ExecutionStatus` | 不新增导出 |
| `SimConfig` | `AgeStructuredConfig` | 不新增导出 |
| `DiscreteConfig` | `DiscreteGenerationConfig` | 不新增导出 |
| `Params` | `EcologyParams` | 保持现有边界形式 |
| `TensorSet` | `GeneticsTensors` | 保持现有边界形式 |

PyO3 类型重命名时使用显式 Python 名称映射，保留其他现有 `pyclass` 属性。先检查现有声明，不能只改 Rust struct 名称而无意改变扩展 API。Blueprint、HookProgram、HookTransaction、HistoryStore 等名称已经清楚，保留。

## 5. 测试目录：保持单元测试身份

当前 crate 是 `cdylib`。不要将全部内部测试直接作为 `rust/tests/*.rs` 集成测试搬走，再通过扩大可见性或更改 crate-type 修补编译。

- 大段单元测试统一存放 `rust/tests/unit/`，通过所属生产模块中 `#[cfg(test)]` 与 `#[path = "..."] mod tests;` 挂载。
- 保留测试的父模块上下文，使 `use super::*` 仍访问原模块私有实现。每个挂载路径按最终源文件位置计算并实际编译验证。
- 跨模块的 `runtime_boundaries` 测试由 crate 根在 `cfg(test)` 下挂载，保留它原本的访问范围。
- 小型内联测试可以一起搬出以统一布局；不要拆散原有测试断言或更改其数学预期。
- 仅测试辅助代码留在 `cfg(test)` 下；不能把生产逻辑误搬入测试文件，也不能让 release 编译测试代码。
- 本轮不要求新增真正的 Rust 外部集成测试。已有 Python 扩展合同测试继续验证实际边界。

测试移动后先比较名称与数量，再验证测试意义。数量与基线不一致必须逐项解释，不得用 skip、xfail 或删除断言消除差异。

## 6. 拆分边界和依赖约束

`model/` 保留原类型的数据拥有关系与验证顺序。先将对应类型及其 impl 成组移动，再提取确有多处调用的 Python 提取函数；不要同时重写解析框架。生产代码已经依赖 PyO3，不把“model 必须完全无 PyO3”作为本轮附加目标。

`output/history.rs` 拥有 ring、行数据及共享历史句柄；`output/parameter_log.rs` 拥有参数日志和日志值转换；`output/observation.rs` 拥有数值投影及现有 Python 投影入口。拆分不得改变 Arc/Mutex 的共享关系、锁范围、复制行为和淘汰顺序。

将共享生态快照逻辑从年龄结构 session 移走，避免另外两种 session 因复用工具而依赖一个特定模型的实现模块。仅移动真正共享的辅助逻辑，不发明通用 session 基类。

实施前记录现有跨模块依赖；目标是使边界更可读，不强行建立与现状不符的单向层次。发现循环依赖时优先把共同数据或纯辅助函数移到其真正归属模块，不能通过全局 `pub use` 网或回调框架掩盖它。

## 7. 实施顺序与建议 commit

以下为实施阶段建议；是否实际创建 commit 取决于用户授权。每步保持可编译、可检查，不要把全部搬迁和职责拆分混在一个提交中。

| 步骤 | 建议 commit 标题 | 工作与退出条件 |
|---|---|---|
| S0 | 不必提交 | 固定实际 HEAD、确认用户修改、读取规范；保存 Rust 测试清单、扩展导出/签名及可用性能基线。若 HEAD 已变化，记录新基线并评估差异 |
| S1 | `refactor(rust): organize kernel and session modules` | 搬迁内核、会话、hooks、生成文件；同步全部引用与生成器路径。暂保留类型名和 contract/history 文件，避免同时拆分；Rust gate、生成器检查及相关 Python 测试通过 |
| S2 | `refactor(rust): clarify native type names` | 按表重命名内部类型，固定 PyO3 对外名称；普通/离散/空间构造及运行通过，扩展导出/签名无差异 |
| S3 | `refactor(rust): separate model contracts by responsibility` | 拆 contract 与共享生态快照辅助，保持所有输入验证和原子失败语义；参数边界、所有权、恢复、hook 事务测试通过 |
| S4 | `refactor(rust): separate history logs and observation` | 拆 output 并提取 lib.rs 的独立适配器；历史、检查点、参数日志、投影测试通过 |
| S5 | `test(rust): move internal tests into dedicated files` | 迁移内联与跨模块测试并保留私有访问；59 项基线测试无无故消失，release 不包含测试模块 |
| S6 | `docs(rust): align module documentation with the sole engine` | 更新实际受影响的模块文档、生成器说明、目录引用及 Cargo 描述；完成最终完整验证、独立审查和移交记录 |

S1–S5 中必要的准确注释可以随代码同步；S6 用于收尾，不应让错误路径说明长期存在。每步使用 `git diff --find-renames` 核对搬迁与实质修改，不做全仓库字符串盲替换。

需要清理的已确认陈旧说明包括：hooks 中“回退 Python/Numba”、Cargo 中“prototype”、lib 中将所有 session 数据交换笼统描述为原地零复制。仍有数值推导价值的历史来源说明可保留并注明实现已退役，不机械删除所有 Reference 字样。

## 8. 验证与证据

### 8.1 阶段检查

- 使用现有 `scripts/check_rust.py` 验证 fmt、clippy、check 和 Rust 单元测试；不要猜测 PyO3 链接参数或关闭检查。
- 修改生成文件路径后运行 `python scripts/generate_param_tables.py --check`；必要时先运行生成器，并验证重复生成无内容漂移。同步 TARGET、提示信息、文档及引用该路径的测试。
- 扩展必须重建后再跑 Python 合同测试，防止加载旧 `.so` 造成假通过；记录导入路径、Python 版本和扩展哈希。
- 重点复用 `test_native_session_contracts.py`、`test_native_parameter_boundaries.py`、`test_native_spatial_handoff.py`、`test_hook_transaction_contract.py`、`test_native_history_contracts.py` 和已有冻结 API/模型数值测试。列举不是固定完整清单，按每步实际影响补充。
- 对纯移动不新写镜像测试；需要新增测试时，说明它能捕获什么具体错误。

### 8.2 最终验收

由独立 evaluator 对最终产物运行规范中的四项完整门禁：`pytest`、`pyright`、`ruff check src demos`、`python scripts/check_rust.py`。公开 stub 只有实际导出表面变化时才需要重新生成；本轮预期表面无变化，仍须核对 `_engine_rs.pyi`。

覆盖率依规范逐模块报告。Rust 新模块仍须满足 95%，不能因为文件是搬迁来的就直接豁免，也不能拿 Python 覆盖率代替 Rust 覆盖率。为重命名建立旧/新路径映射，报告新增可执行行的比较基线、分母、未覆盖行和命令；测试文件与 `cfg(test)` 代码不冒充生产覆盖率。

作为共享基础设施重构，执行受影响示例和可运行文档示例，并验证干净隔离环境中的 release wheel。GUI 启动与页面加载、实际按钮操作、数值正确性是不同证据，不得混称。

性能使用 `scripts/perf_freeze.py` 的原有八个工作负载和 `scripts/rust_only_perf_baseline.json` 的 10% 阈值。与重构前相同环境、release 编译方式对照；小耗时场景有噪声时采用事先声明的九次采样方法，保存全部原始样本，不能挑最快结果。构建与运行是否计时必须前后一致。

文件搬迁和内部命名调整不应改变科学数值或随机流。固定种子、相同配置的前后版本结果应一致；如果出现差异，先当作回归定位，不更新 golden 数据消除失败。新增数值容差必须有数学依据。

### 8.3 可参考但不可依赖的历史资料

- 当前提交对应最终 release 八场景记录曾保存在 `/tmp/natal-perf-final-release7.json`；冻结基线在仓库内。
- 当前 Drive-RIDL 默认完整双扫描共 17,640 次模拟，进程约 35.56 秒、111.83 MiB RSS。这是一次运行记录，不是新的硬性能阈值，也不是论文复现验证。
- 原生历史压力曾以 120 种基因型、max_rows=2 验证到 1024 步，只有末两份检查点可恢复。

`/tmp` 文件可能失效，接手者不得将其存在作为前提。缺少所需证据时在实际基线重新测量，不能凭上述数字宣布本轮通过。Python tracemalloc 只表示 Python 分配，实际总内存需另测进程 RSS。

## 9. 完成标准与交接格式

完成时同时满足：目标职责可从路径识别；无失效模块引用和旧路径兼容网；生成器无漂移；测试保持有效；扩展导出和四项用户合同不变；全部适用检查有证据；独立 evaluator 给出 `APPROVED`。

每个阶段交接记录至少包括：实际基线、完成步骤、旧/新文件映射、是否存在非机械修改、测试命令与结果、扩展身份、未完成项。最终报告另外给出覆盖率与性能证据、独立审查结论、commit 列表（若获授权）。不得把缺失验证记为通过。

建议给接手 agent 的任务指令：

> 按 RUST_MODULE_ORGANIZATION_PLAN.md 从 S0 开始实施。先核实当前 HEAD 和工作区，不覆盖用户修改。保留科学计算、RNG、事务、恢复和所有 Python 外部接口行为，只整理 Rust 内部命名与职责。按 S1–S6 分步推进，每步验证后再继续；高风险部分落实 tester 和独立 evaluator。提交操作需用户授权。遇到缺陷先隔离并报告，不把行为修复混入机械搬迁；完成后提供实际验证证据和剩余事项。
