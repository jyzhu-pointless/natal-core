# CLAUDE.md

> 这是中文版。另见：[English version](./CLAUDE.en.md)
>
> 任一一版更新时，另一版必须同步更新。

## 语言

默认使用中文回答。仅在用户明确使用英文提问时用英文回复。

## 规范引用

以下文件定义本项目的编码、文档和测试规范，按优先级排列：

1. `docstring_spec.md`
2. `quality_checks_spec.md`
3. `docstring_spec_cn.md`（中文说明）
4. `quality_checks_spec_cn.md`（中文说明）

规范冲突时以英文版为准。

## 行为指南

遵循 Karpathy 编码指南（Claude Code 命令：`/andrej-karpathy-skills:karpathy-guidelines`）：

1. **先想后写** — 不确定时停下来问，不要猜。有更简单方案时直接说。
2. **简洁优先** — 只写解决问题所需的最少代码。不为单次使用建抽象，不为不可能的场景加错误处理。
3. **精准修改** — 只碰必须改的。不动相邻代码，不改格式，不顺手重构。
4. **目标驱动** — 把任务转化为可验证目标。多步骤任务先列计划再执行。

项目补充规则：

- 任何方案、计划或非平凡修改，必须先向用户说明并获得批准后再执行。不得擅自实施。
- 倾向于写注释。注释应解释 WHY（设计意图、约束、非显而易见的逻辑），而非 WHAT（代码本身已经说明）。
- 不要创建文档文件（*.md），除非用户明确要求。
- 不要过早抽象。不为假想需求设计。
- 修改后使用文字详细解释你的修改内容，包括为什么修改以及修改后的效果。避免使用模糊的表述，如“我改了这个函数”，而是具体说明改了什么（如“我将 `foo` 函数的参数从 `x` 改为 `y`，以支持新的用例”）。
- 在文字表述中，尽可能使用通俗易懂的语言。如引入专业的软件工程术语（本项目架构中已引入的概念除外），必须详细解释其含义。例子：
  - 不这样说："preset 参数由 Configurator 的 deferred 管理，运行时也是 Configurator 来改。"
  - 而是说："遗传预设（preset）的参数比较特殊——它不能在构建过程中直接写入配置，必须等 Population 对象创建完成后才能生效。所以 Configurator 会先把这些参数暂存起来（deferred），等 `build()` 真正执行时再统一应用。运行时修改 preset 参数也是通过 Configurator 的 `update()` 入口。"

- **善用选择题澄清细节**：当需要用户在多个可行方案之间做决策时（设计取舍、命名选择、参数值等），使用 AskUserQuestion 工具给出 2-4 个具体选项让用户直接选择，避免开放式追问造成沟通往返。
- **维护 Tasks 列表**：多步骤任务使用 TaskCreate/TaskUpdate 跟踪进度。创建 task 后及时更新状态（pending → in_progress → completed），不要遗留僵尸 task。单步琐碎操作无需创建 task。
- **优先使用专用工具**：Read/Glob/Grep/Edit/Write 等专用工具比 shell 命令更可靠（不会被沙箱拦截、输出格式稳定、渲染更友好）。仅在批量操作、管道组合、或专用工具无法实现时使用 Bash。
- **禁止主动 commit / push**：除非用户明确要求，不得执行 `git commit`、`git push` 或任何形式的提交操作。
- **禁止修改 `.gitignore`**：除非用户明确要求，不得改动 `.gitignore` 文件。

## 门禁检查

每次修改后，必须运行以下命令：

```bash
pytest                          # 运行全部测试
pyright                         # 类型检查（strict mode）
ruff check src demos             # Lint 检查
ruff check src demos --fix       # Lint 自动修复
python scripts/check_rust.py    # Rust：fmt + clippy + check（rust-analyzer 可选诊断）
python scripts/generate_init_pyi.py  # 公开 API 变更后重新生成 stub
```

虚拟环境已自动激活，直接运行命令即可。

提交前必须通过 **Python 三项与 Rust 硬门禁**：`pytest` + `pyright` + `ruff check src demos`，以及 `python scripts/check_rust.py`（其内部的 `cargo fmt --check`、`cargo clippy -- -D warnings`、`cargo check --all-targets` 任一失败即门禁失败；rust-analyzer 诊断为可选，不阻断）。不压制、不绕过。

### 审查流程

每次完成代码修改后，必须按以下顺序执行，**禁止以自我审查代替**：

1. **`@tester`** — 根据 `numerical-verification` 和 `adversarial-review` 标准，为新代码或变更代码生成严格测试。必须覆盖五类测试：**负向合同测试**（assert 已删除接口不可访问）、**所有权测试**（assert 返回的 ndarray/list/dict 是副本或只读）、**状态转换测试**（restore→run、finish→snapshot、import→run、clear→record）、**轴组合测试**（配置轴的笛卡尔积枚举）、**错误路径测试**（无效输入抛正确异常且状态不变）。每条断言必须证明一个数值不变量。
2. **运行 `python scripts/generate_init_pyi.py`** — 公开 API 变更后重新生成 stub，确保后续 pyright 检查基于最新 stub。
3. **`@evaluator`** — 执行**对抗性**审查。审查者采取攻击者立场，主动寻找代码不符合 spec 的证据，而非仅验证门禁通过。必须：
   - 从 spec 建立**账本**：must-exist、must-not-exist、invariants 三份清单
   - 对每个 must-not-exist 项做**全仓搜索**（src/tests/demos/docs/stub），确认无法访问
   - 对每个 invariant 执行**攻击**：所有权攻击（检查 ndarray 引用/写保护）、状态机攻击（restore→run、finish→snapshot）、轴组合攻击（笛卡尔积枚举）
   - **实际执行**所有 demo 脚本和文档示例代码
   - 运行 `pytest` / `pyright` / `ruff` / `python scripts/check_rust.py` 门禁，加载 `code-review`、`numerical-verification`、`adversarial-review` skill
   - 判定为**机械规则**：`APPROVED` ⇔ 零 hard-blocker
4. **若变更涉及公开 API（签名、参数、默认值、模块重命名等），主 agent 调用 `@docs`** 同步 `docs/zh/` 和 `docs/en/` 中的文档和示例代码。
5. **主 agent 不得自行运行 `pytest`、`pyright`、`ruff`、`python scripts/check_rust.py` 并声称"已通过审查"**。这些命令的结果必须由 evaluator 独立验证并出具结构化报告。

只有当 evaluator 给出 `APPROVED` 判定后，修改才算完成。

#### evaluator 强制拒绝标准

以下任一情况发生时，evaluator **必须**返回 `REJECTED`。**严重度不软化 hard-blocker**：一个缺少注释的 `Any` 和一项语义破坏同等对待。

- **测试失败**：`pytest` 出现任何 FAILED。
- **Rust 硬门禁失败**：`cargo fmt --check`、`cargo clippy -- -D warnings`、`cargo check --all-targets` 任一非零（通过 `python scripts/check_rust.py` 执行）。rust-analyzer 可选诊断不构成拒绝理由。
- **覆盖率不足**：新模块或已有模块新增代码的行覆盖率 **< 95%**。
- **Hard-blocker 存在**：以下任一类问题出现一条即 REJECTED：
  - **Must-not-exist 违规**：spec 要求删除的接口仍可访问（含 `getattr`、`__init__` 重导出）
  - **Invariant 破坏**：所有权泄漏（ndarray 引用/非写保护）、状态机错误（restore 后 tick 不同步）、轴组合崩溃
  - **Demo/doc 崩溃**：demo 脚本或文档示例代码实际运行报错
  - **未注释的 `Any` / `object`**
  - **未注释的 `# type: ignore`**
  - **`cast(Any, …)`**
  - **非 Google 风格 docstring section** 或 **缺少类型标注的参数/返回值/属性**
- **负向合同缺失**：spec 标注为删除的接口，tests 中没有对应的 assert-not-exists 测试。
- **所有权测试缺失**：返回容器（ndarray/list/dict）的公开方法，tests 中没有对应的只读/副本验证。
- **状态转换未覆盖**：restore→run、finish→snapshot、import→run、clear→record 等关键生命周期序列未被测试。

### 修复策略

- **修改的文件**：所有 pyright / ruff / pytest / cargo fmt / cargo clippy / cargo check 报错必须修。不存在"预先存在的失败"——如果 main 上通过而本分支失败，就是本分支引入的回归，必须立即修复。
- **被改动波及的文件**：签名或 import 变更导致的其他文件报错也必须修。
- **未修改文件的既有问题**：指出并分析；修复推荐但不强制当前提交必须完成。
- **`cast(Any, …)` 禁止**。不能用它绕过类型检查。出现即 REJECTED。
- **禁止滥用 `Any` 和 `object`**：参数、返回值、变量类型注解必须指向具体类型，不要偷懒用 `Any` 或 `object`。为导入类型而添加新的 import 是值得的。仅在有具体、书面理由（如泛型 `Callable[..., Any]` 表示"任意可调用对象"）时才可用 `Any`。
- **`cast(T, x)`** 仅在静态分析完全无法证明 `x: T` 时可用（如 guard 后 narrow Optional）。优先用类型窄化断言或重构。
- **`# type: ignore`** 是最后手段。每个 ignore 必须附带简短原因。缺少注释即 REJECTED。

### 测试覆盖

- **新模块**：≥95% 行覆盖。
- **已有模块新增代码**：≥95% 行覆盖。
- **确定性模拟** (`stochastic=False`)：精确数值断言。
- **随机模拟**：需统计验证（多次运行、置信区间或分布检验），单次通过不算。
- **优先使用 pytest-collected 测试**，而非脚本式 smoke test。

### Docstring 规范

- 仅使用 Google 风格 section（`Args:`、`Returns:`、`Raises:` 等）。不发明新的 section 名称。
- docstring 内容使用**英文**。
- 所有参数、返回值、属性必须显式标注类型（优先使用 annotation）。

### 变更说明

每次修改完成后，必须包含以下四项：
1. 变更的文件
2. 行为变化
3. 执行的验证命令
4. 残余风险或后续事项（如有）

## 架构

`natal` 包使用 **lazy loading**——顶层 `__init__.py` 通过 AST 静态解析各公开子包的字面量 `__all__`，构建名称到模块的索引；首次访问名称时，`__getattr__` 才导入对应子包。

### 核心模块

| 层 | 子包 | 职责 |
|---|---|---|
| 遗传领域与索引 | `genetics/`, `patterns/`, `registry/` | 定义 Species / Chromosome / Locus 和遗传实体；解析基因型模式；将 ZType / GType 映射为引擎使用的整数索引 |
| 配置与数据 | `configurator/`, `data/` | Configurator 链式构建及运行时修改；PopulationConfig / DiscretePopulationConfig 和对应的 State 数组容器 |
| 种群模型 | `population/`, `spatial/` | 年龄结构与离散代种群；多 deme 容器、空间拓扑和迁移配置 |
| 遗传效应 | `modifiers/`, `presets/`, `fitness/` | 配子/合子转换规则；HomingDrive、ToxinAntidoteDrive、Wolbachia 等预设；适应度补丁的解析与写入 |
| Hook 系统 | `hooks/` | 编译声明式操作和 `@hook` 函数，按 first / early / late / finish 事件及优先级执行 |
| 引擎与加速 | `engine/`, `numba/` | 年龄结构、离散代和空间生命周期；迁移内核；Numba 开关、兼容层及动态 wrapper 生成 |
| 输出与界面 | `output/`, `ui/` | 观测筛选、历史记录、可读格式转换和交互式可视化 |

### 数据流

```
Species → Configurator → IndexRegistry + PopulationConfig / DiscretePopulationConfig
  → 预设、修饰器与适应度 → 可选 ZType / GType 压缩
  → Population + PopulationState / DiscretePopulationState

标准每 tick: first Hook → 繁殖 → early Hook → 密度调节 + 存活
  → late Hook → 年龄推进 → tick + 1

离散代 extreme-speed: first Hook → 融合的 Wright-Fisher tick → tick + 1
  （不执行 early / late Hook）

空间模型: 各 deme 生命周期 → Migration → 可选 Observation / History 记录
```

`finish` Hook 不属于单个 tick；它在种群结束模拟时执行。

### 关键设计

- **引擎数据与可变参数分离**：Config 和 State 都是 NamedTuple。标量元数据不可原地替换，数组内容可以修改；9 个生态参数使用 0-d ndarray，因此 Configurator 和 Hook 可通过 `config.field[()] = value` 原地更新。
- **扁平类型索引**：IndexRegistry 直接维护 `ZType = (Genotype, slab)` 与 `GType = (HaploidGenotype, glab)` 的索引。构建时可按可达性压缩两套索引及相关配置数组。
- **统一配置入口**：`AgeStructuredPopulation.setup(...)`、`DiscreteGenerationPopulation.setup(...)` 和 `pop.update()` 使用同一套 Configurator 链式 API。构建阶段通过 ConfigContext 在 Population 创建前应用预设和修饰器，并在需要时重建遗传映射与 offspring tensor；`.fitness()` 则直接写入 Config。
- **双路径 Hook**：声明式操作编译为连续数组和偏移表（CSR），自定义函数通过 `@hook` 注册；两者可按优先级混合。函数签名为 `hook(state, config, deme_id) -> int`。
- **Numba-first, Python-fallback**：`njit_switch` 默认启用 Numba，也可切换到 Python 实现以便调试。空间 wrapper 在 Numba 路径中并行运行各 deme 的生命周期，再统一迁移。
- **稳定身份的 codegen 缓存**：Hook 和生命周期 wrapper 根据函数的 `module:qualname`、分派结构及选择器生成稳定哈希。生成源码与 Numba 产物写入 `.numba_cache/`；相同组合沿用稳定路径，并在缓存有效时跨运行复用编译结果。
