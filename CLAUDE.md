# CLAUDE.md

> 这是中文版。另见：[English version](./CLAUDE.en.md)
>
> 任一一版更新时，另一版必须同步更新。

## 语言

默认使用中文回答。仅在用户明确使用英文提问时用英文回复。

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

## 门禁检查

每次修改后，必须运行以下命令：

```bash
pytest                          # 运行全部测试
pyright                         # 类型检查（strict mode）
ruff check src demos             # Lint 检查
ruff check src demos --fix       # Lint 自动修复
python scripts/generate_init_pyi.py  # 公开 API 变更后重新生成 stub
```

虚拟环境已自动激活，直接运行命令即可。

提交前必须通过 **全部三项**：`pytest` + `pyright` + `ruff check src demos`。不压制、不绕过。

### 修复策略

- **修改的文件**：所有 pyright / ruff / pytest 报错必须修，无论是否预先存在。
- **被改动波及的文件**：签名或 import 变更导致的其他文件报错也必须修。
- **未修改文件的既有问题**：指出即可，不强制修。
- **`cast(Any, …)` 禁止**。不能用它绕过类型检查。
- **函数参数禁止 `Any`**，除非有具体、书面的理由。
- **`cast(T, x)`** 仅在静态分析完全无法证明 `x: T` 时可用（如 guard 后 narrow Optional）。优先用类型窄化断言或重构。
- **`# type: ignore`** 是最后手段。每个 ignore 必须附带简短原因。

### 测试覆盖

- **新模块**：≥95% 行覆盖。
- **已有模块新增代码**：≥95% 行覆盖。
- **确定性模拟** (`stochastic=False`)：精确数值断言。
- **随机模拟**：需统计验证（多次运行、置信区间或分布检验），单次通过不算。

## 架构

`natal` 包使用 **lazy loading**——`__init__.py` 通过 AST 解析 `__all__` 构建名称索引，`__getattr__` 按需导入，启动近乎即时。

### 核心模块

| 层 | 模块 | 职责 |
|---|---|---|
| 遗传结构 | `genetic_structures.py`, `genetic_entities.py` | Species / Chromosome / Locus 不可变蓝图；Gene / Genotype / Haplotype 实体 |
| 配置与状态 | `population_config.py`, `population_state.py` | PopulationConfig（静态 NamedTuple，9 个生态参数为 0-d ndarray）+ PopulationState（可变数组） |
| 群体模型 | `discrete_generation_population.py`, `age_structured_population.py`, `spatial_population.py` | Wright-Fisher / 年龄结构 / 空间多 deme 模拟 |
| Builder | `population_builder.py`, `configurator.py` | 链式建造 API + Configurator 运行时修改 |
| Hook 系统 | `hooks/` | 事件驱动干预（init/first/early/late/finish），声明式 + Python 函数式 |
| 引擎 | `engine/` | Numba 加速模拟循环 + codegen 动态生成 wrapper |
| 预设 | `genetic_presets.py` | HomingDrive / ToxinAntidoteDrive，封装 modifier + fitness |
| 参数注册 | `parameters.py` | ParamDescriptor 注册表 + Numba setter codegen |

### 数据流

```
Species → IndexRegistry → PopulationConfig + PopulationState
  → 每 tick: Hooks → Reproduction → Competition → Survival → Observation
  → 空间模型: per-deme 模拟 → migration
```

### 关键设计

- **0-d ndarray**：9 个生态参数（K, eggs, sex_ratio 等）可在 hook 内 `config.field[()] = v` 原地修改。
- **Hook 签名**：`hook(state, config, deme_id) → int`，config 直接传入。
- **Numba-first, Python-fallback**：核心模拟兼容两种模式，`njit_switch` 自动降级。
- **Codegen 缓存**：生成的 kernel wrapper 按内容哈希缓存，重复运行复用编译结果。
