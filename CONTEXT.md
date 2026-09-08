# NATAL Core — 领域与架构上下文

> 本文件记录项目的领域术语（Ubiquitous Language）和模块架构设计决策。
> 不含实现细节、不含代码示例。纯粹的概念地图。
>
> **状态**：模块重组已完成（2026-07-03），15 个子包全部就位。

---

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
| 模型草稿 | ModelDraft | 构建期声明草稿（不可变结构），编译为 Rust 会话的 Blueprint/Params/遗传表 |
| 种群状态 | PopulationState | 状态快照（数组容器）；Rust 会话拥有权威运行状态，读取返回独立拷贝 |
| 配置器 | Configurator | 用户链式 DSL；将声明编译为会话输入并支持运行时更新 |
| Rust 会话 | EngineSession / SpatialEngineSession 变体 | 唯一运行时权威：拥有计数、tick、生态、遗传、随机流、历史与检查点 |
| 索引注册表 | IndexRegistry | 将基因型/单倍型映射为引擎使用的整数索引 |
| Hook | Hook | 模拟过程中的事件干预点（first / early / late / finish） |
| 种群 | Population | 具体的种群模型实例（年龄结构型 / 离散代型 / 空间型） |
| 空间拓扑 | SpatialTopology | 空间种群的区域布局（六边形网格、方形网格等） |
| 迁移 | Migration | 空间种群中个体在区域间的移动 |
| 观测 | Observation | 模拟过程中对特定基因型频率的记录和过滤规则 |
| 区域 | Deme | 空间种群中的一个局部子种群 |

---

## 目录结构

> `_` 前缀 = 内部模块，不通过 `__init__.py` 暴露
> 顶层只有三个实体包树：`frontend/`、`backends/`、`contracts/`。
> 垫片目录（`natal.data`、`natal.hooks` …）与 numba 后端已在 ⑥ 全部移除；
> 顶层惰性导出按规则扫描这三个包树中声明了字面量 `__all__` 的子包。

```
src/natal/
│
├── __init__.py                # 惰性加载入口（AST 扫描 frontend/backends/contracts 的 __all__）
├── __init__.pyi               # 生成器产物（scripts/generate_init_pyi.py）
├── parameters.jsonc           # 参数注册表（ParamDescriptor 的数据源）
├── _engine_rs.*               # Rust 原生扩展（maturin 构建）
│
├── frontend/                  # 🖥️ 用户面 ✅
│   ├── genetics/              # 遗传结构/实体（structures/, entities/ + __init__）
│   ├── patterns/              # 基因型模式匹配（elements/, parser, selector）
│   ├── registry/              # IndexRegistry
│   ├── configurator/          # Configurator 链式 API（_base/_factory/_params/_routes/_writers/...）
│   ├── data/                  # ModelDraft + PopulationState/DiscretePopulationState（NamedTuple）
│   ├── utils/                 # Sex/Age/GameteLabel 类型、helpers、参数注册表 loader
│   ├── population/            # BasePopulation/AgeStructuredPopulation/DiscreteGenerationPopulation
│   ├── spatial/               # SpatialPopulation/DemeSlice/SpatialParamsView/topology/migration(CSR 折叠)
│   ├── modifiers/             # 配子/合子转换规则（conditions/gamete_conversion/zygote_conversion）
│   ├── presets/               # HomingDrive/ToxinAntidoteDrive/Wolbachia/...（_base/_types/cytoplasmic/...）
│   ├── fitness/               # 适应度补丁（_patch/_writer/_types）
│   ├── hooks/                 # @hook + Op（entry/、compile/（CSR 容器）、runtime/、tick_context.py、types.py）
│   ├── output/                # History/Observation/record/translation
│   └── ui/                    # Dashboard/可视化（依赖 matplotlib 等，可选导入）
│
├── backends/                  # 🔌 引擎适配层
│   └── rust/                  # Rust 原生扩展适配（rust_backend.py：会话桥 + 检查点/错误转换；唯一执行引擎）
│
├── contracts/                 # 前后端契约（blueprint/params/state/materialize 等）
│
└── py.typed
```

> 注意：`natal.frontend.spatial.migration` 中的 migration CSR 折叠与
> `natal.contracts` 的契约层共同构成 slice-5 数据面；
> `natal.backends.reference` / `natal.engine` / `natal.numba` 等旧路径已不存在。

## 依赖方向

```
utils → genetics → patterns → presets → modifiers
                                    ↘ fitness
genetics + patterns → registry → data
data → configurator → population → spatial → output → ui
hooks → engine → population
```

## 关键设计决策

1. **`data/` 独立于 `configurator/`**：config 和 state 是面向引擎的纯数据结构，不依赖配置器自身。
2. **`patterns/` 与 `genetics/` 平行**：patterns 被 hooks、configurator、modifiers 等多个模块依赖，不是 genetics 的子概念。
3. **`registry/` 独立顶层**：IndexRegistry 是遗传领域到引擎整数空间的桥梁。
4. **`fitness/` 已激活**：fitness 逻辑已从 presets 和 configurator 提取到独立的 `fitness/` 子包，使用 `FitnessPopulationView` Protocol 作为统一接口，`_patch.py` 为唯一写入层，`_writer.py` 为 DSL 解析层。
5. **`modifiers/` 独立**：修饰器是连接 presets 和引擎的独立抽象层。
6. **500 行单模块上限**：每个 `.py` 文件不超过 500 行，超限需拆分。
7. **执行引擎只有一个**：`backends.rust`（原生扩展 `natal._engine_rs`）；引擎会话拥有运行状态，`backends.reference` 与 `auto` 选择器已删除。
8. **旧 Builder 已废弃**：`population_builder.py` 中的 Builder 类已删除，统一使用 Configurator API。
9. **旧模块导入路径已全部更新**：`genetic_structures`、`genetic_entities`、`genetic_patterns`、`population_config`、`population_state` 等旧路径不再存在。

## 后续重构（待办）

### ✅ 已完成

#### `fitness/` — 适应度系统已激活

fitness 逻辑已从 presets 和 configurator 提取到独立的 `fitness/` 子包。实现内容：

- `fitness/_types.py`：`FitnessPopulationView` Protocol（`config` + `species` + `index_registry`）
- `fitness/_patch.py`：唯一写入层，`apply_preset_fitness_patch` 及 9 个 `_apply_*_scaling` 函数
- `fitness/_writer.py`：DSL 解析层，`write_fitness_field` 解析 pattern → 委托 `_patch.py` 统一写入
- `modifiers/gamete_conversion.py` + `modifiers/zygote_conversion.py`：从 `presets/` 迁入，原名不变
- `presets/__init__.py`、`configurator/_fitness.py`：保留向后兼容的 shim

#### `output/` — History / Observation 重构（2026-07-15）

重构观测和历史记录的数据模型与生命周期。核心变更：

- `patterns/individual_selector.py`：从 Observation 中提取独立的个体选择器，供 Hook、Preset 等复用
- `output/_recording.py`：RecordingPlan，在构建期冻结 schema + mask + meta
- `output/history.py`：History / HistorySchema / HistoryBatch 自描述存储，支持 `max_rows` 环形缓冲
- `output/observation.py`：ObservationResult 结构化结果、auto-identity observation
- `configurator/_base.py` + `spatial/configurator.py`：`record_history()` 构建期配置入口
- `population/base.py`：`pop.observation` / `pop.observe()` / `pop.record_snapshot()` / `pop.restore_checkpoint()`
- `population/_mixins/`：ObservationMixin + OutputMixin（记录写入与 checkpoint）

#### `population/base.py` — BasePopulation mixin 拆分

BasePopulation 从 1743 行拆分为一组 mixin + 532 行核心 ABC：

- `HookManagerMixin`：Hook 程序构建、编译缓存管理
- `ModifierPresetMixin`：修饰器和预设集合管理
- `ObservationMixin`：Observation 属性访问
- `OutputMixin`：历史记录写入、snapshot / checkpoint

### 🔴 高优先级

（当前无高优先级待办）

### 🟡 中优先级

#### `spatial/population.py`（2,041 行）+ `spatial/configurator.py`（1,678 行）

超大文件，需拆分为子模块。PRD #28 中标记为 Out of Scope。

#### `engine/simulation/age_structured.py`（1,342 行）

按生命周期阶段拆分。PRD #28 中标记为 Out of Scope。

### 🟢 低优先级

#### UI 类型检查修复

`ui/` 下多个文件使用全文件 `# type: ignore`，需恢复逐行类型检查。

#### 测试覆盖率提升

核心模块间集成测试覆盖不足，优先补充 fitness 和 modifier 的路径测试。

### ✅ 已确认不变更

以下文件超过 500 行但经评估不适合拆分——类的边界即自然边界：

| 文件 | 行数 | 理由 |
|---|---|---|
| `configurator/_base.py` | 1,164 | Configurator 是完整 DSL 类，内聚性高 |
| `configurator/_base.py` | （见上） | Configurator 是完整 DSL 类，内聚性高 |
| `genetics/entities/genotype.py` | 649 | 基因型构造 + 重组逻辑，单一职责 |
| `patterns/parser.py` | 613 | GenotypePatternParser 是递归下降解析器 |
| `frontend/modifiers/gamete_conversion.py` | 675 | 配子转换规则集，内聚性高（从 presets/ 迁入） |
| `frontend/modifiers/zygote_conversion.py` | 654 | 合子转换规则集，内聚性高（从 presets/ 迁入） |
