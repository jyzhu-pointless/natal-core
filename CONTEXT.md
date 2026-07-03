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
| 种群配置 | PopulationConfig | 引擎的静态配置（不可变 NamedTuple），包含所有生态和遗传参数 |
| 种群状态 | PopulationState | 引擎的可变状态（数组容器），记录当前个体分布 |
| 配置器 | Configurator | 构建和修改 PopulationConfig 的链式 API |
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
> 🔴 = 标记后续重构

```
src/natal/
│
├── __init__.py                # 惰性加载入口
│
├── genetics/                  # 🧬 遗传结构定义 ✅
│   ├── structures/
│   │   ├── _base.py           # GeneticStructure 基类
│   │   ├── _registry.py       # RegistryBase / EntityRegistry / ChildStructureRegistry
│   │   ├── _types.py          # SexChromosomeType + 类型别名
│   │   ├── species.py         # Species 核心（构造、CRUD、缓存）
│   │   ├── species_dict.py    # from_dict + 字符串解析方法
│   │   ├── species_iteration.py  # 基因型/单倍型迭代枚举
│   │   ├── species_pattern.py # 模式匹配方法
│   │   ├── species_maps.py    # gamete/zygote maps + 配置蓝图 + 辅助函数
│   │   ├── chromosome.py      # Chromosome 核心
│   │   ├── chromosome_map.py  # RecombinationMap
│   │   └── locus.py           # Locus
│   ├── entities/
│   │   ├── _base.py           # GeneticEntity 基类
│   │   ├── gene.py            # Gene
│   │   ├── haplotype.py       # Haplotype + HaploidGenotype
│   │   └── genotype.py        # Genotype + 重组逻辑
│   └── __init__.py
│
├── patterns/                  # 🔍 基因型模式匹配 ✅
│   ├── elements/
│   │   ├── _base.py           # PatternElement(ABC) + PatternParseError
│   │   ├── atom.py            # AllelePattern, WildcardPattern, SetPattern, LabPattern, LocusPattern
│   │   ├── chromosome.py      # HaplotypePath, ChromosomePairPattern
│   │   ├── haploid.py         # HaploidGenomePattern, GameteTypePattern
│   │   └── diploid.py         # GenotypePattern, ZygoteTypePattern
│   ├── parser.py              # GenotypePatternParser
│   ├── selector.py            # GenotypeSelector + resolve_zygote_type
│   └── __init__.py
│
├── presets/                   # 🎯 遗传预设 ✅
│   ├── _types.py              # 配置类型别名 + TypeGuard
│   ├── _fitness.py            # fitness patch 构建/应用引擎
│   ├── _base.py               # GeneticPreset(ABC) + apply_preset_to_population
│   ├── homing.py              # HomingDrive
│   ├── toxin_antidote.py      # ToxinAntidoteDrive
│   ├── cytoplasmic.py         # CytoplasmicPreset + Wolbachia + TransgenicBackground
│   ├── gamete_conversion.py   # 配子等位基因转换
│   ├── zygote_conversion.py   # 合子等位基因转换
│   └── __init__.py
│
├── configurator/              # 🔧 种群配置器 ✅
│   ├── _base.py               # Configurator 基类
│   ├── _factory.py            # PopulationConfigBuilder build() 逻辑（旧 Builder 废弃）
│   ├── _params.py             # 参数解析辅助
│   ├── discrete.py            # DiscreteConfigurator
│   ├── age_structured.py      # AgeStructuredConfigurator
│   └── __init__.py
│
├── data/                      # 📦 引擎面数据结构 ✅
│   ├── config.py              # PopulationConfig + DiscretePopulationConfig (NamedTuple)
│   ├── state.py               # PopulationState + DiscretePopulationState
│   ├── _builders.py           # build_population_config 等工厂函数
│   ├── _extract.py            # extract_gamete_frequencies 等提取函数
│   ├── constants.py           # 增长模式常量 (NO_COMPETITION, FIXED, LOGISTIC, CONCAVE...)
│   └── __init__.py
│
├── registry/                  # 📋 索引注册表 ✅
│   ├── index.py               # IndexRegistry
│   └── __init__.py
│
├── population/                # 👥 种群模型 ✅
│   ├── base.py                # BasePopulation(ABC) 🔴
│   ├── age_structured.py      # AgeStructuredPopulation
│   ├── discrete_generation.py # DiscreteGenerationPopulation
│   └── __init__.py
│
├── spatial/                   # 🗺️ 空间模型 ✅
│   ├── population.py          # SpatialPopulation 🔴
│   ├── configurator.py        # SpatialConfigurator 🔴
│   ├── topology.py            # SpatialTopology
│   └── __init__.py
│
├── output/                    # 📤 数据输出 ✅
│   ├── observation.py         # Observation 类、GroupsInput、过滤规则
│   ├── record.py              # CompactMeta、引擎端记录构建器
│   ├── translation.py         # 状态格式化输出
│   └── __init__.py
│
├── modifiers/                 # 🔀 遗传修饰器 ✅
│   ├── module.py              # GameteModifier / ZygoteModifier Protocol
│   └── __init__.py
│
├── fitness/                   # 💪 适应度系统 🔴 空壳，后续从 presets/configurator 提取
│   └── __init__.py
│
├── numba/                     # ⚡ Numba 基础设施 ✅
│   ├── utils.py               # njit_switch, enable/disable, cache 管理
│   ├── compat.py              # 双实现兼容层
│   └── __init__.py
│
├── utils/                     # 🛠 通用工具 ✅
│   ├── types.py               # Sex(IntEnum), Age, GameteLabel
│   ├── helpers.py             # resolve_sex_label, validate_name
│   ├── parameters.py          # ParamDescriptor + JSONC 注册表
│   └── __init__.py
│
├── hooks/                     # 🪝 Hook 系统（保持现有结构）
├── engine/                    # ⚙️ 模拟引擎 🔴 后续按生命周期阶段拆分
└── ui/                        # 🖥️ Web 界面 🔴 后续拆分
```

---

## 依赖方向

```
utils → genetics → patterns → presets → modifiers
                                    ↘ fitness (后续)
genetics + patterns → registry → data
data → configurator → population → spatial → output → ui
hooks → engine → population
```

## 关键设计决策

1. **`data/` 独立于 `configurator/`**：config 和 state 是面向引擎的纯数据结构，不依赖配置器自身。
2. **`patterns/` 与 `genetics/` 平行**：patterns 被 hooks、configurator、modifiers 等多个模块依赖，不是 genetics 的子概念。
3. **`registry/` 独立顶层**：IndexRegistry 是遗传领域到引擎整数空间的桥梁。
4. **`fitness/` 空壳**：fitness 逻辑散落在 presets 和 configurator 中，预留子包待提取。
5. **`modifiers/` 独立**：修饰器是连接 presets 和引擎的独立抽象层。
6. **500 行单模块上限**：每个 `.py` 文件不超过 500 行，超限需拆分。
7. **Numba cache 位于工作目录**：`numba/utils.py` 中 `NUMBA_CACHE_DIR` 默认指向 `cwd/.numba_cache`，而非项目根目录。
8. **旧 Builder 已废弃**：`population_builder.py` 中的 Builder 类已删除，统一使用 Configurator API。
9. **旧模块导入路径已全部更新**：`genetic_structures`、`genetic_entities`、`genetic_patterns`、`population_config`、`population_state` 等旧路径不再存在。

## 后续重构（待办）

- `population/base.py` 🔴 — BasePopulation ABC 职责过多，混杂 Hook 管理、历史记录、事件派发
- `spatial/population.py` + `spatial/configurator.py` 🔴 — 均超 1,600 行，需拆分
- `engine/simulation/age_structured.py` 🔴 — 1,342 行，按生命周期阶段拆分
- `fitness/` 🔴 — 从 presets + configurator 提取适应度逻辑
- `engine/` 🔴 — 整体按生命周期阶段重构
