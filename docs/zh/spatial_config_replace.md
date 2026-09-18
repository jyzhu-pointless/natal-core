# SpatialPopulationBuilder 异构 Config 共享机制

> **实现说明**：本页描述异构构建中 `ModelDraft._replace` 共享大数组的内部机制。该机制仍在使用，但**不是**异构构建的全部内容——声明冻结、按签名分组与模板克隆见 [SpatialPopulationBuilder：空间种群批量构造](spatial_population_builder.md)。

## 问题

`SpatialPopulationBuilder._build_heterogeneous_demes()` 按**遗传学**签名分组编译（`_genetics_batch_names()` 决定哪些 batch kwarg 属于遗传学；仅生态参数的差异不会拆分组）。每个 deme 都需要自己的 `ModelDraft`；若每个都从头编译，所有大数组（`zygotes_to_gametes_map`、`gametes_to_zygotes_map`、`viability_fitness`、`fecundity_fitness` 等）都会被复制，造成内存浪费。

## 方案：声明投影 + 组内共享遗传产物

每个 deme 的具体声明（按该 deme 批值解析后的日志）经唯一声明解释器（`builder/_declarations.py`）投影到全新基线——不重新执行任何 builder 方法。遗传组经 `compile_definition` 编译一次；组内每个 deme 原样挂接该组的遗传产物字段，大数组由此共享：

```
组编译：  投影组 0 声明 → compile_definition → products
deme i：  仅投影与组值不同的声明到 products.config
          → 共享该组的遗传产物数组
```


## 方案：`_replace` 快路径

`ModelDraft` 是 `NamedTuple`，其 `_replace()` 方法创建新实例时**共享所有未被替换字段的引用**。利用这一特性，组内第一个 variant 完整编译，后续 variant 仅替换差异字段：

```
variant 0: 完整编译管线 → base_config（所有数组）
variant 1: base_config._replace(carrying_capacity=2000)       → 共享所有大数组
variant 2: base_config._replace(carrying_capacity=3000)       → 共享所有大数组
...
variant N: base_config._replace(initial_individual_count=arr) → 只重建 initial_individual_count
```

## 哪些参数可以按 deme 不同

不再有映射表。差量投影复用链式方法自身的写入路径（`competition()` / `reproduction()` / `survival()` / `initial_state()` 背后的路由表 writer），因此这些方法接受的每个参数都可按 deme 投影：标量（`carrying_capacity`、`eggs_per_female`、`sex_ratio` 等）、按年龄向量（`female_age_based_survival` 等），以及初始分布（`individual_count` / `sperm_storage` 字典由 `natal.frontend.model.initial_state` 的纯解析函数转成数组）。

影响遗传学的 kwarg（`presets`、`fitness` 各行、自定义 modifier）不走差量投影：它们把 deme 拆成不同的遗传组，每组经 `compile_definition` 编译一次。遗传内容与模板相同的组继承模板的编译缓存，配方不会为相同内容重跑。派生标量（例如 `expected_num_new_adult_females` 推导的 Champer 卵覆盖）冻结于组计算值，除非用户为该 deme 重新声明。


## 平衡态指标在读取时派生

`carrying_capacity`、`eggs_per_female`、`sex_ratio` 变化会影响 `expected_competition_strength` 和 `expected_survival_rate`，但 draft 不存储这两个值：`ModelDraft` 没有平衡态字段，`_replace` 之后也没有任何重算步骤。两个指标在读取时实时派生 —— `pop.params.expected_competition_strength` / `pop.params.expected_survival_rate` 调用 `derive_equilibrium_metrics_from_draft()`（`src/natal/frontend/model/ecology.py`），因此 `_replace` 得到的 variant 无需任何同步即自动保持一致。

## 数组字段的转换

`individual_count` 和 `sperm_storage` 的值是用户传入的 dict（如 `{"female": {"WT|WT": 100}}`），需要先转换为 numpy 数组才能 `_replace`。转换由 `natal.frontend.model.initial_state` 中的普通解析函数完成：

- 年龄结构：`resolve_age_structured_initial_individual_count(species, distribution, n_ages, new_adult_age)`
- 年龄结构精子库：`resolve_age_structured_initial_sperm_storage(species, sperm_storage, n_ages, new_adult_age)`
- 离散世代：`resolve_discrete_initial_individual_count(species, distribution)`

结果匹配 builder 行为。

## Clone 与初始状态

`_clone_deme()` 委托给 `PopulationInstance._clone()`（`src/natal/frontend/population/base.py`），后者从模板 deme 复制 state 数组（`individual_count`，年龄结构下还有 `sperm_storage`）。clone 总是与模板共享同一个 config 对象，因此复制来的 state 本就与 config 一致 —— 不需要 clone 后再覆写。

初始状态不同的 deme 从不通过 clone 产生：它们的 variant config（由解析函数新算出的 `initial_individual_count` / `initial_sperm_storage` 数组）走正常的发布路径（`_publish_and_build()`），由 config 初始化种群状态。

## `_build_heterogeneous_demes` 流程

```
_build_heterogeneous_demes()
  │
  ├─ 1. 把所有 batch_settings 展开为按 deme 的值列表
  ├─ 2. _genetics_batch_names() → 影响遗传学段的 batch kwarg
  ├─ 3. 按仅遗传学签名分组（生态差异不拆组）
  │
  └─ 4. 每个遗传组——编译候选：
       │
       ├─ _projected_group(values[first])
       │     解析该 deme 日志 → 投影到全新基线
       │     （唯一解释器；零 builder 方法执行）
       ├─ _carrier_from_projection(...)
       │     遗传与模板匹配的组 0 继承模板编译缓存（配方不重跑）
       ├─ carrier._compile_products() → compile_definition
       │
       └─ 组内其余 deme：
            ├─ 完整签名与先前 deme 相同 → 复用其候选
            └─ _projected_variant_config(group_config, values_i, values_first)
                  仅投影*不同*的声明；派生标量冻结于组计算值
  │
  ├─ 5. _spatial_projection() → 共享 registry 投影（各组路由表之并）
  │
  └─ 6. 每个遗传组——发布：每个唯一 config → builder._publish_and_build()

## 内存效果

以 2601 个 deme、仅 `carrying_capacity` 不同为例：

| 项目 | 优化前 | 优化后 |
|---|---|---|
| `zygotes_to_gametes_map` | 2601 份 | 1 份（共享） |
| `gametes_to_zygotes_map` | 2601 份 | 1 份（共享） |
| `viability_fitness` | 2601 份 | 1 份（共享） |
| `fecundity_fitness` | 2601 份 | 1 份（共享） |
| `carrying_capacity`（标量） | 2601 个 | 2601 个（~60KB） |
| `initial_individual_count` | 2601 份 | 1 份（所有 deme 同构） |

以 2601 个 deme、仅 `initial_individual_count` 不同为例：

| 项目 | 优化前 | 优化后 |
|---|---|---|
| `zygotes_to_gametes_map` | 2601 份 | 1 份（共享） |
| `gametes_to_zygotes_map` | 2601 份 | 1 份（共享） |
| 所有 fitness 数组 | 2601 份 | 1 份（共享） |
| `initial_individual_count` | 2601 份 | 2601 份（必须不同） |

## 文件位置

相关实现集中在 `src/natal/frontend/spatial/builder.py`：

| 符号 | 作用 |
|---|---|
| `_projected_group()` | 解析 deme 日志并投影到全新基线 |
| `_projected_variant_config()` | 把 deme 与组不同的声明投影到组 config |
| `builder/_declarations.py` | 两条路径共享的唯一声明解释器 |
| `_genetics_batch_names()` | 选出会拆分遗传学组的 batch kwarg |
| `SpatialPopulationBuilder._build_heterogeneous_demes()` | 异构构建主流程 |
| `_clone_deme()` → `PopulationInstance._clone()` | 克隆已发布的 deme，共享编译状态与 config |
