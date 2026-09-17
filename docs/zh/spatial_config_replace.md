# SpatialPopulationBuilder 异构 Config 共享机制

> **实现说明**：本页描述异构构建中 `ModelDraft._replace` 共享大数组的内部机制。该机制仍在使用，但**不是**异构构建的全部内容——声明冻结、按签名分组与模板克隆见 [SpatialPopulationBuilder：空间种群批量构造](spatial_population_builder.md)。

## 问题

`SpatialPopulationBuilder._build_heterogeneous_demes()` 按**遗传学**签名分组编译 builder 管线（`_genetics_batch_names()` 决定哪些 batch kwarg 属于遗传学；仅生态参数的差异不会拆分组）。组内每个额外的 variant 仍需要自己的 `ModelDraft`：要么从该组的基础 config 派生，要么由 `_builder_for_group()` 完整重放 builder 管线（`setup → … → build()`），每次都编译出全新的 `ModelDraft`。

如果不存在 `_replace` 快路径，每个仅生态参数不同的 variant 都需要完整重放，所有大数组（`zygotes_to_gametes_map`、`gametes_to_zygotes_map`、`viability_fitness`、`fecundity_fitness` 等）都会被重复创建，造成内存浪费。

```
2601 个 deme，每个有唯一的 carrying_capacity
→ 2601 个完整 ModelDraft（每个 deme 一次完整重放）
→ 大数组被复制 2601 次
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

## 参数发现机制

不维护硬编码的白名单，而是通过分层策略自动发现可 `_replace` 的参数：

### 1. 数组字段（显式）

需要 dict → numpy array 转换的 builder 参数，定义在 `_ARRAY_KWARGS`：

| Builder kwarg | Config 字段 | 转换方式 |
|---|---|---|
| `individual_count` | `initial_individual_count` | `initial_state.resolve_*_initial_individual_count()` |
| `sperm_storage` | `initial_sperm_storage` | `initial_state.resolve_age_structured_initial_sperm_storage()` |

### 2. 多字段映射（显式）

离散世代标量 builder kwarg 各自写入统一 `(2, n_ages)` 向量 config 字段的**一个单元格**，定义在 `_DISCRETE_VECTOR_CELLS`。写入前会先复制目标向量，variant 不会与 base 产生别名：

| Builder kwarg | Config 字段 | 单元格 |
|---|---|---|
| `female_age0_survival` | `age_based_survival_rates` | `(0, 0)` |
| `male_age0_survival` | `age_based_survival_rates` | `(1, 0)` |
| `female_adult_mating_rate` | `age_based_mating_rates` | `(0, 1)` |
| `male_adult_mating_rate` | `age_based_mating_rates` | `(1, 1)` |

### 3. 重命名（显式）

builder kwarg 名与 config 字段名不同，定义在 `_KWARG_RENAMES`：

| Builder kwarg | Config 字段 |
|---|---|
| `eggs_per_female` | `eggs_per_female` |

### 4. 动态发现（隐式）

不在上述三类中的 kwarg，通过 `hasattr(base_config, kwarg_name)` 检测是否为有效 config 字段。例如 `low_density_growth_rate`、`juvenile_growth_mode`、`sex_ratio`、`sperm_displacement_rate` 等，由于 builder kwarg 名与 config 字段名一致，**无需任何映射配置即可自动支持**。

添加新的 batch-able 标量参数通常不需要修改映射表 —— 只要 builder kwarg 名与 config 字段名相同即可。

影响遗传学的 kwarg（`presets`、`fitness` 的行参数如 `viability` / `fecundity`、自定义 modifier 等）不会走 `_replace`：它们把 deme 拆分进不同的遗传学组，每组各走一次完整 builder 重放。组内未通过 `hasattr` 检测的非遗传学 kwarg 同样回退到完整重放；重放得到的 variant 随后与基础 config 共享遗传学产物字段。

### 故意不支持异构的参数

`stochastic` 和 `continuous_sampling` 是 simulation mode 级别的参数，不应在不同 deme 间变化。`setup()` 直接接收它们，不经过 batch 机制，因此这些参数**无法**通过 `batch_setting` 传递。

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
  ├─ 1. 展开所有 batch_setting 为 per-deme 值列表
  ├─ 2. _genetics_batch_names() → 找出影响遗传学的 batch kwarg
  ├─ 3. 按纯遗传学签名对 deme 分组
  │     （生态参数差异不会拆分组）
  │
  └─ 4. 每个遗传学组 —— 编译候选：
       │
       ├─ 第一个 deme → 模板 builder 的副本（deme 0）
       │              或 _builder_for_group() 完整重放
       │              → _compile_products() → base_config
       │
       └─ 组内其他 deme：
            ├─ 与更早 deme 完整签名相同 → 复用其候选
            ├─ _can_use_replace(生态 kwarg, base_config)
            │   ├─ 是 → _build_variant_config(sig_map, base_config)
            │   │        │
            │   │        ├─ 数组字段 → initial_state 的 resolve_* → _replace
            │   │        ├─ 离散标量 → 复制向量、写一个单元格 → _replace
            │   │        ├─ 重命名 → _replace(renamed_field=val)
            │   │        └─ 动态发现 → hasattr → _replace
            │   └─ 否 → _builder_for_group() 完整重放，再
            │            从 base_config _replace 遗传学产物字段
  │
  ├─ 5. _spatial_projection() → 共享注册表投影
  │     （所有组路由表的并集）
  │
  └─ 6. 每个遗传学组 —— 发布：
       ├─ 每个唯一 config → builder._publish_and_build()
       │    （首个发布的 deme 成为遗传学模板）
       └─ 共享签名的 deme → _clone_deme() → _clone()
            （从模板复制 state 数组；大数组共享）
```

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
| `_ARRAY_KWARGS` | 需 dict→array 转换的参数集合 |
| `_DISCRETE_VECTOR_CELLS` | 离散标量 kwarg → 统一向量字段的一个单元格 |
| `_KWARG_RENAMES` | builder kwarg → config 字段重命名 |
| `_genetics_batch_names()` | 选出会拆分遗传学组的 batch kwarg |
| `SpatialPopulationBuilder._build_heterogeneous_demes()` | 异构构建主流程 |
| `SpatialPopulationBuilder._builder_for_group()` | 将一个组重放为完整轴的未发布 builder |
| `SpatialPopulationBuilder._can_use_replace(sig_map, base_config)` | 判断是否可用 `_replace` |
| `SpatialPopulationBuilder._build_variant_config()` | 创建 variant config |
| `_clone_deme()` → `PopulationInstance._clone()` | 克隆已发布的 deme，共享编译状态与 config |
