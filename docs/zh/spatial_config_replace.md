# SpatialBuilder 异构 Config 共享机制

## 问题

`SpatialBuilder._build_heterogeneous()` 为每个 config 等价组调用 `_build_template_for_group()`，该函数完整重放 builder 管线（`setup → … → build()`），每次都调用 `build_population_config()` 创建全新的 `PopulationConfig`。

如果只有少数参数在组间不同，所有大数组（`genotype_to_gametes_map`、`gametes_to_zygote_map`、`viability_fitness`、`fecundity_fitness` 等）仍会被重复创建，造成内存浪费。

```
2601 个 deme，每个有唯一的 carrying_capacity
→ 2601 个完整 PopulationConfig
→ 大数组被复制 2601 次
```

## 方案：`_replace` 快路径

`PopulationConfig` 是 `NamedTuple`，其 `_replace()` 方法创建新实例时**共享所有未被替换字段的引用**。利用这一特性，第一个组完整构建，后续组仅替换差异字段：

```
组 0: 完整 builder 管线 → base_config（所有数组）
组 1: base_config._replace(carrying_capacity=2000)       → 共享所有大数组
组 2: base_config._replace(carrying_capacity=3000)       → 共享所有大数组
...
组 N: base_config._replace(initial_individual_count=arr) → 只重建 initial_individual_count
```

## 参数发现机制

不维护硬编码的白名单，而是通过分层策略自动发现可 `_replace` 的参数：

### 1. 数组字段（显式）

需要 dict → numpy array 转换的 builder 参数，定义在 `_ARRAY_KWARGS`：

| Builder kwarg | Config 字段 | 转换方式 |
|---|---|---|
| `individual_count` | `initial_individual_count` | `PopulationConfigBuilder.resolve_*_initial_individual_count()` |
| `sperm_storage` | `initial_sperm_storage` | `PopulationConfigBuilder.resolve_age_structured_initial_sperm_storage()` |

### 2. 多字段映射（显式）

一个 builder kwarg 对应多个 config 字段（需要特殊缩放处理），定义在 `_KWARG_MULTI_FIELD`：

| Builder kwarg | Config 字段 |
|---|---|
| `carrying_capacity` | `base_carrying_capacity`（原值）+ `carrying_capacity`（× population_scale） |
| `age_1_carrying_capacity` | 同上 |
| `old_juvenile_carrying_capacity` | 同上 |

### 3. 重命名（显式）

builder kwarg 名与 config 字段名不同，定义在 `_KWARG_RENAMES`：

| Builder kwarg | Config 字段 |
|---|---|
| `eggs_per_female` | `expected_eggs_per_female` |
| `expected_num_adult_females` | `base_expected_num_adult_females` |

### 4. 动态发现（隐式）

不在上述三类中的 kwarg，通过 `hasattr(base_config, kwarg_name)` 检测是否为有效 config 字段。例如 `low_density_growth_rate`、`juvenile_growth_mode`、`sex_ratio`、`sperm_displacement_rate` 等，由于 builder kwarg 名与 config 字段名一致，**无需任何映射配置即可自动支持**。

添加新的 batch-able 标量参数通常不需要修改映射表 —— 只要 builder kwarg 名与 config 字段名相同即可。

不在上述任何类别中的参数（`presets`、`fitness`、survival 的 rate 数组等）会回退到完整 builder 重放。

### 故意不支持异构的参数

`stochastic` 和 `use_continuous_sampling` 是 simulation mode 级别的参数，不应在不同 deme 间变化。`setup()` 方法不经过 `_detect_and_delegate`，因此这些参数**无法**通过 `batch_setting` 传递。

## 平衡态重算

`carrying_capacity`、`eggs_per_female`、`sex_ratio` 变化会影响 `expected_competition_strength` 和 `expected_survival_rate`。这些参数在 `_EQUILIBRIUM_SENSITIVE_KWARGS` 中标记，`_replace` 完成后自动调用 `compute_equilibrium_metrics()` 重算。

## 数组字段的转换

`individual_count` 和 `sperm_storage` 的值是用户传入的 dict（如 `{"female": {"WT|WT": 100}}`），需要先转换为 numpy 数组才能 `_replace`。转换通过 `PopulationConfigBuilder` 的静态方法完成：

- 年龄结构：`resolve_age_structured_initial_individual_count(species, distribution, n_ages, new_adult_age)`
- 离散世代：`resolve_discrete_initial_individual_count(species, distribution)`

结果乘以 `population_scale` 以匹配 builder 行为。

## Clone 后的状态覆写

`_clone_deme()` 从模板复制 state 数据。当 `individual_count` 或 `sperm_storage` 在组间不同时，`_replace` 路径在 clone 完成后额外覆写对应的 state 数组：

```python
group_template = _clone_deme(base_template, config=variant_config, name=...)
# _clone 从 base_template 复制 state；用 config 中的值覆写
state = group_template._require_state()
if "individual_count" in sig_map:
    state.individual_count[:] = variant_config.initial_individual_count
if "sperm_storage" in sig_map:
    ss = getattr(state, 'sperm_storage', None)
    if ss is not None:
        ss[:] = variant_config.initial_sperm_storage
```

## `_build_heterogeneous` 流程

```
_build_heterogeneous()
  │
  ├─ 1. 展开所有 batch_setting 为 per-deme 值列表
  ├─ 2. 计算每个 deme 的 config 签名（hash）
  ├─ 3. 按签名分组
  │
  └─ 4. 对每个组：
       │
       ├─ 第一个组 → _build_template_for_group()
       │             完整 builder 管线，产出 base_config + base_template
       │
       ├─ 后续组 + _can_use_replace(sig_map, base_config)
       │   │  所有 kwarg 通过 显式映射 或 hasattr 动态发现
       │   │
       │   ├─ _build_variant_config(sig_map, base_config)
       │   │   │
       │   │   ├─ 数组字段 → PopulationConfigBuilder.resolve_* → _replace
       │   │   ├─ 多字段 → _replace(base=raw, scaled=raw*pop_scale)
       │   │   ├─ 重命名 → _replace(renamed_field=val)
       │   │   ├─ 动态发现 → hasattr → _replace
       │   │   └─ 平衡态敏感 → compute_equilibrium_metrics()
       │   │
       │   └─ _clone_deme(base_template, variant_config)
       │        └─ 覆写 array-valued batch setting 对应的 state
       │
       └─ 后续组 + 非 replaceable
           └─ _build_template_for_group()  （完整重放，行为不变）
```

## 内存效果

以 2601 个 deme、仅 `carrying_capacity` 不同为例：

| 项目 | 优化前 | 优化后 |
|---|---|---|
| `genotype_to_gametes_map` | 2601 份 | 1 份（共享） |
| `gametes_to_zygote_map` | 2601 份 | 1 份（共享） |
| `viability_fitness` | 2601 份 | 1 份（共享） |
| `fecundity_fitness` | 2601 份 | 1 份（共享） |
| `carrying_capacity`（标量） | 2601 个 | 2601 个（~60KB） |
| `initial_individual_count` | 2601 份 | 1 份（所有 deme 同构） |

以 2601 个 deme、仅 `initial_individual_count` 不同为例：

| 项目 | 优化前 | 优化后 |
|---|---|---|
| `genotype_to_gametes_map` | 2601 份 | 1 份（共享） |
| `gametes_to_zygote_map` | 2601 份 | 1 份（共享） |
| 所有 fitness 数组 | 2601 份 | 1 份（共享） |
| `initial_individual_count` | 2601 份 | 2601 份（必须不同） |

## 文件位置

所有改动集中在 `src/natal/spatial_builder.py`：

| 符号 | 作用 |
|---|---|
| `_ARRAY_KWARGS` | 需 dict→array 转换的参数集合 |
| `_KWARG_MULTI_FIELD` | 多字段映射（carrying_capacity 变体） |
| `_KWARG_RENAMES` | builder kwarg → config 字段重命名 |
| `_EQUILIBRIUM_SENSITIVE_KWARGS` | 需重算平衡态的参数集合 |
| `SpatialBuilder._build_heterogeneous()` | 主构建逻辑 |
| `SpatialBuilder._can_use_replace(sig_map, base_config)` | 判断是否可用 `_replace` |
| `SpatialBuilder._build_variant_config()` | 创建 variant config |
