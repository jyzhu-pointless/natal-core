# 参数参考表

严格对齐 `population_builder.py` 中 Builder / Configurator 链式 API 的每个参数。

**修改方式**：A=`config.field[()]=v` B=`config.array[idx]=v` C=`set_param(config,"name",v)` D=`set_config_param(config,id,v)` E=`pop.update().method(kwarg=v)` F=`sync_equilibrium_metrics(config)`

**图例**：✅ 支持　❌ 不支持　— 不适用

---

### `setup(name, stochastic, use_continuous_sampling, use_fixed_egg_count, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `stochastic` | `is_stochastic` | Python `bool` | — | — | ❌ | ❌ | ✅ | Python 标量，E 通过 `_replace` |
| `use_continuous_sampling` | `use_continuous_sampling` | Python `bool` | — | — | ❌ | ❌ | ✅ | 同上 |
| `use_fixed_egg_count` | `use_fixed_egg_count` | Python `bool` | — | — | ❌ | ❌ | ✅ | 同上 |

### `age_structure(n_ages, new_adult_age, generation_time, equilibrium_distribution, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `n_ages` | `n_ages` | Python `int` | — | — | ❌ | ❌ | ✅ | E 重建 Config |
| `new_adult_age` | `new_adult_age` | Python `int` | — | — | ❌ | ❌ | ✅ | E 重建 Config |
| `generation_time` | `generation_time` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | |
| `equilibrium_distribution` | — | 中间参数 | — | — | — | — | — | |

### `initial_state(individual_count, sperm_storage)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `individual_count` | `initial_individual_count` | 3-D `float64` | — | ✅ | ❌ | ❌ | ✅ | is_tensor，C/D 拒绝 |
| `sperm_storage` | `initial_sperm_storage` | 3-D `float64` | — | ✅ | ❌ | ❌ | ✅ | 同上 |

### `survival(female_age_based_survival_rates, male_age_based_survival_rates, generation_time, equilibrium_distribution, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `female_age_based_survival_rates` | `age_based_survival_rates[0]` | 1-D | — | ✅ | — | — | ✅ | |
| `male_age_based_survival_rates` | `age_based_survival_rates[1]` | 1-D | — | ✅ | — | — | ✅ | |
| `generation_time` | 见 age_structure |

### `reproduction(...)` ← 参数多，分两表

**第 1 组：age-based 数组**

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `female_age_based_mating_rates` | `age_based_mating_rates[0]` | 1-D | — | ✅ | — | — | ✅ | |
| `male_age_based_mating_rates` | `age_based_mating_rates[1]` | 1-D | — | ✅ | — | — | ✅ | |
| `female_age_based_reproduction_rates` | `age_based_reproduction_rates` | 1-D | — | ✅ | — | — | ✅ | |
| `female_age_based_relative_fertility` | `female_age_based_relative_fertility` | 1-D | — | ✅ | — | — | ✅ | |

**第 2 组：标量**

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `eggs_per_female` | `expected_eggs_per_female` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | 需同步(F) |
| `sex_ratio` | `sex_ratio` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | 需同步(F) |
| `sperm_displacement_rate` | `sperm_displacement_rate` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | |
| `use_fixed_egg_count` | `use_fixed_egg_count` | Python `bool` | — | — | ❌ | ❌ | ✅ | |
| `use_sperm_storage` | — | — | — | — | — | — | — | 未传入 Config |

### `competition(competition_strength, juvenile_growth_mode, low_density_growth_rate, age_1_carrying_capacity, old_juvenile_carrying_capacity, expected_num_adult_females, equilibrium_distribution, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `competition_strength` | `age_based_relative_competition_strength[1]` | 标量 | — | ✅ | ✅ | ✅ | ✅ | |
| `juvenile_growth_mode` | `juvenile_growth_mode` | 0-d `int64` | ✅ | — | ✅ | ✅ | ✅ | |
| `low_density_growth_rate` | `low_density_growth_rate` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | |
| `age_1_carrying_capacity` | `carrying_capacity` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | 需同步(F)，别名同 K |
| `old_juvenile_carrying_capacity` | 同上 | — | — | — | — | — | — | 别名 |
| `expected_num_adult_females` | — | 中间参数 | — | — | — | — | — | |

### `fitness(viability, fecundity, sexual_selection, zygote_viability, mode)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `viability` | `viability_fitness` | 3-D | — | ✅ | ❌ | ❌ | deferred | is_tensor |
| `fecundity` | `fecundity_fitness` | 2-D | — | ✅ | ❌ | ❌ | deferred | is_tensor |
| `sexual_selection` | `sexual_selection_fitness` | 2-D | — | ✅ | ❌ | ❌ | deferred | is_tensor |
| `zygote_viability` | `zygote_viability_fitness` | 2-D | — | ✅ | ❌ | ❌ | deferred | is_tensor |

### `presets(*preset_list)`, `modifiers(gamete_modifiers, zygote_modifiers)`, `hooks(*hook_items)`, `custom(**kwargs)`

| 方法 | 运行时修改 |
|---|---|
| `presets` | deferred（需 Population + baseline），通过 E + `build()` |
| `modifiers` | deferred（需 Population），通过 E + `build()` |
| `hooks` | 注册即生效。E 中 `hooks()` 存入 `_hook_items`，`build()` 时传入 |
| `custom(**kwargs)` | ✅ nopython:`config.custom['name'][()]=v`　❌ C/D 不适用（动态字段不在 parameters.py） |

---

## DiscreteGenerationPopulationBuilder

与 AgeStructured 同名参数相同，仅列出差异。

### `reproduction(eggs_per_female, sex_ratio, female_adult_mating_rate, male_adult_mating_rate, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `eggs_per_female` | `expected_eggs_per_female` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | Discrete 无需 F（内联） |
| `sex_ratio` | `sex_ratio` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | 同上 |
| `female_adult_mating_rate` | `age_based_mating_rates[0, 1]` | 标量 | — | ✅ | ✅ | ✅ | ✅ | |
| `male_adult_mating_rate` | `age_based_mating_rates[1, 1]` | 标量 | — | ✅ | ✅ | ✅ | ✅ | |

### `survival(female_age0_survival, male_age0_survival, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `female_age0_survival` | `age_based_survival_rates[0, 0]` | 标量 | — | ✅ | ✅ | ✅ | ✅ | |
| `male_age0_survival` | `age_based_survival_rates[1, 0]` | 标量 | — | ✅ | ✅ | ✅ | ✅ | |

### `competition(juvenile_growth_mode, low_density_growth_rate, carrying_capacity, **kwargs)`

| 参数 | Config 字段 | 类型 | A | B | C | D | E | 注 |
|---|---|---|---|---|---|---|---|---|
| `carrying_capacity` | `carrying_capacity` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | Discrete 无需 F（内联） |
| `low_density_growth_rate` | `low_density_growth_rate` | 0-d `float64` | ✅ | — | ✅ | ✅ | ✅ | |
| `juvenile_growth_mode` | `juvenile_growth_mode` | 0-d `int64` | ✅ | — | ✅ | ✅ | ✅ | |

---

## 修改方式速查

| 代码 | 方法 | 适用 |
|---|---|---|
| A | `config.field[()] = v` | 0-d ndarray（9 个生态参数） |
| B | `config.array[idx] = v` | ndarray 索引（1-D/2-D/3-D 位置） |
| C | `set_param(config, "name", v)` | Python / objmode，字符串名路由 |
| D | `set_config_param(config, id, v)` | nopython hook，PARAM_IDS 整数路由 |
| E | `pop.update().method(kwarg=v)` | Python between-tick 链式修改 |
| F | `sync_equilibrium_metrics(config)` | K/eggs/sr 改后手动 sync（Age-structured） |

## 特殊说明

- **需同步(F) 的参数**：`carrying_capacity`、`eggs_per_female`、`sex_ratio`。Age-structured 模型通过 E 修改时自动调 F，hook 内 A/C/D 修改后需手动调 F。Discrete 模型内联计算，无需 F。
- **Python 标量**（`is_stochastic`、`n_ages` 等）：A/B 不适用，C/D 跳过——只能通过 E 的 `_replace`。
- **is_tensor 参数**（fitness 数组、`initial_*`）：C/D 拒绝（`set_param` 检查 `is_tensor`），只能 A/B。
- **Custom 字段**：不在 `parameters.py` 中，C/D 不适用。nopython 直接 `config.custom['name'][()] = v`。
- **Deferred 方法**（`fitness`/`presets`/`modifiers`）：E 收集到 `_deferred`，需 `build()` 执行。
