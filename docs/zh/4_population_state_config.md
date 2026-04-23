# `PopulationState` 与 `PopulationConfig`

`PopulationState` 和 `PopulationConfig` 是 NATAL 模拟框架中最关键的两个数据对象：

- `PopulationState`（以及离散世代对应的 `DiscretePopulationState`）负责保存模拟过程中的动态状态
- `PopulationConfig` 负责保存模拟参数与遗传映射，是运行内核时读取的配置对象

理解这两个对象有助于更稳定地组织初始化、运行与结果解释过程。

## 概述

用户通过 Builder 或 `setup(...).build()` 完成种群构建后，框架内部会形成以下流程：

```text
用户输入参数
  → PopulationConfig（静态配置）
  → PopulationState / DiscretePopulationState（动态状态）
  → run(...) / run_tick() 持续更新 state
```

可以将其理解为：

- `PopulationConfig` 回答"模型规则是什么"
- `PopulationState` 回答"当前系统处于什么状态"

## `PopulationState`：年龄结构模型的状态对象

`PopulationState` 定义在 `src/natal/population_state.py`，本质上是 `NamedTuple` 容器。

### 字段结构

```python
class PopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
    sperm_storage: NDArray[np.float64]     # (n_ages, n_genotypes, n_genotypes)
```

各字段含义：

- `n_tick`：当前时间步
- `individual_count`：个体计数主张量，按"性别-年龄-基因型"索引
- `sperm_storage`：用于表达雌体储精结构，按"年龄-雌性基因型-雄性基因型"索引

## `DiscretePopulationState`：离散世代模型的状态对象

离散世代模型使用 `DiscretePopulationState`，同样定义在 `src/natal/population_state.py`。

### 字段结构

```python
class DiscretePopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
```

与 `PopulationState` 的主要区别：

- 不包含 `sperm_storage` 字段
- 状态更新由离散世代流程维护
- 当前离散世代实现中，配置会规范为 `n_ages=2`、`new_adult_age=1`

## `PopulationConfig`：模型规则与映射配置

`PopulationConfig` 定义在 `src/natal/population_config.py`，包含运行模型所需的固定参数与矩阵。

### 配置内容分组

1. **维度与控制参数**
  - `n_sexes`, `n_ages`, `n_genotypes`, `n_haploid_genotypes`, `n_glabs`
  - `is_stochastic`, `use_continuous_sampling`, `sex_ratio`

2. **年龄相关参数**
  - `age_based_survival_rates`
  - `age_based_mating_rates`
  - `female_age_based_relative_fertility`
  - `age_based_relative_competition_strength`

3. **适应度参数**
  - `viability_fitness`（形状：`(n_sexes, n_ages, n_genotypes)`）
  - `fecundity_fitness`（形状：`(n_sexes, n_genotypes)`）
  - `sexual_selection_fitness`（形状：`(n_genotypes, n_genotypes)`）

4. **遗传映射矩阵**
  - `genotype_to_gametes_map`（形状：`(n_sexes, n_genotypes, n_haploid_genotypes * n_glabs)`）
  - `gametes_to_zygote_map`（形状：`(n_hg*n_glabs, n_hg*n_glabs, n_genotypes)`）

5. **初始分布与缩放参数**
  - `initial_individual_count`
  - `initial_sperm_storage`
  - `population_scale`, `base_carrying_capacity` 等

### 使用时应关注什么

`PopulationConfig` 是一个**静态对象**，包含模型的所有固定参数与遗传映射矩阵。**它不能也不应在模拟中被修改。**

可以打印输出 `PopulationConfig` 的字段值，以确认模型参数是否符合预期：

```python
cfg = pop.config
print(cfg.n_ages, cfg.n_genotypes)
print(cfg.viability_fitness.shape)
```

## 最简示例：查看 state 与 config

```python
from natal.genetic_structures import Species
from natal.age_structured_population import AgeStructuredPopulation
from natal.discrete_generation_population import DiscreteGenerationPopulation

sp = Species.from_dict(name="Demo", structure={"chr1": {"A": ["A1", "A2"]}})

age_pop = (
    AgeStructuredPopulation
    .setup(sp, stochastic=False)
    .age_structure(n_ages=4, new_adult_age=2)
    .build()
)

dis_pop = (
    DiscreteGenerationPopulation
    .setup(sp, stochastic=False)
    .build()
)

print(type(age_pop.state).__name__)  # PopulationState
print(type(dis_pop.state).__name__)  # DiscretePopulationState

print(age_pop.config.n_ages, age_pop.config.new_adult_age)  # 4, 2
print(dis_pop.config.n_ages, dis_pop.config.new_adult_age)  # 2, 1
```

<!--TODO：可能需要介绍history；需要介绍Population对象的主要方法-->

## 状态翻译为可读字典/JSON

为便于日志记录、前后端通信与调试，NATAL 提供了将状态对象翻译为人类可读结构的能力。

相关 API 位于 `natal.state_translation`：

- `population_state_to_dict` / `population_state_to_json`
- `discrete_population_state_to_dict` / `discrete_population_state_to_json`
- `population_to_readable_dict` / `population_to_readable_json`
- `population_history_to_readable_dict` / `population_history_to_readable_json`
- `population_to_observation_dict` / `population_to_observation_json`

其中：

- `PopulationState` 翻译结果包含 `individual_count` 与 `sperm_storage`
- `DiscretePopulationState` 翻译结果包含 `individual_count`（无 `sperm_storage`）

示例：

```python
import natal as nt

# 假设 pop 是任意已构建 population（年龄结构或离散世代）
readable = nt.population_to_readable_dict(pop)
print(readable["state_type"], readable["tick"])

# JSON 输出（便于持久化或传输）
payload = nt.population_to_readable_json(pop, indent=2)
print(payload[:200])

# 历史记录输出（由扁平快照转换）
hist_view = nt.population_history_to_readable_dict(pop)
print(hist_view["n_snapshots"], hist_view["snapshots"][-1]["tick"])
```

如果需要在翻译时直接应用 observation rules（详见 [种群观测规则](2_data_output.md)），可使用观测集成接口：

```python
observed = nt.population_to_observation_dict(
    pop,
    groups={
        "adult_wt_female": {
            "genotype": ["WT|WT"],
            "sex": "female",
            "age": [1],
        }
    },
    collapse_age=False,
)
print(observed["observed"]["adult_wt_female"])
```

如果直接操作 `PopulationState` / `DiscretePopulationState`，也可以调用对应的函数，并显式传入标签：

```python
from natal.state_translation import population_state_to_dict

data = population_state_to_dict(
    state,
    genotype_labels=["WT|WT", "WT|Drive", "Drive|Drive"],
    sex_labels=["female", "male"],
)
```

---

## 相关章节

- [种群初始化](2_population_initialization.md)
- [模拟内核深度解析](4_simulation_kernels.md)
- [Modifier 机制](3_modifiers.md)
- [Hook 系统](2_hooks.md)
- [种群观测规则](2_data_output.md)
