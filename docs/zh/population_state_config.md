# `PopulationState` 与 `PopulationConfig`

本章介绍 NATAL 模拟中最关键的两个数据对象：

- `PopulationState`（以及离散世代对应的 `DiscretePopulationState`）：用于保存模拟过程中的动态状态。
- `PopulationConfig`：用于保存模拟参数与遗传映射，是运行内核时读取的配置对象。

理解这两个对象，可以帮助你更稳定地组织初始化、运行与结果解释。

## 1. 先建立整体心智模型

在用户层，通常通过 Builder 或 `setup(...).build()` 完成模型构建。构建完成后，框架内部会形成：

```text
用户输入参数
  → PopulationConfig（静态配置）
  → PopulationState / DiscretePopulationState（动态状态）
  → run(...) / run_tick() 持续更新 state
```

可以把它理解为：

- `PopulationConfig` 回答“模型规则是什么”。
- `PopulationState` 回答“当前系统处于什么状态”。

## 2. `PopulationState`：年龄结构模型的状态对象

`PopulationState` 定义在 `src/natal/population_state.py`，本质是 `NamedTuple` 容器。

### 2.1 字段结构

```python
class PopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
    sperm_storage: NDArray[np.float64]     # (n_ages, n_genotypes, n_genotypes)
```

各字段含义：

- `n_tick`：当前时间步。
- `individual_count`：个体计数主张量，按“性别-年龄-基因型”索引。
- `sperm_storage`：用于表达雌体储精结构，按“年龄-雌性基因型-雄性基因型”索引。

### 2.2 为什么是 `NamedTuple`

这种设计兼顾了“结构清晰”和“数值计算效率”：

- 标量字段（如 `n_tick`）语义稳定。
- 数组字段可原地更新，便于模拟循环中高频写入。

### 2.3 推荐创建方式

```python
from natal.population_state import PopulationState

state = PopulationState.create(
    n_genotypes=6,
    n_sexes=2,
    n_ages=8,
    n_tick=0,
)
```

> 提示：通常不需要手动创建 state；框架会在 population 初始化时自动创建并维护。

## 3. `DiscretePopulationState`：离散世代模型的状态对象

离散世代模型使用 `DiscretePopulationState`。它同样定义在 `src/natal/population_state.py`。

### 3.1 字段结构

```python
class DiscretePopulationState(NamedTuple):
    n_tick: int
    individual_count: NDArray[np.float64]  # (n_sexes, n_ages, n_genotypes)
```

与 `PopulationState` 的主要区别：

- 不包含 `sperm_storage` 字段。
- 状态更新由离散世代流程维护。
- 当前离散世代实现中，配置会规范为 `n_ages=2`、`new_adult_age=1`。

### 3.2 推荐创建方式

```python
from natal.population_state import DiscretePopulationState

state = DiscretePopulationState.create(
    n_sexes=2,
    n_ages=2,
    n_genotypes=6,
    n_tick=0,
)
```

## 4. `PopulationConfig`：模型规则与映射配置

`PopulationConfig` 定义在 `src/natal/population_config.py`。它包含运行模型所需的固定参数与矩阵。

### 4.1 配置内容分组

1. 维度与控制参数

- `n_sexes`, `n_ages`, `n_genotypes`, `n_haploid_genotypes`, `n_glabs`
- `is_stochastic`, `use_continuous_sampling`, `sex_ratio`

2. 年龄相关参数

- `age_based_survival_rates`
- `age_based_mating_rates`
- `female_age_based_relative_fertility`
- `age_based_relative_competition_strength`

3. 适应度参数

- `viability_fitness`（形状：`(n_sexes, n_ages, n_genotypes)`）
- `fecundity_fitness`（形状：`(n_sexes, n_genotypes)`）
- `sexual_selection_fitness`（形状：`(n_genotypes, n_genotypes)`）

4. 遗传映射矩阵

- `genotype_to_gametes_map`（形状：`(n_sexes, n_genotypes, n_haploid_genotypes * n_glabs)`）
- `gametes_to_zygote_map`（形状：`(n_hg*n_glabs, n_hg*n_glabs, n_genotypes)`）

5. 初始分布与缩放参数

- `initial_individual_count`
- `initial_sperm_storage`
- `population_scale`, `base_carrying_capacity` 等

### 4.2 使用时应关注什么

对多数用户而言，最常见的是读取配置而不是手动构造配置：

```python
cfg = pop.config
print(cfg.n_ages, cfg.n_genotypes)
print(cfg.viability_fitness.shape)
```

如果你需要修改某些系数，请先确认目标字段的维度和生物学语义，再进行写入。

## 5. 从输入到运行：编译步骤在做什么

从用户视角，构建阶段主要完成四件事：

1. 解析输入

- 将基因型字符串解析为内部对象。
- 将初始计数统一整理为固定形状数组。

2. 构建遗传映射

- 生成基因型到配子的概率映射。
- 生成配子结合到合子的概率映射。

3. 组装生命史与适应度参数

- 汇总年龄相关生存/交配/繁殖参数。
- 写入 viability / fecundity / sexual selection 张量。

4. 创建可运行对象

- 绑定 `PopulationConfig`。
- 初始化对应的 `PopulationState` 或 `DiscretePopulationState`。

## 6. 两类模型该如何理解

### 6.1 AgeStructuredPopulation

适用于年龄层次明确、年龄转移与储精过程需要显式表达的场景。

### 6.2 DiscreteGenerationPopulation

适用于非重叠世代场景；状态结构更紧凑，流程语义更简洁。

## 7. 最小示例：查看 state 与 config

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

## 8. 实践建议

1. 优先通过 Builder 或 `setup(...).build()` 生成对象，避免手动拼接底层数组。
2. 修改配置数组前，先核对维度，再核对生物学含义。
3. 在离散世代模型中，建议始终按 age 0/1 的语义组织输入与分析。

## 9. 状态翻译为可读字典/JSON

为了便于日志记录、前后端通信与调试，NATAL 提供了将状态对象翻译为人类可读结构的能力。

相关 API 位于 `natal.state_translation`：

- `population_state_to_dict` / `population_state_to_json`
- `discrete_population_state_to_dict` / `discrete_population_state_to_json`
- `population_to_readable_dict` / `population_to_readable_json`
- `population_to_observation_dict` / `population_to_observation_json`

其中：

- `PopulationState` 翻译结果包含 `individual_count` 与 `sperm_storage`。
- `DiscretePopulationState` 翻译结果包含 `individual_count`（无 `sperm_storage`）。

示例：

```python
import natal as nt

# 假设 pop 是任意已构建 population（年龄结构或离散世代）
readable = nt.population_to_readable_dict(pop)
print(readable["state_type"], readable["tick"])

# JSON 输出（便于持久化或传输）
payload = nt.population_to_readable_json(pop, indent=2)
print(payload[:200])
```

如果需要在翻译时直接应用 observation rules（详见 [种群观测规则](observation_rules.md)），可使用观测集成接口：

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

如果你直接操作 `PopulationState` / `DiscretePopulationState`，也可以调用对应的函数，并显式传入标签：

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

- [Builder 系统](builder_system.md)
- [模拟内核深度解析](simulation_kernels.md)
- [Modifier 机制](modifiers.md)
- [Hook 系统](hooks.md)
- [种群观测规则](observation_rules.md)
