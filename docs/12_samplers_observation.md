# Samplers：观察过滤系统

::: warnings
此模块目前位于 NATAL Inferencer，预计后续才会迁移至 NATAL Core。
:::

本章介绍 `samplers.observation` 模块，用于从种群状态中提取和聚合数据。这是进行数据同化 (PMCMC) 和统计推断的关键组件。

---

## 核心概念

### 为什么需要 ObservationFilter？

在模拟过程中，我们通常需要：
1. 从完整的 `individual_count` 数组（全基因型、全性别、全年龄）中提取特定的子群体
2. 将多维数据压缩为一维向量进行统计比较
3. 支持灵活的分组（如"所有成年雌性"或"特定基因型的幼虫"）

### 三个关键对象

| 对象 | 作用 |
|------|------|
| **ObservationFilter** | 创建过滤规则和应用规则的主类 |
| **规则 (rule)** | NumPy 掩码数组，shape: `(n_groups, n_sexes, [n_ages], n_genotypes)` |
| **观察结果 (observed)** | 应用规则后的聚合数据，shape: `(n_groups, n_sexes, [n_ages])` |

### 设计特点

- **纯函数设计**: `apply_rule()` 是无副作用的 NumPy 操作
- **灵活的选择器**: 支持多种格式指定基因型、年龄、性别
- **无序基因型**: `unordered=True` 自动处理 "A|a" 和 "a|A" 等价性
- **性能优化**: 全部用 NumPy 向量化操作，避免循环

---

## ObservationFilter API

### 构造函数

```python
from natal.observation import ObservationFilter
from natal.index_registry import IndexRegistry

# 创建过滤器
registry = pop.registry  # IndexRegistry 实例
filter = ObservationFilter(registry)
```

**参数**：
- `registry`: IndexRegistry 对象，用于基因型名称解析

### build_filter 方法

主方法，构建过滤规则。

```python
def build_filter(
    self,
    pop_or_state: Union[PopulationState, BasePopulation],
    *,
    diploid_genotypes: Optional[Union[Sequence, Species, BasePopulation]] = None,
    groups: Optional[Union[List, Tuple, Dict]] = None,
    collapse_age: bool = False,
) -> Tuple[np.ndarray, List[str]]
```

**参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `pop_or_state` | PopulationState \| BasePopulation | 种群对象或状态 |
| `diploid_genotypes` | Sequence \| Species \| BasePopulation | 基因型列表（支持多种格式）|
| `groups` | None \| List \| Dict | 分组规范 |
| `collapse_age` | bool | 是否将所有年龄合并为一维 |

**返回**：
```python
(rule, labels)
# rule: np.ndarray, shape (n_groups, n_sexes, [n_ages], n_genotypes)
# labels: List[str], 每个分组的名称
```

#### groups 格式

##### 1. groups=None（默认，每个基因型一组）

```python
rule, labels = filter.build_filter(pop, diploid_genotypes=pop.species)
# labels = ['g0', 'g1', 'g2', ...]，一个标签对应一个基因型
```

##### 2. groups 是列表（无名分组）

```python
groups = [
    {"genotype": ["WT|WT"], "sex": "female"},
    {"genotype": ["WT|Drive"], "age": [2, 3, 4]},
]
rule, labels = filter.build_filter(pop, groups=groups)
# labels = ['group_0', 'group_1']
```

##### 3. groups 是字典（有名分组）

```python
groups = {
    "all_females": {"sex": "female"},
    "adults": {"age": [2, 3, 4, 5, 6, 7]},
    "drive_carriers": {"genotype": ["WT|Drive", "Drive|Drive"]},
    "juvenile_drive": {
        "genotype": ["WT|Drive"],
        "age": [0, 1],
        "sex": "female"
    },
}
rule, labels = filter.build_filter(pop, groups=groups)
# labels = ['all_females', 'adults', 'drive_carriers', 'juvenile_drive']
```

#### 选择器格式 (Selector Specs)

在 groups 字典中，每个分组规范支持以下键：

##### genotype / genotypes

指定基因型。支持多种格式：

```python
# 字符串（逗号分隔）
{"genotype": "WT|WT"}

# 字符串列表
{"genotype": ["WT|WT", "WT|Drive", "Drive|Drive"]}

# 整数索引
{"genotype": [0, 2, 3]}

# 通配符（所有基因型）
{"genotype": "*"}

# 不指定（默认所有基因型）
{}
```

##### sex

指定性别。支持格式：

```python
# 字符串
{"sex": "female"}  或  {"sex": "male"}
{"sex": "f"}       或  {"sex": "m"}

# 整数（Sex.FEMALE = 0, Sex.MALE = 1）
{"sex": 0}  或  {"sex": 1}

# 列表
{"sex": ["female", "male"]}

# 不指定或 None（两性）
{}
```

##### age

指定年龄。支持多种格式：

```python
# 显式列表
{"age": [2, 3, 4]}

# 闭区间元组 [start, end]（包含）
{"age": [2, 7]}  # ages 2,3,4,5,6,7

# 区间列表（并集）
{"age": [[0, 1], [4, 6]]}  # ages 0,1,4,5,6

# 可调用对象（断言函数）
{"age": lambda a: a >= 2}

# 不指定（所有年龄）
{}
```

##### unordered

处理无序基因型（"A|a" 和 "a|A" 视为等同）：

```python
# 启用无序匹配
{"genotype": ["A|a"], "unordered": True}
# 会同时匹配 "A|a" 和 "a|A"

# 禁用（默认）
{"genotype": ["A|a"], "unordered": False}
# 只匹配 "A|a"
```

---

## 构建过滤规则

### 示例 1：简单性别分组

```python
from natal.observation import ObservationFilter

filter = ObservationFilter(pop.registry)

groups = {
    "females": {"sex": "female"},
    "males": {"sex": "male"},
}

rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,
    groups=groups
)

print(labels)  # ['females', 'males']
print(rule.shape)  # (2, 2, 8, n_genotypes) if 8 ages
```

### 示例 2：年龄分层

```python
groups = {
    "juveniles": {"age": [0, 1]},
    "young_adults": {"age": [2, 3]},
    "old_adults": {"age": [4, 7]},
}

rule, labels = filter.build_filter(pop, groups=groups)
```

### 示例 3：基因型特异性观察

```python
# 关注特定基因型的时间动态
groups = {
    "WT_WT_all": {"genotype": ["WT|WT"]},
    "WT_WT_females": {"genotype": ["WT|WT"], "sex": "female"},
    "WT_WT_juvenile_f": {"genotype": ["WT|WT"], "sex": "female", "age": [0, 1]},
    "drive_all": {"genotype": ["WT|Drive", "Drive|Drive"]},
}

rule, labels = filter.build_filter(pop, groups=groups)
```

### 示例 4：无序基因型（抑制/驱动系统）

在抑制系统（如 CRISPR-based suppression）中，"S|+" 和 "+|S" 可能是等价的：

```python
groups = {
    "suppressed_hetero": {
        "genotype": ["S|+"],
        "unordered": True,  # 同时匹配 "S|+" 和 "+|S"
    },
    "suppressed_homo": {"genotype": ["S|S"]},
}

rule, labels = filter.build_filter(pop, groups=groups)
```

### 示例 5：年龄合并

有时我们想忽略年龄维度，直接比较性别和基因型：

```python
rule, labels = filter.build_filter(
    pop,
    groups={"females": {"sex": "female"}},
    collapse_age=True
)

# rule.shape 会是 (1, 2, n_genotypes)，而不是 (1, 2, 8, n_genotypes)
```

---

## 应用规则

### apply_rule 函数

```python
from natal.observation import apply_rule
import numpy as np

# 从规则 rule 和个体计数数组得到观察值
observed = apply_rule(pop.state.individual_count, rule)

# observed shape: (n_groups, n_sexes, [n_ages]) 或 (n_groups, n_sexes)
```

**工作原理**：
1. 将规则数组（全为 0 或 1）与个体计数相乘
2. 在基因型维度上求和，得到每组的个体总数

**示例**：

```python
# pop.state.individual_count shape: (2, 8, 50)
# rule shape: (3, 2, 8, 50)
# observed shape: (3, 2, 8)

# observed[i, j, k] = 第 i 组、性别 j、年龄 k 的个体总数
```

### 完整工作流

```python
from natal.observation import ObservationFilter, apply_rule

# 1. 创建过滤器
filter = ObservationFilter(pop.registry)

# 2. 定义分组（例如 PMCMC 似然计算）
groups = {
    "female_drive": {"genotype": ["WT|Drive", "Drive|Drive"], "sex": "female"},
    "male_drive": {"genotype": ["WT|Drive", "Drive|Drive"], "sex": "male"},
}

# 3. 构建规则
rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,
    groups=groups
)

# 4. 在每个时间步应用规则
observations = []
for _ in range(100):
    pop.step()
    observed = apply_rule(pop.state.individual_count, rule)
    observations.append(observed)

# 5. 转换为数组用于统计
observations = np.array(observations)  # shape: (100, 2, 2, 8)
```

---

## 实际例子

### 例子 1：基因驱动扩散监测

```python
from natal.observation import ObservationFilter, apply_rule
import numpy as np

pop = AgeStructuredPopulation(
    species=species,
    initial_individual_count={...},
    n_ages=8,
)

# 定义观察目标：监测驱动等位基因的频率
filter = ObservationFilter(pop.registry)

groups = {
    "WT": {"genotype": ["WT|WT"]},
    "heterozygous": {"genotype": ["WT|Drive"]},
    "homozygous_drive": {"genotype": ["Drive|Drive"]},
}

rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,
    groups=groups,
    collapse_age=True  # 不关心年龄分布
)

# 记录时间序列
times = []
drive_freq = []

for t in range(100):
    pop.step()
    observed = apply_rule(pop.state.individual_count, rule)

    # observed shape: (3, 2)
    # 计算驱动等位基因频率
    het_count = observed[1].sum()  # 杂合体
    hom_count = observed[2].sum()  # 纯合体
    total_alleles = 2 * observed[0].sum() + het_count + 2 * hom_count
    drive_alleles = het_count + 2 * hom_count

    drive_freq.append(drive_alleles / max(1, total_alleles))
    times.append(t)

# 可视化
import matplotlib.pyplot as plt
plt.plot(times, drive_freq)
plt.xlabel("Time (generations)")
plt.ylabel("Drive allele frequency")
plt.show()
```

### 例子 2：年龄特异性监测

```python
# 关注不同年龄群体的基因型分布
groups = {
    "juvenile_WT": {"age": [0, 1], "genotype": ["WT|WT"]},
    "juvenile_Drive": {"age": [0, 1], "genotype": ["WT|Drive", "Drive|Drive"]},
    "adult_WT": {"age": [2, 7], "genotype": ["WT|WT"]},
    "adult_Drive": {"age": [2, 7], "genotype": ["WT|Drive", "Drive|Drive"]},
}

rule, labels = filter.build_filter(pop, groups=groups)

# 观察每一步
for t in range(100):
    pop.step()
    observed = apply_rule(pop.state.individual_count, rule)

    # observed shape: (4, 2)
    # observed[0] = juvenile_WT females & males
    # observed[1] = juvenile_Drive females & males
    # ...

    print(f"t={t}: juveniles={observed[:2].sum():.0f}, "
          f"adults={observed[2:].sum():.0f}")
```

### 例子 3：PMCMC 似然计算

```python
from natal.observation import ObservationFilter, apply_rule
from scipy.stats import poisson

# 假设我们有实际数据 observed_data
observed_data = np.array([
    [1000, 800],  # t=0: female, male counts
    [950, 750],
    [900, 700],
    # ...
])

# 设置观察过滤
filter = ObservationFilter(pop.registry)
groups = {
    "females": {"sex": "female"},
    "males": {"sex": "male"},
}
rule, labels = filter.build_filter(pop, groups=groups, collapse_age=True)

# 计算似然
log_likelihood = 0.0
for t, data_t in enumerate(observed_data):
    pop.step()

    # 获取模拟的观察值
    simulated = apply_rule(pop.state.individual_count, rule)
    # simulated shape: (2, 2) - 2 groups, 2 sexes

    # 对每性别求和
    sim_females = simulated[0, 0] + simulated[1, 0]  # females
    sim_males = simulated[0, 1] + simulated[1, 1]    # males

    # Poisson 似然
    log_likelihood += (
        poisson.logpmf(data_t[0], sim_females) +
        poisson.logpmf(data_t[1], sim_males)
    )

print(f"Log-likelihood: {log_likelihood:.2f}")
```

### 例子 4：与 PMCMC 参数管理集成（推荐）

当前 PMCMC 设计中，参数到模型配置的映射由 PMCMC 层统一管理，
而不是在 particle filter 内部执行。推荐通过
`params_to_model_fn` 回调进行更新。

```python
import numpy as np
from samplers.pmcmc import run_pmcmc
from samplers.likelihood import (
    make_init_sampler,
    make_transition_fn,
    make_obs_loglik_fn,
    LogLikelihoodEvaluator,
)
from samplers.parameter_mapping import make_fitness_config_applier

# 已有: config, observations, obs_rule, shapes, state_flat, param_idx, geno_idx
init_sampler = make_init_sampler(state_flat, n_sexes=shapes[0][0], n_ages=shapes[0][1], n_genotypes=shapes[0][2])
transition_fn, transition_args = make_transition_fn(config, shapes, param_idx=param_idx, geno_idx=geno_idx)
obs_loglik_fn = make_obs_loglik_fn(10.0, obs_rule, apply_rule, shapes=shapes)

evaluator = LogLikelihoodEvaluator(
    config=config,
    observations=observations,
    initial_state=state_flat,
    shapes=shapes,
    n_particles=300,
    sigma=10.0,
    obs_rule=obs_rule,
)

# 参数映射函数（可替换为自定义逻辑，修改任意 PopulationConfig 字段）
params_to_model_fn = make_fitness_config_applier(
    config=config,
    param_idx=param_idx,
    geno_idx=geno_idx,
)

result = run_pmcmc(
    observations=observations,
    n_particles=300,
    init_sampler=init_sampler,
    transition_fn=transition_fn,
    obs_loglik_fn=obs_loglik_fn,
    params_init=np.array([0.5, 0.5]),
    n_iter=1000,
    step_sizes=np.array([0.1, 0.1]),
    log_prior_fn=lambda p: 0.0,
    transition_args=transition_args,
    params_to_model_fn=params_to_model_fn,
    loglik_evaluator=evaluator,
)
```

---

## 性能提示

### 1. 重用规则

如果在多次模拟中使用同一分组，提前构建并重用规则：

```python
# ✅ 推荐
rule, labels = filter.build_filter(pop, groups=groups)
for particle in range(100):
    pop.reset()
    for t in range(n_steps):
        pop.step()
        observed = apply_rule(pop.state.individual_count, rule)

# ❌ 低效
for particle in range(100):
    pop.reset()
    rule, _ = filter.build_filter(pop, groups=groups)  # 重复构建
    for t in range(n_steps):
        pop.step()
        observed = apply_rule(pop.state.individual_count, rule)
```

### 2. 矢量化 apply_rule

`apply_rule` 支持批量应用：

```python
# 如果有多个规则
rules = [rule1, rule2, rule3]
results = [apply_rule(pop.state.individual_count, r) for r in rules]
```

### 3. 尽早合并年龄

如果不需要年龄分辨率，使用 `collapse_age=True` 减少维度：

```python
# 更快
rule, _ = filter.build_filter(pop, groups=groups, collapse_age=True)

# 较慢
rule, _ = filter.build_filter(pop, groups=groups, collapse_age=False)
```

---

## 常见错误

### 错误 1：diploid_genotypes 为 None

```python
# ❌ 错误
rule, labels = filter.build_filter(pop, groups=groups)

# ✅ 正确
rule, labels = filter.build_filter(
    pop,
    diploid_genotypes=pop.species,  # 必须提供
    groups=groups
)
```

### 错误 2：基因型字符串不匹配

```python
# 假设 pop 的基因型是通过 to_string() 生成的
gt = pop.species.get_all_genotypes()[0]
print(gt.to_string())  # 输出: "WT|WT"

# ❌ 不会匹配（拼写错误）
groups = {"target": {"genotype": ["wt|wt"]}}

# ✅ 正确（区分大小写）
groups = {"target": {"genotype": ["WT|WT"]}}
```

### 错误 3：年龄范围超出

```python
# 如果 pop.n_ages = 8（年龄 0-7）

# ❌ 错误（年龄 8 不存在）
groups = {"old": {"age": [6, 8]}}

# ✅ 正确
groups = {"old": {"age": [6, 7]}}
```

---

## 下一步

- [API 入口](api/genetic_structures.md) - 查看完整的方法签名
- [Hook 系统](09_hooks.md) - 在 Hook 中应用观察过滤
- [Numba 优化](07_numba_optimization.md) - 性能调优

---

**返回到目录**: [完整文档索引](index.md)
