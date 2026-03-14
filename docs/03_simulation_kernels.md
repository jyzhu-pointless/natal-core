# Simulation Kernels 深度解析

本章逐步讲解 NATAL 的核心数值计算引擎 `simulation_kernels` 模块。这是高性能模拟的秘密所在。

## 核心设计哲学

Simulation Kernels 遵循 **纯函数化** 设计：

```
AgeStructuredPopulation (高层对象)
    ↓ [导出]
(PopulationState, PopulationConfig, history)
    ↓ [传入]
simulation_kernels.run_tick() / run() (纯函数)
    ↓ [计算]
(new_state, result_code) (不修改输入)
    ↓ [导入]
AgeStructuredPopulation 更新状态
```

**优势**：
- 完全无副作用，易于测试和调试
- 可以导出状态进行批量 Monte Carlo 模拟
- 所有函数都可被 Numba JIT 编译

## 数据结构

### PopulationState：运行时状态

包含种群的所有 **动态** 数据：

```python
@jitclass
class PopulationState:
    # 当前时间步
    n_tick: int
    
    # 个体计数：(n_sexes, n_ages, n_genotypes)
    # 例：individual_count[1, 3, 5] = 150
    #     表示雌性（sex=1），年龄3，基因型5 有150只个体
    individual_count: np.ndarray[np.float64]
    
    # 精子存储：(n_ages, n_female_genotypes, n_male_genotypes)
    # 例：sperm_storage[3, 5, 8] = 200.5
    #     表示年龄3的雌性（基因型5）存储了来自雄性（基因型8）的200.5份精子
    sperm_storage: np.ndarray[np.float64]
```

### PopulationConfig：静态配置

包含所有 **静态** 参数和映射矩阵。这是初始化时"编译"的结果。

```python
@jitclass
class PopulationConfig:
    # 维度
    n_sexes: int
    n_ages: int
    n_genotypes: int
    n_haploid_genotypes: int
    n_glabs: int  # gamete labels 的数量
    
    # 生命史参数
    adult_ages: np.ndarray[np.int32]  # 成年年龄列表
    female_survival_rates: np.ndarray[np.float64]
    male_survival_rates: np.ndarray[np.float64]
    female_mating_rates: np.ndarray[np.float64]
    male_mating_rates: np.ndarray[np.float64]
    
    # === 关键映射矩阵 ===
    # 基因型→配子（包括 gamete label）
    # shape: (n_sexes, n_genotypes, n_haploid_genotypes, n_glabs)
    # genotype_to_gametes_map[sex, gt_idx, haploid_idx, label_idx]
    #   = 基因型 gt_idx 产生配子 (haploid_idx, label_idx) 的概率
    genotype_to_gametes_map: np.ndarray[np.float64]
    
    # 配子→合子
    # shape: (n_haploid_genotypes*n_glabs, n_haploid_genotypes*n_glabs, n_genotypes)
    # gametes_to_zygote_map[mat_glab_idx, pat_glab_idx, result_gt_idx]
    #   = 雌配子 mat_glab_idx 和雄配子 pat_glab_idx 产生基因型 result_gt_idx 的概率
    gametes_to_zygote_map: np.ndarray[np.float64]
    
    # 适应度矩阵
    viability_fitness: np.ndarray[np.float64]  # shape: (n_sexes, n_ages, n_genotypes)
    fecundity_fitness: np.ndarray[np.float64]  # shape: (n_genotypes,) 仅对雌性
    sexual_selection_fitness: np.ndarray[np.float64]  # shape: (n_genotypes, n_genotypes)
    
    # 其他参数
    expected_eggs_per_female: float
    sperm_displacement_rate: float
    use_sperm_storage: bool
    is_stochastic: bool
    # ... 等等
```

## 算法流程：run_tick

一个完整的 tick（时间步）包含四个阶段：

```
tick t:  [current state]
    ↓ step 1: 繁殖 (run_reproduction)
    ↓ step 2: 生存 (run_survival)
    ↓ step 3: 衰老 (run_aging)
    ↓
tick t+1: [new state]
```

### 概览代码

```python
@njit
def run_tick(
    state: PopulationState,
    config: PopulationConfig,
    reproduction_hook: Callable,  # Hook 函数
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
) -> Tuple[Tuple[NDArray, NDArray, int], int]:
    """执行一个 tick，支持 Hook 回调"""
    
    ind_count = state.individual_count.copy()
    sperm_store = state.sperm_storage.copy()
    tick = state.n_tick
    
    # ===== REPRODUCTION =====
    ind_count, sperm_store = run_reproduction(ind_count, sperm_store, config)
    
    # Hook：reproduction 后可以修改状态
    result = reproduction_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== EARLY =====
    result = early_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== SURVIVAL =====
    ind_count = run_survival(ind_count, config)
    
    result = survival_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    # ===== AGING =====
    ind_count, sperm_store = run_aging(ind_count, sperm_store, config)
    
    # ===== LATE =====
    result = late_hook(ind_count, tick)
    if result != RESULT_CONTINUE:
        return (ind_count, sperm_store, tick), RESULT_STOP
    
    return (ind_count, sperm_store, tick + 1), RESULT_CONTINUE
```

### 第一步：繁殖 (run_reproduction)

**目标**：根据交配产生后代，更新年龄 0 个体和精子存储。

```python
@njit
def run_reproduction(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Args:
        ind_count: (n_sexes, n_ages, n_genotypes)
        sperm_store: (n_ages, n_female_genotypes, n_male_genotypes)
        config: PopulationConfig
        
    Returns:
        (new_ind_count, new_sperm_store) with age-0 updated
    """
```

#### 算法细节

1️⃣ **收集成年个体**

```python
# 只有在 adult_ages 中的年龄才能交配
# 例：adult_ages = [2, 3, 4, 5]
adult_females = {}  # {genotype_idx: count}
adult_males = {}

for age in config.adult_ages:
    for gt_idx in range(n_genotypes):
        adult_females[gt_idx] += ind_count[FEMALE, age, gt_idx]
        adult_males[gt_idx] += ind_count[MALE, age, gt_idx]
```

2️⃣ **计算交配产生的配子**

```python
# 对每个雌性基因型，计算其产生的配子
for female_gt in adult_females:
    # 该基因型的配子产生概率
    # genotype_to_gametes_map[FEMALE, female_gt, haploid_idx, label_idx]
    gamete_probs = config.genotype_to_gametes_map[FEMALE, female_gt]
    # shape: (n_haploid_genotypes, n_glabs)
    
    # 与所有雄性配子进行受精（可能考虑精子存储）
    for male_gt in adult_males:
        # ...复杂的多项分布计算...
```

3️⃣ **精子替换**

```python
# 新精子存储 = 旧精子 * (1 - displacement) + 新精子 * displacement
new_sperm_store = (
    sperm_store * (1 - displacement_rate) +
    new_sperm * displacement_rate
)
```

4️⃣ **年龄 0 个体更新**

```python
# 清空年龄 0（来自上一 tick 的婴儿已长大）
ind_count[FEMALE, 0, :] = newborn_females
ind_count[MALE, 0, :] = newborn_males
```

**复杂性**：这一步涉及：
- 多项分布采样（随机性）
- 精子存储机制（非标准遗传学）
- 配子 Modifier（修改配子比例）
- 合子 Modifier（修改受精结果）

### 第二步：生存 (run_survival)

**目标**：根据生存率过滤个体，应用存活力适应度。

```python
@njit
def run_survival(
    ind_count: NDArray[np.float64],
    config: PopulationConfig,
) -> NDArray[np.float64]:
    """
    应用生存率和生存适应度。
    
    新计数 = 旧计数 × 生存率[年龄] × 适应度[性别, 年龄, 基因型]
    """
    
    for sex in [FEMALE, MALE]:
        for age in range(n_ages):
            for gt in range(n_genotypes):
                # 获取生存参数
                sr = survival_rates[sex][age]  # 基于年龄的基础生存率
                
                # 获取适应度
                fitness = viability_fitness[sex, age, gt]
                
                # 应用（在确定性模式下）
                # 在随机模式下，使用二项分布采样
                if is_stochastic:
                    count = binom(ind_count[sex, age, gt], sr * fitness)
                else:
                    count = ind_count[sex, age, gt] * sr * fitness
                
                ind_count[sex, age, gt] = count
    
    return ind_count
```

**关键点**：
- 生存既取决于年龄，也取决于基因型
- 适应度在这里应用（而非繁殖或衰老）
- 随机性来自二项分布

### 第三步：衰老 (run_aging)

**目标**：个体年龄增长一岁，超龄的死亡。

```python
@njit
def run_aging(
    ind_count: NDArray[np.float64],
    sperm_store: NDArray[np.float64],
    config: PopulationConfig,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    从年龄 age → age+1
    年龄最大值之后的个体被移除。
    """
    
    # 创建新的数组（年龄向前滑动）
    new_ind = np.zeros_like(ind_count)
    new_sperm = np.zeros_like(sperm_store)
    
    # 每个年龄向前推进一岁
    for age in range(n_ages - 1):
        new_ind[:, age + 1, :] = ind_count[:, age, :]
        new_sperm[age + 1, :, :] = sperm_store[age, :, :]
    
    # 年龄 0 在后续 tick 的 reproduction 中会被重新填充
    # 年龄 >= n_ages 的个体死亡（自动被丢弃）
    
    return new_ind, new_sperm
```

**简单性**：这是三个阶段中最直接的。

## 完整 tick 的流程示例

以一个简单的 2 个基因型、2 个年龄的例子说明：

```
初始状态 (tick 0):
  ind_count[FEMALE] = [[0, 0],     # age 0: 无新生儿
                       [100, 50]]   # age 1: 100 个 WT|WT，50 个 WT|Drive
  ind_count[MALE] = [[0, 0],
                     [50, 50]]
  sperm_store = [[..., ...],       # age 0 的精子（少）
                 [..., ...]]        # age 1 的精子（多）

========== REPRODUCTION ==========
成年个体（age 1）进行交配，产生新生儿
新的 ind_count[FEMALE, 0] = [150, 80]  （根据交配和适应度计算）
新的 ind_count[MALE, 0] = [140, 85]

========== SURVIVAL ==========
应用生存率和适应度
假设年龄1的生存率为 0.9，无额外适应度成本：
新的 ind_count[FEMALE, 1] = [100 * 0.9, 50 * 0.9] = [90, 45]
新的 ind_count[MALE, 1] = [50 * 0.9, 50 * 0.9] = [45, 45]

========== AGING ==========
年龄向前推进
年龄0的新生儿（150, 80）仍留在年龄0
年龄1的成年个体（90, 45）向前推进到年龄2（但由于只有2个年龄，死亡）
年龄2及以上的个体死亡

final state (tick 1):
  ind_count[FEMALE] = [[150, 80],  # age 0: 从繁殖得到
                       [90, 45]]    # age 1: 从上一 tick age 0 推进
  ind_count[MALE] = [[140, 85],
                     [45, 45]]
```

## 高阶函数：运行多个 ticks

### run() 函数

运行 N 个 ticks，支持 Hook 和历史记录。

```python
@njit
def run(
    state: PopulationState,
    config: PopulationConfig,
    n_ticks: int,
    reproduction_hook: Callable,
    early_hook: Callable,
    survival_hook: Callable,
    late_hook: Callable,
    record_history: bool = False
) -> Tuple[
    Tuple[NDArray, NDArray, int],  # final state
    Optional[NDArray],              # history
    bool                            # was_stopped
]:
    """运行多个 ticks"""
    
    current_state = (state.individual_count, state.sperm_storage, state.n_tick)
    history = None
    
    if record_history:
        # 预分配历史数组
        history_size = 1 + ind_count.size + sperm.size
        history = np.zeros((n_ticks + 1, history_size), dtype=np.float64)
        history_count = 0
    
    was_stopped = False
    
    for tick_i in range(n_ticks):
        # 执行一个 tick
        current_state, result_code = run_tick(
            state, config,
            reproduction_hook, early_hook, survival_hook, late_hook
        )
        
        if record_history:
            # 展平并保存状态
            ind_count, sperm, tick = current_state
            flattened = np.concatenate((
                np.array([float(tick)]),
                ind_count.flatten(),
                sperm.flatten()
            ))
            history[history_count, :] = flattened
            history_count += 1
        
        if result_code == RESULT_STOP:
            was_stopped = True
            break
    
    return current_state, history[:history_count], was_stopped
```

### 历史记录的存储格式

```python
# history 是一个二维数组：(n_records, history_size)
# 其中 history_size = 1 + n_sexes*n_ages*n_genotypes + n_ages*n_genotypes*n_genotypes
#                   = 1 + ind_count.size + sperm.size

# 每一行的格式：
# [tick, ind_count_flattened..., sperm_flattened...]

# 恢复状态：
tick = history[i, 0]
ind_count = history[i, 1:1+ind_size].reshape((n_sexes, n_ages, n_genotypes))
sperm = history[i, 1+ind_size:].reshape((n_ages, n_genotypes, n_genotypes))
```

## 导出/导入机制

### export_state()

从 `AgeStructuredPopulation` 对象导出状态：

```python
def export_state(pop: AgeStructuredPopulation):
    """导出状态供 simulation_kernels 使用"""
    
    state = pop._state  # PopulationState 对象
    config = pop._config  # PopulationConfig 对象
    history = pop.get_history()  # numpy 数组
    
    return state, config, history
```

### import_state()

将状态导入回 `AgeStructuredPopulation` 对象：

```python
def import_state(pop: AgeStructuredPopulation, state: PopulationState, 
                 history: Optional[NDArray] = None):
    """导入状态"""
    
    pop._state = state
    if history is not None:
        pop._history_array = history
    
    # 更新 tick 计数器
    pop._tick = state.n_tick
```

## 批量 Monte Carlo 模拟

对于大规模随机模拟，可以使用 `batch_ticks()` 进行并行采样：

```python
def batch_ticks(
    initial_state: PopulationState,
    config: PopulationConfig,
    n_particles: int,
    n_steps_per_particle: int,
    rng: np.random.Generator,
    record_history: bool = False
) -> List[PopulationState]:
    """
    运行 n_particles 个独立的模拟轨迹。
    
    Args:
        initial_state: 共享的初始状态
        config: 共享的配置
        n_particles: 模拟数量
        n_steps_per_particle: 每个模拟的步数
        rng: 随机数生成器
        record_history: 是否记录历史
        
    Returns:
        List[PopulationState]: n_particles 个最终状态
    """
    
    results = []
    for i in range(n_particles):
        # 复制初始状态（重要！不能共享）
        state_copy = initial_state.copy()
        
        # 运行模拟
        final_state, _ = run(
            state_copy, config,
            n_steps_per_particle,
            hooks...,
            record_history=record_history
        )
        
        results.append(final_state)
    
    return results
```

### 使用示例

```python
from natal.simulation_kernels import export_state, batch_ticks

pop = AgeStructuredPopulation(...)
state, config, _ = export_state(pop)

# 运行 1000 个独立的 Monte Carlo 轨迹
particles = batch_ticks(
    state, config,
    n_particles=1000,
    n_steps_per_particle=100,
    rng=np.random.default_rng(seed=42),
    record_history=False
)

# 分析结果
for i, final_state in enumerate(particles):
    final_pop_size = final_state.individual_count.sum()
    print(f"Particle {i}: {final_pop_size:.0f}")
```

## 性能优化

### Numba JIT 编译

所有关键函数都用 `@njit` 标注：

```python
from natal.numba_utils import njit_switch

@njit_switch(cache=True)
def run_reproduction(ind_count, sperm_store, config):
    # Numba 会编译这个函数为机器码
    # 首次调用时：编译耗时 1-3 秒
    # 后续调用：直接执行，快 100-1000 倍
```

**编译缓存**：

```
First run (no cache):  2-5 seconds (包括编译)
Subsequent runs:       50-100 ms (直接执行)
```

### 内存布局

避免频繁的内存分配：

```python
# 高效：原地操作
ind_count *= survival_rates[:, np.newaxis, np.newaxis]

# 低效：创建临时数组
ind_count = ind_count * survival_rates[:, np.newaxis, np.newaxis]
```

## Hook 集成

Hook 在 tick 的四个关键点被调用：

```python
def run_tick(..., reproduction_hook, early_hook, survival_hook, late_hook):
    
    # Step 1
    ind_count, sperm_store = run_reproduction(ind_count, sperm_store, config)
    
    # Hook 1: reproduction
    result = reproduction_hook(ind_count, tick)
    if result == RESULT_STOP:
        return ..., RESULT_STOP
    
    # Hook 2: early
    result = early_hook(ind_count, tick)
    if result == RESULT_STOP:
        return ..., RESULT_STOP
    
    # Step 2
    ind_count = run_survival(ind_count, config)
    
    # Hook 3: survival
    result = survival_hook(ind_count, tick)
    if result == RESULT_STOP:
        return ..., RESULT_STOP
    
    # Step 3
    ind_count, sperm_store = run_aging(ind_count, sperm_store, config)
    
    # Hook 4: late
    result = late_hook(ind_count, tick)
    if result == RESULT_STOP:
        return ..., RESULT_STOP
    
    return ..., RESULT_CONTINUE
```

Hook 的签名：

```python
# Numba 兼容的 Hook
@njit
def my_hook(ind_count: NDArray[np.float64], tick: int) -> int:
    """
    Args:
        ind_count: (n_sexes, n_ages, n_genotypes) 当前个体计数
        tick: 当前时间步
        
    Returns:
        int: 0 (继续) 或 1 (停止)
    """
    # 读取或修改 ind_count
    if tick == 50:
        ind_count[1, 2, 0] = 0  # 杀死所有女性年龄2基因型0
    
    return 0  # 继续
```

> 详细的 Hook 写法见 [Hook DSL 系统](07_hooks_dsl.md)

## 完整工作流程示例

```python
from natal.nonWF_population import AgeStructuredPopulation
from natal.simulation_kernels import export_state, run, import_state
import numpy as np

# 1. 创建和初始化种群
pop = AgeStructuredPopulation(...)

# 2. 导出状态和配置
state, config, history = export_state(pop)

# 3. 定义 Hook（Numba 兼容）
from numba import njit

@njit
def my_hook(ind_count, tick):
    if tick % 10 == 0:
        print(f"Tick {tick}, total: {ind_count.sum()}")
    return 0

# 4. 运行模拟（纯函数）
final_state, history_array, was_stopped = run(
    state, config,
    n_ticks=100,
    reproduction_hook=my_hook,
    early_hook=my_hook,
    survival_hook=my_hook,
    late_hook=my_hook,
    record_history=True
)

# 5. 导入回种群对象
import_state(pop, final_state, history_array)

# 6. 查看结果
print(f"Final population size: {pop.get_total_count()}")
print(f"History records: {len(pop.history)}")
```

---

## 🎯 本章总结

| 概念 | 说明 | 作用 |
|------|------|------|
| **PopulationState** | 动态状态（个体计数、精子存储） | 输入/输出数据结构 |
| **PopulationConfig** | 静态配置（映射矩阵、参数） | 编译一次后的配置 |
| **run_reproduction** | 交配产生后代 | 种群增长的主要贡献者 |
| **run_survival** | 应用生存率和适应度 | 种群管理的选择压力 |
| **run_aging** | 年龄增长和死亡 | 人口结构演变 |
| **run_tick** | 完整的一个时间步 | 核心模拟单元 |
| **run** | 多个 ticks + 历史 | 完整模拟循环 |
| **batch_ticks** | 并行 Monte Carlo | 大规模随机模拟 |

**关键洞察**：
1. 三个阶段（繁殖、生存、衰老）的分离使得各阶段可以独立优化
2. 纯函数设计允许导出/导入和批量模拟
3. Numba 加速使得 Python 代码有 C 语言的性能
4. Hook 机制在不影响性能的前提下提供灵活性

---

## 📚 相关章节

- [PopulationState & PopulationConfig](04_population_state_config.md) - 深入数据结构
- [Hook DSL 系统](07_hooks_dsl.md) - Hook 的高级写法
- [Numba 优化指南](08_numba_optimization.md) - 性能优化技巧
- [IndexCore 索引机制](05_index_core.md) - 对象索引在计算中的应用

---

**准备了解 "编译" 步骤吗？** [前往下一章：PopulationState & PopulationConfig →](04_population_state_config.md)
