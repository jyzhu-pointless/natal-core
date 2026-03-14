# PopulationState & PopulationConfig：编译与配置

本章深入讲解 NATAL 的"编译"步骤和两个核心数据结构：`PopulationState` 和 `PopulationConfig`。

## 高层视角：编译步骤

```
AgeStructuredPopulation(...) 构造函数
    ↓
1️⃣ 解析参数（生存率、交配率等）
    ↓
2️⃣ 构建映射矩阵（基因型→配子、配子→合子）
    ↓
3️⃣ 编译成 PopulationConfig（Numba 兼容）
    ↓
4️⃣ 初始化 PopulationState（numpy 数组）
    ↓
✅ 种群就绪，可运行模拟
```

**目的**：将高层的、面向对象的用户输入转换为底层的、数值化的计算配置。

## PopulationState：动态状态

这个数据结构存储种群的 **动态** 数据，在每个 tick 都会改变。

### 数据结构定义

```python
@jitclass  # Numba 兼容
class PopulationState:
    # 当前时间步
    n_tick: int
    
    # 核心状态：个体计数
    # shape: (n_sexes, n_ages, n_genotypes)
    # individual_count[sex, age, genotype_idx] = count
    individual_count: np.ndarray[np.float64]
    
    # 精子存储（用于模拟精子存储物种）
    # shape: (n_ages, n_female_genotypes, n_male_genotypes)
    # sperm_storage[age, female_gt_idx, male_gt_idx] = count
    sperm_storage: np.ndarray[np.float64]
```

### 理解三个维度

#### 维度 1：性别 (n_sexes = 2)

```python
state.individual_count[0, :, :] = 雄性个体
state.individual_count[1, :, :] = 雌性个体

# 在 natal/type_def.py 中定义
from natal.type_def import Sex
Sex.MALE = 0
Sex.FEMALE = 1
```

#### 维度 2：年龄 (n_ages)

```python
# 例：8 个年龄类别 (0-7)
state.individual_count[:, 0, :] = 婴儿（刚出生）
state.individual_count[:, 1, :] = 少年
state.individual_count[:, 2, :] = 成年（可能开始交配）
state.individual_count[:, 3, :] = 成年
...
state.individual_count[:, 7, :] = 老年（下一 tick 死亡）
```

#### 维度 3：基因型 (n_genotypes)

```python
# 整数索引（由 IndexCore 分配）
# 例：3 个等位基因 (A1, A2, A3) → 6 个基因型
state.individual_count[:, :, 0] = A1|A1
state.individual_count[:, :, 1] = A1|A2
state.individual_count[:, :, 2] = A1|A3
state.individual_count[:, :, 3] = A2|A2
state.individual_count[:, :, 4] = A2|A3
state.individual_count[:, :, 5] = A3|A3
```

### 精子存储的特殊性

精子存储是一个 3D 数组，反映了昆虫中的精子库现象：

```python
state.sperm_storage[age, female_gt, male_gt] = 精子数量

# 解释：
# - 年龄 `age` 的雌性
# - 基因型 `female_gt` 的
# - 存储了来自基因型 `male_gt` 的雄性的精子
```

### 基本操作

```python
# 创建状态
state = PopulationState(
    n_genotypes=6,
    n_sexes=2,
    n_ages=8
)

# 访问个体计数
total = state.individual_count.sum()
females = state.individual_count[1, :, :].sum()
young_females = state.individual_count[1, 0:3, :].sum()

# 修改个体计数
state.individual_count[1, 3, 5] = 150  # 雌性，年龄3，基因型5

# 复制状态（深拷贝）
state_copy = state.copy()

# 获取当前时间步
tick = state.n_tick
```

## PopulationConfig：静态配置

这个数据结构存储所有 **静态** 参数和映射矩阵。初始化一次后，整个模拟期间不再改变。

### 核心组件

#### 1. 维度信息

```python
config.n_sexes = 2
config.n_ages = 8
config.n_genotypes = 6  # 所有基因型总数
config.n_haploid_genotypes = ?  # 单倍体基因型数
config.n_glabs = 2  # gamete labels 数量（如 ["default", "Cas9_deposited"]）
```

#### 2. 生命史参数

```python
# 生存率：按年龄
config.female_survival_rates = np.array([1.0, 1.0, 5/6, 4/5, ...])
config.male_survival_rates = np.array([1.0, 1.0, 2/3, 1/2, ...])

# 交配率：按年龄和性别
config.female_mating_rates = np.array([0.0, 0.0, 1.0, 1.0, ...])
config.male_mating_rates = np.array([0.0, 0.0, 1.0, 1.0, ...])

# 或用 adult_ages 列表
config.adult_ages = np.array([2, 3, 4, 5])  # 成年年龄
```

#### 3. 关键映射矩阵 ⭐

这些是编译时生成的最重要的对象：

##### 3.1 基因型→配子映射

```python
# shape: (n_sexes, n_genotypes, n_haploid_genotypes, n_glabs)
config.genotype_to_gametes_map

# 解释：
# config.genotype_to_gametes_map[sex, gt_idx, haploid_idx, label_idx]
# = 性别为 sex、基因型为 gt_idx 的个体
#   产生"(haploid_idx, label_idx)配子"的概率

# 例：
# 假设有 2 个等位基因 (A1, A2)，2 个 labels (default, Cas9)
# 基因型 A1|A2 (idx=1) 的雌性产生：
config.genotype_to_gametes_map[FEMALE, 1, :, :] =
    [[0.5, 0.0],      # A1-default: 50%, A1-Cas9: 0%
     [0.5, 0.0]]      # A2-default: 50%, A2-Cas9: 0%
# 总概率为 1.0（每个配子要么是 default，要么是 Cas9）
```

**生成方式**：

```python
# 对于每个基因型和性别：
# 1. 根据孟德尔遗传规律，确定产生每个配子的基础概率
# 2. 应用 gamete_modifiers，修改配子比例（如 gene drive）
# 3. 归一化为概率
```

##### 3.2 配子→合子映射

```python
# shape: (n_haploid_with_labels, n_haploid_with_labels, n_genotypes)
# 其中 n_haploid_with_labels = n_haploid_genotypes * n_glabs
config.gametes_to_zygote_map

# 解释：
# config.gametes_to_zygote_map[mat_idx, pat_idx, result_gt_idx]
# = 雌配子 mat_idx 和雄配子 pat_idx 结合
#   产生基因型 result_gt_idx 的概率

# 例：
# 两个配子结合，产生后代基因型
# (A1-default, A2-default) → A1|A2 (idx=1)
config.gametes_to_zygote_map[mat_a1_default, pat_a2_default, 1] = 1.0
# (A1-default, A2-Cas9) → A1|A2 (idx=1) 且标记为 Cas9
# 这可能通过 zygote_modifiers 修改（如胚胎拯救）
```

**生成方式**：

```python
# 对于每个雌配子和雄配子对：
# 1. 根据孟德尔遗传，确定产生的基因型和 label 组合
# 2. 应用 zygote_modifiers，修改结果概率（如细胞质不兼容）
# 3. 归一化
```

#### 4. 适应度矩阵

```python
# 存活力适应度（按性别、年龄、基因型）
# shape: (n_sexes, n_ages, n_genotypes)
config.viability_fitness[sex, age, gt] = [0.0, 1.0]

# 生育力适应度（仅对雌性）
# shape: (n_genotypes,)
config.fecundity_fitness[gt] = [0.0, 1.0]

# 性选择适应度（雌性偏好）
# shape: (n_genotypes, n_genotypes)  # (female_gt, male_gt)
config.sexual_selection_fitness[female_gt, male_gt] = [0.0, 2.0]
```

#### 5. 其他参数

```python
config.expected_eggs_per_female = 100.0      # 每只雌性产卵数
config.sperm_displacement_rate = 0.05    # 精子库替换率
config.use_sperm_storage = True          # 是否启用精子存储
config.is_stochastic = False             # 是否为随机模型
```

## 编译过程详解

### 步骤 1：参数解析

```python
# 用户输入（高层）
initial_individual_count = {
    "female": {
        "WT|WT": [600, 500, 400, ...],
        "WT|Drive": [100, 80, 60, ...],
    },
    ...
}

# 内部处理
# 1. 遍历所有基因型字符串
# 2. 通过 species.get_genotype_from_str() 转换为 Genotype 对象
# 3. 通过 IndexCore 获取整数索引
# 4. 创建 numpy 数组

individual_count = np.zeros((n_sexes, n_ages, n_genotypes))
individual_count[FEMALE, :, idx_wt_wt] = [600, 500, 400, ...]
individual_count[FEMALE, :, idx_wt_drive] = [100, 80, 60, ...]
...
```

### 步骤 2：映射矩阵生成

#### 2.1 基因型→配子映射

```python
# 对于每个（性别，基因型）对：

for sex in [MALE, FEMALE]:
    for gt_idx, genotype in enumerate(all_genotypes):
        # 1. 获取该基因型产生的配子（孟德尔遗传）
        # 例：WT|Drive 的雌性产生 50% WT + 50% Drive
        gametes_freq = mendelian_inheritance(genotype)
        
        # 2. 每个配子与两个 label 组合
        # 例：WT-default, WT-Cas9, Drive-default, Drive-Cas9
        for gamete_idx, freq in enumerate(gametes_freq):
            for label_idx in range(n_glabs):
                # 初始化为标准孟德尔比例
                config.genotype_to_gametes_map[sex, gt_idx, gamete_idx, label_idx] = ...
        
        # 3. 应用 gamete_modifiers
        # 例：gene drive 修饰器改变配子比例
        for modifier in gamete_modifiers:
            modified = modifier(population)
            # modified 是一个字典：{基因型: {(配子, label): 频率}}
            if genotype in modified:
                # 覆盖原有的配子频率
                config.genotype_to_gametes_map[sex, gt_idx, :, :] = modified[genotype]
        
        # 4. 归一化为概率
        total = config.genotype_to_gametes_map[sex, gt_idx, :, :].sum()
        config.genotype_to_gametes_map[sex, gt_idx, :, :] /= total
```

#### 2.2 配子→合子映射

```python
# 对于每对（雌配子，雄配子）：

for mat_haploid_idx in range(n_haploid_genotypes):
    for pat_haploid_idx in range(n_haploid_genotypes):
        for mat_label_idx in range(n_glabs):
            for pat_label_idx in range(n_glabs):
                # 1. 孟德尔遗传：两个单倍体基因组结合产生的基因型
                result_genotype = maternal + paternal
                result_gt_idx = index_core.genotype_index(result_genotype)
                
                # 2. Label 处理（取决于性别和具体规则）
                # 例：后代继承母亲的 cytoplasm label
                result_label = mat_label_idx
                
                # 3. 初始化概率
                config.gametes_to_zygote_map[...] = 1.0
                
                # 4. 应用 zygote_modifiers
                # 例：CI（细胞质不兼容）导致部分后代死亡
                for modifier in zygote_modifiers:
                    modified = modifier(population)
                    # 修改概率
                
                # 5. 归一化
```

### 步骤 3：适应度矩阵

```python
# 从 fitness map 构建矩阵
fitness_map = {
    genotype_obj: fitness_value,
    ...
}

for gt_idx, genotype in enumerate(all_genotypes):
    if genotype in fitness_map:
        config.viability_fitness[sex, age, gt_idx] = fitness_map[genotype]
    else:
        config.viability_fitness[sex, age, gt_idx] = 1.0  # 默认值
```

## 示例：完整的编译过程

```python
from natal.genetic_structures import Species
from natal.nonWF_population import AgeStructuredPopulation

# === 定义遗传架构 ===
sp = Species.from_dict(
    name="Test",
    structure={"chr1": {"A": ["A1", "A2"]}}
)

# === 定义初始分布 ===
initial_dist = {
    "female": {"A1|A1": [100, 100], "A1|A2": [50, 50]},
    "male": {"A1|A1": [50, 50]},
}

# === 创建种群（开始编译） ===
pop = AgeStructuredPopulation(
    species=sp,
    n_ages=2,
    is_stochastic=False,
    initial_individual_count=initial_dist,
    female_survival_rates=[1.0, 0.8],
    male_survival_rates=[1.0, 0.6],
    expected_eggs_per_female=10,
)

# === 编译过程输出 ===
print(f"n_genotypes: {pop._config.n_genotypes}")          # 3
print(f"n_haploid_genotypes: {pop._config.n_haploid_genotypes}")  # 2
print(f"genotype_to_gametes shape: {pop._config.genotype_to_gametes_map.shape}")
# (2, 3, 2, 1) = (sex, genotype, haploid, label)
print(f"gametes_to_zygote shape: {pop._config.gametes_to_zygote_map.shape}")
# (2, 2, 3) = (mat_haploid*label, pat_haploid*label, result_genotype)

# === 查看初始状态 ===
state = pop._state
print(f"个体计数 shape: {state.individual_count.shape}")  # (2, 2, 3)
print(f"总个体数: {state.individual_count.sum()}")  # 400
```

## 修改配置

初始化后，可以动态修改某些参数（注意：修改映射矩阵需要重新初始化）：

```python
# 修改适应度（安全）
config = pop._config
wt_wt = sp.get_genotype_from_str("A1|A1")
wt_wt_idx = pop.registry.genotype_index(wt_wt)
config.viability_fitness[FEMALE, 1, wt_wt_idx] = 0.5  # 修改 age=1 的生存率

# 修改配子比例（需要重新生成映射矩阵）
# 使用 set_gamete_modifier() 而非直接修改 config
pop.set_gamete_modifier(my_modifier_func, hook_name="mymod")

# 修改 offspring 数量（安全）
config.expected_eggs_per_female = 150.0
```

## 性能影响

### 初始化耗时

```
n_genotypes = 6     → 100-200 ms
n_genotypes = 30    → 500-1000 ms
n_genotypes = 100   → 5-10 s
```

**原因**：生成映射矩阵的复杂度约为 O(n_genotypes³)。

### 运行效率

一旦编译完成，每个 tick 的耗时与 `n_genotypes` 成线性关系：

```
n_genotypes = 6     → 1-5 ms/tick
n_genotypes = 100   → 10-50 ms/tick
n_genotypes = 1000  → 100-500 ms/tick
```

## 总结：编译的意义

```
用户友好的高层 API         PopulationConfig
  ↓                         ↓
字符串和对象            数值化的矩阵和数组
  ↓                         ↓
灵活的建模能力           Numba 加速的高性能计算
  ↓                         ↓
即时修改 Modifier        一次编译，多次运行
```

编译步骤将"建模灵活性"与"计算性能"有机结合：

- **用户看到**：高层、直观的 Python 对象
- **计算看到**：低层、数值化的 numpy/Numba 数组

---

## 📚 相关章节

- [遗传结构与实体](02_genetic_structures.md) - Genotype 对象的详细说明
- [Simulation Kernels 深度解析](03_simulation_kernels.md) - 配置在计算中的使用
- [IndexCore 索引机制](05_index_core.md) - 对象→索引的映射机制
- [Modifier 机制](06_modifiers.md) - 如何修改映射矩阵

---

**准备理解索引机制了吗？** [前往下一章：IndexCore 索引机制 →](05_index_core.md)
