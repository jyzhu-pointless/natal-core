# 快速开始：15 分钟上手 NATAL

本章基于 `mosquito_population.py` 示例，带你快速上手 NATAL 框架。

## 前置要求

- Python >= 3.9
- NumPy >= 1.21
- Numba >= 0.55

```bash
pip install numpy numba
```

## 1️⃣ 第一步：定义遗传架构（2 分钟）

NATAL 采用 **声明式** 的方式定义遗传架构。使用 `Species.from_dict()` 方法：

```python
from natal.genetic_structures import Species

# 方式1：使用 from_dict（推荐快速定义）
sp = Species.from_dict(
    name="AnophelesGambiae",
    structure={
        "chr1": {
            "A": ["WT", "Drive", "Resistance"]
        }
    }
)
# chr1 是染色体名称
# A 是位点名称
# ["WT", "Drive", "Resistance"] 是等位基因列表
```

### 理解架构中的关键概念

- **Species**: 物种，遗传架构的根
- **Chromosome**: 染色体（如 "chr1"），包含多个位点
- **Locus**: 基因位点（如 "A"），包含多个等位基因
- **Allele**: 等位基因（如 "WT", "Drive"）

### 验证架构

```python
# 查看所有可能的基因型
all_genotypes = sp.get_all_genotypes()
print(f"总共有 {len(all_genotypes)} 种基因型")
# 输出: 总共有 6 种基因型
# (WT|WT, WT|Drive, WT|Resistance, Drive|Drive, Drive|Resistance, Resistance|Resistance)

# 获取特定基因型
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
print(f"WT|WT: {wt_wt}")
print(f"WT|Drive: {wt_drive}")
```

### 链式 API 方式（更灵活）

如果需要更多控制，也可以使用链式 API：

```python
sp2 = Species("Aedes aegypti")

# 添加常染色体
chr1 = sp2.add("chr1")
locus_A = chr1.add("A")
locus_A.add_alleles(["WT", "Drive"])

# 添加性染色体
chr_x = sp2.add("ChrX", sex_type="X")
chr_x.add("white").add_alleles(["w+", "w"])

chr_y = sp2.add("ChrY", sex_type="Y")
chr_y.add("Ymarker").add_alleles(["Y"])
```

> 更多遗传架构的细节，见 [遗传结构与实体](02_genetic_structures.md)

---

## 2️⃣ 第二步：初始化种群（3 分钟）

初始化是最关键的一步，这里会进行"编译"——生成用于数值计算的映射矩阵。

```python
from natal.nonWF_population import AgeStructuredPopulation

# 定义初始种群分布
initial_individual_count = {
    "female": {
        "WT|WT":    [0, 600, 600, 500, 400, 300, 200, 100],
        "WT|Drive": [0, 100, 100, 80, 60, 40, 20, 10],
    },
    "male": {
        "WT|WT":    [0, 300, 300, 200, 100, 0, 0, 0],
        "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0],
    },
}
# 外层键：性别 ("female" 或 "male")
# 中层键：基因型字符串 (如 "WT|WT")
# 内层值：每个年龄的个体数量列表

# 创建种群实例
pop = AgeStructuredPopulation(
    species=sp,
    name="MosquitoPop",
    n_ages=8,  # 8 个年龄类别 (0-7)
    is_stochastic=False,  # False: 确定性模型; True: 随机性模型
    
    # === 初始条件 ===
    initial_individual_count=initial_individual_count,
    
    # === 生存率 ===
    # 按年龄的生存率（0-7 岁）
    female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
    male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0],
    
    # === 繁殖相关 ===
    # 成熟年龄（年龄 >= 2 时，性别才能参与交配）
    female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    
    # === 精子存储 ===
    use_sperm_storage=True,  # 启用精子存储机制
    sperm_displacement_rate=0.05,  # 每次交配时旧精子被替换的比例
    gamete_labels=["default", "Cas9_deposited"],  # 配子标签（用于标记细胞质特征）
    
    # === 生育力 ===
    expected_eggs_per_female=100,  # 每只雌性产卵数
    
    # === 幼体生长 ===
    juvenile_growth_mode=1,  # 0: 无竞争，1: 固定，2: 逻辑斯谛，3: Beverton-Holt
    old_juvenile_carrying_capacity=1200,  # 幼体承载量
    
    # === 其他 ===
    expected_num_adult_females=2100,
    effective_population_size=0,
)

print(f"初始化完成！")
print(f"总种群大小: {pop.get_total_count():.0f}")
print(f"雌性总数: {pop.get_female_count():.0f}")
print(f"雄性总数: {pop.get_male_count():.0f}")
```

### 初始化做了什么？（"编译"过程）

高层看起来只是构造函数，但底层发生了很多事：

1. **索引注册**: 所有基因型被分配整数索引，存储在 `pop.registry` (IndexCore)
2. **映射矩阵生成**: 生成两个关键矩阵：
   - `基因型→配子`: 规定每个基因型产生什么配子
   - `配子→合子`: 规定配子组合产生什么基因型
3. **配置编译**: 所有参数被编译成 `PopulationConfig` 对象，为 Numba JIT 优化做准备
4. **状态初始化**: 根据初始分布创建 `PopulationState` 对象（包含 numpy 数组）

这个过程对用户透明，但理解它很重要。详见 [PopulationState & PopulationConfig](04_population_state_config.md)。

---

## 3️⃣ 第三步：设置适应度（可选，2 分钟）

使用 `set_viability()` 和 `set_fecundity()` 方法修改基因型的适应度：

```python
# 方式1：修改个别基因型
resistance = sp.get_genotype_from_str("Resistance|Resistance")
pop.set_viability(resistance, 0.7, sex="female")  # Resistance|Resistance 雌性生存率降低

drive_drive = sp.get_genotype_from_str("Drive|Drive")
pop.set_fecundity(drive_drive, 0.0, sex="female")  # Drive|Drive 雌性不育

# 方式2：批量设置（更高效）
fitness_map = {
    wt_wt: 1.0,
    wt_drive: 0.95,
    "Resistance|Resistance": 0.8,
}
pop.set_viability_batch(fitness_map, sex="female")
```

> 适应度的高层原理见 [Modifier 机制](06_modifiers.md)

---

## 4️⃣ 第四步：定义遗传规则 - Modifier（3 分钟）

### 配子修饰器（Gamete Modifier）

用来改变某些基因型产生配子的频率。典型场景：**基因驱动**。

```python
# 定义基因驱动：Drive|WT 杂合子的 Drive 配子比例异常高
def gene_drive_modifier(pop):
    """
    返回一个字典：
    {
        基因型字符串: {
            (等位基因, gamete_label): 频率,
            ...
        }
    }
    """
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,  # 95% 是 Drive 配子，且标记为 Cas9 沉积
            ("WT", "Cas9_deposited"): 0.05,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        },
    }

# 注册修饰器
pop.set_gamete_modifier(gene_drive_modifier, hook_name="gene_drive")
```

### 合子修饰器（Zygote Modifier）

用来改变配子组合产生基因型的频率。典型场景：**细胞质不兼容**、**胚胎拯救**。

```python
def embryo_resistance_modifier(pop):
    """
    返回一个字典：
    {
        (女性配子, 男性配子): {
            基因型字符串: 频率,
            ...
        }
    }
    """
    return {
        (("Drive", "Cas9_deposited"), ("WT", "default")): {
            "Resistance|Resistance": 0.5,  # 部分后代获得抗性
            "Drive|Resistance": 0.3,
            "WT|Resistance": 0.2,
        },
    }

pop.set_zygote_modifier(embryo_resistance_modifier, hook_name="embryo_resistance")
```

> 详细讲解见 [Modifier 机制](06_modifiers.md)

---

## 5️⃣ 第五步：定义模拟逻辑 - Hook（2 分钟）

Hook 允许你在模拟的特定阶段注入自定义逻辑。

### 声明式 Hook（最简单）

```python
from natal.hook_dsl import hook, Op

# 定义一个在 "first" 事件触发的钩子
@hook(event='first')
def release_drive_males():
    """在 tick == 10 时释放携带驱动的雄性"""
    return [
        Op.add(
            genotypes='WT|Drive',  # 选择 WT|Drive 基因型
            ages=[2, 3, 4, 5, 6, 7],  # 成年年龄
            delta=500,  # 增加 500 只
            when='tick == 10'  # 条件
        )
    ]

# 注册到种群
release_drive_males.register(pop)
```

### 原生 Numba Hook（高性能）

```python
from numba import njit

@njit
def release_hook(ind_count, tick):
    """原生 Numba 钩子——完全由用户控制"""
    if tick == 10:
        # ind_count shape: (n_sexes, n_ages, n_genotypes)
        # 这里需要自己做索引查询
        pass
    return 0  # 0: 继续，1: 停止

# 注册（需要自己管理索引）
# pop.set_hook("first", release_hook)
```

> 详细的 Hook 写法见 [Hook DSL 系统](07_hooks_dsl.md)

---

## 6️⃣ 第六步：运行模拟（1 分钟）

```python
# 运行 100 个时间步，每 10 步记录一次历史
pop.run(n_steps=100, record_every=10)

# 或运行直到特定条件（定义在 Hook 中）
pop.run(n_steps=200, record_every=5, finish=False)
```

### 查看结果

```python
# 获取最终状态
print(f"最终种群大小: {pop.get_total_count():.0f}")
print(f"最终雌性数: {pop.get_female_count():.0f}")
print(f"最终雄性数: {pop.get_male_count():.0f}")

# 查看历史记录
print(f"记录点数: {len(pop.history)}")

# 获取特定基因型的历史
history_objects = pop.get_history_as_objects()
for tick, state in history_objects:
    # state 是 PopulationState 对象
    # state.individual_count shape: (n_sexes, n_ages, n_genotypes)
    total = state.individual_count.sum()
    print(f"Tick {tick}: {total:.0f} individuals")
```

---

## 📊 完整示例

将所有步骤合并成一个完整的脚本：

```python
from natal.genetic_structures import Species
from natal.nonWF_population import AgeStructuredPopulation
from natal.hook_dsl import hook, Op

# === 第一步：定义遗传架构 ===
sp = Species.from_dict(
    name="AnophelesGambiae",
    structure={
        "chr1": {"A": ["WT", "Drive", "Resistance"]}
    }
)

# === 第二步：初始化种群 ===
initial_individual_count = {
    "female": {"WT|WT": [0, 600, 600, 500, 400, 300, 200, 100]},
    "male": {"WT|WT": [0, 300, 300, 200, 100, 0, 0, 0],
             "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0]},
}

pop = AgeStructuredPopulation(
    species=sp,
    name="MosquitoPop",
    n_ages=8,
    is_stochastic=False,
    initial_individual_count=initial_individual_count,
    female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
    male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0],
    female_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    male_age_based_mating_rates=[0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    expected_eggs_per_female=100,
    use_sperm_storage=True,
    gamete_labels=["default", "Cas9_deposited"],
)

# === 第三步：设置适应度（可选）===
drive_drive = sp.get_genotype_from_str("Drive|Drive")
pop.set_fecundity(drive_drive, 0.0, sex="female")

# === 第四步：定义 Modifier ===
def gene_drive_mod(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        },
    }

pop.set_gamete_modifier(gene_drive_mod, hook_name="gene_drive")

# === 第五步：定义 Hook ===
@hook(event='first')
def release_drive():
    return [
        Op.add(genotypes='Drive|*', ages=[2, 3, 4, 5, 6, 7],
               delta=100, when='tick == 10')
    ]

release_drive.register(pop)

# === 第六步：运行 ===
pop.run(n_steps=100, record_every=10)

# === 查看结果 ===
print(f"最终种群: {pop.get_total_count():.0f}")
```

---

## 🎯 下一步

现在你已经掌握了基础知识！接下来可以：

1. **深入学习遗传架构**：[遗传结构与实体](02_genetic_structures.md)
2. **理解性能**：[Simulation Kernels 深度解析](03_simulation_kernels.md)
3. **掌握高级功能**：[Modifier 机制](06_modifiers.md) 和 [Hook DSL](07_hooks_dsl.md)
4. **优化性能**：[Numba 优化指南](08_numba_optimization.md)

---

## ❓ 常见问题

### Q: 什么是 "gamete_labels"?
**A**: 用来标记配子的附加维度。例如 "default" 和 "Cas9_deposited" 可以区分有没有 Cas9 蛋白沉积的配子。在计算合子时，会同时考虑配子的等位基因和标签。

### Q: 为什么初始化很慢？
**A**: 初始化时要生成两个映射矩阵，复杂度与基因型数量的 3-4 次方有关。这只发生一次。之后的每个 tick 速度很快。

### Q: "确定性" vs "随机性" 是什么区别？
**A**: 
- `is_stochastic=False`: 使用多项分布期望，结果完全确定
- `is_stochastic=True`: 使用随机抽样，结果随机波动

### Q: 如何进行多次运行（Monte Carlo）?
**A**: 使用 `pop.export_state()` 和 `simulation_kernels.batch_ticks()` 进行批量模拟。详见 [Simulation Kernels](03_simulation_kernels.md)。

---

**准备好更深入的学习了吗？** [前往下一章：遗传结构与实体 →](02_genetic_structures.md)
