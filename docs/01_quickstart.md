# 快速开始：15 分钟上手 NATAL

本指南将通过一个典型的蚊媒种群模拟示例，带你快速熟悉 NATAL 的核心建模流程与可视化分析工具。

## 前置要求

- Python >= 3.9
- NumPy >= 2.0.0
- Numba >= 0.60.0
- Pandas >= 2.1.0
- Plotly >= 6.0.0
- SciPy >= 1.10.0
- NiceGUI[highcharts] >= 3.0.0

## 1️⃣ 第一步：定义遗传架构（2 分钟）

NATAL 采用**声明式**语法定义遗传架构。你可以通过 `Species.from_dict()` 快速描述复杂的位点、染色体以及配子标记：

```python
from natal.genetic_structures import Species

# 定义遗传架构
sp = Species.from_dict(
    name="AnophelesGambiae",
    structure={
        "chr1": {    # Chromosome
            "A": ["WT", "Drive", "Resistance"]    # Locus: [Alleles]
        }
    },
    gamete_labels=["default", "Cas9_deposited"]    # 可选：定义配子标签，用于模拟细胞质沉积效应
)
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

> 更多遗传架构的细节，见 [遗传结构与实体](02_genetic_structures.md)

---

## 2️⃣ 第二步：初始化种群（3 分钟）

初始化是模型构建的关键。在此阶段，NATAL 会执行“编译”过程，将高层对象转换为高效的数值映射矩阵。推荐使用 **Builder 模式**进行直观的链式配置：

```python
from natal.population_builder import AgeStructuredPopulationBuilder

# 使用Builder模式创建和配置种群
pop = (AgeStructuredPopulationBuilder(sp)
    .setup(name="MosquitoPop", stochastic=False)  # False: 确定性模型; True: 随机性模型
    
    # === 年龄结构 ===
    .age_structure(n_ages=8, new_adult_age=2)  # 8个年龄类别，2岁为成年
    
    # === 初始种群分布 ===
    .initial_state({
        "female": {
            "WT|WT":    [0, 600, 600, 500, 400, 300, 200, 100],
        },
        "male": {
            "WT|WT":    [0, 300, 300, 200, 100, 0, 0, 0],
            "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0],
        }
    })
    
    # === 生存率 ===
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0]
    )
    
    # === 繁殖相关 ===
    .reproduction(
        eggs_per_female=100,           # 每只雌性产卵数
        sex_ratio=0.5,                 # 后代性别比例
        use_sperm_storage=True,        # 启用精子存储机制
    )
    
    # === 幼体竞争 ===
    .competition(
        juvenile_growth_mode=1,        # 1: 固定竞争模式
        old_juvenile_carrying_capacity=1200 # 幼体承载量
    )
    
    .build())  # 构建最终种群实例

print(f"初始化完成！")
print(f"总种群大小: {pop.get_total_count():.0f}")
print(f"雌性总数: {pop.get_female_count():.0f}")
print(f"雄性总数: {pop.get_male_count():.0f}")
```

### 离散世代种群（DiscreteGenerationPopulation）

如果你的模型不需要年龄结构（如理论模型、果蝇等实验室生物），可以使用更简单的离散世代模型：

```python
from natal.population_builder import DiscreteGenerationPopulationBuilder

# 离散世代模型（无年龄结构）
pop = (DiscreteGenerationPopulationBuilder(sp)
    .setup(name="FruitFlyPop", stochastic=True)
    .initial_state({
        "female": {"WT|WT": 1000},  # 简单数量，不是年龄分布
        "male": {"WT|WT": 1000}
    })
    .reproduction(
        eggs_per_female=50,         # 每只雌性产卵数
        sex_ratio=0.5                # 后代性别比例
    )
    .build())

print(f"初始种群: {pop.get_total_count()}")
```

**两种种群类型对比：**

| 特性 | AgeStructuredPopulation | DiscreteGenerationPopulation |
|------|------------------------|----------------------------|
| 年龄结构 | ✅ 支持 | ❌ 不支持 |
| 生存率 | 按年龄配置 | 固定概率 |
| 精子存储 | ✅ 支持 | ❌ 不支持 |
| 适用场景 | 自然种群、混合笼养种群 | 实验室连续传代种群 |
| 复杂度 | 较高 | 较低 |

### 深入理解：初始化时的“编译”过程

虽然高层代码直观易读，但底层在 `build()` 时完成了一系列复杂操作：

1. **索引注册**: 所有基因型被分配整数索引，存储在 `pop.registry` (IndexRegistry)
2. **映射矩阵生成**: 生成两个关键矩阵：
   - `基因型→配子`: 规定每个基因型产生什么配子
   - `配子→合子`: 规定配子组合产生什么基因型
3. **配置编译**: 所有参数被编译成 `PopulationConfig` 对象，为 Numba JIT 优化做准备
4. **状态初始化**: 根据初始分布创建 `PopulationState` 对象（包含 numpy 数组）

这个过程对用户透明，但理解它很重要。详见 [PopulationState & PopulationConfig](04_population_state_config.md)。

---

## 3️⃣ 第三步：使用遗传预设系统（2 分钟）

针对基因驱动、点突变等常见的遗传现象，NATAL 提供了**遗传预设（Genetic Presets）**系统。相比手动编写底层的映射修饰函数，预设系统更加简洁、可重用且易于维护。

### 使用基因驱动预设

```python
from natal.genetic_presets import HomingDrive

# 创建基因驱动预设
drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive", 
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,  # 95%转换效率
    late_germline_resistance_formation_rate=0.03
)

# 在Builder中添加预设
pop = (AgeStructuredPopulationBuilder(sp)
    .setup(name="MosquitoPop", stochastic=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({...})
    .survival(female_rates=[...], male_rates=[...])
    .reproduction(eggs_per_female=100)
    .presets(drive)  # 应用基因驱动预设
    .build())
```

### 使用其他预设

预设系统支持多种遗传修饰：

- **HomingDrive**: CRISPR/Cas9基因驱动
- **PointMutation**: 简单点突变
- **CustomPresets**: 用户自定义预设

### 适应度设置（可选）

如果需要设置适应度效应，可以在builder中配置：

```python
pop = (AgeStructuredPopulationBuilder(sp)
    .setup(name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .fitness(viability={
        "Resistance|Resistance": {"female": 0.7},  # 抗性纯合子生存率降低
        "Drive|Drive": {"female": 0.0}              # 驱动纯合子不育
    })
    .presets(drive)
    .build())
```

> **💡 提示**: 对于需要自定义复杂遗传规则的高级用户，可以参考[Modifier机制](06_modifiers.md)手动编写modifier函数。但对于大多数常见场景，预设系统更简单可靠。

---

## 4️⃣ 第四步：定义模拟逻辑 - Hook（2 分钟）

**Hook 系统**允许你在模拟循环的关键节点（如每步开始、生存筛选后等）注入自定义干预或监测逻辑。使用声明式 `Op` 语法最为高效直观：

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
            sex='male',  # 仅释放雄性
            delta=500,  # 增加 500 只
            when='tick == 10'  # 条件
        )
    ]

# 注册到种群
release_drive_males.register(pop)
```

> **💡 提示**: 对于需要高性能或复杂逻辑的高级用户，可以使用原生 Numba Hook。详见 [Hook 系统](07_hooks.md)

---

## 5️⃣ 第五步：运行模拟与结果分析（1 分钟）

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

将所有步骤合并成一个完整的脚本，使用推荐的Builder模式：

```python
from natal.genetic_structures import Species
from natal.population_builder import AgeStructuredPopulationBuilder
from natal.genetic_presets import HomingDrive
from natal.hook_dsl import hook, Op

# === 第一步：定义遗传架构 ===
sp = Species.from_dict(
    name="AnophelesGambiae",
    structure={
        "chr1": {"A": ["WT", "Drive", "Resistance"]}
    },
    gamete_labels=["default", "Cas9_deposited"]
)

# === 第二步：使用Builder模式创建和配置种群 ===
pop = (AgeStructuredPopulationBuilder(sp)
    .setup(name="MosquitoPop", stochastic=False)
    .age_structure(n_ages=8, new_adult_age=2)
    
    # 初始种群分布
    .initial_state({
        "female": {"WT|WT": [0, 600, 600, 500, 400, 300, 200, 100]},
        "male": {
            "WT|WT": [0, 300, 300, 200, 100, 0, 0, 0],
            "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0]
        }
    })
    
    # 生存率配置
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0]
    )
    
    # 繁殖配置
    .reproduction(
        eggs_per_female=100,
        sex_ratio=0.5,
        use_sperm_storage=True
    )
    
    # 添加基因驱动预设（替代手动modifier）
    .presets(HomingDrive(
        name="MyDrive",
        drive_allele="Drive",
        target_allele="WT", 
        resistance_allele="Resistance",
        drive_conversion_rate=0.95,
        late_germline_resistance_formation_rate=0.03
    ))
    
    # 设置适应度（Drive|Drive雌性不育）
    .fitness(viability={
        "Drive|Drive": {"female": 0.0}  # 雌性Drive纯合子完全不育
    })
    
    .build())  # 构建最终种群实例

# === 第三步：定义 Hook（释放驱动个体）===
@hook(event='first')
def release_drive():
    return [
        Op.add(genotypes='Drive|*', ages=[2, 3, 4, 5, 6, 7],
               delta=100, when='tick == 10')
    ]

release_drive.register(pop)

# === 第四步：运行模拟 ===
pop.run(n_steps=100, record_every=10)

# === 第五步：查看结果 ===
print(f"最终种群: {pop.get_total_count():.0f}")
print(f"等位基因频率: {pop.compute_allele_frequencies()}")
```

### 离散世代种群完整示例

如果使用离散世代模型，代码更简洁：

```python
from natal.genetic_structures import Species
from natal.population_builder import DiscreteGenerationPopulationBuilder
from natal.genetic_presets import HomingDrive
from natal.hook_dsl import hook, Op

# === 第一步：定义遗传架构 ===
sp = Species.from_dict(
    name="FruitFly",
    structure={
        "chr1": {"A": ["WT", "Drive", "Resistance"]}
    }
)

# === 第二步：使用离散世代Builder创建种群 ===
pop = (DiscreteGenerationPopulationBuilder(sp)
    .setup(name="FruitFlyPop", stochastic=True)
    
    # 初始种群分布（简单数量，无需年龄分布）
    .initial_state({
        "female": {"WT|WT": 500},
        "male": {"WT|WT": 500}
    })
    
    # 繁殖配置
    .reproduction(
        eggs_per_female=50,
        sex_ratio=0.5
    )
    
    # 添加基因驱动预设
    .presets(HomingDrive(
        name="MyDrive",
        drive_allele="Drive",
        target_allele="WT",
        resistance_allele="Resistance",
        drive_conversion_rate=0.95,
        late_germline_resistance_formation_rate=0.03
    ))
    
    .build())

# === 第三步：定义 Hook ===
@hook(event='first')
def release_drive():
    return [
        Op.add(genotypes='Drive|WT', delta=50, when='tick == 10')
    ]

release_drive.register(pop)

# === 第四步：运行模拟 ===
pop.run(n_steps=100, record_every=10)

# === 第五步：查看结果 ===
print(f"最终种群: {pop.get_total_count():.0f}")
print(f"等位基因频率: {pop.compute_allele_frequencies()}")
```

---

## 🎯 下一步

现在你已经掌握了基础知识！接下来可以：

1. **深入学习遗传预设系统**：[遗传预设系统](15_genetic_presets_guide.md) - 学习如何创建自定义预设
2. **理解遗传架构**：[遗传结构与实体](02_genetic_structures.md) - 深入了解Species、Chromosome等概念
3. **掌握高级功能**：[Hook 系统](07_hooks.md) - 学习如何注入自定义模拟逻辑
4. **需要自定义遗传规则**：[Modifier 机制](06_modifiers.md) - 手动编写gamete/zygote修饰器
5. **性能优化**：[Numba 优化指南](08_numba_optimization.md) - 提升模拟性能

---

## ❓ 常见问题

### Q: 什么是 "gamete_labels"?
**A**: 用来标记配子的附加维度。例如 "default" 和 "Cas9_deposited" 可以区分有没有 Cas9 蛋白沉积的配子。在计算合子时，会同时考虑配子的等位基因和标签。

### Q: 为什么初始化很慢？
**A**: 初始化时要生成两个映射矩阵，复杂度与基因型数量的 3-4 次方有关。这只发生一次。之后的每个 tick 速度很快。

### Q: 为什么要使用Builder模式？
**A**: Builder模式提供了更清晰、更灵活的API：

```python
# Builder模式 - 链式调用，参数组织清晰
pop = (AgeStructuredPopulationBuilder(species)
    .setup(name="MyPop", stochastic=True)
    .age_structure(n_ages=8)
    .initial_state({...})
    .survival(female_rates=[...], male_rates=[...])
    .reproduction(eggs_per_female=100)
    .presets(my_preset)  # 添加预设
    .build())
```

Builder模式的优势：
- **可读性更好**：每个配置步骤都有明确的方法名
- **参数验证**：在构建时进行参数检查
- **链式调用**：支持流畅的API设计
- **扩展性**：易于添加新的配置选项

### Q: 什么时候使用离散世代种群？
**A**: 当你的模型不需要年龄结构时，使用`DiscreteGenerationPopulationBuilder`更简单：

```python
from natal.population_builder import DiscreteGenerationPopulationBuilder

# 离散世代模型（无年龄结构）
pop = (DiscreteGenerationPopulationBuilder(species)
    .setup(name="SimplePop", stochastic=True)
    .initial_state({
        "female": {"WT|WT": 1000},
        "male": {"WT|WT": 1000}
    })
    .reproduction(eggs_per_female=50)
    .presets(my_drive)
    .build())
```

适用于：
- 果蝇等实验室模型
- 理论模型研究
- 不需要年龄相关效应的模拟

### Q: "确定性" vs "随机性" 是什么区别？
**A**: 
- `is_stochastic=False`: 使用多项分布期望，结果完全确定
- `is_stochastic=True`: 使用随机抽样，结果随机波动

### Q: 如何进行多次运行（Monte Carlo）?
**A**: 使用 `pop.export_state()` 和 `simulation_kernels.batch_ticks()` 进行批量模拟。详见 [Simulation Kernels](03_simulation_kernels.md)。

---

**准备好更深入的学习了吗？** [前往下一章：遗传结构与实体 →](02_genetic_structures.md)