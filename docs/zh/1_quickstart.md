# 快速开始：15 分钟上手 NATAL

本章将通过一个**离散世代种群**和一个**年龄结构种群**的示例，带你快速熟悉 NATAL 的核心建模流程与可视化分析工具。
如您还没有安装 `natal-core`，请参考[首页](index.md)完成安装。

---

## 1️⃣ 第一步：定义遗传架构

NATAL 采用**声明式**语法定义遗传架构。你可以通过 `Species.from_dict()` 快速描述复杂的位点、染色体以及配子标记：

```python
import natal as nt

# 定义遗传架构
sp = nt.Species.from_dict(
    name="TestSpecies",
    structure={
        "chr1": {    # 染色体
            "A": ["WT", "Drive", "Resistance"]    # 位点: [等位基因列表]
        }
    },
    gamete_labels=["default", "Cas9_deposited"]    # 可选：配子标签，用于模拟细胞质沉积效应
)
```

### 理解遗传架构中的关键概念

本框架将生物遗传信息分为两个层面：**遗传学结构**（静态模板）和**遗传学实体**（动态实例）。

#### 遗传学结构（genetic structures） —— 物种的“设计图”

定义“有哪些东西”，只描述可能性，不涉及具体选择：

- **`Species`**：二倍体生物物种，包含多对同源染色体，是基因组信息的顶层容器。
- **`Chromosome`**：染色体（如 `chr1`），包含多个位点。
- **`Locus`**：基因位点（如 `A`），定义该位置可能出现的等位基因名称（如 `["WT", "Drive"]`）。

> 结构层相当于 “户型图”——几室几厅、每个房间可以放什么样的家具。

#### 遗传学实体（genetic entities） —— 具体的“装修实例”

基于设计图，在每个位点上实际选取一个等位基因，形成模拟中真实存在的遗传物质：

- **`Gene`（或称 `Allele`）**：特定的等位基因（如 `WT`、`Drive`）。它是在位点上被实例化的具体变体。
- **`Haplotype`（单倍型）**：一条染色体上所有位点选定的等位基因组合。
- **`HaploidGenotype`（单倍基因型）**：物种每对染色体各贡献一条单倍型，共同构成一组完整的单倍体基因组。
- **`Genotype`（二倍体基因型）**：母本和父本的两套单倍基因型组合而成，代表个体的全部遗传信息。
  - **注意：** 基因型严格区分母本来源和父本来源的单倍基因型。在字符串表达中，遵循 `Maternal|Paternal` 的顺序。`A|a` 和 `a|A` 会被认为是不同的基因型。

> 实体层相当于 “装修好的房子” ——每个房间已经选定了具体的家具款式，窗户也确定了是圆是方。

#### 为什么这样区分？

- **结构** 是模型级的、不可变的配置（例如“这个物种有 A、B 两个位点”），可以在模拟开始前一次性定义。
- **实体** 是种群级的、动态出现的实例（例如“当前种群中有 `WT|WT`、`WT|Drive`、`Drive|WT`、`Drive|Drive` 四种基因型”），通过遗传规则产生和传递。

这种分离使得模拟可以灵活定义复杂的遗传架构，同时在运行时保持高效的计算。

> 完整概念、对象关系与更多示例请参考 [遗传结构与实体](2_genetics.md)。

### 验证架构

```python
# 查看所有可能的基因型
all_genotypes = sp.get_all_genotypes()
print(f"总共有 {len(all_genotypes)} 种基因型")
# 输出: 总共有 9 种基因型
# (WT|WT, WT|Drive, WT|Resistance, Drive|WT, Drive|Drive, Drive|Resistance, Resistance|WT, Resistance|Drive, Resistance|Resistance)

# 获取特定基因型
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
print(f"WT|WT: {wt_wt}")
print(f"WT|Drive: {wt_drive}")
```

> 更多遗传架构的细节，见 [遗传结构与实体](2_genetics.md)

---

## 2️⃣ 第二步：初始化种群

初始化是模型构建的关键。在此阶段，NATAL 会执行“编译”过程，将高层对象转换为高效的数值映射矩阵。

### 离散世代种群（DiscreteGenerationPopulation）—— 最简单的起点

如果你的模型不需要年龄结构（如理论模型、果蝇等实验室生物），可以使用更简单的离散世代模型：

```python
# 离散世代模型（无年龄结构）
pop = (nt.DiscreteGenerationPopulation
    .setup(
        species=sp,
        name="FruitFlyPop",
        stochastic=True                # True: 随机性模型; False: 确定性模型
    )
    .initial_state({
        "female": {"WT|WT": 1000},
        "male":   {"WT|WT": 1000}
    })
    .reproduction(
        eggs_per_female=50,            # 每只雌性产卵数
        sex_ratio=0.5                  # 后代性别比例
    )
    .build()
)

print(f"初始种群: {pop.get_total_count()}")
```

### 年龄结构种群（AgeStructuredPopulation）—— 更贴近自然

对于需要年龄结构的自然种群或混合笼养种群，使用年龄结构模型：

```python
# 年龄结构模型
pop = (nt.AgeStructuredPopulation
    .setup(
        species=sp,
        name="MosquitoPop",
        stochastic=False               # False: 确定性模型; True: 随机性模型
    )
    .age_structure(
        n_ages=8,                      # 8个年龄类别
        new_adult_age=2                # 2岁为成年
    )
    .initial_state({
        "female": {
            "WT|WT":    [0, 600, 600, 500, 400, 300, 200, 100],
        },
        "male": {
            "WT|WT":    [0, 300, 300, 200, 100, 0, 0, 0],
            "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0],
        }
    })
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0]
    )
    .reproduction(
        eggs_per_female=100,
        sex_ratio=0.5,
        use_sperm_storage=True,        # 启用精子存储机制
    )
    .competition(
        juvenile_growth_mode=1,        # 1: 固定竞争模式
        age_1_carrying_capacity=1200
    )
    .build()
)

print(f"初始化完成！")
print(f"总种群大小: {pop.get_total_count():.0f}")
print(f"雌性总数: {pop.get_female_count():.0f}")
print(f"雄性总数: {pop.get_male_count():.0f}")
```

### 两种种群类型对比

| 特性 | DiscreteGenerationPopulation | AgeStructuredPopulation |
|------|-----------------------------|------------------------|
| 年龄结构 | ❌ 不支持 | ✅ 支持 |
| 生存率 | 固定概率 | 按年龄配置 |
| 精子存储 | ❌ 不支持 | ✅ 支持 |
| 适用场景 | 实验室人工传代种群 | 自然种群、混合笼养种群 |
| 复杂度 | 较低 | 较高 |

---

## 3️⃣ 第三步：使用遗传预设系统

针对基因驱动、点突变等常见的遗传现象，NATAL 提供了**遗传预设（Genetic Presets）**系统。相比手动编写底层的映射修饰函数，预设系统更加简洁、可重用且易于维护。

### 使用基因驱动预设

```python
# 创建基因驱动预设
drive = nt.HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.03
)

# 在离散世代种群中添加预设
pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="FruitFlyPop", stochastic=True)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .presets(drive)                 # 应用基因驱动预设
    .build()
)
```

### 使用其他预设

目前预设系统包括 [HomingDrive](api/genetic_presets.md#natal.genetic_presets.HomingDrive) 和 [ToxinAntidoteDrive](api/genetic_presets.md#natal.genetic_presets.ToxinAntidoteDrive) 两类，未来会持续扩展更多预设类型。

你也可以自定义预设，详见 [设计你自己的预设](3_custom_presets.md)。

### 适应度设置（可选）

如果需要设置适应度效应，可以在 `fitness()` 方法中配置：

```python
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .fitness(viability={
        "Resistance|Resistance": {"female": 0.7},   # 抗性纯合子生存率降低
        "Drive|Drive": {"female": 0.0}              # 驱动纯合子不育
    })
    .presets(drive)
    .build()
)
```

> **💡 提示**: 对于需要自定义复杂遗传规则的高级用户，可以参考[Modifier机制](3_modifiers.md)手动编写modifier函数。但对于大多数常见场景，预设系统更简单可靠。

---

## 4️⃣ 第四步：定义模拟逻辑 - Hook

**Hook 系统**允许你在模拟循环的关键节点（如每步开始、生存筛选后等）注入自定义干预或监测逻辑。使用声明式 `Op` 语法最为高效直观：

```python
from natal.hook_dsl import hook, Op

@hook(event='first')
def release_drive_males():
    """在 tick == 10 时释放携带驱动的雄性"""
    return [
        Op.add(
            genotypes='WT|Drive',    # 选择 WT|Drive 基因型
            ages=2,                  # 成年年龄（仅对年龄结构模型有效）
            sex='male',              # 仅释放雄性
            delta=500,               # 增加 500 只
            when='tick == 10'        # 条件
        )
    ]

# 注册到种群
release_drive_males.register(pop)

# 也可在构建过程中注册
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MyPop")
    # ...（其他初始化方法）
    .hooks(release_drive_males)
    .build()
)
```

> **💡 提示**: 对于需要高性能或复杂逻辑的高级用户，可以使用原生 Numba Hook。详见 [Hook 系统](2_hooks.md)

---

## 5️⃣ 第五步：运行模拟与结果分析

```python
# 运行 100 个时间步，每 10 步记录一次历史
pop.run(n_steps=100, record_every=10)

# 或运行直到特定条件（定义在 Hook 中）
pop.run(n_steps=200, record_every=5, finish=False)
```

### 查看结果

```python
# 1) 用可读字典查看当前状态（适合日志/调试/接口返回）
state_view = nt.population_to_readable_dict(pop)
print(state_view["state_type"], state_view["tick"])
print(state_view["individual_count"]["female"].keys())

# 2) 如需 JSON，可直接导出
state_json = nt.population_to_readable_json(pop, indent=2)
print(state_json[:240])

# 3) 定义可复用 observation 规则（推荐通过 population API）
observation = pop.create_observation(
    groups={
        "adult_drive_female": {
            "genotype": "Drive::*",
            "sex": "female",
            "age": [2, 3, 4, 5, 6, 7],
        },
        "all_adults": {
            "age": [2, 3, 4, 5, 6, 7],
        },
    },
    collapse_age=False,
)

# 4) 导出当前快照（状态翻译 + observation）
current_obs = pop.output_current_state(
    observation=observation,
    include_zero_counts=False,
)
print(current_obs["labels"])
print(current_obs["observed"]["adult_drive_female"])

# 5) 导出历史 observation（可直接用于绘图/导出）
history_obs = pop.output_history(
    observation=observation,
    include_zero_counts=False,
)
print(history_obs["n_snapshots"])
print(history_obs["snapshots"][0]["observed"]["all_adults"])
```

推荐将 `output_current_state()` 与 `output_history()` 搭配使用：

- `observation` 定义观测对象（分组与筛选规则）
- 状态翻译 API 定义导出形式（可读 dict/JSON）

如果你更偏好模块级函数，也可以使用
`nt.output_current_state(...)` 和 `nt.output_history(...)`；语义与
population 方法等价。

### 对应可运行示例

下面脚本与上述流程直接对应（在仓库根目录运行）：

```bash
python demos/observation_history_demo.py
python demos/discrete.py
python demos/mosquito.py
```

更多场景可参考 `demos/` 目录。

### 🎛️ 使用内置可视化面板（可选）

NATAL 提供了一个基于 NiceGUI 的实时可视化面板，可以在浏览器中观察种群动态：

```python
import natal as nt
from natal.ui import launch

# ... 定义遗传架构、构建种群 ...

# 启动面板
launch(pop, port=8080, title="My Simulation")
```

启动后，在浏览器中打开 <http://localhost:8080> 即可查看种群数量变化、基因型频率等动态图表。

---

## 深入理解：初始化时的“编译”过程

虽然高层代码直观易读，但底层在 `build()` 时完成了一系列复杂操作：

1. **索引注册**: 所有基因型被分配整数索引，存储在 `pop.registry` (IndexRegistry)
2. **映射矩阵生成**: 根据遗传预设和遗传映射的 `modifiers` 生成两个关键矩阵：
   - `基因型→配子`: 规定每个基因型产生什么配子
   - `配子→合子`: 规定配子组合产生什么基因型
3. **配置编译**: 所有参数被编译成 `PopulationConfig` NamedTuple，为 Numba JIT 优化做准备
4. **Hooks 编译**: 用户自定义的 Hooks 被编译为执行计划，将在对应的时机被调用
5. **状态初始化**: 根据初始分布创建 `PopulationState` NamedTuple（包含 numpy 数组）

这个过程对用户透明，但理解它很重要。详见：
- [Index registry](4_index_registry.md)
- [PopulationState & PopulationConfig](4_population_state_config.md)
- [Modifiers 系统](3_modifiers.md) 和 [遗传预设系统](2_genetic_presets.md)
- [Hooks 系统](2_hooks.md)
- [Numba 优化指南](4_numba_optimization.md)---

## 📊 完整示例

### 示例一：离散世代种群 + 基因驱动 + Hook

```python
import natal as nt
from natal.genetic_presets import HomingDrive
from natal.hook_dsl import hook, Op

sp = nt.Species.from_dict(
    name="FruitFly",
    structure={"chr1": {"A": ["WT", "Drive", "Resistance"]}}
)

drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.03
)

@hook(event='first')
def release_drive():
    return [Op.add(genotypes='Drive|WT', delta=50, when='tick == 10')]

pop = (nt.DiscreteGenerationPopulation
    .setup(species=sp, name="FruitFlyPop", stochastic=True)
    .initial_state({"female": {"WT|WT": 500}, "male": {"WT|WT": 500}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .presets(drive)
    .hooks(release_drive)              # 注册 Hook
    .build()
)

pop.run(n_steps=100, record_every=10)
print(f"最终种群: {pop.get_total_count():.0f}")
print(f"等位基因频率: {pop.compute_allele_frequencies()}")
```

### 示例二：年龄结构种群 + 基因驱动 + 适应度 + Hook

```python
import natal as nt
from natal.genetic_presets import HomingDrive
from natal.hook_dsl import hook, Op

sp = nt.Species.from_dict(
    name="AnophelesGambiae",
    structure={"chr1": {"A": ["WT", "Drive", "Resistance"]}},
    gamete_labels=["default", "Cas9_deposited"]
)

drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,
    late_germline_resistance_formation_rate=0.03
)

@hook(event='first')
def release_drive():
    return [Op.add(genotypes='Drive|WT', ages=[2,3,4,5,6,7], delta=100, when='tick == 10')]

pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="MosquitoPop", stochastic=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({
        "female": {"WT|WT": [0, 600, 600, 500, 400, 300, 200, 100]},
        "male": {
            "WT|WT": [0, 300, 300, 200, 100, 0, 0, 0],
            "WT|Drive": [0, 300, 300, 200, 100, 0, 0, 0]
        }
    })
    .survival(
        female_age_based_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
        male_age_based_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0]
    )
    .reproduction(eggs_per_female=100, sex_ratio=0.5, use_sperm_storage=True)
    .competition(juvenile_growth_mode=1, age_1_carrying_capacity=1200)
    .fitness(viability={"Drive|Drive": {"female": 0.0}})
    .presets(drive)
    .hooks(release_drive)
    .build()
)

pop.run(n_steps=100, record_every=10)
print(f"最终种群: {pop.get_total_count():.0f}")
print(f"等位基因频率: {pop.compute_allele_frequencies()}")
```

---

## 🎯 下一步

现在你已经掌握了基础知识！接下来可以：

1. **深入学习遗传预设系统**：[遗传预设系统](2_genetic_presets.md) - 学习如何创建自定义预设
2. **理解遗传架构**：[遗传结构与实体](2_genetics.md) - 深入了解Species、Chromosome等概念
3. **掌握高级功能**：[Hook 系统](2_hooks.md) - 学习如何注入自定义模拟逻辑
4. **需要自定义遗传规则**：[Modifier 机制](3_modifiers.md) - 手动编写gamete/zygote修饰器
5. **性能优化**：[Numba 优化指南](4_numba_optimization.md) - 提升模拟性能

---

## ❓ 常见问题

### Q: 什么是 "gamete_labels"?
**A**: 用来标记配子的附加维度。例如 "default" 和 "Cas9_deposited" 可以区分有没有 Cas9 蛋白沉积的配子。在计算合子时，会同时考虑配子的等位基因和标签。

### Q: 为什么初始化较慢？
**A**: 初始化时要生成两个映射矩阵，复杂度与基因型数量的 3-4 次方有关，根据 numba 缓存情况可能还需要不同程度的编译。对于相对简单（仅有几十种基因型）的遗传学设定，预计需要数秒至数十秒时间。这只发生一次。之后的每个 tick 速度很快。

### Q: 什么时候使用离散世代种群？
**A**: 当你的模型不需要年龄结构时，使用`DiscreteGenerationPopulationBuilder`更简单，适用于：
- 果蝇等实验室模型
- 理论模型研究
- 不需要年龄相关效应的模拟

### Q: "确定性" vs "随机性" 是什么区别？
**A**:
- `is_stochastic=False`: 使用多项分布期望，结果完全确定（允许小数）
- `is_stochastic=True`: 使用随机抽样，结果随机波动（必为整数）

---

**准备好更深入的学习了吗？** [前往下一章：遗传结构与实体 →](2_genetics.md)
