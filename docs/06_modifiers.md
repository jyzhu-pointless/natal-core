# Modifier 机制

本章讲解如何通过 Modifier 修改种群遗传规则，实现基因驱动、细胞质不兼容等复杂遗传现象。

> **💡 提示**: 对于常见的遗传修饰（如基因驱动），推荐使用[遗传预设系统](15_genetic_presets_guide.md)，它提供了更简洁的API。本章介绍底层Modifier机制，适合需要自定义复杂规则的高级用户。

## 核心原理

Modifier 的本质是 **修改映射矩阵**：

```
未修改的映射矩阵:
  A1|A2 雌性 → [50% A1, 50% A2]
                    ↓ Gamete Modifier
修改后的映射矩阵:
  A1|A2 雌性 → [95% A1 (with Cas9), 5% A2 (with Cas9)]
  
未修改的合子分布:
  (A1-Cas9) + (A2-default) → [25% A1|A2, 25% A2|A2, ...]
                    ↓ Zygote Modifier
修改后的合子分布:
  (A1-Cas9) + (A2-default) → [10% A1|A2, 50% A1|Resistance, ...]
```

## 两类 Modifier

### 类型 1：配子修饰器（Gamete Modifier）

修改特定基因型产生配子的频率。

#### 使用场景

- **基因驱动**：某个等位基因过度分离（drive allele）
- **配子选择**：某类配子优先存活
- **标记配子**：标记特定的配子类型（如 Cas9 蛋白沉积）

#### 现代Builder模式中的使用

```python
from natal.population_builder import AgeStructuredPopulationBuilder

def my_gamete_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        }
    }

# 在Builder中使用
pop = (AgeStructuredPopulationBuilder(species)
    .setup(name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .modifiers(gamete_modifiers=[(None, "my_modifier", my_gamete_modifier)])
    .build())
```



#### 例 1：基因驱动（Homing Endonuclease Gene Drive）

```python
def heg_drive_modifier(pop):
    """HEG 驱动：Drive|WT 杂合子的 Drive 配子比例极高"""
    
    ic = pop.registry
    sp = pop.species
    
    # 定义哪些基因型会产生驱动配子
    # 假设驱动在杂合子中表现（Drive|WT, WT|Drive）
    # 纯合子（Drive|Drive）可能完全不育或产生正常配子
    
    return {
        # 杂合子：驱动等位基因过度分离
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.98,  # 98% 是驱动
            ("WT", "Cas9_deposited"): 0.02,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.98,
            ("WT", "Cas9_deposited"): 0.02,
        },
        
        # 纯合子：正常分离（或其他行为）
        # 如果不列出，保持默认的孟德尔分离
    }

pop.set_gamete_modifier(heg_drive_modifier, hook_name="HEG_drive")
```

#### 例 2：性连锁驱动

```python
def sex_linked_drive(pop):
    """仅雌性产生驱动配子，雄性正常"""
    
    return {
        "Drive|WT": {
            # 仅在 population.female_gamete_modifiers 中应用
            ("Drive", "default"): 0.95,
            ("WT", "default"): 0.05,
        }
    }

# 注册时可以指定性别（通过额外参数或修饰器内部判断）
pop.set_gamete_modifier(sex_linked_drive, hook_name="sex_drive")
```

#### 例 3：标记配子（标记细胞质沉积）

```python
def cas9_deposition_modifier(pop):
    """Drive|WT 和 WT|Drive 雌性产生的配子携带 Cas9 蛋白标记"""
    
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.5,
            ("WT", "Cas9_deposited"): 0.5,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.5,
            ("WT", "Cas9_deposited"): 0.5,
        },
        
        # WT|WT 不产生 Cas9 标记
        "WT|WT": {
            ("WT", "default"): 1.0,
        }
    }

pop.set_gamete_modifier(cas9_deposition_modifier, hook_name="Cas9_mark")
```

### 类型 2：合子修饰器（Zygote Modifier）

修改配子组合产生基因型的结果。

#### 使用场景

- **细胞质不兼容**（Cytoplasmic Incompatibility, CI）：某些配子组合导致胚胎死亡
- **胚胎拯救**：某些配子组合的存活率降低或增高
- **复杂的遗传比例**：不遵循孟德尔比例的遗传
- **孤雌生殖触发**：某些配子组合产生全雌后代

#### 现代Builder模式中的使用

```python
from natal.population_builder import AgeStructuredPopulationBuilder

def my_zygote_modifier(pop):
    return {
        (("Drive", "Cas9_deposited"), ("WT", "default")): {
            "Resistance|Resistance": 0.5,
            "Drive|Resistance": 0.3,
            "WT|Resistance": 0.2,
        },
    }

# 在Builder中使用
pop = (AgeStructuredPopulationBuilder(species)
    .setup(name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .modifiers(zygote_modifiers=[(None, "my_modifier", my_zygote_modifier)])
    .build())
```

#### 传统方式（直接设置）

```python
def zygote_modifier(pop) -> Dict:
    """
    Args:
        pop: AgeStructuredPopulation 对象
        
    Returns:
        Dict: {(母配子, 父配子): {基因型: 频率, ...}}
        
        母配子和父配子格式：(等位基因, label)
    """
    return {
        (("A1", "default"), ("A2", "default")): {
            "A1|A2": 0.5,
            "A2|A2": 0.5,
        },
    }


```
```

#### 例 1：细胞质不兼容（CI）

```python
def ci_modifier(pop):
    """
    Wolbachia 引起的 CI：
    - 感染雄性 × 未感染雌性 → 胚胎死亡
    - 其他组合正常
    """
    
    return {
        # 未感染雌性 × 感染雄性的精子 → 胚胎不存活
        (("Allele1", "uninfected"), ("Allele1", "Wolbachia")): {
            # 所有基因型概率为 0（胚胎死亡）
            # 实际上会被跳过
        },
        
        # 其他组合使用默认的孟德尔比例
    }

pop.set_zygote_modifier(ci_modifier, hook_name="CI")
```

#### 例 2：胚胎拯救与驱动

```python
def embryo_rescue_modifier(pop):
    """
    Cas9 驱动系统：
    - Cas9+ 雌性 (Drive|WT) × WT 雄性
    - 不匹配的后代（WT|WT）通过 Cas9 切割被拯救为 Drive|WT
    """
    
    return {
        # Cas9-bearing female × WT male
        (("Drive", "Cas9_deposited"), ("WT", "default")): {
            "Drive|WT": 0.4,      # 正常分离
            "WT|WT": 0.0,         # 原本是 0.4，但被 Cas9 切割
            "Drive|Drive": 0.6,   # 拯救后成为 Drive|Drive (或 Drive|WT)
        },
    }

pop.set_zygote_modifier(embryo_rescue_modifier, hook_name="rescue")
```

#### 例 3：性比失衡驱动

```python
def sex_ratio_distorter(pop):
    """
    通过修改合子性别比例实现的驱动
    （注意：这种方式改变基因型频率，不改变配子）
    """
    
    return {
        # 某些配子组合产生过多雌性
        (("Drive", "default"), ("WT", "default")): {
            "Drive|WT": 0.6,  # 正常基因型（假设）
            # 但产生的后代中，雌性比例升高
            # （这需要与 sex determination 系统交互）
        },
    }
```

## 配子标签（Gamete Labels）

Gamete Label 是附加在配子上的标记，用于区分具有不同细胞质或其他特征的配子。

### 概念

```
标准配子：("A1", None) 或 ("A1", "default")
标记配子：("A1", "Cas9_deposited")

两者遗传的 A1 等位基因是同一个，但标签不同
```

### 使用场景

1. **Cas9 蛋白沉积**：标记产生 Cas9 蛋白的雌性的配子
2. **细胞质标记**：区分携带不同内共生菌的配子
3. **条件表达**：标记在特定条件下才激活的基因

### 配置

```python
pop = AgeStructuredPopulation(
    ...,
    gamete_labels=["default", "Cas9_deposited"],
    # 或
    gamete_labels=["WT_cytotype", "Wolbachia_infected"],
)
```

### 在 Modifier 中使用

```python
def marker_modifier(pop):
    """产生带标记的配子"""
    
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.5,     # 标记配子
            ("WT", "Cas9_deposited"): 0.5,
            # 注意：("Drive", "default") 不出现，因为所有配子都标记
        },
    }
```

### 标签在后代中的传递

```
母配子：("A1", "Cas9_deposited")
父配子：("A2", "default")
后代的 "Cas9" label 取决于：
  - 通常，label 来自母配子（细胞质继承）
  - 或可以在合子修饰器中自定义规则
```

## 注册 Modifier

### 方式 1：构造时传入

```python
def my_gamete_mod(pop):
    return {"A1|A2": {("A1", "default"): 0.7, ("A2", "default"): 0.3}}

def my_zygote_mod(pop):
    return {}

pop = AgeStructuredPopulation(
    ...,
    gamete_modifiers=[
        (0, "my_name", my_gamete_mod),  # (priority, name, function)
    ],
    zygote_modifiers=[
        (0, "my_name", my_zygote_mod),
    ],
)
```

### 方式 2：动态注册

```python
pop.set_gamete_modifier(my_gamete_mod, hook_name="my_drive")
pop.set_zygote_modifier(my_zygote_mod, hook_name="my_rescue")
```

### 优先级（Priority）

当多个 Modifier 同时作用时，按优先级顺序应用：

```python
# 优先级低的先应用
pop.set_gamete_modifier(base_mod, priority=1, hook_name="base")
pop.set_gamete_modifier(drive_mod, priority=2, hook_name="drive")

# 执行顺序：base_mod → drive_mod
```

## 完整示例：CRISPR 基因驱动

```python
from natal.genetic_structures import Species
from natal.nonWF_population import AgeStructuredPopulation
from natal.hook_dsl import hook, Op

# 1. 定义遗传架构
sp = Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "drive_locus": ["WT", "Drive", "Resistance"]
        }
    }
)

# 2. 初始化种群（全 WT）
pop = AgeStructuredPopulation(
    species=sp,
    n_ages=8,
    initial_individual_count={
        "female": {"WT|WT": [0, 600, 600, 500, 400, 300, 200, 100]},
        "male": {"WT|WT": [0, 300, 300, 200, 100, 0, 0, 0]},
    },
    female_survival_rates=[1.0, 1.0, 5/6, 4/5, 3/4, 2/3, 1/2, 0],
    male_survival_rates=[1.0, 1.0, 2/3, 1/2, 0, 0, 0, 0],
    expected_eggs_per_female=100,
    gamete_labels=["default", "Cas9_deposited"],
)

# 3. 适应度设置
drive_wt = sp.get_genotype_from_str("Drive|WT")
pop.set_viability(drive_wt, 1.0)  # 正常存活

# 4. 配子修饰器：基因驱动
def crispr_drive(population):
    """Drive|WT 或 WT|Drive 的驱动配子比例 95%"""
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        },
    }

pop.set_gamete_modifier(crispr_drive, hook_name="CRISPR_drive")

# 5. 合子修饰器：Cas9 介导的重排（可选）
def cas9_induced_repair(population):
    """Cas9 介导的同源重定向修复"""
    return {
        # (Cas9_Drive_female, WT_male) → 大部分后代都有 Drive
        (("Drive", "Cas9_deposited"), ("WT", "default")): {
            "Drive|WT": 0.7,
            "WT|WT": 0.0,  # 被切割
            "Drive|Drive": 0.3,
        },
    }

pop.set_zygote_modifier(cas9_induced_repair, hook_name="Cas9_repair")

# 6. Hook：释放驱动型
@hook(event='first')
def release_drive():
    return [
        Op.add(
            genotypes='Drive|WT',
            ages=[2, 3, 4, 5],
            delta=100,
            when='tick == 20'
        )
    ]

release_drive.register(pop)

# 7. 运行
pop.run(n_steps=200, record_every=10)

# 8. 分析结果
print(f"最终种群: {pop.get_total_count():.0f}")
history = pop.get_history_as_objects()
for tick, state in history[-5:]:  # 最后 5 条记录
    print(f"Tick {tick}: {state.individual_count.sum():.0f}")
```

## 常见陷阱

### ❌ 陷阱 1：忘记基因型字符串

```python
# 错误
def wrong_modifier(pop):
    return {
        ("A1", "A2"): {  # ❌ 这是元组，不是基因型字符串
            ("A1", "default"): 0.5,
        }
    }

# 正确
def correct_modifier(pop):
    return {
        "A1|A2": {  # ✅ 基因型字符串
            ("A1", "default"): 0.5,
            ("A2", "default"): 0.5,
        }
    }
```

### ❌ 陷阱 2：概率不归一化

```python
# 错误（概率和 ≠ 1）
def wrong_modifier(pop):
    return {
        "A1|A2": {
            ("A1", "default"): 0.4,
            ("A2", "default"): 0.4,
            # 总计 0.8，不是 1.0！
        }
    }

# 正确
def correct_modifier(pop):
    return {
        "A1|A2": {
            ("A1", "default"): 0.5,
            ("A2", "default"): 0.5,  # 总计 1.0
        }
    }
```

### ❌ 陷阱 3：在 Modifier 中修改种群状态

```python
# 错误（Modifier 不应有副作用）
def wrong_modifier(pop):
    pop.state.individual_count[:] = 0  # ❌ 不要这样做！
    return {}

# 正确（Modifier 只返回映射修改）
def correct_modifier(pop):
    # 可以读取种群信息来做决定
    total = pop.get_total_count()
    
    # 但只返回映射修改
    if total > 1000:
        return {"A1|A2": {("A1", "default"): 0.6, ("A2", "default"): 0.4}}
    else:
        return {}
```

## 性能优化

### 尽量减少 Modifier 的复杂性

```python
# ❌ 低效（每次都遍历所有基因型）
def slow_modifier(pop):
    result = {}
    for gt in pop.species.get_all_genotypes():
        if "Drive" in str(gt):
            result[str(gt)] = {...}
    return result

# ✅ 高效（只指定需要修改的基因型）
def fast_modifier(pop):
    return {
        "Drive|WT": {...},
        "WT|Drive": {...},
    }
```

### 缓存预算计算

```python
# ❌ 每次都计算比例
def slow_modifier(pop):
    drive_freq = pop.get_allele_frequency("Drive")  # 每次计算
    return {"Drive|WT": {("Drive", "default"): drive_freq * 0.95, ...}}

# ✅ 预先计算（如果可能）
drive_freq = pop.get_allele_frequency("Drive")
def fast_modifier(pop):
    return {"Drive|WT": {("Drive", "default"): drive_freq * 0.95, ...}}
```

---

## 🎯 本章总结

| 类型 | 修改对象 | 典型应用 |
|------|---------|---------|
| **Gamete Modifier** | 基因型→配子 | 基因驱动、配子选择 |
| **Zygote Modifier** | 配子→合子 | CI、胚胎拯救 |
| **Gamete Label** | 配子标记 | 细胞质标记、Cas9 标记 |

**关键原理**：
1. 修饰器修改映射矩阵，影响遗传学计算
2. 返回格式为字典，格式严格
3. 概率必须归一化为 1.0
4. 支持多个修饰器层级应用

---

## 📚 相关章节

- [快速开始](01_quickstart.md) - Modifier 的基本使用
- [遗传结构与实体](02_genetic_structures.md) - Genotype 的字符串化
- [IndexRegistry 索引机制](05_index_registry.md) - 后台的对象→索引映射
- [Hook 系统](07_hooks.md) - 与 Hook 的配合

---

**准备使用 Hook 系统了吗？** [前往下一章：Hook 系统 →](07_hooks.md)

---

## 选择指南：Modifier vs Genetic Presets

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| **基因驱动** | ✅ Genetic Presets | 内置HomingDrive，参数化配置 |
| **简单突变** | ✅ Genetic Presets | 几行代码实现，自动处理底层细节 |
| **复杂自定义规则** | ⚖️ 两者皆可 | 预设提供框架，modifier提供完全控制 |
| **特殊转换逻辑** | ✅ Modifier | 需要手动控制映射矩阵时 |
| **性能优化** | ✅ Modifier | 直接操作索引，避免额外开销 |
| **教学/学习** | ✅ Genetic Presets | 更直观，隐藏底层复杂性 |

### 从Modifier迁移到Presets

如果你有现有的modifier函数，可以很容易地包装成预设：

```python
from natal.genetic_presets import GeneticPreset

class LegacyModifierPreset(GeneticPreset):
    def __init__(self, gamete_modifier_func=None, zygote_modifier_func=None):
        super().__init__(name="LegacyModifier")
        self.gamete_modifier_func = gamete_modifier_func
        self.zygote_modifier_func = zygote_modifier_func
    
    def gamete_modifier(self, population):
        return self.gamete_modifier_func(population) if self.gamete_modifier_func else None
    
    def zygote_modifier(self, population):
        return self.zygote_modifier_func(population) if self.zygote_modifier_func else None

# 使用现有的modifier函数
legacy_preset = LegacyModifierPreset(
    gamete_modifier_func=your_existing_gamete_modifier,
    zygote_modifier_func=your_existing_zygote_modifier
)

population.apply_preset(legacy_preset)
```

### Builder模式中的Modifier使用

现代NATAL推荐使用Builder模式创建种群，modifier可以在builder中配置：

```python
from natal.population_builder import AgeStructuredPopulationBuilder

def my_gamete_modifier(pop):
    return {"Drive|WT": {("Drive", "Cas9_deposited"): 0.95}}

def my_zygote_modifier(pop):
    return {(("Drive", "Cas9_deposited"), ("WT", "default")): {"Resistance|Resistance": 0.5}}

# 在Builder中配置modifier
pop = (AgeStructuredPopulationBuilder(species)
    .setup(name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .modifiers(
        gamete_modifiers=[(None, "my_gamete", my_gamete_modifier)],
        zygote_modifiers=[(None, "my_zygote", my_zygote_modifier)]
    )
    .build())
```

### 性能对比

- **Genetic Presets**: 轻微的性能开销（规则编译），但提供更好的抽象
- **直接Modifier**: 零额外开销，完全控制，但需要手动处理所有细节
- **实际差异**: 在大多数模拟中，差异可以忽略不计

### 总结

- **新手/常见任务**: 从Genetic Presets开始
- **高级用户/特殊需求**: 使用Modifier或组合两者
- **代码复用**: 将常用的modifier包装成预设
- **团队协作**: 预设提供更好的接口抽象
- **现代API**: 优先使用Builder模式创建种群

---

## 选择指南：Modifier vs Genetic Presets

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| **基因驱动** | ✅ Genetic Presets | 内置HomingDrive，参数化配置 |
| **简单突变** | ✅ Genetic Presets | 几行代码实现，自动处理底层细节 |
| **复杂自定义规则** | ⚖️ 两者皆可 | 预设提供框架，modifier提供完全控制 |
| **特殊转换逻辑** | ✅ Modifier | 需要手动控制映射矩阵时 |
| **性能优化** | ✅ Modifier | 直接操作索引，避免额外开销 |
| **教学/学习** | ✅ Genetic Presets | 更直观，隐藏底层复杂性 |

### 从Modifier迁移到Presets

如果你有现有的modifier函数，可以很容易地包装成预设：

```python
from natal.genetic_presets import GeneticPreset

class LegacyModifierPreset(GeneticPreset):
    def __init__(self, gamete_modifier_func=None, zygote_modifier_func=None):
        super().__init__(name="LegacyModifier")
        self.gamete_modifier_func = gamete_modifier_func
        self.zygote_modifier_func = zygote_modifier_func
    
    def gamete_modifier(self, population):
        return self.gamete_modifier_func(population) if self.gamete_modifier_func else None
    
    def zygote_modifier(self, population):
        return self.zygote_modifier_func(population) if self.zygote_modifier_func else None

# 使用现有的modifier函数
legacy_preset = LegacyModifierPreset(
    gamete_modifier_func=your_existing_gamete_modifier,
    zygote_modifier_func=your_existing_zygote_modifier
)

population.apply_preset(legacy_preset)
```

### 性能对比

- **Genetic Presets**: 轻微的性能开销（规则编译），但提供更好的抽象
- **直接Modifier**: 零额外开销，完全控制，但需要手动处理所有细节
- **实际差异**: 在大多数模拟中，差异可以忽略不计

### 总结

- **新手/常见任务**: 从Genetic Presets开始
- **高级用户/特殊需求**: 使用Modifier或组合两者
- **代码复用**: 将常用的modifier包装成预设
- **团队协作**: 预设提供更好的接口抽象

---

## 选择指南：Modifier vs Genetic Presets

| 场景 | 推荐方法 | 原因 |
|------|----------|------|
| **基因驱动** | ✅ Genetic Presets | 内置HomingDrive，参数化配置 |
| **简单突变** | ✅ Genetic Presets | 几行代码实现，自动处理底层细节 |
| **复杂自定义规则** | ⚖️ 两者皆可 | 预设提供框架，modifier提供完全控制 |
| **特殊转换逻辑** | ✅ Modifier | 需要手动控制映射矩阵时 |
| **性能优化** | ✅ Modifier | 直接操作索引，避免额外开销 |
| **教学/学习** | ✅ Genetic Presets | 更直观，隐藏底层复杂性 |

### 从Modifier迁移到Presets

如果你有现有的modifier函数，可以很容易地包装成预设：

```python
from natal.genetic_presets import GeneticPreset

class LegacyModifierPreset(GeneticPreset):
    def __init__(self, gamete_modifier_func=None, zygote_modifier_func=None):
        super().__init__(name="LegacyModifier")
        self.gamete_modifier_func = gamete_modifier_func
        self.zygote_modifier_func = zygote_modifier_func
    
    def gamete_modifier(self, population):
        return self.gamete_modifier_func(population) if self.gamete_modifier_func else None
    
    def zygote_modifier(self, population):
        return self.zygote_modifier_func(population) if self.zygote_modifier_func else None

# 使用现有的modifier函数
legacy_preset = LegacyModifierPreset(
    gamete_modifier_func=your_existing_gamete_modifier,
    zygote_modifier_func=your_existing_zygote_modifier
)

population.apply_preset(legacy_preset)
```

### 性能对比

- **Genetic Presets**: 轻微的性能开销（规则编译），但提供更好的抽象
- **直接Modifier**: 零额外开销，完全控制，但需要手动处理所有细节
- **实际差异**: 在大多数模拟中，差异可以忽略不计

### 总结

- **新手/常见任务**: 从Genetic Presets开始
- **高级用户/特殊需求**: 使用Modifier或组合两者
- **代码复用**: 将常用的modifier包装成预设
- **团队协作**: 预设提供更好的接口抽象