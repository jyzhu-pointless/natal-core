# 遗传预设使用指南 (Genetic Presets Guide)

本文档介绍如何使用和创建遗传预设（Genetic Presets），包括基因驱动、突变系统和其他遗传修饰。

## 概述

**遗传预设**（Genetic Presets）是NATAL框架中用于定义可重用遗传修饰的机制。预设可以：

- 修改配子生成规则（如基因驱动的过度分离）
- 改变合子发育过程（如胚胎抗性形成）
- 调整适应度参数（如驱动等位基因的成本）

## 在构建器中应用预设
```python
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="TestPop")
       .presets(preset1, preset2)  # 可以应用多个预设
       .build())
```

## 内置预设

### HomingDrive - 同源重组基因驱动

实现CRISPR/Cas9类型的同源重组基因驱动：

```python
from natal.genetic_presets import HomingDrive

# 创建基本的基因驱动
drive = HomingDrive(
    name="MyDrive",
    drive_allele="Drive",
    target_allele="WT",
    resistance_allele="Resistance",
    drive_conversion_rate=0.95,  # 95%的转换效率
    late_germline_resistance_formation_rate=0.03  # 3%形成抗性
)

# 应用到种群
population.apply_preset(drive)
```

#### 高级配置
```python
# 性别特异性参数
drive = HomingDrive(
    name="SexSpecificDrive",
    drive_allele="Drive",
    target_allele="WT",
    drive_conversion_rate={"female": 0.98, "male": 0.92},  # 性别差异
    late_germline_resistance_formation_rate=(0.02, 0.04),  # 元组形式 (female, male)
    embryo_resistance_formation_rate=0.01,
    functional_resistance_ratio=0.2,  # 20%的抗性等位基因是功能性的

    # 适应度成本
    viability_scaling=0.9,      # 10%生存力成本
    fecundity_scaling=0.95,     # 5%繁殖力成本
    sexual_selection_scaling=0.85  # 15%性选择劣势
)
```

### ToxinAntidoteDrive - 毒素-解毒剂驱动（TARE/TADE）

`ToxinAntidoteDrive` 用于建模“驱动等位基因触发目标位点破坏，破坏等位基因产生适应度损失，而驱动等位基因提供救援”的系统。

```python
from natal.genetic_presets import ToxinAntidoteDrive

ta_drive = ToxinAntidoteDrive(
    name="TARE_Drive",
    drive_allele="Drive",
    target_allele="WT",
    disrupted_allele="Disrupted",
    conversion_rate=0.95,
    embryo_disruption_rate={"female": 0.30, "male": 0.0},
    viability_scaling=0.0,
    fecundity_scaling=1.0,
    viability_mode="recessive",
    fecundity_mode="recessive",
    cas9_deposition_glab="cas9",
)

population.apply_preset(ta_drive)
```

常见参数说明：

1. `conversion_rate`：生殖系中 `target -> disrupted` 的转换概率，支持 `float`、`(female, male)` 或按性别字典。
2. `embryo_disruption_rate`：胚胎期转换概率，可与 `cas9_deposition_glab` / `use_paternal_deposition` 联合建模母源/父源沉积效应。
3. `viability_scaling` 与 `viability_mode`：用于定义 `disrupted` 等位基因的毒素效应；TARE 常用 `viability_scaling=0.0` 且 `viability_mode="recessive"`。
4. `fecundity_scaling` 与 `fecundity_mode`：定义繁殖力成本。
5. `sexual_selection_scaling`（可选）：定义性选择效应；支持标量或二元组 `(default_male, carrier_male)`，配合 `sexual_selection_mode` 使用。

示例：加入性选择成本

```python
ta_drive_with_mating_cost = ToxinAntidoteDrive(
    name="TA_WithMatingCost",
    drive_allele="Drive",
    target_allele="WT",
    disrupted_allele="Disrupted",
    sexual_selection_scaling=(1.0, 0.8),
    sexual_selection_mode="dominant",
)
```

## 创建自定义预设

### 与模式匹配结合（强烈推荐）

当规则作用范围复杂时，建议不要用 `lambda gt: "X" in str(gt)` 这类脆弱字符串判断，而是使用物种提供的模式解析能力生成 `genotype_filter`。

```python
class PatternBasedPreset(GeneticPreset):
    def __init__(self, pattern: str, conversion_rate: float = 0.95):
        super().__init__(name="PatternBasedPreset")
        self.pattern = pattern
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, population):
        from natal.gamete_allele_conversion import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("PatternBased")
        pattern_filter = population.species.parse_genotype_pattern(self.pattern)

        ruleset.add_convert(
            from_allele="WT",
            to_allele="Drive",
            rate=self.conversion_rate,
            genotype_filter=pattern_filter,
        )
        return ruleset.to_gamete_modifier(population)
```

实践建议：

1. 在配置文件中维护 pattern 字符串。
2. Preset 内部负责编译 pattern。
3. Observation 分组也使用同一 pattern 或由同一 pattern 展开，确保统计口径与规则口径一致。

### 基础模板

所有自定义预设都应该继承自`GeneticPreset`：

```python
from natal.genetic_presets import GeneticPreset, PresetFitnessPatch
from natal.modifiers import GameteModifier, ZygoteModifier
from typing import Optional

class MyCustomPreset(GeneticPreset):
    """自定义遗传修饰预设"""

    def __init__(self, name: str = "MyCustom", species=None):
        super().__init__(name=name, species=species)
        # 自定义参数
        self.custom_param = 0.5

    def gamete_modifier(self, population) -> Optional[GameteModifier]:
        """定义配子阶段的修饰逻辑"""
        # 返回GameteModifier或None
        return None

    def zygote_modifier(self, population) -> Optional[ZygoteModifier]:
        """定义合子阶段的修饰逻辑"""
        # 返回ZygoteModifier或None
        return None

    def fitness_patch(self) -> PresetFitnessPatch:
        """定义适应度效应"""
        # 返回适应度配置字典或None
        return None
```

### 实现要点

1. **所有方法都是可选的** - 可以实现1-3个方法
2. **至少实现一个方法** - 否则预设不会有任何效果
3. **可以返回None** - 表示该阶段不需要修饰
4. **支持延迟物种绑定** - 可以在创建时不指定species

## 实用示例

### 示例1：简单点突变
```python
from natal.genetic_presets import GeneticPreset, PresetFitnessPatch
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.population_builder import AgeStructuredPopulationBuilder

class PointMutation(GeneticPreset):
    """简单点突变：WT以一定频率突变为Mutant"""

    def __init__(self, mutation_rate: float = 1e-5):
        super().__init__(name="PointMutation")
        self.mutation_rate = mutation_rate

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("PointMutation")
        ruleset.add_convert("WT", "Mutant", rate=self.mutation_rate)
        return ruleset.to_gamete_modifier(population)

    def fitness_patch(self):
        return {
            "viability_allele": {"Mutant": 0.98}  # 轻微有害
        }

# 使用示例
species = Species.from_dict("TestSpecies", {
    "chr1": {"GeneA": ["WT", "Mutant"]}
})

mutation = PointMutation(mutation_rate=1e-5)

pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="MutationTest", stochastic=False)
       .age_structure(n_ages=5)
       .initial_state({"female": {"WT|WT": [0, 100, 100, 100, 100]}})
       .presets(mutation)
       .build())
```

### 示例2：双向突变平衡
```python
class BidirectionalMutation(GeneticPreset):
    """双向突变平衡"""

    def __init__(self, forward_rate: float = 1e-5, backward_rate: float = 1e-6):
        super().__init__(name="BidirectionalMutation")
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

    def gamete_modifier(self, population):
        from natal.gamete_allele_conversion import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("BidirectionalMutation")

        # A → B (正向突变)
        ruleset.add_convert("A", "B", rate=self.forward_rate)
        # B → A (回复突变)
        ruleset.add_convert("B", "A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(population)
```

### 示例3：条件突变（基因型依赖）
```python
class ConditionalMutation(GeneticPreset):
    """条件突变 - 只在特定基因型背景下发生"""

    def __init__(self, target_allele: str = "B", required_background: str = "A"):
        super().__init__(name="ConditionalMutation")
        self.target_allele = target_allele
        self.required_background = required_background

    def gamete_modifier(self, population):
        from natal.gamete_allele_conversion import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("ConditionalMutation")

        # 只在携带背景等位基因时才发生突变
        ruleset.add_convert(
            from_allele=self.target_allele,
            to_allele=f"{self.target_allele}_mutant",
            rate=1e-4,
            genotype_filter=lambda gt: self.required_background in str(gt)
        )

        return ruleset.to_gamete_modifier(population)
```

### 示例4：复杂基因驱动
```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.zygote_allele_conversion import ZygoteConversionRuleSet
from natal.population_builder import AgeStructuredPopulationBuilder

class ComplexDrive(GeneticPreset):
    """复杂基因驱动，包含多个阶段的转换"""

    def __init__(self):
        super().__init__(name="ComplexDrive")

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("ComplexDrive")

        # 阶段1: 驱动转换 (WT → Drive)
        ruleset.add_convert("WT", "Drive", rate=0.95,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        # 阶段2: 抗性形成 (剩余WT → Resistance)
        ruleset.add_convert("WT", "Resistance", rate=0.05,
                           genotype_filter=lambda gt: "Drive" in str(gt))

        return ruleset.to_gamete_modifier(population)

    def zygote_modifier(self, population):
        ruleset = ZygoteConversionRuleSet("ComplexDrive_Embryo")

        # 胚胎阶段的额外修饰
        ruleset.add_convert(
            from_allele="WT",
            to_allele="Resistance",
            rate=0.02,
            maternal_glab="Cas9_deposited"  # 需要母源Cas9沉积
        )

        return ruleset.to_zygote_modifier(population)

    def fitness_patch(self):
        return {
            "viability_allele": {
                "Drive": 0.9,      # 驱动等位基因成本
                "Resistance": 1.0   # 抗性等位基因中性
            },
            "fecundity_allele": {
                "Drive": 0.95
            }
        }

# 使用示例
species = Species.from_dict("DriveSpecies", {
    "chr1": {"drive_locus": ["WT", "Drive", "Resistance"]}
})

complex_drive = ComplexDrive()

pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="ComplexDrivePop", stochastic=False)
       .age_structure(n_ages=8)
       .initial_state({
           "female": {"WT|WT": [0, 500, 500, 400, 300, 200, 100, 50]},
           "male": {"WT|WT": [0, 250, 250, 200, 150, 100, 50, 25]}
       })
       .presets(complex_drive)
       .build())
```

## 适应度配置详解

### 适应度效应类型

```python
def fitness_patch(self):
    return {
        # 1. 基因型特异性适应度
        "viability": {
            "Drive|Drive": 0.8,      # 特定基因型
            "Drive|WT": 0.9,
            "WT|WT": 1.0
        },

        # 2. 等位基因驱动的适应度（推荐）
        "viability_allele": {
            "Drive": 0.9,            # 按等位基因拷贝数倍增
            "Resistance": 1.0
        },

        # 3. 繁殖力效应
        "fecundity_allele": {
            "Drive": 0.95            # 仅影响雌性
        },

        # 4. 性选择效应
        "sexual_selection_allele": {
            "Drive": (1.0, 0.8)      # (默认雄性, 携带者选择)
        }
    }
```

### 性别和年龄特异性

```python
def fitness_patch(self):
    return {
        # 性别特异性
        "viability_allele": {
            "Drive": {
                "female": 0.95,      # 雌性中
                "male": 0.85         # 雄性中更严重
            }
        },

        # 年龄特异性
        "viability_allele": {
            "Drive": {
                0: 1.0,               # 年龄0（幼体）
                1: 0.95,              # 年龄1
                2: 0.90               # 年龄2+
            }
        },

        # 组合：性别+年龄
        "viability_allele": {
            "Drive": {
                "female": {0: 0.98, 1: 0.96, 2: 0.94},
                "male": {0: 0.92, 1: 0.88, 2: 0.84}
            }
        }
    }
```

## 高级主题

### 多个预设的组合

```python
# 创建多个预设
mutation = PointMutation(mutation_rate=1e-5)
drive = HomingDrive(name="GeneDrive", drive_allele="Drive", target_allele="WT")
selection = FitnessSelection(target_allele="Deleterious", cost=0.1)

# 同时应用多个预设
population.apply_preset(mutation, drive, selection)

# 或在构建器中
pop = (DiscreteGenerationPopulationBuilder(species)
       .setup(name="ComplexModel")
       .presets(mutation, drive, selection)
       .build())
```

### 预设的顺序和交互

多个预设按应用顺序执行：
1. 配子修饰器按顺序应用
2. 合子修饰器按顺序应用
3. 适应度补丁合并应用（后面的覆盖前面的）

### 性能优化

```python
class OptimizedPreset(GeneticPreset):
    """性能优化的预设实现"""

    def __init__(self):
        super().__init__(name="OptimizedPreset")
        # 预计算常用数据
        self._precomputed_rules = None

    def gamete_modifier(self, population):
        # 缓存规则集避免重复创建
        if self._precomputed_rules is None:
            from natal.gamete_allele_conversion import GameteConversionRuleSet
            self._precomputed_rules = GameteConversionRuleSet("Optimized")
            # ... 添加规则

        return self._precomputed_rules.to_gamete_modifier(population)
```

## 故障排除

### 常见问题

1. **预设没有效果**
   - 检查是否所有方法都返回None
   - 确认等位基因名称拼写正确
   - 验证转换率是否在0-1范围内

2. **物种绑定错误**
   - 确保预设和种群使用相同的物种
   - 使用延迟绑定（创建时不指定species）

3. **性能问题**
   - 避免在修饰器中创建大量临时对象
   - 使用规则集缓存
   - 考虑简化复杂的规则链

### 调试技巧

```python
class DebugPreset(GeneticPreset):
    def gamete_modifier(self, population):
        print(f"应用预设到物种: {population.species.name}")
        print(f"可用等位基因: {list(population.species.gene_index.keys())}")

        # 创建修饰器并返回
        # ...
```

## 相关文档

- [等位基因转换规则](allele_conversion_rules.md) - 详细的转换规则系统
- [模式匹配与可扩展配置](genotype_pattern_matching_design.md) - 语法规则与 pattern 设计
- [种群观测规则](observation_rules.md) - pattern 在观察分组中的使用
- [Modifier机制](modifiers.md) - 底层修饰器原理
- [快速开始](quickstart.md) - 基础使用教程

## 参考实现

查看测试文件获取完整示例：
- `tests/test_recipe_species_binding.py` - 物种绑定测试
- `tests/test_recipe_fitness_patch.py` - 适应度补丁测试

