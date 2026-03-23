# 等位基因转换规则 (Allele Conversion Rules)

## 概述

`GameteConversionRuleSet` 提供了一种**通用的、灵活的接口**来定义等位基因之间的转换规则。这些规则在配子生成时应用，直接覆盖现有的 zygote-to-gamete 映射。

核心思想：**定义从某个等位基因转换为另一个等位基因，以一定的概率在配子中发生**。

## 核心类

### 1. GameteAlleleConversionRule

单个转换规则的数据容器。

#### 基本用法

```python
from natal.gamete_allele_conversion import GameteAlleleConversionRule

# 最简单的形式：50% 概率从 A 转换为 B
rule = GameteAlleleConversionRule(
    from_allele="A",
    to_allele="B", 
    rate=0.5
)
```

#### 高级选项

```python
# 仅在雄性中应用
rule_male_only = GameteAlleleConversionRule(
    from_allele="A",
    to_allele="B",
    rate=0.7,
    sex_filter="male"  # "male", "female", 或 "both" (默认)
)

# 仅在特定基因型中应用
def my_filter(genotype: Genotype) -> bool:
    """自定义过滤器：仅在携带drive的个体中应用"""
    for hap in genotype.maternal.haplotypes:
        for locus in genotype.species.chromosomes[0].loci:
            gene = hap.get_gene_at_locus(locus)
            if gene and gene.name == "drive":
                return True
    return False

rule_het_only = GameteAlleleConversionRule(
    from_allele="A",
    to_allele="B",
    rate=0.5,
    genotype_filter=my_filter
)

# 自定义名称
rule_named = GameteAlleleConversionRule(
    from_allele="drive",
    to_allele="resistance",
    rate=0.8,
    name="drive_to_resistance_conversion"
)
```

### 2. GameteConversionRuleSet

管理多个转换规则的集合。

#### 创建规则集

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="my_drive_system")
```

#### 添加规则 - 方法 1：add_rule

```python
rule1 = GameteAlleleConversionRule(from_allele="A", to_allele="B", rate=0.5)
rule2 = GameteAlleleConversionRule(from_allele="B", to_allele="C", rate=0.3)

ruleset.add_rule(rule1).add_rule(rule2)  # 支持链式调用
```

#### 添加规则 - 方法 2：add_convert (推荐)

```python
ruleset.add_convert("A", "B", rate=0.5) \
       .add_convert("B", "C", rate=0.3) \
       .add_convert("C", "A", rate=0.1)
```

#### 转换为 GameteModifier

```python
# 将规则集转换为可以应用于种群的 GameteModifier
gamete_modifier = ruleset.to_gamete_modifier(population)

# 将其注册到种群
population.add_gamete_modifier(
    gamete_modifier,
    name="allele_conversions"
)
```

## 完整示例

### 场景：信使基因驱动 (Homing Gene Drive)

想象一个 CRISPR 型的信使基因驱动，在杂合的个体中：
- 50% 的配子中，野生型等位基因 (W) 转换为驱动等位基因 (D)
- 这导致驱动快速传播

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.population_builder import DiscreteGenerationPopulationBuilder
from natal.genetic_structures import Species

# 1. 定义物种
species = Species.from_dict(
    name="DriveSpecies",
    structure={"chr1": {"drive_locus": ["W", "D"]}}
)

# 2. 创建种群（使用现代Builder模式）
pop = (DiscreteGenerationPopulationBuilder(species)
    .setup(name="DrivePop", stochastic=True)
    .initial_state({
        "female": {"W|W": 500, "W|D": 200},
        "male": {"W|W": 500, "W|D": 200}
    })
    .build())

# 3. 定义转换规则
ruleset = GameteConversionRuleSet("homing_drive")
ruleset.add_convert(
    from_allele="W",      # 野生型
    to_allele="D",        # 驱动
    rate=0.5,
    sex_filter="both"     # 在所有性别中应用
)

# 4. 应用到种群
gamete_mod = ruleset.to_gamete_modifier(pop)
pop.add_gamete_modifier(gamete_mod, name="homing")

# 5. 运行模拟
pop.run(n_steps=100, record_every=10)
# ... 种群会在每代中看到 W -> D 的转换
```

### 场景：多向转换 (Multi-way conversion)

抑制因子系统，其中多个等位基因可以相互转换：

```python
ruleset = GameteConversionRuleSet("suppression_system")

# 配子中的转换链
ruleset.add_convert("A", "B", rate=0.6) \
       .add_convert("B", "C", rate=0.4) \
       .add_convert("C", "A", rate=0.2)  # 形成循环

gamete_mod = ruleset.to_gamete_modifier(population)
population.add_gamete_modifier(gamete_mod, name="cyclic_conversion")
```

### 场景：性别特异性转换

某些系统只在特定性别的配子中有效：

```python
ruleset = GameteConversionRuleSet("sex_specific_drive")

# 仅在雄性配子中的转换
ruleset.add_convert(
    "W", "male_drive",
    rate=0.8,
    sex_filter="male"
)

# 仅在雌性配子中的不同转换
ruleset.add_convert(
    "W", "female_suppressor",
    rate=0.5,
    sex_filter="female"
)

gamete_mod = ruleset.to_gamete_modifier(population)
population.add_gamete_modifier(gamete_mod, name="sex_specific")
```

## 工作原理

### 等位基因转换流程

1. **检查适用性**: 对每对 (sex, genotype)，检查是否有适用的规则
   - 通过 `sex_filter` 检查性别
   - 通过 `genotype_filter` 检查基因型

2. **计算配子频率**: 对于每个二倍体基因型，计算其产生的配子中的等位基因频率

3. **应用规则**: 对每条单倍体基因组应用第一个匹配的转换规则
   - 原始等位基因频率: `(1 - rate)`
   - 转换后的等位基因频率: `rate`

4. **返回结果**: 返回转换后的配子分布作为 GameteModifier

### 频率计算示例

假设一个杂合子 (D/W)：

**未应用规则时**:
- 50% 的配子是 D
- 50% 的配子是 W

**应用 W→D (rate=0.5) 后**:
- 原始 D 配子: 50% × 1.0 = 50%
- 原始 W 配子转换: 50% × 0.5 = 25%
- W 配子保留: 50% × 0.5 = 25%

**最终结果**:
- D 配子: 50% + 25% = 75%
- W 配子: 25%

## 集成到 GeneticPreset

虽然 `GameteConversionRuleSet` 是独立的，但你也可以将其集成到自定义的遗传修饰预设中：

```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet

class MyCustomDrive(GeneticPreset):
    def __init__(self):
        super().__init__(name="CustomDrive")
        self.conversion_ruleset = GameteConversionRuleSet()
        self.conversion_ruleset.add_convert("A", "B", rate=0.5)
    
    def gamete_modifier(self, population):
        # 使用转换规则创建修饰器
        return self.conversion_ruleset.to_gamete_modifier(population)
        
    def fitness_patch(self):
        # 可选：定义适应度效应
        return {
            "viability_allele": {"B": 0.95}  # B等位基因轻微有害
        }
```

# 使用现代API应用预设：
```python
from natal.population_builder import AgeStructuredPopulationBuilder
from natal.genetic_structures import Species

species = Species.from_dict("TestSpecies", {
    "chr1": {"test_locus": ["A", "B"]}
})

drive = MyCustomDrive()
pop = (AgeStructuredPopulationBuilder(species)
       .setup(name="TestPop", stochastic=False)
       .age_structure(n_ages=5)
       .initial_state({"female": {"A|A": 100}})
       .presets(drive)  # 应用预设
       .build())
```

## 性能考虑

1. **缓存**: 规则集在创建 GameteModifier 时进行一次编译，之后每代都快速应用
2. **复杂度**: O(n_rules × n_genotypes × n_sexes) 用于编译，然后查表应用
3. **建议**: 规则数量通常保持在 < 10 个，以获得最佳性能

## 当前限制

1. **单一等位基因转换**: 每条单倍体基因组仅应用第一个匹配的规则
2. **占位符实现**: `_convert_haploid_genotype()` 当前返回 None
   - 需要根据 HaploidGenotype 的具体结构来实现
3. **无频率检查**: 不检查转换规则是否冲突或概率非法

## 未来改进

- [ ] 支持多个等位基因的同时转换
- [ ] 基于频率的条件转换
- [ ] 与 zygote_modifier 的交互规则
- [ ] 规则优先级管理