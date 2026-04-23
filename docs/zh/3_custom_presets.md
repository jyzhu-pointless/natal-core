# 设计你自己的预设

本部分将指导你从零开始设计、实现、验证和发布自定义遗传预设（Genetic Preset）。

## 1. 从等位基因转换规则开始

`GeneticPreset` 的设计过程始于遗传机制的清晰表达。对多数驱动系统而言，这一步通常体现在**等位基因转换规则**的制定。

### 定义机制目标

在编写任何代码之前，需要明确回答三个关键问题：

1. 哪个等位基因会转换（`from_allele`）？
2. 转成什么（`to_allele`）？
3. 转换概率是多少（`rate`）？

例如，一个最简驱动假设可以表述为：

- 在配子生成阶段，`W -> D`，概率 `0.5`。

### 规则对象与规则集

NATAL 提供两层结构来组织转换规则：

- `GameteAlleleConversionRule`：单条转换规则
- `GameteConversionRuleSet`：规则集合

可以将其理解为：

- Rule 是"一个句子"
- RuleSet 是"一个段落"

### 最简可用示例

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_convert(from_allele="W", to_allele="D", rate=0.5)
```

这个示例已经足以描述一个最简的转换机制。

### Zygote 转换规则（受精卵阶段）

等位基因转换还可以在受精卵（zygote）阶段发生，通常用于模拟以下机制：

- **基因驱动的修复**：在合子中表达的修复系统（例如 Cas9 切割修复）
- **等位基因特异性死亡**：某些基因型受精卵生活力降低
- **分生组后期转换**：发育过程中的等位基因转换

#### 从 Gamete 到 Zygote 的关键区别

| 阶段 | 输入 | 机制 | 适用场景 |
|------|------|------|---------|
| **Gamete** | 配子（单倍体）| 配子生成时的转换 | 配子驱动系统 |
| **Zygote** | 受精卵（二倍体）| 受精后立即的转换 | 合子驱动、合子修复 |

#### 使用 ZygoteConversionRuleSet

```python
from natal.gamete_allele_conversion import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# 在受精卵中，只要A位点含有D等位基因，就转换W->D
def has_d_at_a(genotype) -> bool:
    # 伪代码，实际取决于你的Genotype结构
    return "D" in str(genotype)

ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    genotype_filter=has_d_at_a,
)

zygote_mod = ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="zygote_repair")
```

#### Gamete + Zygote 的组合使用

通常驱动系统会同时使用两类规则：

```python
# 配子阶段：W -> D（偏向）
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_convert("W", "D", rate=0.99)

# 受精卵阶段：实现复制（确保纯和）
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_convert(
    "W", "D",
    rate=0.95,
    genotype_filter=lambda g: "D" in str(g)
)

pop.add_gamete_modifier(gamete_ruleset.to_gamete_modifier(pop))
pop.add_zygote_modifier(zygote_ruleset.to_zygote_modifier(pop))
```

### 设计规则时的注意事项

1. 从一条规则开始，不要一上来写十几条
2. 每增加一条规则，先跑 20-50 步检查方向是否符合预期
3. 记录"生物学假设 -> 参数值"映射，避免后续难以解释

### 基础模板

在开始设计复杂的转换规则之前，了解 `GeneticPreset` 的基础模板很重要：

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

实现要点：

1. **所有方法都是可选的** - 可以实现 1~3 个方法
2. **至少实现一个方法** - 否则预设不会有任何效果
3. **可以返回 None** - 表示该阶段不需要修饰
4. **支持延迟物种绑定** - 可以在创建时不指定 `Species`

### 简单示例

#### 简单点突变

```python
from natal.genetic_presets import GeneticPreset, PresetFitnessPatch
from natal.gamete_allele_conversion import GameteConversionRuleSet

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
```

#### 双向突变平衡

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

## 2. 用 genotype_filter 控制规则生效范围

在完成转换规则定义后，`genotype_filter` 用于解决一个关键问题：**同一条转换规则通常不应对所有基因型都生效**。

`genotype_filter` 将模式匹配的结果应用到规则作用范围中，实现规则的精准控制。

### 理解 genotype_filter

`genotype_filter` 是一个函数，接受 `Genotype` 作为输入，返回 `True` 或 `False`：

- 返回 `True`：规则对该基因型生效
- 返回 `False`：规则对该基因型不生效

```python
def my_filter(genotype):
    return True  # 或 False
```

### 核心示例：只在 W::D 杂合子中发生 W->D

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet


def is_wd_heterozygote(genotype) -> bool:
    name = str(genotype)
    return name in {"W|D", "D|W"}


ruleset = GameteConversionRuleSet("homing_drive")
ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    genotype_filter=is_wd_heterozygote,
)
```

这样就把机制作用范围明确地定义出来了。

### 常见过滤模式

- **携带某等位基因**：适合"只要携带 drive 就触发"的场景。
- **指定杂合/纯合**：适合"仅在杂合子切割"或"仅在纯合子生效"的场景。
- **组合逻辑**：可把多个过滤器组合成与/或/非逻辑，保持规则可读。

### 与模式匹配语法联动

当规则条件复杂时，建议直接复用第13章的 pattern 语法，而不是手写字符串包含判断。

```python
def build_filter_from_pattern(species, pattern: str):
    return species.parse_genotype_pattern(pattern)


ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    genotype_filter=build_filter_from_pattern(
        population.species,
        "A1/B1|A2/B2; C1/D1|C2/D2",
    ),
)
```

这样做的好处：

1. 语义统一：和 Observation 章节中的 pattern 展开规则一致
2. 可维护：pattern 可直接放进实验配置文件
3. 可测试：可以独立验证 pattern 命中集合

### 设计过滤器的实践建议

1. 过滤器应当"单一职责"
2. 先写最简单可读版本，再做性能优化
3. 对复杂过滤器做单元测试，避免误筛选
4. 在实验日志里记录过滤器名称和语义

### 与模式匹配结合（推荐做法）

当规则作用范围复杂时，建议使用物种提供的模式解析能力生成 `genotype_filter`，避免使用脆弱的字符串判断。

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

1. 在配置文件中维护 pattern 字符串
2. Preset 内部负责编译 pattern
3. Observation 分组也使用同一 pattern 或由同一 pattern 展开，确保统计口径与规则口径一致

### 条件突变（基因型依赖）

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

### 与 Observation 保持统计口径一致

推荐把同一个 pattern 同时用于：

1. Preset 的 `genotype_filter`（决定谁会被规则影响）
2. Observation 的 `groups["genotype"]`（决定统计谁）

若两边使用不同定义，常见症状是"规则看起来生效了，但观测指标不动"或"观测变化与机制预期不一致"。

### 调试方法

当过滤器效果不符合预期时，可以：

1. 打印过滤器命中结果
2. 检查 pattern 编译是否正确
3. 验证基因型字符串表示
4. 对比预期和实际的基因型集合

## 3. 封装、验证与发布前检查

本章是"设计自己的 Preset"主线的最后一章。在前两章中，你已经完成了：

1. 规则定义（Gamete 与 Zygote 转换）
2. 规则生效范围的精细化控制

本章将学习如何把这些内容封装为**可复用 Preset**，进行充分验证，最后发布使用。

### 封装成 Preset 的价值

如果只在脚本中编写规则，后期会遇到三个问题：

1. 难复用：每个实验都要复制逻辑
2. 难追溯：很难说清"这个版本到底用了哪组规则"
3. 难维护：规则、适应度、Hook 分散在多个文件

Preset 的价值就是把这些内容收敛成一个稳定配置单元。

### 推荐的 Preset 结构

一个实用 Preset 建议包含：

1. 机制规则（转换规则与过滤器）
2. 适应度补丁（如需要）
3. 可选参数（例如转换率、性别限制）
4. 清晰的名称与版本标记

### 示例：封装一个最小 DrivePreset

```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet


class DrivePreset(GeneticPreset):
    def __init__(self, conversion_rate: float = 0.5):
        super().__init__(name="DrivePreset")
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("drive_rules")

        def is_wd_heterozygote(genotype) -> bool:
            name = str(genotype)
            return name in {"W|D", "D|W"}

        ruleset.add_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            genotype_filter=is_wd_heterozygote,
        )

        return ruleset.to_gamete_modifier(population)
```

### 在 Builder 中应用 Preset

```python
pop = (
    AgeStructuredPopulationBuilder(species)
    .setup(name="DriveExperiment", stochastic=True)
    .age_structure(n_ages=8)
    .initial_state({...})
    .presets(DrivePreset(conversion_rate=0.55))
    .build()
)
```

这就是"Preset 作为配置组件"最推荐的接入方式。

### 验证清单（强烈建议）

在做大规模实验前，至少完成以下检查：

1. 机制检查：转换方向和目标等位基因是否正确
2. 过滤检查：`genotype_filter` 命中范围是否符合预期
3. 质量守恒检查：频率归一化是否成立
4. 对照检查：与无 Preset 的 baseline 对比趋势是否合理
5. 稳定性检查：更换随机种子后结论是否稳健

### 实验记录建议

建议把 Preset 配置写入实验元数据：

- Preset 名称
- 关键参数（如 `conversion_rate`）
- 代码版本或 commit
- 随机种子

这样可以显著降低"结果无法复现"的风险。

### 复杂基因驱动示例

```python
from natal.genetic_presets import GeneticPreset
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.zygote_allele_conversion import ZygoteConversionRuleSet

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
            maternal_glab="cas9"  # 需要母源Cas9沉积
        )

        return ruleset.to_zygote_modifier(population)

    def fitness_patch(self):
        return {
            "viability_per_allele": {
                "Drive": 0.9,      # 驱动等位基因成本
                "Resistance": 1.0   # 抗性等位基因中性
            },
            "fecundity_per_allele": {
                "Drive": 0.95
            },
            "zygote_per_allele": {
                "Drive": 0.8,     # 合子阶段生存率降低
                "Resistance": 1.0   # 抗性等位基因中性
            }
        }
```

### 常见错误与调试

#### 参数验证错误
- 验证转换率是否在 [0, 1] 范围内

#### 物种绑定错误
- 确保预设和种群使用相同的物种
- 使用延迟绑定（创建时不指定 `Species`）

#### 性能问题
- 避免在修饰器中创建大量临时对象
- 使用规则集缓存
- 考虑简化复杂的规则链

#### 调试技巧

```python
class DebugPreset(GeneticPreset):
    def gamete_modifier(self, population):
        print(f"应用预设到物种: {population.species.name}")
        print(f"可用等位基因: {list(population.species.gene_index.keys())}")

        # 创建修饰器并返回
        # ...
```

### 发布前检查清单

在发布 Preset 前建议完成：

- [ ] 单元测试覆盖主要功能
- [ ] 文档说明清晰完整
- [ ] 参数范围验证通过
- [ ] 与现有系统兼容性测试
- [ ] 性能基准测试

### 本章小结

🎉 恭喜！你已经完成了"设计自己的 Preset"的完整主线：

1. 规则定义（Gamete 与 Zygote 转换
2. 规则生效范围精细化（genotype_filter）
3. Preset 工程化、验证与发布

现在你已经掌握了从零开始设计、实现、验证和发布自定义 Preset 的完整流程。