# 设计自己的 Preset（1）：从等位基因转换规则开始

`GeneticPreset` 的设计过程始于遗传机制的清晰表达。对多数驱动系统而言，这一步通常体现在**等位基因转换规则**的制定。

## 定义机制目标

在编写任何代码之前，需要明确回答三个关键问题：

1. 哪个等位基因会转换（`from_allele`）？
2. 转成什么（`to_allele`）？
3. 转换概率是多少（`rate`）？

例如，一个最简驱动假设可以表述为：

- 在配子生成阶段，`W -> D`，概率 `0.5`。

## 规则对象与规则集

NATAL 提供两层结构来组织转换规则：

- `GameteAlleleConversionRule`：单条转换规则
- `GameteConversionRuleSet`：规则集合

可以将其理解为：

- Rule 是"一个句子"
- RuleSet 是"一个段落"

## 最简可用示例

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_convert(from_allele="W", to_allele="D", rate=0.5)
```

这个示例已经足以描述一个最简的转换机制。

## Zygote 转换规则（受精卵阶段）

等位基因转换还可以在受精卵（zygote）阶段发生，通常用于模拟以下机制：

- **基因驱动的修复**：在合子中表达的修复系统（例如 Cas9 切割修复）
- **等位基因特异性死亡**：某些基因型受精卵生活力降低
- **分生组后期转换**：发育过程中的等位基因转换

### 从 Gamete 到 Zygote 的关键区别

| 阶段 | 输入 | 机制 | 适用场景 |
|------|------|------|---------|
| **Gamete** | 配子（单倍体）| 配子生成时的转换 | 配子驱动系统 |
| **Zygote** | 受精卵（二倍体）| 受精后立即的转换 | 合子驱动、合子修复 |

### 使用 ZygoteConversionRuleSet

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

### Gamete + Zygote 的组合使用

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

## 设计规则时的注意事项

1. 从一条规则开始，不要一上来写十几条
2. 每增加一条规则，先跑 20-50 步检查方向是否符合预期
3. 记录"生物学假设 -> 参数值"映射，避免后续难以解释

## 基础模板

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

## 简单示例

### 简单点突变

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

### 双向突变平衡

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

## 本章小结

你已经完成 Preset 设计的第一步：定义等位基因转换规则。下一步将学习如何控制规则的作用范围。
