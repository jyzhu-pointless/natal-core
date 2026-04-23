# 设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围

在完成转换规则定义后，`genotype_filter` 用于解决一个关键问题：**同一条转换规则通常不应对所有基因型都生效**。

`genotype_filter` 将模式匹配的结果应用到规则作用范围中，实现规则的精准控制。

## 理解 genotype_filter

`genotype_filter` 是一个函数，接受 `Genotype` 作为输入，返回 `True` 或 `False`：

- 返回 `True`：规则对该基因型生效
- 返回 `False`：规则对该基因型不生效

```python
def my_filter(genotype):
    return True  # 或 False
```

## 核心示例：只在 W::D 杂合子中发生 W->D

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

## 常见过滤模式

### 携带某等位基因
适合"只要携带 drive 就触发"的场景。

### 指定杂合/纯合
适合"仅在杂合子切割"或"仅在纯合子生效"的场景。

### 组合逻辑
可把多个过滤器组合成与/或/非逻辑，保持规则可读。

## 与模式匹配语法联动

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

## 设计过滤器的实践建议

1. 过滤器应当"单一职责"
2. 先写最简单可读版本，再做性能优化
3. 对复杂过滤器做单元测试，避免误筛选
4. 在实验日志里记录过滤器名称和语义

## 与模式匹配结合（推荐做法）

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

## 条件突变（基因型依赖）

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

## 与 Observation 保持统计口径一致

推荐把同一个 pattern 同时用于：

1. Preset 的 `genotype_filter`（决定谁会被规则影响）
2. Observation 的 `groups["genotype"]`（决定统计谁）

若两边使用不同定义，常见症状是"规则看起来生效了，但观测指标不动"或"观测变化与机制预期不一致"。

## 调试方法

当过滤器效果不符合预期时，可以：

1. 打印过滤器命中结果
2. 检查 pattern 编译是否正确
3. 验证基因型字符串表示
4. 对比预期和实际的基因型集合

## 本章小结

通过 `genotype_filter`，你可以精确控制转换规则的作用范围。下一章将学习如何将这些规则封装为可复用的 Preset。
