# 设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围

上一章你完成了转换规则的定义。本章继续主线，解决一个关键问题：

**同一条转换规则，通常不该对所有基因型都生效。**

这就是 `genotype_filter` 的作用——把模式匹配的结果应用到规则作用范围中。

## 1. genotype_filter 是什么

`genotype_filter` 是一个函数：

- 输入：`Genotype`
- 输出：`True` 或 `False`

当返回 `True` 时，规则对该基因型生效；返回 `False` 时，不生效。

```python
def my_filter(genotype):
    return True  # 或 False
```

## 2. 主线示例：只在 W|D 杂合子中发生 W->D

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

这样就把“机制作用人群”明确出来了。

## 3. 常见过滤模式

### 3.1 携带某等位基因

适合“只要携带 drive 就触发”的场景。

### 3.2 指定杂合/纯合

适合“仅在杂合子切割”或“仅在纯合子生效”的场景。

### 3.3 组合逻辑

可把多个过滤器组合成与/或/非逻辑，保持规则可读。

## 4. 与模式匹配语法联动（第13章）

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

1. 语义统一：和 Observation 章节中的 pattern 展开规则一致。
2. 可维护：pattern 可直接放进实验配置文件。
3. 可测试：可以独立验证 pattern 命中集合。

## 5. 设计过滤器的实践建议

1. 过滤器应当“单一职责”。
2. 先写最简单可读版本，再做性能优化。
3. 对复杂过滤器做单元测试，避免误筛选。
4. 在实验日志里记录过滤器名称和语义。

## 6. 与 Observation 保持统计口径一致

推荐把同一个 pattern 同时用于：

1. Preset 的 `genotype_filter`（决定谁会被规则影响）。
2. Observation 的 `groups["genotype"]`（决定统计谁）。

若两边使用不同定义，常见症状是“规则看起来生效了，但观测指标不动”或“观测变化与机制预期不一致”。

## 7. 调试方法

建议在小样本下做“命中率检查”：

1. 枚举当前关注基因型。
2. 对每个基因型打印 `genotype_filter` 结果。
3. 核对是否与生物学预期一致。

这一步往往能提前避免大量无效模拟。

## 8. 让规则更可维护

当规则增多时，建议把过滤器按语义拆分：

- `is_drive_carrier`
- `is_target_heterozygote`
- `is_male_specific_target`

再通过组合函数构建最终过滤器，而不是写一个巨型函数。

## 9. 本章小结

你已经完成 Preset 设计的核心两步：

1. 定义规则（第 1 章）。
2. 精细化规则生效范围（这一章）。

下一章将把这些内容工程化，讲如何把规则与过滤封装成可复用的 Preset 类。

---

## 下一章

- [设计自己的 Preset（3）：封装、验证与发布前检查](preset_encapsulation_and_validation.md)
- [模式匹配与可扩展配置](genotype_pattern_matching_design.md)
- [种群观测规则](observation_rules.md)
