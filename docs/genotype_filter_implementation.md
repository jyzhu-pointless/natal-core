# 设计自己的 Preset（3）：用 genotype_filter 控制规则生效范围

上两章你已经学了：

1. 如何定义转换规则（第 1 章）。
2. 如何用模式匹配精确描述基因型（第 2 章）。

本章继续主线，把两者结合起来，解决一个关键问题：

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

## 4. 设计过滤器的实践建议

1. 过滤器应当“单一职责”。
2. 先写最简单可读版本，再做性能优化。
3. 对复杂过滤器做单元测试，避免误筛选。
4. 在实验日志里记录过滤器名称和语义。

## 5. 调试方法

建议在小样本下做“命中率检查”：

1. 枚举当前关注基因型。
2. 对每个基因型打印 `genotype_filter` 结果。
3. 核对是否与生物学预期一致。

这一步往往能提前避免大量无效模拟。

## 6. 让规则更可维护

当规则增多时，建议把过滤器按语义拆分：

- `is_drive_carrier`
- `is_target_heterozygote`
- `is_male_specific_target`

再通过组合函数构建最终过滤器，而不是写一个巨型函数。

## 7. 本章小结

你已经掌握：

1. 定义规则（第 1 章）。
2. 用模式描述基因型（第 2 章）。
3. 用过滤器精化规则的作用范围（这一章）。

下一章将把这些内容工程化，讲如何把规则与过滤封装成可复用的 Preset 类。

---

## 下一章

- [设计自己的 Preset（4）：封装、验证与发布前检查](preset_encapsulation_and_validation.md)
