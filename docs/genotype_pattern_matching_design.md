# 设计自己的 Preset（4）：模式匹配与可扩展配置

这是“设计自己的 Preset”主线最后一章。

前面三章已经完成：

1. 规则定义。
2. 规则筛选。
3. Preset 封装。

本章解决进一步扩展时最常见的问题：

**当目标基因型越来越多，如何避免写大量硬编码判断？**

答案是：引入模式匹配思路。

## 1. 为什么需要模式匹配

当系统复杂到以下程度时，硬编码会迅速失控：

- 多染色体、多位点。
- 多个等位基因组合。
- 需要按“某类基因型集合”批量施加规则。

模式匹配可以把“具体基因型列表”提升为“可读规则表达式”，从而：

1. 降低规则数量。
2. 提高可读性。
3. 提高后续扩展效率。

## 2. 设计模式语法时的原则

如果你准备做自定义语法，建议遵循三条原则：

1. 少而稳：先支持最常用表达，再逐步扩展。
2. 可解释：表达式能被团队成员快速读懂。
3. 可测试：每个语法单元都有明确测试用例。

## 3. 与 Preset 的结合方式

推荐把“模式解析”放在 Preset 内部，而不是散落在脚本里。

一个实用结构是：

1. Preset 参数接收 pattern 字符串或 pattern 列表。
2. 初始化阶段将 pattern 编译为过滤函数。
3. 规则执行时直接调用编译后的过滤函数。

这样运行期更稳定，调试也更清晰。

## 4. 示例：参数化 Preset

```python
class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def _build_filter(self, species):
        # 这里可接入你自己的 pattern parser
        # 返回 Callable[[Genotype], bool]
        return species.parse_genotype_pattern(self.target_pattern)

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("pattern_rules")
        pattern_filter = self._build_filter(population.species)

        ruleset.add_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            genotype_filter=pattern_filter,
        )
        return ruleset.to_gamete_modifier(population)
```

这个结构让 Preset 从“固定规则”升级为“可配置规则模板”。

## 5. 测试策略

模式匹配相关功能建议按三层测试：

1. 语法层：每个模式输入是否正确解析。
2. 过滤层：每个模式对应的基因型命中是否正确。
3. 集成层：Preset 应用后群体轨迹是否符合预期方向。

## 6. 常见风险

1. 语法过度复杂，导致团队难以维护。
2. 模式命中范围过宽，造成意外规则扩散。
3. 缺少回归测试，后续修改解析器后结果漂移。

避免这些问题的关键是：

- 维持最小语法集。
- 为每种模式保留示例与测试。
- 把 pattern 与实验配置一起归档。

## 7. 主线总结：如何设计一个自己的 Preset

可以用四步完成：

1. 把机制写成转换规则。
2. 用过滤器限定生效范围。
3. 封装为可复用 Preset。
4. 用模式匹配提升可配置性。

这四步能覆盖从“快速原型”到“稳定复现实验”的大多数需求。

---

## 回看与延伸

- [设计自己的 Preset（1）：从等位基因转换规则开始](allele_conversion_rules.md)
- [设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围](genotype_filter_implementation.md)
- [设计自己的 Preset（3）：封装、验证与发布前检查](filtering_api_reference.md)
- [遗传预设使用指南](genetic_presets.md)
