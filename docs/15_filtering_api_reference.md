# 设计自己的 Preset（3）：封装、验证与发布前检查

本章继续主线，讲如何把前两章的规则与过滤逻辑封装为一个**可复用 Preset**。

## 1. 为什么要封装成 Preset

如果你只在一个脚本里写规则，后期会遇到三个问题：

1. 难复用：每个实验都要复制逻辑。
2. 难追溯：很难说清“这个版本到底用了哪组规则”。
3. 难维护：规则、适应度、Hook 分散在多个文件。

Preset 的价值就是把这些内容收敛成一个稳定配置单元。

## 2. 推荐的 Preset 结构

一个实用 Preset 建议包含：

1. 机制规则（转换规则与过滤器）。
2. 适应度补丁（如需要）。
3. 可选参数（例如转换率、性别限制）。
4. 清晰的名称与版本标记。

## 3. 示例：封装一个最小 DrivePreset

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

## 4. 在 Builder 中应用 Preset

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

这就是“Preset 作为配置组件”最推荐的接入方式。

## 5. 验证清单（强烈建议）

在做大规模实验前，至少完成以下检查：

1. 机制检查：转换方向和目标等位基因是否正确。
2. 过滤检查：`genotype_filter` 命中范围是否符合预期。
3. 质量守恒检查：频率归一化是否成立。
4. 对照检查：与无 Preset 的 baseline 对比趋势是否合理。
5. 稳定性检查：更换随机种子后结论是否稳健。

## 6. 实验记录建议

建议把 Preset 配置写入实验元数据：

- Preset 名称
- 关键参数（如 `conversion_rate`）
- 代码版本或 commit
- 随机种子

这样可以显著降低“结果无法复现”的风险。

## 7. 本章小结

现在你已经能把规则设计变成可复用 Preset，并在 Builder 中稳定应用。

下一章将给出主线最后一步：**如何用模式匹配把 Preset 做成可扩展的规则模板**。

---

## 下一章

- [设计自己的 Preset（4）：模式匹配与可扩展配置](16_genotype_pattern_matching_design.md)
