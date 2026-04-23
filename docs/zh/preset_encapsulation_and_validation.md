# 设计自己的 Preset（3）：封装、验证与发布前检查

本章是"设计自己的 Preset"主线的最后一章。在前两章中，你已经完成了：

1. 规则定义（Gamete 与 Zygote 转换）
2. 规则生效范围的精细化控制

本章将学习如何把这些内容封装为**可复用 Preset**，进行充分验证，最后发布使用。

## 封装成 Preset 的价值

如果只在脚本中编写规则，后期会遇到三个问题：

1. 难复用：每个实验都要复制逻辑
2. 难追溯：很难说清"这个版本到底用了哪组规则"
3. 难维护：规则、适应度、Hook 分散在多个文件

Preset 的价值就是把这些内容收敛成一个稳定配置单元。

## 推荐的 Preset 结构

一个实用 Preset 建议包含：

1. 机制规则（转换规则与过滤器）
2. 适应度补丁（如需要）
3. 可选参数（例如转换率、性别限制）
4. 清晰的名称与版本标记

## 示例：封装一个最小 DrivePreset

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

## 在 Builder 中应用 Preset

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

## 验证清单（强烈建议）

在做大规模实验前，至少完成以下检查：

1. 机制检查：转换方向和目标等位基因是否正确
2. 过滤检查：`genotype_filter` 命中范围是否符合预期
3. 质量守恒检查：频率归一化是否成立
4. 对照检查：与无 Preset 的 baseline 对比趋势是否合理
5. 稳定性检查：更换随机种子后结论是否稳健

## 实验记录建议

建议把 Preset 配置写入实验元数据：

- Preset 名称
- 关键参数（如 `conversion_rate`）
- 代码版本或 commit
- 随机种子

这样可以显著降低"结果无法复现"的风险。

## 复杂基因驱动示例

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

## 常见错误与调试

### 参数验证错误
- 验证转换率是否在 [0, 1] 范围内

### 物种绑定错误
- 确保预设和种群使用相同的物种
- 使用延迟绑定（创建时不指定 `Species`）

### 性能问题
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

## 发布前检查清单

在发布 Preset 前建议完成：

- [ ] 单元测试覆盖主要功能
- [ ] 文档说明清晰完整
- [ ] 参数范围验证通过
- [ ] 与现有系统兼容性测试
- [ ] 性能基准测试

## 本章小结

🎉 恭喜！你已经完成了"设计自己的 Preset"的完整主线：

1. 规则定义（Gamete 与 Zygote 转换
2. 规则生效范围精细化（genotype_filter）
3. Preset 工程化、验证与发布

现在你已经掌握了从零开始设计、实现、验证和发布自定义 Preset 的完整流程。
