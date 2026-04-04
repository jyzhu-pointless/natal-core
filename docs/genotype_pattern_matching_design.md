# 模式匹配与可扩展配置

本章解决一个关键基础问题：

**如何用高效、可读、可维护的方式描述和过滤基因型？**

模式匹配是 NATAL 的基础能力，既用于 Observation（观察与聚合）中的分组定义，也用于 Preset 规则里的 `genotype_filter`。如果两边语义不一致，就会出现“规则作用范围”和“观测统计范围”不一致的问题。

## 1. 为什么需要模式匹配

当模型复杂到以下程度时，硬编码基因型列表会迅速失控：

- 多染色体、多位点。
- 大量等位基因组合。
- 需要按“某类基因型集合”批量定义规则或观察分组。

模式匹配把“显式枚举基因型列表”升级为“语义表达式”，收益是：

1. 规则数量更少。
2. 可读性更高。
3. 与实验配置（YAML/JSON）更容易绑定。

## 2. 语法规则总览（先看这一节）

模式字符串按“从外到内”三层解析：

1. 染色体层：用 `;` 分隔多个染色体段。
2. 同源染色体层：每段必须包含 `|` 或 `::`。
3. 染色体位点层：每条同源染色体拷贝内部用 `/` 分隔位点模式。

可先记住骨架：

`<chr1_hap1>/<...>|<chr1_hap2>/<...>; <chr2_hap1>/<...>|<chr2_hap2>/<...>`

### 2.1 分隔符含义

| 语法元素 | 含义 | 示例 |
|---|---|---|
| `;` | 分隔不同染色体段 | `A/B|C/D; E/F|G/H` |
| `|` | 有序匹配：`Maternal|Paternal` | `A/B|C/D` |
| `::` | 无序匹配：同源染色体两条拷贝可交换 | `A/B::C/D` |
| `/` | 分隔单条同源染色体拷贝内部位点 | `A/B/C` |

### 2.2 位点原子模式

| 模式 | 含义 | 示例 |
|---|---|---|
| `X` | 精确匹配等位基因 `X` | `A1` |
| `*` | 通配任意等位基因 | `*` |
| `{A,B,C}` | 枚举集合之一 | `{A1,A2}` |
| `!X` | 非 `X` | `!A1` |

### 2.3 组合示例

1. 精确匹配：`A1/B1|A2/B2; C1/D1|C2/D2`
2. 通配混合：`A1/*|A2/B2; */D1|C2/*`
3. 集合匹配：`{A1,A2}/B1|A3/B2; C1/D1|C2/D2`
4. 否定匹配：`!A1/B1|A2/B2; C1/D1|C2/D2`

### 2.4 常见错误与修正

1. 错误：`Chromosome pattern must contain '|' or '::'`
原因：某个染色体段未写同源染色体双拷贝分隔。
修正：不要写 `C1/C1`，改成完整的 `...|...` 或 `...::...` 形式。

2. 错误：染色体段数量不匹配。
原因：`;` 分段数和物种染色体数不同。
修正：按物种定义逐段补齐。

3. 错误：位点数量不匹配。
原因：`/` 分隔后的位点模式数量和该染色体位点数不同。
修正：逐位点补齐，或使用 `*` 占位。

## 3. 与 Observation 的结合方式

Observation 章节里的 `groups["genotype"]` 已经直接走 GeneticPattern 解析。推荐直接传入 pattern 字符串，而不是手写长列表。

推荐流程：

1. 在实验配置中维护 pattern 字符串。
2. 直接把 pattern 字符串放入 `groups["genotype"]`。
3. 由 Observation 内部统一解析与匹配。

```python
groups = {
    "target_group": {
        # 有序匹配：Maternal|Paternal
        "genotype": "A1/B1|A2/B2; C1/D1|C2/D2",
        "sex": "female",
    },
    "target_group_unordered": {
        # 无序匹配：同源染色体两条拷贝可交换
        "genotype": "A1/B1::A2/B2; C1/D1::C2/D2",
        "sex": "female",
    }
}
```

如需调试命中集合，再使用 `species.enumerate_genotypes_matching_pattern(...)` 做离线展开检查。

这样可确保 Observation 与 Preset 复用同一语义来源。

## 4. 与 Preset 的结合方式

推荐把“模式解析”放在 Preset 内部，而不是散落在脚本里。

实用结构：

1. Preset 接收 pattern 参数。
2. 初始化/绑定阶段把 pattern 编译为过滤函数。
3. 在 `add_convert(..., genotype_filter=...)` 中复用。

## 5. 示例：参数化 Preset

```python
class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def _build_filter(self, species):
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

## 6. 测试策略

建议按三层测试：

1. 语法层：pattern 能否正确解析。
2. 过滤层：pattern 对各基因型命中是否符合预期。
3. 集成层：规则应用后总体轨迹方向是否合理。

## 7. 常见风险

1. 语法集合过大，团队难以维护。
2. 模式命中范围过宽，导致规则误扩散。
3. 缺少回归测试，解析器调整后行为漂移。

规避建议：

- 只保留必要语法子集。
- 每种语法至少保留一个示例和一个测试。
- 将 pattern 与实验参数一并版本化。

## 8. 主线总结：如何设计一个自己的 Preset

四步闭环：

1. 把机制写成转换规则。
2. 用 `genotype_filter` 限定作用范围。
3. 封装为可复用 Preset。
4. 用模式匹配把 Preset 参数化。

---

## 回看与延伸

- [种群观测规则](observation_rules.md)
- [设计自己的 Preset（1）：从等位基因转换规则开始](allele_conversion_rules.md)
- [设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围](genotype_filter.md)
- [设计自己的 Preset（3）：封装、验证与发布前检查](preset_encapsulation_and_validation.md)
- [遗传预设使用指南](genetic_presets.md)
