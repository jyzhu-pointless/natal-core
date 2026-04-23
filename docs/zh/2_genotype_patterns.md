# 基因型模式匹配

本章详细介绍 NATAL 的模式匹配机制，该机制允许用户用格式化的字符串描述和批量筛选基因型。模式匹配是精确基因型字符串格式的自然延伸，支持对二倍体基因型（`Genotype`）和单倍体基因型（`HaploidGenotype`）进行灵活的模式描述。

## 概述

### 为什么需要模式匹配

当遗传模型复杂度达到以下程度时，硬编码基因型列表会变得难以维护：

- 多染色体、多位点
- 大量等位基因组合
- 需要按"某类基因型集合"批量定义规则或观察分组

模式匹配将显式枚举基因型列表升级为**语义表达式**，既避免了冗长的枚举，又提供了更直观易懂的表达方式。

### 支持的匹配类型

NATAL 支持两种模式匹配：

1. **`GenotypePattern`**：用于二倍体基因型的模式匹配
2. **`HaploidGenotypePattern`**：用于单倍体基因型的模式匹配

两种模式共享相同的语法基础，但在染色体层级处理上有所不同。

## 语法基础

### 基本结构

模式字符串按"从外到内"三层解析：

1. **染色体层**：用 `;` 分隔多个染色体段
2. **同源染色体层**：每段必须包含 `|` 或 `::`（仅 `GenotypePattern`）
3. **染色体位点层**：每条同源染色体内部用 `/` 分隔位点模式

### 分隔符含义

| 语法元素 | 含义 | 适用模式 | 示例 |
|---|---|---|---|
| `;` | 分隔不同染色体段 | 两者 | `A/B|C/D; E/F|G/H` |
| `|` | 有序匹配：`Maternal|Paternal` | GenotypePattern | `A/B|C/D` |
| `::` | 无序匹配：同源染色体可交换 | GenotypePattern | `A/B::C/D` |
| `/` | 分隔单条染色体内部位点 | 两者 | `A/B/C` |

### 位点原子模式

| 模式 | 含义 | 示例 |
|---|---|---|
| `X` | 精确匹配等位基因 `X` | `A1` |
| `*` | 通配任意等位基因 | `*` |
| `{A,B,C}` | 枚举集合中的任一元素 | `{A1,A2}` |
| `!X` | 排除 `X`，匹配其他等位基因 | `!A1` |

## GenotypePattern：二倍体基因型匹配

### 基本语法

`GenotypePattern` 用于匹配二倍体基因型，其基本语法与精确字符串格式相同：

`<chr1_hap1>/<...>|<chr1_hap2>/<...>; <chr2_hap1>/<...>|<chr2_hap2>/<...>`

### 组合示例

1. **精确匹配**：`A1/B1|A2/B2; C1/D1|C2/D2`
2. **通配混合**：`A1/*|A2/B2; */D1|C2/*`
3. **集合匹配**：`{A1,A2}/B1|A3/B2; C1/D1|C2/D2`
4. **无序匹配**：`A1/B1::A2/B2; C1/D1::C2/D2`

### 有序 vs 无序匹配

- **`|`（有序）**：严格区分母本和父本顺序
- **`::`（无序）**：同源染色体两条拷贝可交换

```python
# 有序匹配：Maternal|Paternal 严格区分
pattern1 = "A1/B1|A2/B2"

# 无序匹配：同源染色体可交换
pattern2 = "A1/B1::A2/B2"
```

## HaploidGenotypePattern：单倍体基因型匹配

### 基本语法

`HaploidGenotypePattern` 用于匹配单倍体基因型，语法更简洁：

`<chr1_hap>/<...>; <chr2_hap>/<...>`

### 组合示例

1. **精确匹配**：`A1/B1; C1/D1`
2. **通配混合**：`A1/*; */D1`
3. **集合匹配**：`{A1,A2}/B1; C1/D1`
4. **排除匹配**：`!A1/B1; C1/D1`

### 使用示例

```python
# 单倍体基因型模式匹配
pattern = sp.parse_haploid_genome_pattern("A1/*; C1")

# 过滤符合条件的单倍体基因型
matching_haploids = [hg for hg in all_haploids if pattern(hg)]

# 或者使用枚举方法
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=10):
    print(f"匹配的单倍体基因型: {hg}")
```

## 高级语法特性

### 小括号语法：同一对染色体内部分隔

小括号 `(...)` 允许在同一对染色体内使用 `;` 进行进一步分隔，特别适用于混合有序和无序匹配的复杂场景：

```python
# 同一对染色体内部分隔：位点A有序匹配，位点B无序匹配
pattern1 = "(A1|A2);(B1::B2)"
# 等价于：A位点必须严格按母本|父本顺序，B位点可以交换

# 混合有序和无序匹配
pattern2 = "(A1|A2);(B1::B2);(C1|C2)"
# A和C位点有序匹配，B位点无序匹配

# 复杂嵌套模式
pattern3 = "(A1/{B1,B2}|A2/{B1,B2});(C1::C2)"
```

小括号语法在单倍体和二倍体模式中都适用，能够显著提升复杂模式的可读性和可维护性。

## 常见错误与修正

### 通用错误

1. **错误**：染色体段数量不匹配
   - **原因**：`;` 分段数与物种染色体数不一致
   - **修正**：按物种定义逐段补齐染色体段

2. **错误**：位点数量不匹配
   - **原因**：`/` 分隔后的位点模式数量与该染色体位点数不一致
   - **修正**：逐位点补齐，或使用 `*` 占位符

### GenotypePattern 特有错误

1. **错误**：`Chromosome pattern must contain '|' or '::'`
   - **原因**：某个染色体段缺少同源染色体双拷贝分隔符
   - **修正**：不要写 `C1/C1`，改为完整的 `...|...` 或 `...::...` 形式

## 应用集成

### 与 Observation 结合

Observation 章节中的 `groups["genotype"]` 支持 `GenotypePattern` 解析：

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

### 与 Preset 结合

推荐将模式解析逻辑封装在 Preset 内部：

```python
class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def _build_filter(self, species):
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

## 调试与验证

如需调试命中集合，可使用以下方法进行离线展开检查：

```python
# 检查 GenotypePattern 匹配结果
for gt in sp.enumerate_genotypes_matching_pattern("A1/*|A2/B2", max_count=5):
    print(f"匹配的基因型: {gt}")

# 检查 HaploidGenotypePattern 匹配结果
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=5):
    print(f"匹配的单倍体基因型: {hg}")
```

---

## 相关章节

- [种群观测规则](2_data_output.md)
- [设计你自己的预设](3_custom_presets.md)
- [遗传预设使用指南](2_genetic_presets.md)
