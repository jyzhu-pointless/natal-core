# 基因型模式匹配

NATAL 的模式匹配机制允许用户用格式化的字符串描述和批量筛选基因型。模式匹配是精确基因型字符串格式的自然延伸，支持对二倍体基因型（`Genotype`）和单倍体基因型（`HaploidGenotype`）进行灵活的模式描述。

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
2. **`HaploidGenomePattern`**：用于单倍体基因型的模式匹配

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

### 标签匹配（@lab）

模式字符串末尾可以用 `@` 附加配子标签（`glab`）或体细胞标签（`slab`）约束。标签会被解析并与模式一起存储，但裸 `GenotypePattern` / `HaploidGenomePattern` 的 `matches()` **不会**检查它；而且只匹配遗传内容的 `Species` 入口（`parse_genotype_pattern`、`enumerate_genotypes_matching_pattern`、`parse_haploid_genome_pattern`、`enumerate_haploid_genomes_matching_pattern`）对带标签的模式**直接报错**（`PatternParseError`），不再"接受后忽略"；`GenotypePatternParser.parse_haploid_genome_pattern` 同样报错。标签过滤只在模式经转换规则过滤器、`ZygoteTypePattern`、`IndividualSelector` 或 `GenotypePatternParser.parse_haplotype_pattern` 编译时才生效。标签语法与等位基因模式一致：

| 模式 | 含义 | 示例 |
|---|---|---|
| `@X` | 精确匹配标签 `X` | `A\|a@Cas9_high` |
| `@!X` | 排除标签 `X` | `A\|a@!wildtype` |
| `@{A,B}` | 匹配集合中的任一标签 | `A\|a@{high,low}` |
| `@!{A,B}` | 排除集合中的标签 | `*|*@!{wildtype,default}` |
| `@*` | 任意标签（等同于不加 @） | `A\|a@*` |

**GenotypePattern** 使用 `@` 匹配体细胞标签（slab），**HaploidGenomePattern** 使用 `@` 匹配配子标签（glab）：

```python
# 匹配携带 Cas9_high 体细胞标签的 A|a 基因型
parser.parse("A|a@Cas9_high")

# 匹配携带 Cas9_deposited 配子标签的单倍体
parser.parse_haplotype_pattern("A@Cas9_deposited")
```

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

- **`|`（单竖线）**：严格有序——`Dr|WT` 只匹配字面顺序 `Dr|WT`，与 Species 的 `unordered` 设置无关。默认 `unordered=True` 物种的规范杂合子是 `WT|Dr`，此时模式 `Dr|WT` 不会命中任何基因型。
- **`::`（双冒号）**：无序匹配——无论 Species 设置如何，同源染色体两条拷贝均可交换。

```python
# 有序匹配：只匹配这一精确的母本/父本相位
pattern1 = "A1/B1|A2/B2"

# 无序匹配：同源染色体可交换
pattern2 = "A1/B1::A2/B2"
```

基于 `unordered` 的宽容确实存在，但位于**选择器层**而非模式解析器：当物种
`unordered=True` 时，`IndividualSelector(ztype=...)` 与
`Species.resolve_single_genotype_selector()` 会先把 `|` 改写为 `::` 再解析，
因此选择器字符串可以匹配任意相位。

## HaploidGenomePattern：单倍体基因型匹配

### 基本语法

`HaploidGenomePattern` 用于匹配单倍体基因型，语法更简洁：

`<chr1_hap>/<...>; <chr2_hap>/<...>`

### 组合示例

1. **精确匹配**：`A1/B1; C1/D1`
2. **通配混合**：`A1/*; */D1`
3. **集合匹配**：`{A1,A2}/B1; C1/D1`
4. **排除匹配**：`!A1/B1; C1/D1`

### 使用示例

```python
# 单倍体基因型模式匹配。
# Species.parse_haploid_genome_pattern() 返回过滤函数（callable），
# 而不是 HaploidGenomePattern 对象本身。
pattern = sp.parse_haploid_genome_pattern("A1/*; C1")

# 过滤符合条件的单倍体基因型
matching_haploids = [hg for hg in all_haploids if pattern(hg)]

# 或者使用枚举方法
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=10):
    print(f"匹配的单倍体基因型: {hg}")
```

## 高级语法特性

### 小括号语法：同一对染色体内部分隔

小括号 `(...)` 将**同一对染色体**上的逐位点二倍体条件分为一组。括号内的分号分隔位点，括号外的分号分隔染色体对。组内每个 `|` 检查对应位点的母源／父源顺序，每个 `::` 则在对应位点独立允许两种顺序，并非将整条染色体的单倍型一起翻转。

```python
import natal as nt

single = nt.Species.from_dict(
    name="PatternSingleChromosome",
    structure={"chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]}},
    unordered=False,
)
pattern1 = single.parse_genotype_pattern("(A1|A2;B1::B2)")
assert pattern1(single.get_genotype_from_str("A1/B1|A2/B2"))
assert pattern1(single.get_genotype_from_str("A1/B2|A2/B1"))
assert not pattern1(single.get_genotype_from_str("A2/B1|A1/B2"))

multiple = nt.Species.from_dict(
    name="PatternTwoChromosomes",
    structure={
        "chr1": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
        "chr2": {"C": ["C1", "C2"]},
    },
    unordered=False,
)
# A and B are on chr1; C is on chr2.
pattern2 = multiple.parse_genotype_pattern("(A1|A2;B1::B2);C1|C2")
assert pattern2(multiple.get_genotype_from_str("A1/B2|A2/B1;C1|C2"))
assert not pattern2(multiple.get_genotype_from_str("A1/B2|A2/B1;C2|C1"))
```

如果三个位点都在同一条染色体上，应写成 `(A1|A2;B1::B2;C1|C2)`。这些逐位点二倍体条件不适用于单倍体模式；单倍体模式使用 `/` 分隔位点。


## 常见错误与修正

### 通用错误

1. **错误**：染色体段数量不匹配
   - **原因**：解析器按"每条常染色体一段 + 每个性染色体组一段"计数（不是按每条性染色体计）。段数超过物种的组数会报错；性染色体组的字符串写法见 [遗传学架构](2_genetics.md) 中"性染色体的字符串格式"一节
   - **修正**：按物种定义，每条常染色体写一段、每个性染色体组写一段

2. **错误**：位点数量不匹配
   - **原因**：`/` 分隔后的位点模式数量与该染色体位点数不一致
   - **修正**：逐位点补齐，或使用 `*` 占位符

### GenotypePattern 特有错误

1. **错误**：`Chromosome pattern must contain '|' or '::'`
   - **原因**：某个染色体段缺少同源染色体双拷贝分隔符
   - **修正**：不要写 `C1/C1`，改为完整的 `...|...` 或 `...::...` 形式

## 应用集成

### 与 Observation 结合

Observation 章节中 `with_observation(groups=...)` 的每个值必须是 `IndividualSelector`，其 `ztype` 字段支持 `GenotypePattern` 解析：

```python
import natal as nt

groups = {
    "target_group": nt.IndividualSelector(
        # 有序匹配：Maternal|Paternal
        ztype="A1/B1|A2/B2; C1/D1|C2/D2",
        sex="female",
    ),
    "target_group_unordered": nt.IndividualSelector(
        # 无序匹配：同源染色体两条拷贝可交换
        ztype="A1/B1::A2/B2; C1/D1::C2/D2",
        sex="female",
    ),
}

# 在构建期传入：.with_observation(groups)
```

### 与 Preset 结合

在 Preset 中保存模式字符串并通过 `filters` 传入，由规则编译器负责解析：

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet

class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def zygote_modifier(self, host):
        return None

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("pattern_rules")

        ruleset.add_allele_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            filters={"parent": self.target_pattern},
        )
        return ruleset.to_gamete_modifier(host)
```

## 调试与验证

如需调试命中集合，可使用以下方法进行离线展开检查：

```python
# 检查 GenotypePattern 匹配结果
for gt in sp.enumerate_genotypes_matching_pattern("A1/*|A2/B2", max_count=5):
    print(f"匹配的基因型: {gt}")

# 检查 HaploidGenomePattern 匹配结果
for hg in sp.enumerate_haploid_genomes_matching_pattern("A1/B1; C1", max_count=5):
    print(f"匹配的单倍体基因型: {hg}")
```

---

## 相关章节

- [种群观测规则](2_data_output.md)
- [设计你自己的预设](3_custom_presets.md)
- [遗传预设使用指南](2_genetic_presets.md)
