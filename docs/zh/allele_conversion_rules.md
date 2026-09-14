# 设计自己的 Preset（1）：从等位基因转换规则开始

## 四类转换规则

所有规则构造函数均使用关键字参数。`rate` 必填，必须有限且在 `[0, 1]` 内；`filters=None` 表示不限制，`name=None` 是可选展示名称。

| 规则 | 必填字段 | 可选字段 |
|---|---|---|
| `GameteGtypeConversionRule` | `to`、`rate` | `filters`、`name` |
| `ZygoteZtypeConversionRule` | `to`、`rate` | `filters`、`name` |
| `GameteAlleleConversionRule` | `from_allele`、`to_allele`、`rate` | `filters`、`name` |
| `ZygoteAlleleConversionRule` | `from_allele`、`to_allele`、`rate` | `filters`、`name`、`side="both"` |

整体转换的 `to` 字符串格式为 `[genotype 或 *]@[label 或 *]`，配子侧为单倍体基因型。两部分必须显式给出。整部分 `*` 保留对应输入，其余目标部分必须精确，不支持内部局部通配符、集合或无序目标候选。

| 合子目标 | 动作 |
|---|---|
| `A\|B@I` | 联合替换 genotype 和 slab |
| `*@I` | 只替换 slab，保留 genotype |
| `A\|B@*` | 只替换 genotype，保留 slab |
| `*@*` | 恒等转换 |

整体规则表示一次概率事件：以 0.4 概率将 `A@S` 变为 `B@I`，得到 60% `A@S` 和 40% `B@I`，并非两部分分别独立转换。独立变化应声明两条规则，筛选条件需覆盖相关分支。

Allele 规则的源和目标均为 gene 名字符串。Gene 名在物种内唯一，因此不需要 `locus` 参数；目标必须属于源所在的位点。Allele 规则保留标签。合子 `side` 可为 `maternal`、`paternal` 或 `both`，每个适用副本独立以 `rate` 转换。`side="both"`、rate 为 0.4 时，有序 `A|A` 输入产生 36% `A|A`、24% `B|A`、24% `A|B` 和 16% `B|B`。

RuleSet 按声明顺序执行，前序规则产生的分支继续进入后续规则，不设数字优先级，也不在首次匹配后停止。合法状态没有源等位基因时保持原样；未知等位基因、跨位点目标和非法目标均报错。五个作用域键参见[filters](genotype_filter.md)。

同一配子或合子修饰器管线注册多个 RuleSet 时，各规则集按注册顺序接收前一步结果。重新构建或刷新模型时，整条管线从未修饰的物种基线重新开始，不在上一次编译结果上重复叠加规则。

以下声明可独立运行；编译需要物种具有对应等位基因和标签。

```python
from natal import (
    GameteGtypeConversionRule, ZygoteZtypeConversionRule,
    GameteAlleleConversionRule, ZygoteAlleleConversionRule,
)

whole_gamete = GameteGtypeConversionRule(
    filters={"current": "A@default"}, to="B@I", rate=0.4,
)
whole_zygote = ZygoteZtypeConversionRule(
    filters={"maternal": "*@I"}, to="*@I", rate=0.9,
)
gamete_allele = GameteAlleleConversionRule(
    from_allele="A", to_allele="B", rate=0.4,
    filters={"parent_sex": "female"},
)
zygote_allele = ZygoteAlleleConversionRule(
    from_allele="A", to_allele="B", rate=0.4, side="both",
    filters={"current": "*@I"},
)
```

通过 `add_rule(rule)` 追加声明。`GameteConversionRuleSet.add_gtype_convert()`、`ZygoteConversionRuleSet.add_ztype_convert()` 和两阶段各自的 `add_allele_convert()` 完整暴露对应构造字段。

新 API 明确不兼容旧接口。原标签专用类和规则别名已移除，改用上述整体规则表达标签转换。不支持旧规则参数、对象／callable 输入和冒号标签格式。标签使用 `@`，既有无序筛选语法 `::` 保持原义。尤其是原来成功后联合改变标签的等位基因转换，应迁移为整体联合转换，不能拆成两个独立事件。

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
from natal.frontend.modifiers import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.5)
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

以下两个片段假定已有种群 `pop`，其物种在同一位点声明 W 和 D。

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# 在受精卵中，仅对已携带 D 等位基因的合子转换 W->D
ruleset.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    filters={"current": "*::D"},  # 仅对已携带 D 的合子生效
)

zygote_mod = ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="zygote_repair")
```

### Gamete + Zygote 的组合使用

通常驱动系统会同时使用两类规则：

```python
# 配子阶段：W -> D（偏向）
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.99)

# 受精卵阶段：实现复制（确保纯和）
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_allele_convert(
    from_allele="W", to_allele="D", rate=0.95,
    filters={"current": "*::D"},
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
from natal.frontend.presets import GeneticPreset, PresetFitnessPatch
from natal.frontend.modifiers import GameteModifier, ZygoteModifier
from typing import Optional

class MyCustomPreset(GeneticPreset):
    """自定义遗传修饰预设"""

    def __init__(self, name: str = "MyCustom", species=None):
        super().__init__(name=name, species=species)
        # 自定义参数
        self.custom_param = 0.5

    def gamete_modifier(self, host) -> Optional[GameteModifier]:
        """定义配子阶段的修饰逻辑"""
        # 返回GameteModifier或None
        return None

    def zygote_modifier(self, host) -> Optional[ZygoteModifier]:
        """定义合子阶段的修饰逻辑"""
        # 返回ZygoteModifier或None
        return None

    def fitness_patch(self) -> Optional[PresetFitnessPatch]:
        """定义适应度效应"""
        # 返回适应度配置字典或None
        return None
```

实现要点：

1. **`gamete_modifier` 与 `zygote_modifier` 必须定义** - `GeneticPreset` 是抽象基类，缺少任一个都无法实例化（可以 `return None` 表示该阶段不修饰）
2. **`fitness_patch` 可选** - 不定义即无适应度效应；定义了也可以返回 `None`
3. **可以返回 None** - 表示该阶段不需要修饰
4. **支持延迟物种绑定** - 可以在创建时不指定 `Species`
5. **`gamete_modifier` / `zygote_modifier` 的入参是 `host`** - 它是一个统一入口（接口约定 `natal.frontend.genetics.compile.RecipeHost`）：运行时指向当前的 Population，编译阶段指向构建中的 PopulationBuilder，两种场景都可以通过它读取 `species`、`config`、`registry`、`index_registry` 四项只读信息

## 简单示例

### 简单点突变

```python
from natal.frontend.presets import GeneticPreset, PresetFitnessPatch
from natal.frontend.modifiers import GameteConversionRuleSet

class PointMutation(GeneticPreset):
    """简单点突变：WT以一定频率突变为Mutant"""

    def __init__(self, mutation_rate: float = 1e-5):
        super().__init__(name="PointMutation")
        self.mutation_rate = mutation_rate

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("PointMutation")
        ruleset.add_allele_convert(from_allele="WT", to_allele="Mutant", rate=self.mutation_rate)
        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None  # 合子阶段不修饰

    def fitness_patch(self):
        return {
            "viability_per_allele": {"Mutant": 0.98}  # 轻微有害
        }
```

> **注意**：NATAL 已内置覆盖该行为的 `PointMutation` 预设（并额外支持多目标竞争与按性别速率），详见[遗传预设](2_genetic_presets.md)。上面的类只是最小化的自定义预设练习，并非内置 API。

### 双向突变平衡

```python
class BidirectionalMutation(GeneticPreset):
    """双向突变平衡"""

    def __init__(self, forward_rate: float = 1e-5, backward_rate: float = 1e-6):
        super().__init__(name="BidirectionalMutation")
        self.forward_rate = forward_rate
        self.backward_rate = backward_rate

    def gamete_modifier(self, host):
        from natal.frontend.modifiers import GameteConversionRuleSet

        ruleset = GameteConversionRuleSet("BidirectionalMutation")

        # A → B (正向突变)
        ruleset.add_allele_convert(from_allele="A", to_allele="B", rate=self.forward_rate)
        # B → A (回复突变)
        ruleset.add_allele_convert(from_allele="B", to_allele="A", rate=self.backward_rate)

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None  # 合子阶段不修饰
```

## 小结

你已经完成 Preset 设计的第一步：定义等位基因转换规则。下一步将学习如何控制规则的作用范围。
