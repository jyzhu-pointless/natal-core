# 设计自己的 Preset（2）：用 filters 控制规则生效范围

`filters` 是作用域名称到既有类型模式字符串的映射，不接受函数或已解析的 Pattern 对象。直接传入原始模式字符串，由规则编译器结合宿主的物种和 registry 解析。

## 支持的键

两种配子规则共用一组键，两种合子规则共用另一组。

| 键 | 配子规则 | 合子规则 |
|---|---|---|
| `current` | 进入当前规则的配子 gtype | 进入当前规则的合子分支 ztype |
| `parent` | 产生配子的亲本 ztype | 不支持 |
| `parent_sex` | `female`、`male` 或 `both` | 不支持 |
| `maternal` | 不支持 | 参与受精的母源配子 gtype |
| `paternal` | 不支持 | 参与受精的父源配子 gtype |

gtype 是 `haploid_genotype@glab`，ztype 是 `genotype@slab`。多个键取 AND。省略某个键、`filters=None` 或空映射表示没有相应限制。未知键、阶段不支持的键、非法模式和未知标签均报错，不会静默解释为未匹配。

每条规则的 `current` 都检查前序规则处理后的分支状态。亲本和参与受精的配子信息在对应阶段中保持固定。不另设 `when` 参数，也不引入新的条件表达式语言。

## 亲本与当前状态条件

以下声明可独立运行。编译时需要物种在同一位点声明 W 和 D。

```python
from natal.frontend.modifiers import GameteConversionRuleSet

ruleset = GameteConversionRuleSet("homing_drive")
ruleset.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    filters={"parent": "W::D", "parent_sex": "female"},
)
```

`W::D` 选择同源染色体任一左右顺序的杂合亲本。在这个单个位点的例子中，`*::D` 则选择任意携带 D 的亲本。多位点模型应提供适用的完整模式，不要在基因型名称上做子串包含判断。

合子阶段若要检查后代自身，应使用 `current`：

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

zygote_rules = ZygoteConversionRuleSet("zygote_copy")
zygote_rules.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    filters={"current": "*::D"},
)
```

## 标签与固定来源

裸遗传组成模式不限制标签。`*@infected` 只限制标签，`@default` 明确指定默认标签。输出目录与模式均使用 `@`，不支持旧冒号标签格式。既有无序配对分隔符 `::` 保持原义。

以下声明保留后代基因型：当母源配子带 infected 标签时，默认标签的后代以 0.9 概率获得 infected slab。宿主需要在配子和体细胞标签目录中分别声明相应标签。

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

infection = ZygoteConversionRuleSet("maternal_transmission")
infection.add_ztype_convert(
    filters={"current": "*@default", "maternal": "*@infected"},
    to="*@infected",
    rate=0.9,
)
```

`maternal` 和 `paternal` 指配子，不是亲本的二倍体基因型或体细胞标签。

## 在预设中复用模式

在配置中保存字符串并直接传入。以下类定义可运行；使用时要求物种在同一位点具有 WT 和 Drive，并提供适用于该物种的亲本模式。

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet

class PatternBasedPreset(GeneticPreset):
    def __init__(self, pattern: str, conversion_rate: float = 0.95):
        super().__init__(name="PatternBasedPreset")
        self.pattern = pattern
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("PatternBased")
        ruleset.add_allele_convert(
            from_allele="WT",
            to_allele="Drive",
            rate=self.conversion_rate,
            filters={"parent": self.pattern},
        )
        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None
```

表达依赖遗传背景的突变时，将背景要求写入亲本模式，将源等位基因写入 `from_allele`。若统计范围相同，可在观察分组中复用同一个类型模式。亲本条件与当前后代条件检查的是不同对象，即使字符串相同也不能混淆。

目标与概率语义参见[转换规则](allele_conversion_rules.md)；完整构建示例参见[预设验证](preset_encapsulation_and_validation.md)。
