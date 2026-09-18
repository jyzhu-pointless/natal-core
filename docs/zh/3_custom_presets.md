# 设计你自己的预设

本部分将指导你从零开始设计、实现、验证和发布自定义遗传预设（Genetic Preset）。

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

## 1. 从等位基因转换规则开始

`GeneticPreset` 的设计过程始于遗传机制的清晰表达。对多数驱动系统而言，这一步通常体现在**等位基因转换规则**的制定。

### 定义机制目标

在编写任何代码之前，需要明确回答三个关键问题：

1. 哪个等位基因会转换（`from_allele`）？
2. 转成什么（`to_allele`）？
3. 转换概率是多少（`rate`）？

例如，一个最简驱动假设可以表述为：

- 在配子生成阶段，`W -> D`，概率 `0.5`。

### 规则对象与规则集

NATAL 提供两层结构来组织转换规则：

- `GameteAlleleConversionRule`：单条转换规则
- `GameteConversionRuleSet`：规则集合

可以将其理解为：

- Rule 是"一个句子"
- RuleSet 是"一个段落"

### 最简可用示例

```python
from natal.frontend.modifiers import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.5)
```

这个示例已经足以描述一个最简的转换机制。

### Zygote 转换规则（受精卵阶段）

等位基因转换还可以在受精卵（zygote）阶段发生，通常用于模拟以下机制：

- **基因驱动的修复**：在合子中表达的修复系统（例如 Cas9 切割修复）
- **等位基因特异性死亡**：某些基因型受精卵生活力降低
- **分生组后期转换**：发育过程中的等位基因转换

#### 从 Gamete 到 Zygote 的关键区别

| 阶段 | 输入 | 机制 | 适用场景 |
|------|------|------|---------|
| **Gamete** | 配子（单倍体）| 配子生成时的转换 | 配子驱动系统 |
| **Zygote** | 受精卵（二倍体）| 受精后立即的转换 | 合子驱动、合子修复 |

#### 使用 ZygoteConversionRuleSet

以下两个片段假定已有种群 `pop`，其物种在同一位点声明 W 和 D。

```python
from natal.frontend.modifiers import ZygoteConversionRuleSet

ruleset = ZygoteConversionRuleSet(name="zygote_drive")

# 在受精卵中，只要A位点含有D等位基因，就转换W->D
ruleset.add_allele_convert(
    from_allele="W",
    to_allele="D",
    rate=0.9,
    filters={"current": "*::D"},
)

zygote_mod = ruleset.to_zygote_modifier(pop)
pop.add_zygote_modifier(zygote_mod, name="zygote_repair")
```

#### Gamete + Zygote 的组合使用

通常驱动系统会同时使用两类规则：

```python
# 配子阶段：W -> D（偏向）
gamete_ruleset = GameteConversionRuleSet("gamete_drive")
gamete_ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.99)

# 受精卵阶段：等位基因转换（确保纯和）
zygote_ruleset = ZygoteConversionRuleSet("zygote_copy")
zygote_ruleset.add_allele_convert(from_allele="W", to_allele="D", rate=0.95,
    filters={"current": "*::D"}
)

pop.add_gamete_modifier(gamete_ruleset.to_gamete_modifier(pop))
pop.add_zygote_modifier(zygote_ruleset.to_zygote_modifier(pop))
```

### 设计规则时的注意事项

1. 从一条规则开始，不要一上来写十几条
2. 每增加一条规则，先跑 20-50 步检查方向是否符合预期
3. 记录"生物学假设 -> 参数值"映射，避免后续难以解释

### 基础模板

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

1. **`gamete_modifier` 与 `zygote_modifier` 是抽象方法** - 两者都必须实现（可以返回 `None`），否则子类无法实例化（`TypeError`）
2. **`fitness_patch` 是可选的** - 不重写时默认返回 `None`
3. **可以返回 None** - 表示该阶段不需要修饰
4. **支持延迟物种绑定** - 可以在创建时不指定 `Species`
5. **`gamete_modifier` / `zygote_modifier` 的入参是 `host`** - 它是一个统一入口（接口约定 `natal.frontend.genetics.compile.RecipeHost`）：运行时指向当前的 Population，编译阶段指向构建中的 PopulationBuilder，两种场景都可以通过它读取 `species`、`config`、`registry`、`index_registry` 四项只读信息

### 简单示例

#### 简单点突变

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
        return None

    def fitness_patch(self):
        return {
            "viability_per_allele": {"Mutant": 0.98}  # 轻微有害
        }
```

> **注意**：NATAL 已内置覆盖该行为的 `PointMutation` 预设（并额外支持多目标竞争与按性别速率），详见[遗传预设](2_genetic_presets.md)。上面的类只是最小化的自定义预设练习，并非内置 API。

#### 双向突变平衡

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
        return None
```

## 2. 用 filters 控制规则生效范围

`filters` 是作用域名称到既有类型模式字符串的映射，不接受函数或已解析的 Pattern 对象。直接传入原始模式字符串，由规则编译器结合宿主的物种和 registry 解析。

### 支持的键

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

### 亲本与当前状态条件

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

### 标签与固定来源

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

### 在预设中复用模式

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

## 3. 封装、验证与发布前检查

在前两章中，你已经完成了：

1. 规则定义（Gamete 与 Zygote 转换）
2. 规则生效范围的精细化控制

下面将这些内容封装为**可复用 Preset**，进行充分验证，最后发布使用。

### 封装成 Preset 的价值

如果只在脚本中编写规则，后期会遇到三个问题：

1. 难复用：每个实验都要复制逻辑
2. 难追溯：很难说清"这个版本到底用了哪组规则"
3. 难维护：规则、适应度、Hook 分散在多个文件

Preset 的价值就是把这些内容收敛成一个稳定配置单元。

### 推荐的 Preset 结构

一个实用 Preset 建议包含：

1. 机制规则（转换规则与过滤器）
2. 适应度补丁（如需要）
3. 可选参数（例如转换率、性别限制）
4. 清晰的名称与版本标记

### 示例：封装一个最小 DrivePreset

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet


class DrivePreset(GeneticPreset):
    def __init__(self, conversion_rate: float = 0.5):
        super().__init__(name="DrivePreset")
        self.conversion_rate = conversion_rate

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("drive_rules")

        ruleset.add_allele_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            filters={"parent": "W::D"},
        )

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        return None
```

### 在 PopulationBuilder 构建链中应用 Preset

```python
import natal as nt

# 物种需要声明 DrivePreset 中用到的等位基因 W、D
species = nt.Species.from_dict(name="DriveExpSpecies", structure={"chr1": {"A": ["W", "D"]}})

pop = (
    nt.AgeStructuredPopulation
    .setup(species=species, name="DriveExperiment", stochastic=True)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state({"female": {"W|W": 500}, "male": {"W|W": 500}})
    .presets(DrivePreset(conversion_rate=0.55))
    .build()
)
```

这就是"Preset 作为配置组件"最推荐的接入方式。

### 验证清单（强烈建议）

在做大规模实验前，至少完成以下检查：

1. 机制检查：转换方向和目标等位基因是否正确
2. 过滤检查：`filters` 命中范围是否符合预期
3. 质量守恒检查：频率归一化是否成立
4. 对照检查：与无 Preset 的 baseline 对比趋势是否合理
5. 稳定性检查：随机性模型（`stochastic=True`）下重复运行，结论是否稳健（当前没有公开的随机种子 API）

### 实验记录建议

建议把 Preset 配置写入实验元数据：

- Preset 名称
- 关键参数（如 `conversion_rate`）
- 代码版本或 commit
- 随机性设置（如 `stochastic`）与运行环境

这样可以显著降低"结果无法复现"的风险。

### 复杂基因驱动示例

```python
from natal.frontend.presets import GeneticPreset
from natal.frontend.modifiers import GameteConversionRuleSet, ZygoteConversionRuleSet

class ComplexDrive(GeneticPreset):
    """复杂基因驱动，包含多个阶段的转换"""

    def __init__(self):
        super().__init__(name="ComplexDrive")

    def gamete_modifier(self, host):
        ruleset = GameteConversionRuleSet("ComplexDrive")

        # 阶段1: 驱动转换 (WT → Drive)
        ruleset.add_allele_convert(from_allele="WT", to_allele="Drive", rate=0.95,
                           filters={"parent": "*::Drive"})

        # 阶段2: 抗性形成 (剩余WT → Resistance)
        ruleset.add_allele_convert(from_allele="WT", to_allele="Resistance", rate=0.05,
                           filters={"parent": "*::Drive"})

        return ruleset.to_gamete_modifier(host)

    def zygote_modifier(self, host):
        ruleset = ZygoteConversionRuleSet("ComplexDrive_Embryo")

        # 胚胎阶段的额外修饰
        ruleset.add_allele_convert(
            from_allele="WT",
            to_allele="Resistance",
            rate=0.02,
            filters={"maternal": "*@cas9"}  # 需要母源Cas9沉积
        )

        return ruleset.to_zygote_modifier(host)

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

### 常见错误与调试

#### 参数验证错误
- 验证转换率是否在 [0, 1] 范围内
- 检查 `fitness_patch` 的每个顶层键：不支持的键会抛 `ValueError`，消息给出该键并列出支持的键（例如把 `viability_per_allele` 写成 `viability_allele`），且不会对模型施加任何修改

#### 物种绑定错误
- 确保预设和种群使用相同的物种
- 使用延迟绑定（创建时不指定 `Species`）

#### 性能问题
- 避免在修饰器中创建大量临时对象
- 使用规则集缓存
- 考虑简化复杂的规则链

#### 调试技巧

```python
class DebugPreset(GeneticPreset):
    def gamete_modifier(self, host):
        print(f"应用预设到物种: {host.species.name}")
        print(f"可用等位基因: {list(host.species.gene_index.keys())}")

        # 创建修饰器并返回
        # ...
```

### 发布前检查清单

在发布 Preset 前建议完成：

- [ ] 单元测试覆盖主要功能
- [ ] 文档说明清晰完整
- [ ] 参数范围验证通过
- [ ] 与现有系统兼容性测试
- [ ] 性能基准测试

### 小结

恭喜！你已经完成了"设计自己的 Preset"的完整主线：

1. 规则定义（Gamete 与 Zygote 转换）
2. 规则生效范围精细化（filters）
3. Preset 工程化、验证与发布

现在你已经掌握了从零开始设计、实现、验证和发布自定义 Preset 的完整流程。