# 设计自己的 Preset（1）：从等位基因转换规则开始

本章开始一条完整主线：**如何设计一个自己的 Preset**。

第一步不是马上写 Preset 类，而是先把“遗传机制”表达清楚。对多数驱动系统而言，这一步通常体现在**等位基因转换规则**。

## 1. 先定义机制目标

在写任何代码前，先回答三件事：

1. 哪个等位基因会转换（`from_allele`）？
2. 转成什么（`to_allele`）？
3. 转换概率是多少（`rate`）？

例如，一个最小驱动假设可以写成：

- 在配子生成阶段，`W -> D`，概率 `0.5`。

## 2. 规则对象与规则集

NATAL 提供两层结构：

- `GameteAlleleConversionRule`：单条规则。
- `GameteConversionRuleSet`：规则集合。

你可以理解为：

- Rule 是“一个句子”。
- RuleSet 是“一个段落”。

## 3. 最小可用示例

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet

ruleset = GameteConversionRuleSet(name="homing_drive")
ruleset.add_convert(from_allele="W", to_allele="D", rate=0.5)
```

这个示例已经足以描述一个最小的转换机制。

## 4. 与 population 的连接

规则集本身只是定义；要生效，需要转换为 modifier 并绑定到 population。

```python
gamete_mod = ruleset.to_gamete_modifier(pop)
pop.add_gamete_modifier(gamete_mod, name="homing")
```

实践上，建议你先在短程模拟里验证频率变化方向，再增加复杂规则。

## 5. 常用控制参数

### 5.1 `sex_filter`

用于指定规则作用于哪个性别：

- `"both"`：两性都应用（默认）。
- `"male"`：仅雄性。
- `"female"`：仅雌性。

示例：

```python
ruleset.add_convert("W", "D", rate=0.8, sex_filter="male")
```

### 5.2 `name`

建议每条规则和规则集都给出可读名称，方便复现实验。

## 6. 规则设计建议

1. 从一条规则开始，不要一上来写十几条。
2. 每增加一条规则，先跑 20-50 步检查方向是否符合预期。
3. 记录“生物学假设 -> 参数值”映射，避免后续难以解释。

## 7. 本章小结

你已经完成 Preset 设计的第一步：

- 把机制写成可执行规则。
- 把规则绑定到 population。

下一章会继续主线：**如何用 genotype_filter 把规则限制在你希望的人群上**。

---

## 下一章

- [设计自己的 Preset（2）：用 genotype_filter 控制规则生效范围](14_genotype_filter_implementation.md)
