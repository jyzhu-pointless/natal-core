# Modifier 机制

Modifier 用于在遗传流程中注入“规则级改变”。

如果你希望表达基因驱动、胚胎拯救、细胞质不兼容等机制，Modifier 是核心工具之一。

> 对常见情形，优先考虑使用 [遗传预设系统](genetic_presets.md)。
> 预设更简洁，Modifier 更灵活。

## 1. Modifier 的作用位置

在模拟中，遗传结果通常由两类映射决定：

1. 基因型到配子的映射。
2. 配子结合到合子的映射。

Modifier 的作用就是对这两类映射进行有控制的改写。

```text
基因型
  →（Gamete Modifier）→ 配子分布
  →（Zygote Modifier）→ 合子分布
```

## 2. 两类 Modifier

### 2.1 Gamete Modifier（配子修饰器）

用于改变某个亲本基因型产生配子的概率。

典型用途：

- drive 等位基因的偏置分离
- 特定配子类型的增强或抑制
- 给配子附加标签（如沉积标记）

### 2.2 Zygote Modifier（合子修饰器）

用于改变“母配子 + 父配子”组合后的后代基因型分布。

典型用途：

- 胚胎死亡或拯救
- 细胞质不兼容
- 非孟德尔比例的后代重分配

## 3. 推荐接入方式（Builder）

在用户实践中，推荐在 Builder 阶段统一注册 Modifier：

```python
from natal.population_builder import AgeStructuredPopulationBuilder


def my_gamete_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.95,
            ("WT", "Cas9_deposited"): 0.05,
        }
    }


pop = (
    AgeStructuredPopulationBuilder(species)
    .setup(name="MyPop")
    .age_structure(n_ages=8)
    .initial_state({...})
    .modifiers(gamete_modifiers=[(None, "drive", my_gamete_modifier)])
    .build()
)
```

这样可以让模型配置、初始状态与遗传规则一次性组织完成。

## 4. Gamete Modifier 示例

### 4.1 基因驱动偏置

```python
def heg_drive_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.98,
            ("WT", "Cas9_deposited"): 0.02,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.98,
            ("WT", "Cas9_deposited"): 0.02,
        },
    }
```

### 4.2 标记配子

```python
def cas9_deposition_modifier(pop):
    return {
        "Drive|WT": {
            ("Drive", "Cas9_deposited"): 0.5,
            ("WT", "Cas9_deposited"): 0.5,
        },
        "WT|Drive": {
            ("Drive", "Cas9_deposited"): 0.5,
            ("WT", "Cas9_deposited"): 0.5,
        },
        "WT|WT": {
            ("WT", "default"): 1.0,
        },
    }
```

## 5. Zygote Modifier 示例

### 5.1 胚胎拯救

```python
def embryo_rescue_modifier(pop):
    return {
        (("Drive", "Cas9_deposited"), ("WT", "default")): {
            "Drive|WT": 0.4,
            "WT|WT": 0.0,
            "Drive|Drive": 0.6,
        },
    }
```

### 5.2 细胞质不兼容

```python
def ci_modifier(pop):
    return {
        (("Allele1", "uninfected"), ("Allele1", "Wolbachia")): {
            # 该组合可按模型需求映射为低存活或无后代
        },
    }
```

## 6. 配子标签（Gamete Labels）

配子标签用于表示“同一等位基因的不同生物学状态”，例如是否携带某种细胞质因子。

例如：

- `(A1, default)`
- `(A1, Cas9_deposited)`

两者等位基因相同，但标签不同，后续在合子阶段可能触发不同规则。

### 6.1 配置标签

```python
pop = AgeStructuredPopulation(
    ...,
    gamete_labels=["default", "Cas9_deposited"],
)
```

## 7. 注册方式与优先级

### 7.1 动态注册

```python
pop.set_gamete_modifier(my_gamete_modifier, hook_name="drive")
pop.set_zygote_modifier(embryo_rescue_modifier, hook_name="rescue")
```

### 7.2 优先级

当多个 Modifier 同时作用时，会按优先级顺序执行。

```python
pop.set_gamete_modifier(base_mod, priority=1, hook_name="base")
pop.set_gamete_modifier(drive_mod, priority=2, hook_name="drive")
```

实践上，建议把“基础规则”放在较低优先级，把“覆盖/修正规则”放在较高优先级。

## 8. 建模建议

1. 先写最小规则，再逐步增加复杂机制。
2. 每增加一个 Modifier，都做小规模可解释性测试。
3. 对关键组合输出中间结果，验证概率是否归一化。
4. 对复杂系统优先使用预设，只有在预设不足时再下沉到自定义 Modifier。

## 9. 最小工作流

```python
# 1) 定义 genetic architecture
# 2) 写一个最小 gamete modifier
# 3) 跑短程模拟并检查基因型频率变化
# 4) 再叠加 zygote modifier
# 5) 最后扩展到完整参数扫描
```

## 10. 本章小结

Modifier 是 NATAL 中表达“遗传规则改写”的核心机制。

- Gamete Modifier 关注“产什么配子”。
- Zygote Modifier 关注“配子结合后产什么后代”。

把两者配合好，就能表达大多数高级遗传机制。

---

## 相关章节

- [遗传预设系统](genetic_presets.md)
- [Hook 系统](hooks.md)
- [模拟内核深度解析](simulation_kernels.md)
- [PopulationState 与 PopulationConfig](population_state_config.md)
