# `IndexRegistry` 索引机制

`IndexRegistry` 是 NATAL 框架中负责将遗传学对象（如 Genotype、HaploidGenotype 等）与整数索引建立关联的核心组件。它作为连接"高层对象世界"与"底层数值计算世界"的关键桥梁，确保用户能够使用直观的遗传学对象，同时底层计算能够高效处理整数索引。

## 核心概念

`IndexRegistry` 维护的对象↔索引映射关系：

```
Genotype("Drive|WT")      ↔  整数索引 2
HaploidGenotype("R1")     ↔  整数索引 3
"default" (glab)          ↔  整数索引 0
"Cas9_deposited" (glab)   ↔  整数索引 1
```

**设计目的**：`IndexRegistry` 解决了 numpy 数组使用整数索引（效率高）与用户习惯使用遗传学对象（直观易懂）之间的矛盾，负责在两者之间进行高效转换。

## `IndexRegistry` 类

```python
class IndexRegistry:
    """稳定的对象→整数索引注册表"""

    # 映射字典
    genotype_to_index: Dict[Genotype, int] = {}
    index_to_genotype: List[Genotype] = []

    haplo_to_index: Dict[HaploidGenotype, int] = {}
    index_to_haplo: List[HaploidGenotype] = []

    glab_to_index: Dict[str, int] = {}
    index_to_glab: List[str] = []
```

## 三个注册维度

### 1. 基因型索引
- 维护 Genotype 对象与整数索引的映射
- 支持基因型的注册和查询
- 提供基因型总数统计

### 2. 单倍基因型索引
- 维护 HaploidGenotype 对象与整数索引的映射
- 支持单倍基因型的注册和查询
- 提供单倍基因型总数统计

### 3. 配子标签索引
- 维护配子标签字符串与整数索引的映射
- 支持标签的注册和查询
- 提供标签总数统计

## 用户接口说明

**重要提示**：`IndexRegistry` 是底层数据表，用户通常无需直接调用其方法。用户访问基因型、单倍基因型或配子标签时，应使用以下高层接口：

### 使用字符串直接访问
```python
# 使用字符串直接操作，无需关心底层索引
pop.state.individual_count[Sex.FEMALE, 3, "A1|A2"]
```

### 使用 Pattern 模式匹配
```python
# 使用 Pattern 进行模式匹配操作
from natal.pattern import Pattern
pattern = Pattern("A1|*")
# 匹配所有包含 A1 的基因型
```

## 框架内部使用

在 NATAL 框架内部，`IndexRegistry` 被用于：

### 1. 状态数据存储
- 个体计数矩阵使用整数索引进行高效存储
- 精子存储矩阵使用基因型索引进行管理
- 所有状态数据都基于索引进行访问

### 2. Modifier 系统
- Modifier 返回的对象字典会被自动转换为索引
- 框架处理对象到索引的转换过程
- 用户只需关注高层对象操作

### 3. Hook 系统
- Numba Hook 使用预计算的索引进行高效操作
- 避免在编译时访问动态注册表
- 通过选择器模式避免硬编码索引

## 性能优化

虽然用户无需直接操作索引，但了解索引机制有助于编写高效代码：

### 缓存索引查询
在需要重复使用相同基因型的场景下，可以缓存索引以提高性能。

### 批量操作
对于多个基因型的操作，使用向量化方式比逐个处理更高效。

## 与全局缓存的关系

`IndexRegistry` 与 Genotype 的全局缓存协同工作：

```
字符串 "A1|A2"
    ↓ Species.get_genotype_from_str()
全局缓存 Species.genotype_cache
    ↓ [命中]
Genotype 对象 (唯一)
    ↓ IndexRegistry.register_genotype()
整数索引 (例如 5)
```

---

## 📚 相关章节

- [遗传结构与实体](2_genetics.md) - Genotype 对象的创建
- [PopulationState & PopulationConfig](4_population_state_config.md) - 配置中的索引应用
- [Modifier 机制](3_modifiers.md) - Modifier 中的 IndexRegistry 使用
- [Hook 系统](2_hooks.md) - 高级 Hook 选择器模式

---

**准备进入配置编译细节了吗？** [前往下一章：PopulationState & PopulationConfig →](4_population_state_config.md)
