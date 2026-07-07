# `IndexRegistry` 索引机制

`IndexRegistry` 是 NATAL 框架中负责将遗传学对象（如 Genotype、HaploidGenotype 等）与整数索引建立关联的核心组件。它作为连接"高层对象世界"与"底层数值计算世界"的关键桥梁，确保用户能够使用直观的遗传学对象，同时底层计算能够高效处理整数索引。

## 核心概念

`IndexRegistry` 维护的对象↔整数索引映射在两个层面上运作：

### ZType（合子类型）— 二倍体个体空间

群体中的每个个体由一个 ZType 标识：

```
ZType = (Genotype, slab_label)

(Genotype("Drive|WT"), "default")       ↔  ZType 索引 0
(Genotype("Drive|WT"), "infected")      ↔  ZType 索引 1
(Genotype("WT|WT"),   "default")        ↔  ZType 索引 2
```

第一个分量是二倍体基因型（如 `"Drive|WT"`）。第二个分量是 **slab**（体细胞标签），用于建模不可遗传的个体特征，如感染状态或转基因标记表达。

**设计目的**：扁平的 ZType 索引允许在压缩时独立地裁剪每个（基因型, slab）对。与公式 `g * n_slabs + s` 不同，每个配对有自己的字典条目，可以独立移除。

### GType（配子类型）— 单倍体配子空间

群体产生的每个配子（精子/卵子）由一个 GType 标识：

```
GType = (HaploidGenotype, glab_label)

(HaploidGenotype("Drive"), "default")        ↔  GType 索引 0
(HaploidGenotype("Drive"), "cas9_deposited") ↔  GType 索引 1
(HaploidGenotype("WT"),    "default")        ↔  GType 索引 2
```

第一个分量是单倍体基因型。第二个分量是 **glab**（配子标签），用于按起源机制对配子进行分类（如由哪个母体基因型产生、是否沉积了 Cas9）。

**设计目的**：GType 空间允许引擎追踪在受精过程中行为不同的配子亚群，而无需扩展二倍体状态空间。

### Slab 和 Glab — 对称的标签系统

| 层次 | 标签 | 含义 | 使用场景 |
|------|------|------|----------|
| 二倍体（ZType） | **slab**（体细胞标签） | 不可遗传的个体状态 | 感染状态、转基因背景 |
| 单倍体（GType） | **glab**（配子标签） | 配子起源分类 | Cas9 沉积、母体效应 |

两个系统的设计是**对称的**：
- 标签在 `Species` 对象上通过 `somatic_labels` / `gamete_labels` 定义。
- 如果未指定，会自动创建一个单一的 `"default"` 标签。
- 引擎会将每个基因型/单倍型与每个标签做叉积，生成完整的 ZType/GType 空间。

Slab 被具体的 Preset 使用，例如 **Wolbachia**（通过 `"wolbachia_infected"` slab 建模细胞质不兼容性）和 **TransgenicBackground**（按个体追踪标记表达）。如果没有这些 Preset，大多数模拟只有一个 `"default"` slab，slab 系统对用户不可见。

### 索引注册表结构

旧版注册表存储扁平的基因型和单倍型列表。新版注册表使用扁平的（实体, 标签）配对列表：

```python
class IndexRegistry:
    """稳定的对象→整数索引注册表"""

    # ZType 空间（二倍体层主索引）
    _ztype_to_index: Dict[Tuple[Genotype, str], int] = {}
    _index_to_ztype: List[Tuple[Genotype, str]] = []

    # GType 空间（配子层主索引）
    _gtype_to_index: Dict[Tuple[HaploidGenotype, str], int] = {}
    _index_to_gtype: List[Tuple[HaploidGenotype, str]] = []

    # 标签元数据（有序列表）
    slab_labels: List[str] = []
    glab_labels: List[str] = []
```

### 与旧版 API 的关系

向后兼容的属性从 ZType/GType 空间重建出扁平列表：

```python
# 旧风格：唯一基因型（从 ZType 空间去重）
registry.index_to_genotype  # [Genotype("A|A"), Genotype("A|a"), ...]
registry.haplo_to_index     # {HaploidGenotype("A"): 0, HaploidGenotype("a"): 1, ...}

# 新风格：包含标签维度
registry.index_to_ztype     # [(Genotype("A|A"), "default"), (Genotype("A|A"), "infected"), ...]
registry.index_to_gtype     # [(HaploidGenotype("A"), "default"), (HaploidGenotype("A"), "cas9_deposited"), ...]
```

计算出的 `N_ztype` 是 `_index_to_ztype` 的长度——这是引擎 `individual_count` 数组最后一个轴（ZType 维度）所消耗的值。

## 注册流程

### 构建时注册

在 `build_registry()` 期间，注册表从 Species 中填充：

```python
字符串 "A1|A2"
    ↓ Species.get_genotype_from_str()
Genotype 对象（唯一）
    ↓ IndexRegistry.register_genotype()   # 自动与 ALL slab_labels 做叉积
ZType 条目：(A1|A2, "default"), (A1|A2, "infected"), ...
```

标签先注册，以便自动叉积覆盖所有 slab/glab 组合：

1. 注册 `glab_labels` → `GType = 单倍型 × all_glabs`
2. 注册 `slab_labels` → `ZType = 基因型 × all_slabs`
3. 注册基因型 → 每个成为 `n_slab` 个 ZType 条目
4. 注册单倍型 → 每个成为 `n_glab` 个 GType 条目

### 注册 API

```python
# 底层：注册单个（基因型, slab）配对
registry.register_ztype(Genotype("A|a"), "default")       # 返回 ZType 索引

# 高层：注册基因型 + 自动与所有 slabs 做叉积
registry.register_genotype(Genotype("A|a"))                # 返回 ZType 索引列表

# GType 空间类似
registry.register_gtype(HaploidGenotype("A"), "default")   # 返回 GType 索引
registry.register_haplogenotype(HaploidGenotype("A"))      # 返回 GType 索引列表

# 标签注册
registry.register_somatic_label("infected")                # 返回 slab 索引
registry.register_gamete_label("cas9_deposited")           # 返回 glab 索引
```

## `@slab` 语法

当基因型字符串包含 `@slab` 后缀时，它同时指定了基因型模式和 slab 约束：

```
"A|a@infected"      → Genotype("A|a") 且 slab = "infected"
"WT|Dr@default"     → Genotype("WT|Dr") 且 slab = "default"
"Drive|WT"          → Genotype("Drive|WT") 无 slab 约束（见下文）
```

`@slab` 后缀由 `ZygoteTypePattern`（定义在 `natal.patterns.elements.diploid` 中）解析。基础基因型模式是最后一个 `@` 之前的所有内容。

### 命名约定：`genotypes` 接受 ZType 字符串

用户面向的参数名为 `genotypes` 以保持熟悉感，但它们实际接受 ZType 模式字符串，可包含可选的 `@slab` 后缀。这是有意为之——大多数用户只需要基因型选择，slab 是仅在使用细胞质 Preset 时才需要的进阶功能。

```python
# 典型用法——无 slab，仅基因型
Op.add(genotypes="Drive|WT", delta=500)

# 进阶用法——slab 约束
Op.add(genotypes="Drive|WT@infected", delta=500)
```

### `@` 缺失的行为：两种不同的规则

"没有 `@slab`"的含义取决于你调用的是哪个 API 函数：

| 上下文 | 解析方法 | 无 `@slab` 的含义 |
|--------|----------|-------------------|
| Hook（`Op.add`、`Op.kill` 等）和 `fitness()` | `resolve_ztype_indices()` | **所有 slabs**——匹配每个已注册的 slab 变体 |
| `initial_state()` | `resolve_default_ztype_index()` | **仅 `@default` slab**——返回第一个匹配的 ZType |

#### 为什么有这种区别？

- **Hook 和 `fitness()`** 使用 `resolve_ztype_indices()`，它返回模式匹配的每个 ZType。当模式没有 slab 约束（`slab is None`）时，`ZygoteTypePattern.matches()` 对**任何** slab 标签都返回 `True`，因此匹配基因型的所有 slab 变体都被选中。这是安全的默认行为——你不希望 hook 仅仅因为个体携带不同的 slab 而漏掉它们。

- **`initial_state()`** 使用 `resolve_default_ztype_index()`，它只返回**第一个**匹配的 ZType。当没有指定 `@slab` 时，这意味着只有 `@default` slab 变体被填充。这是有意为之：初始状态是对个体起始位置的精确规格说明，`@default` slab 是"未标记"状态。要将个体放在非默认 slab 中，需要显式使用 `@slab` 后缀或元组语法：

```python
# 将个体放入 @default slab（第一个匹配的 ZType）
.initial_state(individual_count={"male": {"Drive|WT": 500}})

# 将个体显式放入 @infected slab
.initial_state(individual_count={"male": {"Drive|WT@infected": 500}})

# 元组语法同样有效
.initial_state(individual_count={"male": {("Drive|WT", "infected"): 500}})
```

#### 重要：`initial_state` 键必须精确

传递给 `initial_state()` 的键必须是精确的基因型字符串——像 `"*|*"` 或 `"Drive|*"` 这样的模糊模式不会按预期工作。这样的模式可能只会静默匹配注册顺序中的**第一个** ZType，这几乎可以肯定是不正确的。如果你需要为初始状态使用模式风格的匹配，请改用 `first` 事件 hook 配合 `Op.set_count()`。

## 索引压缩（可达性 BFS）

### 动机

完整的组合空间 `(基因型 × slabs) × (单倍型 × glabs)` 可能很大。大多数基因型和单倍型从未从初始条件可达——它们有零个体，也没有遗传修饰因子产生它们。索引压缩会修剪这些不可达的条目，减小数组大小和计算量。

### BFS 算法

压缩使用不动点 BFS（在 `natal.genetics.structures._helpers` 的 `build_compression_mask` 中实现）。该算法对 GType 和 ZType 层次是对称的：

```
1. 种子：收集可达基因型
   a. initial_individual_count > 0  （初始时就有个体的基因型）
   b. 声明的基因型                     （来自 .declare() 的种子，见下文）

2. 从可达基因型推导可达单倍型：
   for each 可达基因型 g:
       reachable_haplotypes += gametes_produced_by(g)

3. 不动点迭代：
   for each (hl1, hl2) in reachable_haplotypes 的配对:
       for each g that (hl1, hl2) 能形成的基因型:
           if g 是新的:
               reachable_genotypes += g
               reachable_haplotypes += gametes_produced_by(g)
               → 继续迭代

4. 当没有新基因型/单倍型被发现 → 达到不动点。

5. 构建压缩掩码：
   - GType 掩码：-1 表示已修剪的（单倍型, glab）配对，≥0 表示幸存者
   - ZType 掩码：-1 表示已修剪的（基因型, slab）配对，≥0 表示幸存者
```

关键洞察：一旦可达集稳定下来，压缩掩码将旧索引映射到新的压缩后索引。被修剪的条目通过 `registry.compress(mask)` 从注册表中永久移除。

### Declare 语义

`declare` 机制（`setup(compress=True, declared_zygote_types={"AA"})` 或已弃用的 `compress_genotypes(True).declare("AA")` 链式方法）向 BFS 添加**种子**，而不仅仅是基因型列表中的最终条目。

示例：如果初始状态只有 `aa` 个体，而某个 hook 会在第 100 tick 释放 `AA` 个体：

1. 不使用 declare：BFS 仅从 `aa` 开始。可达单倍型是 `{a}`。不动点立即达到——`A` 从未被发现。当 hook 在运行时尝试释放 `AA` 时，其 ZType 索引为 -1（已修剪），导致错误。

2. 使用 `declare("AA")`：`AA` 是一个种子。可达单倍型变为 `{a, A}`。BFS 组合 `A` + `a` → 发现 `Aa`，它又产生 `{A, a}`。不动点：`{AA, Aa, aa}` 全部可达，压缩保留全部三个。

```
初始种子：{aa}
  + declare("AA")
  → reachable_genotypes = {aa, AA}
  → reachable_haplotypes = {a, A}
  → 组合 A + a → 发现 Aa（新！）
  → reachable_genotypes = {aa, AA, Aa}
  → 组合 A + A → AA（已知）
  → 组合 a + a → aa（已知）
  → 达到不动点
  → 全部 3 个基因型幸存于压缩
```

没有 `declare`，`A` 永远不会进入可达单倍型集，`AA`（和 `Aa`）将被修剪。

#### 关键点

- `declared_zygote_types` 在 `.setup(compress=True, declared_zygote_types=...)` 上设置。
- BFS 是**对称的**——声明一个基因型也会引入它产生的所有单倍型，这些单倍型可能组合形成未显式声明的其他基因型。
- 声明的基因型会扩展到**所有 slab 变体**用于 BFS（在内部，它们被视为跨所有 slabs 的可达 ZType）。
- 已弃用的 `.compress_genotypes(True).declare("AA")` 链式方法仍然有效，但推荐使用 `.setup(compress=True, declared_zygote_types={"AA"})`。

## 用户接口说明

**重要提示**：`IndexRegistry` 是底层数据表，用户通常无需直接调用其方法。用户访问基因型、单倍基因型或配子标签时，应使用以下高层接口：

### 使用索引访问

```python
# 通过 IndexRegistry 获取基因型索引后访问
idx = pop.index_registry.genotype_to_index["A1|A2"]
pop.state.individual_count[0, 3, idx]
```

### 使用 GenotypeSelector 模式匹配

```python
# 使用 GenotypeSelector 进行模式匹配操作
from natal.patterns import GenotypeSelector
selector = GenotypeSelector("A1|*", pop.index_registry)
indices = selector.select()  # 返回匹配的整数索引数组
```

**注意**：旧的导入路径 `from natal.genetic_patterns import GenotypeSelector` 已更新为 `from natal.patterns import GenotypeSelector`。

## 框架内部使用

在 NATAL 框架内部，`IndexRegistry` 被用于：

### 1. 状态数据存储

- 个体计数矩阵使用 ZType 索引进行高效存储（最后一个轴是 `n_ztypes`）。
- 精子存储矩阵使用 ZType 索引进行管理。
- 引擎中用于配子动态的数组使用 GType 索引。
- 所有状态数据都基于索引进行访问。

### 2. Modifier 系统

- Modifier 返回的对象字典会被自动转换为索引。
- 框架处理对象到索引的转换过程。
- 用户只需关注高层对象操作。

### 3. Hook 系统

- Numba Hook 使用预计算的索引进行高效操作。
- 避免在编译时访问动态注册表。
- 通过选择器模式避免硬编码索引。

### 4. 索引压缩

- `rebuild_config_maps()`（在 `natal.configurator._registry_builder` 中）运行 BFS。
- 生成的掩码通过 `registry.compress(ztype_mask, gtype_mask)` 应用。
- 压缩后，所有注册表属性只反映幸存的条目。

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
Genotype 对象（唯一）
    ↓ IndexRegistry.register_genotype()
ZType 条目（每个 slab 一个）
```

---

## 相关章节

- [遗传结构与实体](2_genetics.md) — Genotype 和 HaploidGenotype 的创建
- [PopulationState & PopulationConfig](4_population_state_config.md) — 配置中的索引应用
- [Modifier 机制](3_modifiers.md) — Modifier 中的 IndexRegistry 使用
- [Hook 系统](2_hooks.md) — 高级 Hook 选择器模式

---

**准备进入配置编译细节了吗？** [前往下一章：PopulationState & PopulationConfig →](4_population_state_config.md)
