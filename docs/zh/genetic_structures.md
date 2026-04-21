# 遗传架构

本章详细介绍 NATAL 中的遗传学对象系统，涵盖**结构层**（`Species` / `Chromosome` / `Locus`）和**实体层**（`Gene` / `Haplotype` / `HaploidGenotype` / `Genotype`），以及关键的字符串化和全局缓存机制。通过理解这些核心概念，您可以更好地构建和操作遗传模拟模型。

## 遗传学对象的层次结构

NATAL 采用分层架构来组织遗传学对象，将对象划分为两个主要层次：

### 结构层（静态模板）

描述模型允许出现的遗传空间，定义遗传架构的基本框架，不直接代表具体的个体类型：

- **`Species`**：物种级容器，管理染色体结构与全局索引
- **`Chromosome`**：染色体层级，组织基因位点与重组信息
- **`Locus`**：基因位点层级，定义该位置可取的等位基因集合

### 实体层（动态实例）

表示模拟过程中实际出现并演化的遗传对象：

- **`Gene`**：基因位点上的具体等位基因实例
- **`Haplotype`**：单条染色体上多个位点的等位基因组合
- **`HaploidGenotype`**：跨所有染色体的完整单倍体基因型
- **`Genotype`**：由母本（maternal）和父本（paternal）两个单倍体基因型组合形成的二倍体基因型

### 分层设计的优势

在分层架构中：

- **结构层**在建模阶段一次性定义，可复用且保持稳定
- **实体层**在模拟初始化阶段生成，用于构建遗传规则

分层设计既保持了 API 的清晰性，又便于底层实现索引化和高性能计算。

### 通用规则

- 所有遗传结构都继承自基类 `GeneticStructure`，所有遗传实体都继承自基类 `GeneticEntity`。
- 遗传结构和 `Gene` 在创建时需要指定字符串名称 `name`（默认为第一个参数），用于唯一标识该对象，从而可以通过 `get_...` 方法获取。
  - **注意**：同一类型的 `name` 不能重复。如果尝试创建同名对象，系统将返回缓存中的实例并发出警告。
- 创建遗传结构时（最顶层的 `Species` 除外），需要指定上级结构实例；可直接通过上级结构的 `add` 方法创建。
- 创建 `Gene` 时，需要指定所属 `Locus` 实例；可直接通过 `Locus` 的 `add_alleles` 方法创建。
- 其他各级遗传实体在种群初始化时自动创建，通常无需手动管理，可通过相应的字符串格式从 `Species` 中访问（或提前创建）实例。

## 结构层面详解

### Species：物种与遗传架构

`Species` 类是遗传架构的根节点，负责管理所有染色体结构和全局索引，是整个遗传系统的核心容器。

#### 创建方式 1：`from_dict`（快速定义，推荐）

推荐通过 `Species.from_dict` 方法，使用字典格式快速定义物种的遗传架构。

```python
import natal as nt

sp = nt.Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "A": ["WT", "Drive", "Resistance"],  # 位点 A，包含 3 个等位基因
            "B": ["B1", "B2"],                   # 位点 B，包含 2 个等位基因
        },
        "chr2": {
            "C": ["C1", "C2"],
        },
    },
    gamete_labels=["default", "Cas9_deposited"]  # 可选：配子标签，用于模拟母源效应（如 Cas9 蛋白沉积）
)
```

#### 扩展格式：声明性染色体信息

NATAL 支持多种性染色体系统，包括 XY、ZW 等。

当需要定义性染色体时，可以使用扩展格式来明确指定染色体类型：

```python
sp = nt.Species.from_dict(
    name="MosquitoSexAware",
    structure={
        "chrA": {    # 常染色体，无需扩展格式
            "A": ["A1", "A2"],
        },
        "chrX": {
            "sex_type": "X",    # X 染色体
            "loci": {
                "sx": ["X1"],
            },
        },
        "chrY": {
            "sex_type": "Y",    # Y 染色体
            "loci": {
                "sy": ["Y1"],
            },
        },
    },
)
```

可以检查染色体的性质：

```python
# 检查染色体性质
if chr_x.is_sex_chromosome:
    print(f"性染色体类型: {chr_x.sex_type}")  # 输出: "X"
    print(f"性染色体系统: {chr_x.sex_system}")  # 输出: "XY"
```

#### 创建方式 2：链式 API

链式 API 提供了更灵活的构建方式，适合需要动态构建或复杂配置的场景，能够更好地控制构建过程：

```python
sp = nt.Species("Mosquito")

# 常染色体
chr1 = sp.add("chr1")
chr1.add("A").add_alleles(["WT", "Drive", "Resistance"])
chr1.add("B").add_alleles(["B1", "B2"])

# X 染色体
chr_x = sp.add("ChrX", sex_type="X")
chr_x.add("white").add_alleles(["w+", "w"])

# Y 染色体（仅雄性）
chr_y = sp.add("ChrY", sex_type="Y")
chr_y.add("Ymarker").add_alleles(["Y"])

# 也支持 ZW 性染色体系统
# chr_w = sp.add("ChrW", sex_type="W")
```

### Chromosome：染色体

染色体（`Chromosome` 类）负责管理基因位点和重组信息。

染色体会在 `Species.from_dict()` 构建过程中自动创建，也可使用链式 API 创建并添加位点：

```python
chr1 = sp.add("chr1")
chr1.add("A").add_alleles(["A1", "A2"])
chr1.add("B").add_alleles(["B1", "B2"])
chr1.add("C").add_alleles(["C1", "C2"])
```

可以使用 `Species.get_chromosome()` 方法，通过名称获取染色体实例：

```python
# 通过名称获取染色体
chr1 = sp.get_chromosome("chr1")
chr_x = sp.get_chromosome("ChrX")
```

可以从 `Species` 中删除染色体：

```python
sp.remove_chromosome("chr1")
```

删除后，该染色体将从物种的遗传架构中移除，但该 `Chromosome` 实例将保持存在。

#### 重组率和重组图谱

重组率定义了一条染色体上位点之间发生交叉互换并生成重组型配子的概率，在模拟减数分裂生成配子时生效。

重组率通过重组图谱（`RecombinationMap`）管理，该图谱存储了相邻位点之间的重组率。

可使用以下方法，设置相邻位点之间的重组率。对于多个位点间的重组，假定无干涉效应，即每对位点之间的重组相互独立。

```python
# 方法 1：逐个设置相邻位点间的重组率
chr1.set_recombination("A", "B", 0.1)  # A-B 之间 10% 重组率
chr1.set_recombination("B", "C", 0.2)  # B-C 之间 20% 重组率

# 方法 2：批量设置重组率
chr1.set_recombination_bulk({
    ("A", "B"): 0.1,
    ("B", "C"): 0.2
})

# 方法 3：使用 Locus 名称作为索引访问重组图谱
chr1.recombination_map["A", "B"] = 0.1
chr1.recombination_map["B", "C"] = 0.2

# 方法 4：使用切片一次性设置所有相邻区间的重组率（列表长度必须等于位点数-1）
chr1.recombination_map[:] = [0.1, 0.2]
```

相邻位点间的重组率应在 $[0.0, 0.5]$ 范围内，其中 $0.0$ 表示完全连锁，无交换；$0.5$ 表示交换必然发生，近似于自由组合。

相邻位点间未指定重组率时，默认值为 $0.0$，即完全连锁。

> **注意**：重组率设置依赖于位点的顺序，位点顺序由 `position` 参数控制。详见 [Locus：基因位点](#locus基因位点) 中的 position 参数说明。

### Locus：基因位点

位点（`Locus` 类）定义特定位置的等位基因集合。

与染色体一样，位点会在 `Species.from_dict()` 构建过程中自动创建，也可使用链式 API 创建并添加等位基因。

```python
chr1 = sp.get_chromosome("chr1")
chr1.add("A").add_alleles(["A1", "A2"])
chr1.add("B").add_alleles(["B1", "B2"])
chr1.add("C").add_alleles(["C1", "C2"])
```

使用链式 API 创建 `Locus` 时，可以指定以下参数：

- `position`：表示位点在染色体上的相对位置。若未指定 `position` 参数，将默认设置为 `max(现有位点position) + 1`。
- `recombination_rate_with_previous`：表示该位点与前一个位点之间的重组率。若未指定，将默认设置为 $0.0$，即完全连锁。**若是第一个位点**，则表示该位点与下一个位点之间的重组率。

```python
chr1.add("A", position=0.0)
chr1.add("B", position=50.0)
chr1.add("C", position=100.0, recombination_rate_with_previous=0.05)
```

可以通过以下方法获取 `Locus` 实例：

```python
# 通过名称获取位点
locus_A = chr1.get_locus("A")
locus_B = chr1.get_locus("B")

# 在整个 Species 范围内获取位点
locus = sp.get_locus("A")
```

可以从染色体上删除 `Locus` 实例：

```python
chr1.remove_locus("A")
```

删除后，该位点将从染色体上移除，但该 `Locus` 实例将保持存在。该位点两侧的位点成为新的相邻位点，其重组率自动设置为原位点两侧的重组率之和。

#### 关于 `position` 参数

`position` 参数用于定义位点在染色体上的相对位置，**仅作为排序标签使用**，其绝对大小与重组率大小无关。

如果不指定 `position`，系统会自动设置为 `max(现有位点position) + 1`。

请尽量避免在创建后修改 `position` 参数，因为这可能导致预期之外的结果。推荐在创建时一次性设置好 `position` 参数。

> **说明**：若在创建后修改 `position` 参数，且修改后改变了位点的顺序，系统将更新重组率，其行为等价于将该位点移除然后重新添加到指定位置，且与上一位点的重组率为 $0.0$。

## 实体层面详解

### Gene：等位基因实例

`Gene` 类是结构层与实体层的交界点，代表具体的等位基因实例。

**`Gene` 的标识名称 `name` 必须在 `Species` 范围内保持唯一。**

可通过 `Species.get_gene` 快速获取 `Gene` 实例，但一般无需对 `Gene` 实例进行直接操作。在需要指定具体等位基因的场合，通常可以直接使用字符串名称 `name` 来引用等位基因实例。

```python
# 在整个 Species 范围内获取等位基因
gene_wt = sp.get_gene("WT")
gene_drive = sp.get_gene("Drive")
```

### Haplotype：单倍型

`Haplotype` 表示单条染色体上所有位点的等位基因组合。对于包含 $N$ 个位点的染色体，可能的单倍型数量为所有位点等位基因数的乘积：$\prod\_{i=1}^N \text{每个位点的等位基因数}$。

一般无需手动获取 `Haplotype` 实例。

```python
# 获取染色体上所有可能的单倍型
chr1 = sp.get_chromosome("chr1")  # 获取染色体对象
all_haplotypes = chr1.get_all_haplotypes()

# 遍历所有单倍型
for hap in all_haplotypes:
    print(f"单倍型: {hap}")
    # 访问每个位点的等位基因
    for locus in chr1.loci:
        gene = hap.get_gene_at_locus(locus)
        print(f"  {locus.name}: {gene.name}")
```

### HaploidGenotype：单倍体基因型

`HaploidGenotype` 表示完整的单倍体基因型，包含物种所有染色体的单倍型组合。

#### 从格式化字符串获取单倍体基因型

**字符串解析是最灵活的方式**，支持直接从字符串获取单倍体基因型。当打印输出单倍体基因型时，也会自动转换为字符串格式，与输入的字符串格式保持一致。

```python
sp = nt.Species.from_dict(
    name="TestDrive",
    structure={
        "chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
        "chr2": {"X": ["X", "x"], "Y": ["Y", "y"]}
    }
)

# 直接从字符串获取单倍体基因型
hg1 = sp.get_haploid_genotype_from_str("ABC;XY")
hg2 = sp.get_haploid_genotype_from_str("a/b/c;x/y")  # 等价写法

print(f"单倍体基因型: {hg1}")  # 输出: ABC;XY
```

#### 字符串解析语法规则

字符串解析遵循以下语法规则：

- **分号 (;) 分隔不同染色体**：每个分号分隔一个染色体的基因组合
- **斜杠 (/) 分隔同一染色体内的基因**：每个斜杠分隔一个位点的等位基因
- **单字符基因可省略斜杠**：如果所有基因都是单字符，可以省略斜杠分隔符
- **多字符基因必须使用斜杠**：基因名称包含多个字符时，必须用斜杠分隔

```python
# 示例 1：单字符基因，可省略斜杠
hg1 = sp.get_haploid_genotype_from_str("ABC;XY")
# 等价于：hg1 = sp.get_haploid_genotype_from_str("A/B/C;X/Y")

# 示例 2：多字符基因，必须使用斜杠
hg2 = sp.get_haploid_genotype_from_str("WT/Drive/R2;X/Y")

# 示例 3：混合单字符和多字符基因
hg3 = sp.get_haploid_genotype_from_str("A/WT/Drive;X/Y")
```

#### 缓存机制

`HaploidGenotype` 使用 **Species 范围内的缓存机制**，通过可回解析且有序的字符串作为键，确保性能和一致性。

```python
# 字符串解析会自动缓存
hg1 = sp.get_haploid_genotype_from_str("ABC;XY")
hg2 = sp.get_haploid_genotype_from_str("ABC;XY")

print(hg1 is hg2)  # 输出: True（同一个实例）
```

### Genotype：二倍体基因型（核心概念）

`Genotype` 是 NATAL 中最核心的遗传对象，代表一类个体的完整二倍体基因型，由母本（maternal）和父本（paternal）两个单倍体基因型组成。

#### 从格式化字符串获取基因型

与 `HaploidGenotype` 相同，`Genotype` 也支持从字符串直接获取基因型，以及打印输出基因型时自动输出字符串格式。

```python
sp = nt.Species.from_dict(
    name="TestDrive",
    structure={"chr1": {"loc": ["WT", "Drive"]}}
)

# 直接从字符串获取基因型
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
drive_drive = sp.get_genotype_from_str("Drive|Drive")

print(f"基因型: {wt_drive}")  # 输出: WT|Drive（严格保留 maternal|paternal 顺序）
```

#### 字符串解析语法规则

`Genotype` 的字符串解析语法与 `HaploidGenotype` 基本相同，增加了母本和父本的分隔：

- **竖线 (|) 分隔母本和父本**：竖线左侧为母本单倍体基因型，右侧为父本单倍体基因型
- **其他规则与 HaploidGenotype 相同**：包括分号分隔染色体、斜杠分隔基因、单字符基因可省略斜杠等规则

```python
# 示例 1：单字符基因，可省略斜杠
gt1 = sp.get_genotype_from_str("ABC|abc")
# 等价于：gt1 = sp.get_genotype_from_str("A/B/C|a/b/c")

# 示例 2：多字符基因，必须使用斜杠
gt2 = sp.get_genotype_from_str("WT/Drive/R2|WT/Drive/R2")

# 示例 3：混合单字符和多字符基因
gt3 = sp.get_genotype_from_str("A/WT/Drive|a/WT/Drive")
```

#### 缓存机制

与 `HaploidGenotype` 一样，`Genotype` 使用 **Species 范围内的缓存机制**，通过可回解析且有序的字符串作为键，确保性能和一致性。

#### Pattern：字符串格式的自然扩展

在掌握了精确字符串格式的基础上，NATAL 提供了 **Pattern 模式匹配** 作为字符串格式的自然扩展，支持通配符和高级匹配功能。

**Pattern 是精确字符串格式的超集**：在精确字符串格式的基础上增加了以下功能：
- `*` 通配符：匹配任何等位基因
- `{A,B,C}` 集合匹配：匹配集合中的任何等位基因
- `!A` 排除匹配：匹配除 A 外的任何等位基因
- `()` 分组：明确分组染色体位点
- `::` 无序匹配：表示母本和父本顺序无关

详细规则及示例请参考 [基因型模式匹配](genotype_patterns.md)。

**`Species` 中的相关方法**：
- `parse_genotype_pattern(pattern: str)`：解析二倍体基因型模式
- `parse_haploid_genome_pattern(pattern: str)`：解析单倍体基因型模式
- `enumerate_genotypes_matching_pattern(pattern: str)`：枚举匹配模式的基因型
- `enumerate_haploid_genomes_matching_pattern(pattern: str)`：枚举匹配模式的单倍体基因型

Pattern 语法保持了与精确字符串格式的兼容性，所有精确字符串都能被 Pattern 正确匹配。

## 完整示例

```python
# 1. 定义遗传架构
sp = nt.Species.from_dict(
    name="ComplexSpecies",
    structure={
        "chr1": {
            "A": ["A1", "A2"],
            "B": ["B1", "B2", "B3"],
        },
        "chr2": {
            "C": ["C1", "C2"],
        },
    }
)

# 2. 检查规模
all_haploid = sp.get_all_haploid_genotypes()
all_genotypes = sp.get_all_genotypes()
print(f"Haploid genotypes: {len(all_haploid)}")  # 2*3*2 = 12
print(f"Diploid genotypes: {len(all_genotypes)}")  # 12*12 = 144

# 3. 操作特定基因型
gt = sp.get_genotype_from_str("A1|A2")
print(f"Maternal haplotype: {gt.maternal}")
print(f"Paternal haplotype: {gt.paternal}")
```

## 实体层与 IndexRegistry 的协同工作机制

### 实体层的生成时机与 IndexRegistry 管理

在**模拟初始化阶段**，各 `Genotype` 和 `HaploidGenotype` 会生成并注册到 IndexRegistry 进行管理。具体工作机制如下：

```
字符串 "A1|A2"
    ↓ Species.get_genotype_from_str()
全局缓存 Species.genotype_cache
    ↓ [命中缓存]
Genotype 对象 (唯一实例)
    ↓ IndexRegistry.register_genotype()
整数索引 (例如 5)
    ↓
numpy 数组访问 individual_count[:, :, 5]
```

### Genotype 对象与 IndexRegistry 的配合

```python
pop = nt.AgeStructuredPopulation(species=sp, ...)

# 获取 IndexRegistry
registry = pop.registry  # 或 pop._index_registry

# Genotype → 整数索引
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = registry.genotype_index(gt)
print(f"Genotype index: {gt_idx}")

# 反向：整数索引 → Genotype
gt_back = registry.index_to_genotype[gt_idx]
print(f"Genotype: {gt_back}")

# 在 numpy 数组中使用
individual_count = pop.state.individual_count  # shape: (n_sexes, n_ages, n_genotypes)
female_count_of_gt = individual_count[1, :, gt_idx]  # 某基因型所有年龄的雌性数量
```

> 更多关于 IndexRegistry 的细节，见 [IndexRegistry 索引机制](index_registry.md)

***

## 🎯 本章总结

| 层面     | 对象         | 用途          | 创建/获取方式                       |
| ------ | ---------- | ----------- | ---------------------------- |
| **结构** | `Species`    | 定义物种遗传架构    | `from_dict()` 或链式 API        |
| **结构** | `Chromosome`    | 定义染色体和重组    | `species.add()`              |
| **结构** | `Locus`      | 定义基因位点      | `chromosome.add()`           |
| **实体** | `Gene`       | 等位基因实例      | `sp.get_gene()`              |
| **实体** | `HaploidGenotype` | 单倍体基因型 | `sp.get_haploid_genotype_from_str()` |
| **实体** | `Genotype`   | 二倍体基因型 | `sp.get_genotype_from_str()` |

### 关键特性

1. **全局缓存机制**：`Genotype` 使用字符串缓存，确保性能和一致性
2. **双向转换**：字符串和对象可互相转换，支持灵活操作
3. **索引映射**：与 `IndexRegistry` 配合实现对象和索引间的高效映射
4. **分层设计**：结构层与实体层分离，支持复杂遗传架构建模

### 应用价值

遗传架构系统为种群遗传模拟提供强大的建模能力。理解遗传架构系统是使用 NATAL 高级功能（如 `Modifier` 机制）的基础。

***

## 📚 相关章节

- [快速开始：15 分钟上手 NATAL](quickstart.md) - 基本使用示例
- [Builder 系统详解](builder_system.md) - 从 Species 到可运行种群的链式构建
- [IndexRegistry 索引机制](index_registry.md) - 对象索引的详细机制
- [Genotype Pattern 语法与匹配](genotype_patterns.md) - 基因型模式表达、`|`/`::` 顺序规则与匹配示例
- [Modifier 机制](modifiers.md) - 如何基于 Genotype 定义遗传规则

***

**准备开始构建种群了吗？** [前往下一章：Builder 系统详解 →](builder_system.md)
