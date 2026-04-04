# 遗传结构与实体

本章深入讲解 NATAL 中的遗传学对象系统：结构层面（Species/Chromosome/Locus）、实体层面（Gene/Genotype），以及关键的字符串化和全局缓存机制。

## 从快速开始迁移的概念说明

在快速开始中，我们只保留了“结构层 vs 实体层”的最小心智模型。这里给出完整解释：

- **结构层（静态模板）**：描述模型允许出现的遗传空间，不直接代表某个个体。
    - `Species`：物种级容器，管理染色体与全局索引。
    - `Chromosome`：染色体层级，组织位点与重组信息。
    - `Locus`：位点层级，定义该位置可取的等位基因集合。
- **实体层（动态实例）**：表示模拟过程中真实出现并演化的遗传对象。
    - `Gene`：位点上的一个具体等位基因。
    - `Haplotype`：单条染色体上多位点等位基因组合。
    - `HaploidGenotype`：跨所有染色体的单倍体基因组。
    - `Genotype`：由母本与父本两个单倍体组合得到的二倍体基因型。

为什么要分层：

- 结构层在建模阶段定义一次，可复用且稳定。
- 实体层在模拟阶段不断出现、转化和统计。
- 分层后既能保持 API 清晰，也便于底层做索引化与高性能计算。

## 核心概念

NATAL 将遗传学对象分为两层：

### 🔧 结构层面 (Structures)

**静态**、模型级别的蓝图，定义遗传架构的拓扑：

| 类 | 说明 | 示例 |
|---|---|---|
| `Species` | 物种根节点，包含所有染色体 | `Species("Mosquito")` |
| `Chromosome` | 染色体（连锁群），包含位点 | 常染色体、X/Y/W 性染色体 |
| `Locus` | 基因位点，包含等位基因 | "A", "B", "white" |

### 🧬 实体层面 (Entities)

**具体**、个体级别的遗传对象，代表实际的遗传物质：

| 类 | 说明 | 绑定到 | 示例 |
|---|---|---|---|
| `Gene` | 等位基因 | Locus | "w+"、"w"、"Drive" |
| `Haplotype` | 单倍型（单条染色体的等位基因组合） | Chromosome | 可能有数百种 |
| `HaploidGenotype` | 单倍体基因组（所有染色体的单倍型） | Species | $\sim 2^\text{位点数}$ 种 |
| `Genotype` | 二倍体基因型 | Species | $\sim \text{HaploidGenotype数}^2$ 种 |

## 结构层面详解

### Species：物种与遗传架构

#### 创建方式 1：from_dict（推荐快速定义）

```python
from natal.genetic_structures import Species

sp = Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "A": ["WT", "Drive", "Resistance"],  # 位点 A，3 个等位基因
            "B": ["B1", "B2"],                   # 位点 B，2 个等位基因
        },
        "chr2": {
            "C": ["C1", "C2"],
        },
    }
)
```

优点：语法简洁，易于从配置文件加载。

#### 创建方式 2：链式 API（更灵活）

```python
sp = Species("Mosquito")

# 常染色体
chr1 = sp.add("chr1")
chr1.add("A").add_alleles(["WT", "Drive", "Resistance"])
chr1.add("B").add_alleles(["B1", "B2"])

# X 连锁
chr_x = sp.add("ChrX", sex_type="X")
chr_x.add("white").add_alleles(["w+", "w"])

# Y 连锁（仅雄性）
chr_y = sp.add("ChrY", sex_type="Y")
chr_y.add("Ymarker").add_alleles(["Y"])

# W 连锁（仅雌性，某些鸟类/蝶类）
# chr_w = sp.add("ChrW", sex_type="W")
```

### 性染色体

```python
# 检查性质
if chr_x.is_sex_chromosome:
    print(f"Sex type: {chr_x.sex_type}")  # "X"
    print(f"Sex system: {chr_x.sex_system}")  # "XY"
```

### Chromosome：染色体与重组

```python
chr1 = sp.add("chr1")
locus_A = chr1.add("A", position=0.0)      # 位置 0
locus_B = chr1.add("B", position=50.0)     # 位置 50

# 设置重组率（A-B 之间）
chr1.recombination_map[0] = 0.1  # 10% 重组率

# 或通过工厂方法
from natal.genetic_structures import Chromosome
chr2 = Chromosome(
    name="chr2",
    recombination_rates=[0.05, 0.1, 0.05]  # 多个位点间的重组率
)
```

### Locus：基因位点

```python
# 方式1：add 后添加
locus = chr1.add("A")
locus.add_alleles(["A1", "A2", "A3"])

# 方式2：工厂方法
from natal.genetic_structures import Locus
locus = Locus.with_alleles("A", ["A1", "A2"])

# 查看等位基因
for allele in locus.alleles:
    print(allele.name)
```

## 实体层面详解

### Gene：等位基因实例

等位基因是"结构"与"实体"的交界点。

```python
# 获取等位基因（通过 Species）
sp = Species.from_dict(
    name="Test",
    structure={"chr1": {"A": ["a1", "a2"]}}
)

a1 = sp.get_gene("a1")
a2 = sp.get_gene("a2")

print(f"Gene name: {a1.name}")
print(f"Locus: {a1.locus}")  # Locus 对象
print(f"Allele index: {a1.allele_index}")  # 0 或 1
```

### Haplotype：单倍型

单条染色体上的等位基因组合。对于 N 个位点，有 ∏(每个位点的等位基因数) 种可能。

```python
# 获取所有可能的单倍型
chr1 = sp["chr1"]  # 获取染色体
all_haplotypes = chr1.get_all_haplotypes()

for hap in all_haplotypes:
    print(f"Haplotype: {hap}")
    # 访问每个位点的等位基因
    for locus in chr1.loci:
        gene = hap[locus]
        print(f"  {locus.name}: {gene.name}")
```

### HaploidGenotype：单倍体基因组

完整的单倍体基因组，包含所有染色体的单倍型。

```python
# 获取所有可能的单倍体基因组
all_haploid_gts = sp.get_all_haploid_genotypes()
print(f"Total haploid genotypes: {len(all_haploid_gts)}")

# 访问特定染色体的单倍型
hg = all_haploid_gts[0]
hap_chr1 = hg["chr1"]  # Haplotype 对象
```

### Genotype：二倍体基因型（最重要！）

代表一个个体的基因型，由两个单倍体基因组（maternal、paternal）组成。

#### 获取基因型的方式

**方式1：字符串解析（推荐）**

```python
# 最灵活的方式——直接从字符串创建
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
drive_drive = sp.get_genotype_from_str("Drive|Drive")

print(f"Genotype: {wt_drive}")  # 输出: WT|Drive 或 Drive|WT（取决于排序）
```

**方式2：从单倍体基因组**

```python
hg1 = all_haploid_gts[0]
hg2 = all_haploid_gts[1]
genotype = sp.make_genotype(hg1, hg2)
```

**方式3：获取所有可能的基因型**

```python
all_genotypes = sp.get_all_genotypes()
print(f"Total diploid genotypes: {len(all_genotypes)}")

for gt in all_genotypes:
    print(f"  {gt}")
```

#### 字符串化 Genotype（全局缓存）

**这是 NATAL 的一个关键特性**：Genotype 使用全局缓存，通过字符串作为标准化键。

```python
# 关键点1：字符串规范化
wt_drive = sp.get_genotype_from_str("WT|Drive")
drive_wt = sp.get_genotype_from_str("Drive|WT")

# 在 Species 层面，它们被规范化为同一个对象
print(wt_drive == drive_wt)  # True
print(str(wt_drive) == str(drive_wt))  # True

# 关键点2：缓存意味着字符串一致性
gt1 = sp.get_genotype_from_str("WT|Drive")
gt2 = sp.get_genotype_from_str("WT|Drive")
print(gt1 is gt2)  # True（同一对象）

# 关键点3：可以用于字典键
genotype_map = {
    wt_drive: 100,
    drive_drive: 50,
}
```

#### 访问基因型内部结构

```python
gt = sp.get_genotype_from_str("WT|Drive")

# 获取两个单倍体基因组
mat = gt.maternal  # HaploidGenotype
pat = gt.paternal  # HaploidGenotype

# 获取特定染色体的单倍型
mat_chr1_hap = mat["chr1"]  # Haplotype
pat_chr1_hap = pat["chr1"]

# 获取特定位点的等位基因
mat_A = mat_chr1_hap["A"]  # Gene
pat_A = pat_chr1_hap["A"]

print(f"Maternal A: {mat_A.name}")  # "WT" 或 "Drive"
print(f"Paternal A: {pat_A.name}")

# 判断杂合性
is_het_A = mat_A != pat_A
print(f"Heterozygous at A: {is_het_A}")

# 或使用内置方法
print(f"Overall heterozygous: {gt.is_heterozygous()}")
print(f"Heterozygous at locus A: {gt.is_heterozygous(sp['chr1']['A'])}")
```

## 全局缓存机制

### 为什么需要全局缓存？

1. **内存效率**：避免创建重复的 Genotype 对象
2. **哈希一致性**：字符串规范化确保 `"WT|Drive"` 和 `"Drive|WT"` 映射到同一对象
3. **索引稳定性**：与 IndexRegistry 配合，每个 Genotype 对应唯一的整数索引

### 缓存的工作原理

```
字符串 "WT|Drive"
    ↓ [规范化] → 按字母序排序
"Drive|WT"
    ↓ [查缓存]
Species.genotype_cache["Drive|WT"]
    ↓ [命中或创建]
Genotype 对象（全局唯一）
    ↓ [注册到 IndexRegistry]
整数索引（例如 idx=5）
```

### 使用缓存

```python
# 缓存自动管理，用户无需显式操作
gt1 = sp.get_genotype_from_str("WT|Drive")
gt2 = sp.get_genotype_from_str("WT|Drive")
# gt1 和 gt2 是同一对象，无额外内存开销

# 在字典中作为键
fitness_map = {
    sp.get_genotype_from_str("WT|WT"): 1.0,
    sp.get_genotype_from_str("WT|Drive"): 0.95,
}
# 字典查询会自动使用 Genotype 的 __hash__ 和 __eq__

# 在初始化中引用
initial_individual_count = {
    "female": {
        "WT|WT": [600, 500, 400, ...],
        "WT|Drive": [100, 80, 60, ...],
    }
}
# 字符串会在初始化时被解析，由 Species 自动转换为 Genotype 对象
```

## 性别特异的基因型

某些遗传架构中，基因型与性别关联：

```python
# 例：X 连锁遗传
sp = Species.from_dict(
    name="XLinked",
    structure={
        "ChrX": {"white": ["w+", "w"]},
        "ChrY": {"Ymarker": ["Y"]},
    }
)

# X 连锁位点的基因型
# 雌性（XX）：w+|w+, w+|w, w|w（二倍体）
# 雄性（XY）：w+|Y, w|Y（实际上只有一个等位基因）

all_gts = sp.get_all_genotypes()
for gt in all_gts:
    print(f"{gt} (Sex: {gt.sex_type})")
    # 输出：
    # w+|w+ (Sex: both)
    # w+|w (Sex: both)
    # w|w (Sex: both)
    # w+|Y (Sex: male)
    # w|Y (Sex: male)
```

## 遗传实体的完整示例

```python
from natal.genetic_structures import Species

# 1. 定义遗传架构
sp = Species.from_dict(
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
print(f"Diploid genotypes: {len(all_genotypes)}")  # (12+1)*12/2 = 78

# 3. 操作特定基因型
gt = sp.get_genotype_from_str("A1|A2")
print(f"Maternal haplotype: {gt.maternal}")
print(f"Paternal haplotype: {gt.paternal}")

# 4. 遍历所有基因型
for gt in all_genotypes:
    # 在模型中使用
    pop.initial_individual_count["female"][str(gt)] = 0
```

## 与 IndexRegistry 的关系

Genotype 对象与 IndexRegistry（索引机制）紧密配合：

```python
from natal.nonWF_population import AgeStructuredPopulation

pop = AgeStructuredPopulation(species=sp, ...)

# 获取 IndexRegistry
ic = pop.registry  # 或 pop._index_registry

# Genotype → 整数索引
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = ic.genotype_index(gt)
print(f"Genotype index: {gt_idx}")

# 反向：整数索引 → Genotype
gt_back = ic.index_to_genotype[gt_idx]
print(f"Genotype: {gt_back}")

# 在 numpy 数组中使用
individual_count = pop.state.individual_count  # shape: (n_sexes, n_ages, n_genotypes)
female_count_of_gt = individual_count[1, :, gt_idx]  # 某基因型所有年龄的雌性数量
```

> 更多关于 IndexRegistry 的细节，见 [IndexRegistry 索引机制](index_registry.md)

## 常见操作速查

### 创建和查询基因型

```python
# 从字符串获取
gt = sp.get_genotype_from_str("WT|Drive")

# 所有可能的基因型
all_gts = sp.get_all_genotypes()

# 特定基因型是否存在
try:
    gt = sp.get_genotype_from_str("Invalid|Invalid")
except KeyError:
    print("基因型不存在")
```

### 访问基因型内部

```python
gt = sp.get_genotype_from_str("A1|A2")

# 两个单倍体基因组
mat = gt.maternal
pat = gt.paternal

# 特定染色体的单倍型
mat_hap = mat["chr1"]
pat_hap = pat["chr1"]

# 特定位点的等位基因
mat_gene = mat_hap["A"]
pat_gene = pat_hap["A"]

# 基因的名称和属性
print(mat_gene.name)  # "A1"
print(mat_gene.locus)  # Locus 对象
```

### 检查和操作

```python
gt = sp.get_genotype_from_str("A1|A2")
locus = sp["chr1"]["A"]

# 是否杂合
is_het = gt.is_heterozygous(locus)

# 获取对应位点的等位基因
alleles_at_locus = gt.get_alleles_at_locus(locus)  # (mat_gene, pat_gene)
```

---

## 🎯 本章总结

| 层面 | 对象 | 用途 | 创建方式 |
|------|------|------|---------|
| **结构** | Species | 定义物种遗传架构 | `from_dict()` 或链式 API |
| **结构** | Chromosome | 定义染色体和重组 | `species.add()` |
| **结构** | Locus | 定义基因位点 | `chromosome.add()` |
| **实体** | Gene | 等位基因实例 | `sp.get_gene()` |
| **实体** | Genotype | 二倍体基因型（最常用） | `sp.get_genotype_from_str()` |

**关键点**：
1. Genotype 使用全局缓存和字符串规范化
2. 字符串和对象可互相转换
3. Genotype 与 IndexRegistry 配合实现对象↔索引映射
4. 理解这些是使用 Modifier 和高级功能的基础

---

## 📚 相关章节

- [快速开始：15 分钟上手 NATAL](quickstart.md) - 基本使用示例
- [Builder 系统详解](builder_system.md) - 从 Species 到可运行种群的链式构建
- [IndexRegistry 索引机制](index_registry.md) - 对象索引的详细机制
- [Modifier 机制](modifiers.md) - 如何基于 Genotype 定义遗传规则

---

**准备开始构建种群了吗？** [前往下一章：Builder 系统详解 →](builder_system.md)
