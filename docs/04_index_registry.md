# IndexRegistry 索引机制

本章讲解 NATAL 如何将遗传学对象（Genotype、HaploidGenotype 等）与整数索引关联。这是连接"高层对象世界"与"底层数值计算世界"的关键。

## 核心概念

遗传学对象的世界 ↔ 整数索引的世界

```
Genotype("Drive|WT")      ↔  整数索引 2
HaploidGenotype("R1")     ↔  整数索引 3
"default" (glab)          ↔  整数索引 0
"Cas9_deposited" (glab)   ↔  整数索引 1
```

**目的**：numpy 数组使用整数索引（效率高），但用户想用遗传学对象（直观）。IndexRegistry 负责在两者之间转换。

## IndexRegistry 类

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

```python
ic = pop.registry  # IndexRegistry 实例

# 注册一个基因型
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = ic.register_genotype(gt)
# 返回：整数索引，例如 5

# 查询索引
gt_idx = ic.genotype_index(gt)  # 获取已注册基因型的索引

# 反向查询
gt_back = ic.index_to_genotype[gt_idx]  # 获取对应的 Genotype 对象

# 统计
n_gt = ic.num_genotypes()  # 总共有多少基因型
```

### 2. 单倍基因型索引

```python
# 单倍基因型（单倍体基因组）
hg = all_haploid_genotypes[0]

# 注册
hg_idx = ic.register_haplogenotype(hg)

# 查询
hg_idx = ic.haplo_index(hg)
hg_back = ic.index_to_haplo[hg_idx]

# 统计
n_hg = ic.num_haplogenotypes()
```

### 3. 配子标签索引

```python
# 配子标签（如 "default", "Cas9_deposited"）

# 注册
label_idx = ic.register_gamete_label("Cas9_deposited")

# 查询
label_idx = ic.gamete_label_index("Cas9_deposited")
label_back = ic.index_to_glab[label_idx]

# 统计
n_glabs = ic.num_gamete_labels()
```

## 实际使用示例

### 例 1：访问个体计数

```python
from natal.type_def import Sex

state = pop.state
ic = pop.registry

# 获取特定基因型的个体计数
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = ic.genotype_index(gt)

# 雌性、年龄 3、基因型 A1|A2 的个体数量
count = state.individual_count[Sex.FEMALE, 3, gt_idx]

# 所有年龄、基因型 A1|A2 的雌性总数
total = state.individual_count[Sex.FEMALE, :, gt_idx].sum()
```

### 例 2：修改精子存储

```python
female_gt = sp.get_genotype_from_str("A1|A1")
male_gt = sp.get_genotype_from_str("A1|A2")

female_gt_idx = ic.genotype_index(female_gt)
male_gt_idx = ic.genotype_index(male_gt)

# 年龄 3 的 A1|A1 雌性存储来自 A1|A2 雄性的精子
state.sperm_storage[3, female_gt_idx, male_gt_idx] = 500.0
```

### 例 3：遍历所有基因型

```python
ic = pop.registry

for gt_idx in range(ic.num_genotypes()):
    gt = ic.index_to_genotype[gt_idx]

    # 查看这个基因型的总个体数
    total = state.individual_count[:, :, gt_idx].sum()

    print(f"{gt}: {total:.0f} individuals")
```

## 压缩索引（Advanced）

在处理配子和标签组合时，需要处理复合索引：

```python
# 配子可以是：(haploid_idx, label_idx) 的组合
# 例：(A1, default) 和 (A1, Cas9) 是两个不同的"标记配子"

def compress_hg_glab(haploid_idx: int, label_idx: int, n_glabs: int) -> int:
    """将 (haploid, label) 压缩为单个索引"""
    return haploid_idx * n_glabs + label_idx

def decompress_hg_glab(compressed: int, n_glabs: int) -> Tuple[int, int]:
    """将单个索引解压为 (haploid, label)"""
    return compressed // n_glabs, compressed % n_glabs

# 使用示例
n_glabs = 2
haploid_idx = 3
label_idx = 1

compressed = compress_hg_glab(haploid_idx, label_idx, n_glabs)
# compressed = 3 * 2 + 1 = 7

haploid_back, label_back = decompress_hg_glab(compressed, n_glabs)
# (3, 1)
```

**用途**：在映射矩阵中，配子通常被压缩为单个维度以节省空间。

## Modifier 中的使用

在编写 Modifier 时，需要利用 IndexRegistry 获取基因型索引：

```python
def gene_drive_modifier(pop):
    """
    基因驱动 Modifier：Drive|WT 杂合子的驱动等位基因配子比例提高
    """
    ic = pop.registry

    # 找到 Drive|WT 基因型的索引
    drive_wt = pop.species.get_genotype_from_str("Drive|WT")
    drive_wt_idx = ic.genotype_index(drive_wt)

    # 返回映射：基因型 → {(配子, label): 频率}
    return {
        drive_wt: {
            ("Drive", "Cas9_deposited"): 0.95,  # 95% 的驱动配子
            ("WT", "Cas9_deposited"): 0.05,
        }
    }

pop.set_gamete_modifier(gene_drive_modifier, hook_name="drive")
```

## Hook 中的使用

在 Numba Hook 中需要工作于索引而非对象：

```python
from numba import njit

@njit
def my_hook(ind_count, tick):
    """
    Numba 兼容的 Hook——必须使用整数索引

    注意：Hook 在编译时无法访问 IndexRegistry，
    所以需要在 Python 层确定索引，然后硬编码传入
    """
    if tick == 10:
        # 假设我们已经知道基因型 A1|A2 的索引是 5
        gt_idx = 5

        # 杀死所有年龄 2-4 的该基因型雌性
        ind_count[1, 2:5, gt_idx] = 0

    return 0  # 继续

# 使用选择器模式避免硬编码（见 Hook 系统 章节）
```

## 与 Modifier 的交互

Modifier 返回的字典可以使用对象作为键，框架会自动转换为索引：

```python
# 用户编写的 Modifier（高层）
def my_modifier(pop):
    return {
        sp.get_genotype_from_str("A1|A2"): {
            ("A1", "default"): 0.6,
            ("A2", "default"): 0.4,
        }
    }

# 框架处理（自动）
for gt_key, gamete_dict in my_modifier(pop).items():
    gt_idx = pop.registry.genotype_index(gt_key)

    for (allele_name, label), freq in gamete_dict.items():
        # 查询 allele_name 对应的单倍基因型索引
        # 查询 label 对应的标签索引
        # 写入映射矩阵
        pass
```

## 性能提示

### 缓存索引查询

```python
# ❌ 低效：重复查询
for tick in range(100):
    gt_idx = pop.registry.genotype_index(gt)  # 每次都查字典
    # ...

# ✅ 高效：只查一次
gt_idx = pop.registry.genotype_index(gt)
for tick in range(100):
    # 复用 gt_idx
    # ...
```

### 批量操作

```python
# ❌ 低效：逐个修改
for gt in all_genotypes:
    gt_idx = ic.genotype_index(gt)
    config.viability_fitness[FEMALE, 3, gt_idx] = 0.9

# ✅ 高效：向量化
gt_indices = [ic.genotype_index(gt) for gt in all_genotypes]
config.viability_fitness[FEMALE, 3, gt_indices] = 0.9
```

## 常见操作速查

| 操作 | 代码 | 返回值 |
|------|------|--------|
| 获取基因型索引 | `ic.genotype_index(gt)` | int |
| 从索引获取基因型 | `ic.index_to_genotype[idx]` | Genotype |
| 获取单倍基因型索引 | `ic.haplo_index(hg)` | int |
| 获取标签索引 | `ic.gamete_label_index("label")` | int |
| 基因型总数 | `ic.num_genotypes()` | int |
| 单倍基因型总数 | `ic.num_haplogenotypes()` | int |
| 标签总数 | `ic.num_gamete_labels()` | int |

## 与全局缓存的关系

Genotype 使用全局缓存，而 IndexRegistry 维护对象→索引的注册：

```
字符串 "A1|A2"
    ↓ Species.get_genotype_from_str()
全局缓存 Species.genotype_cache
    ↓ [命中]
Genotype 对象 (唯一)
    ↓ IndexRegistry.register_genotype()
整数索引 (例如 5)
    ↓
numpy 数组访问 individual_count[:, :, 5]
```

两个机制协同工作：
1. **全局缓存** 保证对象唯一性
2. **IndexRegistry** 建立对象与索引的映射

---

## 🎯 本章总结

| 概念 | 作用 | 使用频率 |
|------|------|---------|
| **IndexRegistry** | 对象↔索引注册表 | 在 Modifier、Hook 中使用 |
| **genotype_index()** | 获取基因型索引 | 常用 |
| **haplo_index()** | 获取单倍基因型索引 | 少用（高级功能） |
| **gamete_label_index()** | 获取标签索引 | 中等频率 |
| **compress/decompress** | 压缩配子索引 | 低层函数中使用 |

**关键点**：
1. IndexRegistry 自动维护，用户通常通过 `pop.registry` 访问
2. Modifier 返回对象→频率字典，内部自动转换为索引
3. Numba Hook 直接工作于索引（对象信息丢失）
4. 全局缓存 + IndexRegistry 构成完整的对象→索引系统

---

## 📚 相关章节

- [遗传结构与实体](02_genetic_structures.md) - Genotype 对象的创建
- [PopulationState & PopulationConfig](05_population_state_config.md) - 配置中的索引应用
- [Modifier 机制](08_modifiers.md) - Modifier 中的 IndexRegistry 使用
- [Hook 系统](09_hooks.md) - 高级 Hook 选择器模式

---

**准备进入配置编译细节了吗？** [前往下一章：PopulationState & PopulationConfig →](05_population_state_config.md)
