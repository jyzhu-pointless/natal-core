# 筛选 API 参考

本文档列出了 `GameteConversionRuleSet` 系统中所有用于筛选的 API。

## 核心筛选参数

### 1. sex_filter（性别筛选）

**位置**: `GameteAlleleConversionRule.__init__`

**类型**: `Optional[Literal["male", "female", "both"]]`

**默认值**: `"both"`

**作用**: 指定规则仅应用于特定性别的配子生产

#### 选项

| 值 | 含义 | 应用场景 |
|---|-----|--------|
| `"both"` | 应用于所有性别（默认） | 通用转换规则 |
| `"male"` | 仅应用于雄性 | 雄性特异的驱动 |
| `"female"` | 仅应用于雌性 | 雌性特异的抑制 |

#### 使用示例

```python
# 仅在雄性中应用
ruleset.add_convert(
    "W", "D",
    rate=0.8,
    sex_filter="male"
)

# 仅在雌性中应用
ruleset.add_convert(
    "W", "S",
    rate=0.5,
    sex_filter="female"
)

# 在所有性别中应用（默认）
ruleset.add_convert(
    "A", "B",
    rate=0.5,
    sex_filter="both"  # 可省略，这是默认值
)
```

---

### 2. genotype_filter（基因型筛选）

**位置**: `GameteAlleleConversionRule.__init__`

**类型**: `Optional[Callable[[Genotype], bool]]`

**默认值**: `None`（应用于所有基因型）

**作用**: 通过自定义函数判断规则是否应用于特定基因型

#### 函数签名

```python
def genotype_filter(genotype: Genotype) -> bool:
    """
    Args:
        genotype: 待检查的基因型
    
    Returns:
        True: 规则应用于此基因型
        False: 规则不应用于此基因型
    """
    pass
```

#### 使用示例

```python
# 定义过滤器
def my_genotype_filter(genotype: Genotype) -> bool:
    return is_genotype("W", "D")(genotype)

# 使用过滤器
ruleset.add_convert(
    "W", "D",
    rate=0.5,
    genotype_filter=my_genotype_filter
)
```

---

## 内置过滤器工厂函数

这些是推荐的过滤器实现模式（文档中提供的代码片段）。

### has_allele

```python
def has_allele(allele_name):
    """工厂函数：创建检查是否携带指定等位基因的过滤器"""
    def filter_func(genotype: Genotype) -> bool:
        for hap in genotype.maternal.haplotypes + genotype.paternal.haplotypes:
            for locus in genotype.species.chromosomes[0].loci:
                gene = hap.get_gene_at_locus(locus)
                if gene and gene.name == allele_name:
                    return True
        return False
    return filter_func
```

**用途**: 仅在携带特定等位基因的个体中应用规则

**参数**:
- `allele_name` (str): 等位基因名称

**返回**: 过滤函数 `Callable[[Genotype], bool]`

**使用示例**:
```python
ruleset.add_convert("W", "D", rate=0.7,
                    genotype_filter=has_allele("drive"))
```

---

### is_genotype

```python
def is_genotype(allele1, allele2):
    """工厂函数：检查是否为特定基因型组合"""
    def filter_func(genotype: Genotype) -> bool:
        try:
            chromosome = genotype.species.chromosomes[0]
            locus = chromosome.loci[0]
            
            mat_gene = genotype.maternal.haplotypes[0].get_gene_at_locus(locus)
            pat_gene = genotype.paternal.haplotypes[0].get_gene_at_locus(locus)
            
            mat_name = mat_gene.name if mat_gene else None
            pat_name = pat_gene.name if pat_gene else None
            
            return (mat_name == allele1 and pat_name == allele2) or \
                   (mat_name == allele2 and pat_name == allele1)
        except:
            return False
    return filter_func
```

**用途**: 仅在特定基因型中应用规则（不考虑顺序）

**参数**:
- `allele1` (str): 第一个等位基因名称
- `allele2` (str): 第二个等位基因名称

**返回**: 过滤函数 `Callable[[Genotype], bool]`

**使用示例**:
```python
# 仅在 A/B 或 B/A 基因型中应用
ruleset.add_convert("A", "C", rate=0.5,
                    genotype_filter=is_genotype("A", "B"))
```

---

### is_homozygous_with

```python
def is_homozygous_with(allele_name):
    """工厂函数：检查是否在某个等位基因上纯合"""
    def filter_func(genotype: Genotype) -> bool:
        try:
            chromosome = genotype.species.chromosomes[0]
            locus = chromosome.loci[0]
            
            mat_gene = genotype.maternal.haplotypes[0].get_gene_at_locus(locus)
            pat_gene = genotype.paternal.haplotypes[0].get_gene_at_locus(locus)
            
            mat_name = mat_gene.name if mat_gene else None
            pat_name = pat_gene.name if pat_gene else None
            
            return mat_name == allele_name and pat_name == allele_name
        except:
            return False
    return filter_func
```

**用途**: 仅在纯合于特定等位基因的个体中应用规则

**参数**:
- `allele_name` (str): 等位基因名称

**返回**: 过滤函数 `Callable[[Genotype], bool]`

**使用示例**:
```python
# 仅在 D/D 纯合子中应用
ruleset.add_convert("W", "S", rate=0.8,
                    genotype_filter=is_homozygous_with("D"))
```

---

### is_any_heterozygous_locus

```python
def is_any_heterozygous_locus(species):
    """工厂函数：检查任何位点是否杂合"""
    def filter_func(genotype: Genotype) -> bool:
        for chromosome in species.chromosomes:
            mat_hap = genotype.maternal.get_haplotype_for_chromosome(chromosome)
            pat_hap = genotype.paternal.get_haplotype_for_chromosome(chromosome)
            
            for locus in chromosome.loci:
                mat_gene = mat_hap.get_gene_at_locus(locus)
                pat_gene = pat_hap.get_gene_at_locus(locus)
                
                if mat_gene != pat_gene:
                    return True
        return False
    return filter_func
```

**用途**: 仅在至少有一个位点杂合的个体中应用规则

**参数**:
- `species` (Species): 物种对象（用于遍历所有染色体）

**返回**: 过滤函数 `Callable[[Genotype], bool]`

---

## 组合过滤器

提供逻辑组合多个过滤器的工具函数。

### and_filter

```python
def and_filter(*filters):
    """所有条件都满足"""
    def combined(genotype: Genotype) -> bool:
        return all(f(genotype) for f in filters)
    return combined
```

**用途**: 多个条件的与（AND）逻辑

**参数**:
- `*filters`: 可变数量的过滤函数

**返回**: 组合过滤函数

**使用示例**:
```python
# 同时满足：杂合 且 携带 drive
ruleset.add_convert("W", "D", rate=0.6,
                    genotype_filter=and_filter(
                        is_heterozygous,
                        has_allele("drive")
                    ))
```

---

### or_filter

```python
def or_filter(*filters):
    """任一条件满足"""
    def combined(genotype: Genotype) -> bool:
        return any(f(genotype) for f in filters)
    return combined
```

**用途**: 多个条件的或（OR）逻辑

**参数**:
- `*filters`: 可变数量的过滤函数

**返回**: 组合过滤函数

**使用示例**:
```python
# 满足其一：(杂合且携带drive) 或 (纯合D/D)
ruleset.add_convert("W", "D", rate=0.5,
                    genotype_filter=or_filter(
                        and_filter(is_heterozygous, has_allele("drive")),
                        is_genotype_combination("D", "D")
                    ))
```

---

### not_filter

```python
def not_filter(f):
    """反向过滤器"""
    def negated(genotype: Genotype) -> bool:
        return not f(genotype)
    return negated
```

**用途**: 反向条件

**参数**:
- `f`: 待反向的过滤函数

**返回**: 反向过滤函数

**使用示例**:
```python
# 仅在纯合于D的个体中应用
ruleset.add_convert("A", "B", rate=0.5,
                    genotype_filter=is_homozygous_with("D"))
```

---

## 完整 API 矩阵

| 筛选类型 | API | 参数类型 | 返回类型 | 场景 |
|--------|-----|--------|--------|------|
| 性别 | `sex_filter` | `Literal["male"\|"female"\|"both"]` | - | 性别特异性规则 |
| 基因型 | `genotype_filter` | `Callable[[Genotype], bool]` | `bool` | 复杂逻辑筛选 |
| 等位基因 | `has_allele(name)` | `str` | `bool` | 检查特定等位基因 |
| 基因型 | `is_genotype(a1, a2)` | `str, str` | `bool` | 精确基因型匹配 |
| 纯合 | `is_homozygous_with(name)` | `str` | `bool` | 检查纯合等位基因 |
| 多位点 | `is_any_heterozygous_locus(species)` | `Species` | `bool` | 任一位点杂合 |
| 逻辑与 | `and_filter(*filters)` | `*Callable` | `bool` | 多条件AND |
| 逻辑或 | `or_filter(*filters)` | `*Callable` | `bool` | 多条件OR |
| 逻辑非 | `not_filter(f)` | `Callable` | `bool` | 条件反向 |

---

## 常见组合模式

### 模式 1：信使驱动（仅在W/D中激活）

```python
ruleset.add_convert(
    "W", "D",
    rate=0.5,
    sex_filter="both",
    genotype_filter=is_genotype("W", "D")
)
```

### 模式 2：雄性特异的抑制

```python
ruleset.add_convert(
    "W", "S",
    rate=0.8,
    sex_filter="male",
    genotype_filter=has_allele("suppressor")
)
```

### 模式 3：纯合特定等位基因

```python
ruleset.add_convert(
    "W", "X",
    rate=0.5,
    genotype_filter=is_homozygous_with("D")  # 仅在 D/D 纯合子中应用
)
```

### 模式 4：多条件激活

```python
ruleset.add_convert(
    "W", "D",
    rate=0.6,
    genotype_filter=and_filter(
        has_allele("drive"),
        not_filter(is_genotype("D", "D"))
    )
)
```

### 模式 5：条件分支（应用到规则1或规则2）

```python
ruleset.add_convert("W", "D", rate=0.9,
                    genotype_filter=or_filter(
                        is_genotype("W", "D"),
                        is_homozygous_with("D")
                    ))
```

---

## 性能建议

### 快速筛选器排序

性能从快到慢：

1. ✅ **`sex_filter`** - O(1) 常数时间
2. ⚠️ **`is_genotype`** - O(1) 但需要遍历一个位点
3. ⚠️ **`is_homozygous_with`** - O(1) 但需要遍历一个位点
4. ⚠️ **`has_allele`** - O(n_loci) 需要遍历所有位点
5. ❌ **`is_any_heterozygous_locus`** - O(n_chrom × n_loci) 最慢

**优化建议**：
- 先使用快速筛选器（`sex_filter`）
- 再使用中等速度的基因型筛选（`is_genotype`、`is_homozygous_with`）
- 最后使用较慢的筛选器（`has_allele`）
- 避免在内层循环中使用 `is_any_heterozygous_locus`

```python
# ✅ 好的组织顺序
ruleset.add_convert("W", "D", rate=0.5,
                    sex_filter="male",  # 最快
                    genotype_filter=and_filter(
                        is_homozygous_with("D"),  # 次快
                        has_allele("drive")  # 较慢
                    ))
```

---

## 自定义过滤器模板

### 简单过滤器

```python
def my_simple_filter(genotype: Genotype) -> bool:
    """快速返回的简单条件"""
    # 检查条件
    # ...
    return True or False
```

### 工厂函数过滤器

```python
def my_factory(param1, param2):
    """返回一个配置好的过滤函数"""
    def filter_func(genotype: Genotype) -> bool:
        # 使用 param1 和 param2
        # ...
        return True or False
    return filter_func

# 使用
ruleset.add_convert("A", "B", rate=0.5,
                    genotype_filter=my_factory(param1_value, param2_value))
```

### 带缓存的过滤器（性能优化）

```python
from functools import lru_cache

@lru_cache(maxsize=1024)
def cached_heterozygous_check(genotype_id: int) -> bool:
    """缓存过滤结果"""
    # 注意：需要使用 id() 作为 key（lru_cache 需要可哈希参数）
    # 实际应用中较复杂，一般不推荐
    pass
```
