# genotype_filter 实现指南

`genotype_filter` 是一个可选的**过滤函数**，用于决定转换规则是否应用于特定的基因型。

## 基本概念

```python
# genotype_filter 的签名
def my_filter(genotype: Genotype) -> bool:
    """返回 True 表示规则应用于该基因型，False 表示不应用"""
    pass

# 在规则中使用
rule = GameteAlleleConversionRule(
    from_allele="A",
    to_allele="B",
    rate=0.5,
    genotype_filter=my_filter  # 传入过滤函数
)
```

## Genotype 结构回顾

```python
# Genotype 由两个 HaploidGenotype 组成
genotype.maternal  # 母本单倍体基因组
genotype.paternal  # 父本单倍体基因组

# 每个 HaploidGenotype 包含多个 Haplotype（每条染色体一个）
genotype.maternal.haplotypes  # List[Haplotype]

# 访问特定染色体的单倍型
haplotype = genotype.maternal.get_haplotype_for_chromosome(chromosome)

# 访问特定位点的基因
gene = haplotype.get_gene_at_locus(locus)
```

## 常用实现模式

### 模式 1：特定等位基因的存在

#### 检查是否携带特定等位基因

```python
def has_allele(allele_name):
    """创建一个检查是否携带指定等位基因的过滤器"""
    def filter_func(genotype: Genotype) -> bool:
        # 检查所有位点
        for hap in genotype.maternal.haplotypes:
            for locus in genotype.species.chromosomes[0].loci:  # 遍历所有位点
                gene = hap.get_gene_at_locus(locus)
                if gene and gene.name == allele_name:
                    return True
        
        for hap in genotype.paternal.haplotypes:
            for locus in genotype.species.chromosomes[0].loci:
                gene = hap.get_gene_at_locus(locus)
                if gene and gene.name == allele_name:
                    return True
        
        return False
    
    return filter_func

# 使用：仅在携带 "drive" 等位基因的个体中应用规则
ruleset.add_convert(
    "W", "D",
    rate=0.7,
    genotype_filter=has_allele("drive")
)
```

### 模式 3：等位基因组合检查

#### 检查具体的基因型组合

```python
def is_genotype(allele1, allele2):
    """检查基因型是否为特定的等位基因组合"""
    def filter_func(genotype: Genotype) -> bool:
        try:
            # 假设都在第一条染色体的第一个位点
            chromosome = genotype.species.chromosomes[0]
            locus = chromosome.loci[0]
            
            mat_gene = genotype.maternal.haplotypes[0].get_gene_at_locus(locus)
            pat_gene = genotype.paternal.haplotypes[0].get_gene_at_locus(locus)
            
            mat_name = mat_gene.name if mat_gene else None
            pat_name = pat_gene.name if pat_gene else None
            
            # 检查是否匹配（考虑顺序可交换）
            return (mat_name == allele1 and pat_name == allele2) or \
                   (mat_name == allele2 and pat_name == allele1)
        except:
            return False
    
    return filter_func

# 使用：仅在 A/B 或 B/A 杂合子中应用
ruleset.add_convert(
    "A", "C",
    rate=0.5,
    genotype_filter=is_genotype("A", "B")
)
```

### 模式 4：纯合于特定等位基因

#### 检查是否纯合于特定等位基因

```python
def is_homozygous_with(allele_name):
    """检查是否在某个等位基因上纯合"""
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

# 使用：仅在 D/D 纯合子中应用
ruleset.add_convert(
    "W", "S",
    rate=0.8,
    genotype_filter=is_homozygous_with("D")
)
```

### 模式 5：多等位基因不相等

```python
def is_any_heterozygous_locus(species):
    """检查任何位点是否杂合"""
    def filter_func(genotype: Genotype) -> bool:
        for chromosome in species.chromosomes:
            mat_hap = genotype.maternal.get_haplotype_for_chromosome(chromosome)
            pat_hap = genotype.paternal.get_haplotype_for_chromosome(chromosome)
            
            for locus in chromosome.loci:
                mat_gene = mat_hap.get_gene_at_locus(locus)
                pat_gene = pat_hap.get_gene_at_locus(locus)
                
                if mat_gene != pat_gene:  # 该位点是杂合的
                    return True
        
        return False
    
    return filter_func
```

### 模式 5：条件组合（与/或逻辑）

```python
def and_filter(*filters):
    """组合多个过滤器 - 所有条件都满足"""
    def combined(genotype: Genotype) -> bool:
        return all(f(genotype) for f in filters)
    return combined

def or_filter(*filters):
    """组合多个过滤器 - 任一条件满足"""
    def combined(genotype: Genotype) -> bool:
        return any(f(genotype) for f in filters)
    return combined

def not_filter(f):
    """反向过滤器"""
    def negated(genotype: Genotype) -> bool:
        return not f(genotype)
    return negated

# 使用示例：(杂合且携带 drive) 或 (纯合于 D)
ruleset.add_convert(
    "W",
    "D",
    rate=0.5,
    genotype_filter=or_filter(
        has_allele("drive"),
        is_genotype("D", "D")
    )
)
```

## 完整实战示例

### 场景：信使基因驱动中的规则

```python
from natal.gamete_allele_conversion import GameteConversionRuleSet
from natal.genetic_entities import Genotype

# 创建规则集
ruleset = GameteConversionRuleSet("homing_drive")

# W -> D 转换仅在 W/D 基因型中发生
ruleset.add_convert(
    from_allele="W",
    to_allele="D",
    rate=0.5,
    sex_filter="both",
    genotype_filter=is_genotype("W", "D")  # 添加过滤器
)

# 应用到种群
gamete_mod = ruleset.to_gamete_modifier(population)
population.add_gamete_modifier(gamete_mod, name="homing")
```

### 场景：性别特异的抑制系统

```python
ruleset = GameteConversionRuleSet("suppression_system")

# 只在 D/D 纯合的雄性中，W 转换为 S
ruleset.add_convert(
    from_allele="W",
    to_allele="S",
    rate=0.8,
    sex_filter="male",
    genotype_filter=is_genotype("D", "D")
)
```

## 调试技巧

### 1. 添加日志

```python
def my_genotype_filter(genotype: Genotype) -> bool:
    result = is_genotype("W", "D")(genotype)
    print(f"Checking {genotype.name}: matches W/D={result}")
    return result

ruleset.add_convert("W", "D", rate=0.5, 
                    genotype_filter=my_genotype_filter)
```

### 2. 测试过滤器

```python
# 创建一个测试基因型
hg1 = species.get_haploid_genotype_from_str("W")
hg2 = species.get_haploid_genotype_from_str("D")
test_genotype = Genotype(species, hg1, hg2)

# 测试过滤器
my_filter = is_genotype("W", "D")
print(my_filter(test_genotype))  # 应该返回 True
```

## 性能建议

1. **避免复杂的嵌套循环** - 保持过滤器简单快速
2. **缓存结果** - 如果过滤器被频繁调用，可以缓存结果
3. **尽早返回** - 找到匹配条件后立即返回

```python
def efficient_filter(genotype: Genotype) -> bool:
    """高效的过滤器实现"""
    # 尽早返回，避免不必要的迭代
    for locus in limited_loci:  # 只检查关键位点
        if check_condition(locus, genotype):
            return True  # 找到就返回，不继续
    return False
```

## 常见错误

❌ **错误 1**：访问不存在的染色体

```python
# 不要这样做
genotype.maternal.haplotypes[5].get_gene_at_locus(locus)  # 可能 IndexError
```

✅ **正确做法**：先检查存在性

```python
if len(genotype.maternal.haplotypes) > 5:
    gene = genotype.maternal.haplotypes[5].get_gene_at_locus(locus)
```

❌ **错误 2**：假设基因总是非 None

```python
# 不要这样做
gene_name = genotype.maternal.haplotypes[0].get_gene_at_locus(locus).name
```

✅ **正确做法**：检查 None

```python
gene = genotype.maternal.haplotypes[0].get_gene_at_locus(locus)
gene_name = gene.name if gene else None
```

❌ **错误 3**：忘记返回布尔值

```python
# 不要这样做
def my_filter(genotype):
    if genotype.maternal != genotype.paternal:
        print("heterozygous")  # 忘记返回！
```

✅ **正确做法**：明确返回 True/False

```python
def my_filter(genotype):
    result = genotype.maternal != genotype.paternal
    return result  # 明确返回
```
