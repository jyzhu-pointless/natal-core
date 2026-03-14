# 基因型正则表达式模式匹配 - 设计方案

## 概述

设计一个灵活的基因型模式匹配系统，支持通配符、集合、反选等正则表达式风格的语法，用于 `genotype_filter`。

## 核心设计

### 1. 模式语法规范

#### 基本语法（现有）
```
;     分隔染色体（同一个体的不同染色体）
|     分隔 maternal 和 paternal 同源染色体
/     分隔同一条染色体上的等位基因
```

#### 扩展语法

##### 1.1 `::` - 双向匹配（不区分父母本）
```
A::B    匹配 A|B 或 B|A（任意顺序）
```

##### 1.2 `*` - 单个等位基因通配符
```
A/*     匹配 A 和任何其他等位基因的组合（A/B1, A/B2 等）
A::*    匹配 A 和任何其他等位基因，不区分顺序
```

##### 1.3 `{}` - 等位基因集合
```
{A,B,C}    匹配 A 或 B 或 C
{A,B}/C    匹配 (A/C 或 B/C)
```

##### 1.4 `!` - 反选（负集合）
```
!A1        匹配除了 A1 之外的任何等位基因
!{A,B}     匹配除了 A 和 B 之外的任何等位基因
```

##### 1.5 `()` - 同一条染色体的显式分组
```
(A1::A2; B1/B1)    将一对同源染色体的两个位点分开写
                    等价于 A1::A2|?/?;B1/B1|?/?
```

##### 1.6 省略规则
```
A1/B1|A2/B2            省略后续染色体（自动不匹配后续）
A1/B1|A2/B2; *         明确标记第二条染色体为任意（等价于上一行）
A1/*|A2/B2             同一条染色体内，某位点用 * 占位为任意
A1/B1|A2/B2; */C2      第二条染色体，maternal 任意，paternal 为 C2
```

**灵活性规则：**
- ✅ 直接省略后续染色体（最后一条不需要 `;`）
- ✅ 用 `;*` 显式标记整条染色体为任意
- ✅ 用 `*` 占位单个位点
- ✅ 可以混合使用（同一条染色体内既有 `*` 也有具体等位基因）

#### 例子解析

假设物种定义：
```
chr1: A [A1, A2, A3]，B [B1, B2]
chr2: C [C1, C2]
```

| 模式 | 含义 | 匹配例子 |
|-----|------|--------|
| `A1/B1\|A2/B2` | 第一条染色体固定，省略其他 | `A1/B1\|A2/B2; C1\|C1` ✓ |
| `A1/B1\|A2/B2; *` | 第一条固定，第二条任意 | `A1/B1\|A2/B2; C1\|C2` ✓ |
| `A1/*\|A2/B2; C1/C2` | B 位点可以是任何 | `A1/B1\|A2/B2; C1/C2` ✓ |
| `{A2,A3}/B1\|A2/B2; C1/C2` | A 是 A2 或 A3 | `A2/B1\|A2/B2; C1/C2` ✓ |
| `(A1::A2; B1/B1); C1/C1` | 一种染色体用 `::` | `A1/B1\|A2/B1; C1/C1` ✓ |
| `A1::*` | 任何包含 A1 的基因型 | `A1/B1\|B2/C1; C1\|C2` ✓（需要 expand） |

### 2. 核心类结构

```python
# ============================================================================
# 模式元素类工（不同层级的模式节点）
# ============================================================================

class PatternElement(ABC):
    """所有模式元素的基类"""
    
    @abstractmethod
    def matches(self, gene: 'Gene') -> bool:
        """检查单个等位基因是否匹配"""
        pass
    
    @abstractmethod
    def __repr__(self) -> str:
        pass


class AllelePattern(PatternElement):
    """精确匹配单个等位基因"""
    def __init__(self, allele_name: str):
        self.allele_name = allele_name
    
    def matches(self, gene: Optional['Gene']) -> bool:
        return gene is not None and gene.name == self.allele_name


class WildcardPattern(PatternElement):
    """通配符 * - 匹配任何等位基因"""
    def matches(self, gene: Optional['Gene']) -> bool:
        return gene is not None


class SetPattern(PatternElement):
    """集合模式 {A, B, C}"""
    def __init__(self, alleles: Set[str], negate: bool = False):
        self.alleles = alleles
        self.negate = negate  # ! 反选
    
    def matches(self, gene: Optional['Gene']) -> bool:
        if gene is None:
            return False
        result = gene.name in self.alleles
        return (not result) if self.negate else result


# ============================================================================
# 单个位点模式（同一条染色体上）
# ============================================================================

class LocusPattern:
    """单个位点（locus）的模式，表示该位点两条同源染色体的等位基因"""
    
    def __init__(
        self,
        maternal_pattern: PatternElement,
        paternal_pattern: PatternElement,
        unordered: bool = False  # :: 表示不区分顺序
    ):
        self.maternal_pattern = maternal_pattern
        self.paternal_pattern = paternal_pattern
        self.unordered = unordered  # True 表示 :: 模式
    
    def matches(self, mat_gene: Optional['Gene'], pat_gene: Optional['Gene']) -> bool:
        """检查一对等位基因是否匹配"""
        if self.unordered:
            # A::B 可以匹配 A|B 或 B|A
            match_straight = (
                self.maternal_pattern.matches(mat_gene) and
                self.paternal_pattern.matches(pat_gene)
            )
            match_reversed = (
                self.maternal_pattern.matches(pat_gene) and
                self.paternal_pattern.matches(mat_gene)
            )
            return match_straight or match_reversed
        else:
            # 严格按顺序匹配
            return (
                self.maternal_pattern.matches(mat_gene) and
                self.paternal_pattern.matches(pat_gene)
            )


# ============================================================================
# 单条同源染色体对（chromosome pair）的模式
# ============================================================================

class ChromosomePairPattern:
    """一对同源染色体的模式"""
    
    def __init__(
        self,
        locus_patterns: List[LocusPattern],
        explicit_grouping: bool = False  # () 显式分组
    ):
        self.locus_patterns = locus_patterns
        self.explicit_grouping = explicit_grouping
    
    def matches(self, haplotype_pair: Tuple['Haplotype', 'Haplotype']) -> bool:
        """检查一对同源染色体是否匹配"""
        mat_hap, pat_hap = haplotype_pair
        
        # 检查所有位点
        if len(self.locus_patterns) != len(mat_hap.loci):
            return False
        
        for locus_pattern, locus in zip(self.locus_patterns, mat_hap.loci):
            mat_gene = mat_hap.get_gene_at_locus(locus)
            pat_gene = pat_hap.get_gene_at_locus(locus)
            
            if not locus_pattern.matches(mat_gene, pat_gene):
                return False
        
        return True


# ============================================================================
# 完整基因型模式
# ============================================================================

class GenotypePattern:
    """完整的基因型模式"""
    
    def __init__(
        self,
        chromosome_patterns: List[Optional[ChromosomePairPattern]]
        # None 表示任意（省略的染色体）
    ):
        self.chromosome_patterns = chromosome_patterns
    
    def matches(self, genotype: 'Genotype') -> bool:
        """检查一个基因型是否匹配该模式"""
        # 如果模式中指定了某条染色体，则必须精确匹配
        # 如果模式中没有指定（为 None），则忽略该染色体
        
        for i, chr_pattern in enumerate(self.chromosome_patterns):
            if chr_pattern is None:
                # 跳过未指定的染色体
                continue
            
            chromosome = genotype.species.chromosomes[i]
            mat_hap = genotype.maternal.get_haplotype_for_chromosome(chromosome)
            pat_hap = genotype.paternal.get_haplotype_for_chromosome(chromosome)
            
            if not chr_pattern.matches((mat_hap, pat_hap)):
                return False
        
        return True
    
    def to_filter(self) -> Callable[[Genotype], bool]:
        """转换为 genotype_filter 函数"""
        return lambda genotype: self.matches(genotype)


# ============================================================================
# 模式解析器
# ============================================================================

class GenotypePatternParser:
    """将模式字符串解析为 GenotypePattern 对象"""
    
    def __init__(self, species: 'Species'):
        self.species = species
    
    def parse(self, pattern_str: str) -> GenotypePattern:
        """
        解析模式字符串
        
        主流程：
        1. 先分割出模式字符串中的每条染色体部分
        2. 逐个解析每条染色体的模式
        3. 返回 GenotypePattern 对象
        """
        # Split by `;` to get chromosome patterns
        # Handle () grouping for same-chromosome patterns
        
        # TODO: 详细实现在下一阶段
        pass
    
    def _parse_chromosome_pair(self, chr_str: str) -> ChromosomePairPattern:
        """解析单条染色体的模式"""
        # 处理 () 分组
        # 处理 | 分隔
        # 处理 :: vs | 的区别
        # 处理 ; 分隔位点
        pass
    
    def _parse_locus(self, locus_str: str) -> LocusPattern:
        """解析单个位点的模式"""
        # 处理 / 分隔
        # 处理 * 通配符
        # 处理 {} 集合
        # 处理 ! 反选
        pass
    
    def _parse_allele_element(self, allele_str: str) -> PatternElement:
        """解析单个等位基因的模式元素"""
        # 返回 AllelePattern、WildcardPattern 或 SetPattern
        pass
```

### 3. API 集成点

#### 3.1 在 `Species` 中集成（推荐方式）

在 `Species` 类中添加方法：

```python
class Species(GeneticStructure['HaploidGenome']):
    # ... 现有方法 ...
    
    def parse_genotype_pattern(self, pattern: str) -> Callable[['Genotype'], bool]:
        """
        解析基因型模式字符串，返回一个过滤函数。
        
        Args:
            pattern: 模式字符串，如 "A1/B1|A2/B2; C1/C2"
        
        Returns:
            一个接收 Genotype 的过滤函数
        
        Examples:
            >>> species = Species.from_dict("Test", {...})
            >>> pattern_filter = species.parse_genotype_pattern("A1/B1|A2/B2; C1::*")
            >>> genotypes = [gt for gt in population if pattern_filter(gt)]
        """
        from natal.recipes import GenotypePatternParser
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse(pattern)
        return pattern_obj.to_filter()
    
    def filter_genotypes_by_pattern(self, genotypes: Iterable['Genotype'], 
                                     pattern: str) -> List['Genotype']:
        """
        从基因型列表中筛选出匹配模式的基因型。
        
        Args:
            genotypes: 要筛选的基因型迭代器
            pattern: 模式字符串
        
        Returns:
            匹配的基因型列表
        
        Examples:
            >>> matched = species.filter_genotypes_by_pattern(population.genotypes, "A1/*|A2/B2")
        """
        pattern_filter = self.parse_genotype_pattern(pattern)
        return [gt for gt in genotypes if pattern_filter(gt)]
    
    def enumerate_genotypes_matching_pattern(self, pattern: str, 
                                               max_count: Optional[int] = None):
        """
        枚举所有符合模式的基因型组合。
        
        根据模式中指定的等位基因，生成所有可能的基因型。如果模式为 "*" 
        或省略某个位点，则该位置使用所有可用的等位基因。
        
        Args:
            pattern: 模式字符串，如 "A1/B1|A2/B2" 或 "{A1,A2}/B1|A2/B2"
            max_count: 最多返回的基因型数量（防止爆炸）
        
        Yields:
            符合模式的 Genotype 对象
        
        Examples:
            >>> for gt in species.enumerate_genotypes_matching_pattern("A1/*|A2/*"):
            ...     print(gt)  # 会生成所有包含 A1 和 A2 的基因型
        """
        from natal.recipes import GenotypePatternParser, GenotypePattern
        parser = GenotypePatternParser(self)
        pattern_obj = parser.parse(pattern)
        
        # TODO: 实现枚举逻辑，递归生成所有可能的组合
        # 这需要在 GenotypePatternParser 中添加 enumerate() 方法
        pass
```

#### 3.2 在过滤规则中使用
```python
# 方法 1: 直接传递模式字符串（如果 GameteAlleleConversionRule 支持自动解析）
ruleset.add_convert(
    "W", "D",
    rate=0.5,
    genotype_filter="A1/B1|A2/B2; C1/C2"  # 自动调用 species.parse_genotype_pattern()
)

# 方法 2: 显式解析
pattern_filter = species.parse_genotype_pattern("A1/B1|A2/B2; C1/C2")
ruleset.add_convert("W", "D", rate=0.5, genotype_filter=pattern_filter)

# 方法 3: 从种群中直接过滤
matched_genotypes = species.filter_genotypes_by_pattern(
    population.all_genotypes, 
    "A1/*|A2/B2"
)
```

#### 3.3 枚举匹配的基因型
```python
# 枚举所有符合特定模式的基因型
for genotype in species.enumerate_genotypes_matching_pattern("A1/B1|A2/B2; *"):
    print(f"Matched: {genotype}")

# 限制数量（防止数量爆炸）
first_100 = list(
    itertools.islice(
        species.enumerate_genotypes_matching_pattern("*/*|*/*; */*:*"),
        100
    )
)
```

### 4. 实现阶段

#### 第 1 阶段：基础框架
- [ ] 定义 PatternElement 及其子类
- [ ] 定义 LocusPattern、ChromosomePairPattern、GenotypePattern
- [ ] 实现基础匹配逻辑
- [ ] 支持简单模式（无通配符、集合）
- [ ] 在 Species 中添加 `parse_genotype_pattern()` 方法

#### 第 2 阶段：解析器
- [ ] 实现 `_parse_allele_element()`（支持 `*`、`{}`、`!`）
- [ ] 实现 `_parse_locus()`（支持 `/`、`::`、`|`）
- [ ] 实现 `_parse_chromosome_pair()`（支持 `;`、`()`）
- [ ] 实现 `parse()` 主方法
- [ ] 在 Species 中添加 `filter_genotypes_by_pattern()` 方法

#### 第 3 阶段：枚举与集成
- [ ] 在 GenotypePatternParser 中实现 `enumerate_allele_combinations()` 方法
  - 提取模式中的所有具体等位基因和通配符位置
  - 生成所有可能的组合
- [ ] 在 Species 中添加 `enumerate_genotypes_matching_pattern()` 方法
- [ ] 扩展 GameteAlleleConversionRule 支持字符串模式（自动调用 species.parse_genotype_pattern()）
- [ ] 添加辅助函数和单元测试

### 5. 实现注意事项

#### 5.1 省略规则的处理
```python
# 这些模式是等价的，都表示"只管第一条染色体的 A 和 B 位点，其他任意"
pattern1 = parser.parse("A1/B1|A2/B2")           # 直接省略
pattern2 = parser.parse("A1/B1|A2/B2; *")       # 显式用 * 标记第二条
pattern3 = parser.parse("A1/B1|A2/B2; */\*")    # 第二条所有位点都用 *

# 这个模式表示"第一条染色体 A 位点任意，B 位点具体为 B1|B2"
pattern4 = parser.parse("*/B1|*/* B2")          # 位点级别的 * 占位

# 解析器会自动创建：
pattern = parser.parse("A1/B1|A2/B2")
pattern.chromosome_patterns
# [ChromosomePairPattern(...), None, None, ...]
# 或者对于 "A1/B1|A2/B2; *"：
# [ChromosomePairPattern(...), ChromosomePairPattern(all_wildcards), None, ...]
```

#### 5.2 `::` 双向匹配的实现
```python
class LocusPattern:
    def matches(self, mat_gene, pat_gene):
        if self.unordered:
            # 尝试两种顺序
            return (
                (self.mat_pat.matches(mat_gene) and self.pat_pat.matches(pat_gene)) or
                (self.mat_pat.matches(pat_gene) and self.pat_pat.matches(mat_gene))
            )
```

#### 5.3 `()` 分组的处理
```
(A1::A2; B1/B1)
↓ 解析为
A1::A2 （在第一条染色体的第一个位点，`::` 表示不区分顺序）
B1/B1  （在第一条染色体的第二个位点，`/` 表示标准模式）
```

#### 5.4 枚举基因型的核心算法
```python
class GenotypePatternParser:
    def enumerate_allele_combinations(self, pattern: str) -> List[Dict[str, Tuple[str, str]]]:
        """
        从模式中提取所有可能的等位基因组合。
        
        返回一个列表，每个元素是一个字典，表示一个具体的基因型：
        {
            "chromosome_0": {"locus_0": ("allele_mat", "allele_pat"), "locus_1": ...},
            "chromosome_1": ...
        }
        
        处理过程：
        1. 解析模式得到 GenotypePattern
        2. 对每条染色体的每个位点，提取允许的等位基因
           - AllelePattern("A1") → ["A1"]
           - WildcardPattern() → [所有可用的等位基因]
           - SetPattern({A1, A2}) → ["A1", "A2"]
           - SetPattern({A1, A2}, negate=True) → [所有除了 A1, A2 的等位基因]
        3. 将所有可能的组合进行笛卡尔积
        4. 对于 :: 模式，生成两种顺序的组合（或合并为一种，因为最后是 Genotype 对象）
        """
        pattern_obj = self.parse(pattern)
        
        combinations = []
        
        # 遍历每条染色体
        for chr_idx, chr_pattern in enumerate(pattern_obj.chromosome_patterns):
            if chr_pattern is None:
                # 省略的染色体，跳过
                continue
            
            # 对该条染色体的每个位点，提取允许的等位基因
            locus_options = []
            for locus_pattern in chr_pattern.locus_patterns:
                # 获取该位点 maternal 和 paternal 允许的等位基因
                mat_alleles = self._get_allowed_alleles(locus_pattern.maternal_pattern)
                pat_alleles = self._get_allowed_alleles(locus_pattern.paternal_pattern)
                
                locus_options.append({
                    'maternal': mat_alleles,
                    'paternal': pat_alleles,
                    'unordered': locus_pattern.unordered
                })
            
            # 笛卡尔积生成所有可能的组合
            from itertools import product
            for combo in product(*[
                product(opt['maternal'], opt['paternal']) 
                for opt in locus_options
            ]):
                # combo 是一个嵌套元组，表示这条染色体的所有位点
                genotype_spec = {
                    f"chr_{chr_idx}": {
                        f"locus_{loc_idx}": allele_pair
                        for loc_idx, allele_pair in enumerate(combo)
                    }
                }
                combinations.append(genotype_spec)
        
        return combinations
    
    def _get_allowed_alleles(self, pattern_element: PatternElement) -> List['Gene']:
        """获取模式元素允许的等位基因列表"""
        if isinstance(pattern_element, AllelePattern):
            return [pattern_element.allele_name]
        elif isinstance(pattern_element, WildcardPattern):
            # 返回所有可用的等位基因
            return self._get_all_allele_names()
        elif isinstance(pattern_element, SetPattern):
            if pattern_element.negate:
                # 返回除了指定集合之外的所有等位基因
                all_alleles = set(self._get_all_allele_names())
                return list(all_alleles - pattern_element.alleles)
            else:
                return list(pattern_element.alleles)
        else:
            raise ValueError(f"Unknown pattern element type: {type(pattern_element)}")
    
    def _get_all_allele_names(self) -> List[str]:
        """获取物种中所有可用的等位基因名"""
        allele_names = set()
        for chromosome in self.species.chromosomes:
            for locus in chromosome.loci:
                for allele in locus.alleles:
                    allele_names.add(allele.name)
        return sorted(allele_names)
```

然后在 Species 中使用：

```python
class Species:
    def enumerate_genotypes_matching_pattern(self, pattern: str, 
                                               max_count: Optional[int] = None):
        """枚举所有符合模式的基因型"""
        from natal.recipes import GenotypePatternParser
        from itertools import islice
        
        parser = GenotypePatternParser(self)
        
        # 获取所有可能的组合
        combinations = parser.enumerate_allele_combinations(pattern)
        
        # 从组合生成 Genotype 对象
        for combo_spec in islice(combinations, max_count):
            # 将 combo_spec 转换回字符串格式，再用 get_genotype_from_str 解析
            genotype_str = self._convert_spec_to_genotype_str(combo_spec)
            try:
                genotype = self.get_genotype_from_str(genotype_str)
                yield genotype
            except Exception:
                # 跳过无法生成的组合（不应该发生，但防御性编程）
                continue
    
    def _convert_spec_to_genotype_str(self, spec: Dict) -> str:
        """将组合规范转换回 genotype_str 格式"""
        # {chr_0: {locus_0: (A1, A2), locus_1: (B1, B2)}, ...}
        # → "A1/B1|A2/B2;..."
        pass
```

#### 5.5 设置缓存提高性能
```python
class GenotypePatternParser:
    _pattern_cache = {}  # pattern_str -> GenotypePattern
    
    def parse(self, pattern_str: str) -> GenotypePattern:
        key = (id(self.species), pattern_str)
        if key not in self._pattern_cache:
            self._pattern_cache[key] = self._do_parse(pattern_str)
        return self._pattern_cache[key]
```

### 6. 错误处理

应该在解析阶段清楚地报导错误：

```python
class PatternParseError(Exception):
    """模式字符串解析错误"""
    pass

# 在 _parse_allele_element 中
if allele_name not in species.get_all_allele_names():
    raise PatternParseError(
        f"Unknown allele '{allele_name}' in pattern. "
        f"Valid alleles: {species.get_all_allele_names()}"
    )
```

## 伪代码示例

### 完整解析流程

```python
def parse(self, pattern_str: str) -> GenotypePattern:
    """
    "A1/B1|A2/B2"              → 省略后续染色体
    "A1/B1|A2/B2; *"           → 显式标记第二条为任意
    "A1/B1|A2/B2; C1/C2"       → 完整指定
    "A1/*|A2/B2; *"            → 位点级别和染色体级别混合
    
    返回：GenotypePattern，包含 [ChromosomePairPattern, ChromosomePairPattern, None, ...]
    - 不为 None 的位置表示指定的约束
    - None 表示该染色体完全省略
    - ChromosomePairPattern 可能包含 WildcardPattern（对应 * 占位）
    """
    
    # 步骤 1: 按 `;` 分割染色体
    chr_patterns_strs = self._split_by_semicolon(pattern_str)  
    # ["A1/B1|A2/B2", "C1/C2"] 或 ["A1/B1|A2/B2", "*"]
    
    # 步骤 2: 解析每条染色体
    chromosome_patterns = []
    for chr_str in chr_patterns_strs:
        chr_pattern = self._parse_chromosome_pair(chr_str)
        
        # 处理特殊的全通配符染色体标记
        if chr_pattern == "WILDCARD_CHROMOSOME":
            # 这里需要知道染色体有多少个位点
            # 可以在 matches() 时相对处理，或者这里先存一个标记
            # 最简单的方案：创建一个包含所有通配符的 ChromosomePairPattern
            # 但这需要知道位点数，所以先存个标记
            chromosome_patterns.append("WILDCARD_CHROMOSOME")
        else:
            chromosome_patterns.append(chr_pattern)
    
    # 步骤 3: 处理省略的染色体（用 None 表示）
    # 如果指定的染色体数少于总数，后续用 None 填充
    # 如果有明确的 "*..*" 占位，则不填充 None
    
    final_patterns = chromosome_patterns.copy()
    
    # 填充到总染色体数
    while len(final_patterns) < len(self.species.chromosomes):
        final_patterns.append(None)
    
    # 步骤 4: 后处理 "WILDCARD_CHROMOSOME" 标记
    for i, pattern in enumerate(final_patterns):
        if pattern == "WILDCARD_CHROMOSOME":
            # 创建一个与对应染色体位点数相同、所有位点都是通配符的 ChromosomePairPattern
            chromosome = self.species.chromosomes[i]
            num_loci = len(chromosome.loci)
            wildcard_loci = [
                LocusPattern(WildcardPattern(), WildcardPattern())
                for _ in range(num_loci)
            ]
            final_patterns[i] = ChromosomePairPattern(wildcard_loci)
    
    return GenotypePattern(final_patterns)


def _parse_chromosome_pair(self, chr_str: str) -> Optional[ChromosomePairPattern]:
    """
    解析单条染色体的模式
    
    返回 Optional[ChromosomePairPattern]：
    - 如果 chr_str 是 "*" 或 "*/\*|*/\*"，返回一个全通配符的 ChromosomePairPattern
    - 否则解析并返回具体的 ChromosomePairPattern
    
    特殊情况：
    - "*" 单独表示整条染色体都任意
    - None 表示省略的染色体
    """
    
    chr_str = chr_str.strip()
    
    # 如果整条染色体都是 * 或空，返回全通配符的模式
    if chr_str == "*" or chr_str == "":
        # 创建一个所有位点都是通配符的模式
        # 这需要知道有多少个位点，但在这里还不知道
        # 所以返回一个特殊的标记，之后处理
        return "WILDCARD_CHROMOSOME"  # 特殊标记
    
    # 检查是否有 ()
    if '(' in chr_str and ')' in chr_str:
        explicit_grouping = True
        chr_str = chr_str.strip('()')
    else:
        explicit_grouping = False
    
    # 按 | 分割父母本
    if '::' in chr_str:
        # 处理 :: 的情况
        unordered = True
        maternal_str, paternal_str = chr_str.split('::')
    elif '|' in chr_str:
        unordered = False
        maternal_str, paternal_str = chr_str.split('|')
    else:
        raise PatternParseError(f"Invalid chromosome pattern: {chr_str}")
    
    # 按 / 分割位点
    maternal_loci = maternal_str.split('/')
    paternal_loci = paternal_str.split('/')
    
    # 解析每个位点
    locus_patterns = []
    for mat_allele, pat_allele in zip(maternal_loci, paternal_loci):
        mat_pattern = self._parse_allele_element(mat_allele)
        pat_pattern = self._parse_allele_element(pat_allele)
        locus_patterns.append(LocusPattern(mat_pattern, pat_pattern, unordered))
    
    return ChromosomePairPattern(locus_patterns, explicit_grouping)


def _parse_allele_element(self, allele_str: str) -> PatternElement:
    """
    "A1" → AllelePattern("A1")
    "*"  → WildcardPattern()
    "{A,B,C}" → SetPattern({"A", "B", "C"})
    "!A" → SetPattern({所有等位基因} - {"A"}, negate=True) 
        或 SetPattern({"A"}, negate=True)
    """
    
    allele_str = allele_str.strip()
    
    if allele_str == "*":
        return WildcardPattern()
    
    if allele_str.startswith("!"):
        # 反选
        negated_str = allele_str[1:]
        base_pattern = self._parse_set_pattern(negated_str)
        # 需要取反
        return base_pattern.negate()
    
    if allele_str.startswith("{") and allele_str.endswith("}"):
        # 集合
        alleles_str = allele_str[1:-1]
        alleles = {a.strip() for a in alleles_str.split(",")}
        return SetPattern(alleles)
    
    # 普通等位基因
    return AllelePattern(allele_str)
```

## 总结

这个设计提供了：

1. **灵活的 API**：从 Species 中直接使用，无需创建另外的对象
2. **三层级的功能**：
   - 模式验证（`parse_genotype_pattern()`）返回过滤函数
   - 基因型过滤（`filter_genotypes_by_pattern()`）从列表中筛选
   - 基因型枚举（`enumerate_genotypes_matching_pattern()`）生成所有符合模式的基因型
3. **模块化结构**：从最小的元素（PatternElement）到完整基因型（GenotypePattern）
4. **灵活的语法**：支持通配符、集合、反选、双向匹配等
5. **性能考虑**：支持缓存已解析的模式，支持 max_count 限制枚举数量
6. **清晰的错误处理**：在解析阶段提前发现问题

## 使用示例

### 基本过滤
```python
# 创建物种
species = Species.from_dict("Drosophila", {
    "2L": {"W": ["W", "w"], "M": ["M", "m"]},
    "3R": {"D": ["D", "d"]}
})

# 过滤单个基因型
pattern_filter = species.parse_genotype_pattern("W/w|M/m; D/D")
gt = species.get_genotype_from_str("W/w|M/m; D/D")
assert pattern_filter(gt) == True

# 从列表中过滤
all_genotypes = [...]  # 从种群获得
matching = species.filter_genotypes_by_pattern(all_genotypes, "W/w|*/*; D/*")
```

### 模式匹配的强大功能
```python
# 模式 1: 固定模式
"W/w|M/m; D/D"  # 只匹配完全相同的基因型

# 模式 2: 支持通配符
"W/*|M/m; */*"  # maternal 必须有 W，paternal 可是任何；其他位置任意

# 模式 3: 省略规则（灵活）
"W/*|M/m"       # 省略第二条染色体（无需显式 *）
"W/*|M/m; *"    # 显式标记第二条为任意（等价于上一个）
"W/*|M/m; */*"  # 第二条所有位点都用 * 标记（同样等价）

# 模式 4: 支持集合
"{W,w}/W|M/m; D/{D,d}"  # maternal 第一个位点是 W 或 w；paternal 是 W 等

# 模式 5: 支持反选
"!w/w|M/m; D/D"  # 不是 w/w

# 模式 6: 支持双向匹配
"W::w|M::m; D::d"  # W|w 或 w|W 等

# 模式 7: 灵活混合
"(W::M; */*)|(!w/!w; D/*)"  # 复杂的逻辑组合
```

### 枚举和规则
```python
# 枚举所有包含 W 的基因型
for gt in species.enumerate_genotypes_matching_pattern("W/*|*/*; */*"):
    print(f"Found: {gt}")

# 在基因驱动规则中使用
ruleset = GameteConversionRuleSet(species)
ruleset.add_convert(
    "W", "D",                                      # W 转换为 D
    rate=0.5,                                      # 转换率 50%
    genotype_filter=species.parse_genotype_pattern("W/*|W/*")  # 仅在 WW 或 Ww 时触发
)
```

## 下一步

- 等待反馈后进行详细实现
- 可选：先实现**第1阶段**（基础框架）验证设计
- 可选：为常见类型的模式预制等助手函数
