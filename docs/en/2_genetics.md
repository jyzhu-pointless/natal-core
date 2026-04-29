# Genetic Architecture and Objects

This chapter provides a detailed introduction to the genetic object system in NATAL, covering the **structure layer** (`Species` / `Chromosome` / `Locus`) and the **entity layer** (`Gene` / `Haplotype` / `HaploidGenotype` / `Genotype`), along with key stringification and global caching mechanisms. By understanding these core concepts, you can better build and manipulate genetic simulation models.

## Hierarchy of Genetic Objects

NATAL adopts a layered architecture to organize genetic objects, dividing them into two main layers:

### Structure Layer (Static Template)

Describes the genetic space allowed in the model, defining the basic framework of the genetic architecture without directly representing specific individual types:

- **`Species`**: Species-level container, managing chromosome structure and global index
- **`Chromosome`**: Chromosome-level, organizing genetic loci and recombination information
- **`Locus`**: Genetic locus level, defining the set of possible alleles at that position

### Entity Layer (Dynamic Instances)

Represents genetic objects that actually appear and evolve during the simulation:

- **`Gene`**: A concrete allele instance at a genetic locus
- **`Haplotype`**: The combination of alleles across multiple loci on a single chromosome
- **`HaploidGenotype`**: A complete haploid genome spanning all chromosomes
- **`Genotype`**: A diploid genotype formed by combining maternal and paternal haploid genomes

### Advantages of the Layered Design

In the layered architecture:

- **Structure layer** is defined once during the modeling phase, is reusable, and remains stable
- **Entity layer** is generated during the simulation initialization phase and is used to build genetic rules

The layered design keeps the API clean while facilitating low-level indexing and high-performance computing.

### General Rules

- All genetic structures inherit from the base class `GeneticStructure`; all genetic entities inherit from the base class `GeneticEntity`.
- Genetic structures and `Gene` require a string `name` (default is the first parameter) at creation time, which is used to uniquely identify the object and can be retrieved via `get_...` methods.
  - **Note**: `name` must be unique within the same type. If an attempt is made to create an object with a duplicate name, the system will return the cached instance and issue a warning.
- When creating genetic structures (except for the top-level `Species`), you need to specify the parent structure instance; you can create them directly via the parent structure's `add` method.
- When creating a `Gene`, you need to specify the `Locus` instance it belongs to; you can create genes directly via the `Locus`'s `add_alleles` method.
- Other genetic entities at various levels are automatically created during population initialization. Manual management is typically not required; instances can be accessed (or created in advance) through the corresponding string format from the `Species`.

## Structure Layer in Detail

### Species: Species and Genetic Architecture

The `Species` class is the root node of the genetic architecture, responsible for managing all chromosome structures and global indices, serving as the core container of the entire genetic system.

#### Creation Method 1: `from_dict` (Quick Definition, Recommended)

It is recommended to use the `Species.from_dict` method with a dictionary format to quickly define the species' genetic architecture.

```python
import natal as nt

sp = nt.Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "A": ["WT", "Drive", "Resistance"],  # Locus A, containing 3 alleles
            "B": ["B1", "B2"],                   # Locus B, containing 2 alleles
        },
        "chr2": {
            "C": ["C1", "C2"],
        },
    },
    gamete_labels=["default", "Cas9_deposited"]  # Optional: gamete labels for simulating maternal effects (e.g., Cas9 protein deposition)
)
```

#### Extended Format: Declaring Sex Chromosome Information

NATAL supports multiple sex chromosome systems, including XY, ZW, etc.

When sex chromosomes need to be defined, you can use the extended format to explicitly specify the chromosome type:

```python
sp = nt.Species.from_dict(
    name="MosquitoSexAware",
    structure={
        "chrA": {    # Autosome, no extended format needed
            "A": ["A1", "A2"],
        },
        "chrX": {
            "sex_type": "X",    # X chromosome
            "loci": {
                "sx": ["X1"],
            },
        },
        "chrY": {
            "sex_type": "Y",    # Y chromosome
            "loci": {
                "sy": ["Y1"],
            },
        },
    },
)
```

You can check the nature of chromosomes:

```python
# Check chromosome nature
if chr_x.is_sex_chromosome:
    print(f"Sex chromosome type: {chr_x.sex_type}")  # Output: "X"
    print(f"Sex chromosome system: {chr_x.sex_system}")  # Output: "XY"
```

#### Creation Method 2: Chain API

The chain API provides a more flexible way to build, suitable for scenarios requiring dynamic construction or complex configuration, offering better control over the build process:

```python
sp = nt.Species("Mosquito")

# Autosomes
chr1 = sp.add("chr1")
chr1.add("A").add_alleles(["WT", "Drive", "Resistance"])
chr1.add("B").add_alleles(["B1", "B2"])

# X chromosome
chr_x = sp.add("ChrX", sex_type="X")
chr_x.add("white").add_alleles(["w+", "w"])

# Y chromosome (male only)
chr_y = sp.add("ChrY", sex_type="Y")
chr_y.add("Ymarker").add_alleles(["Y"])

# ZW sex chromosome system is also supported
# chr_w = sp.add("ChrW", sex_type="W")
```

### Chromosome: Chromosome

The chromosome (`Chromosome` class) manages genetic loci and recombination information.

Chromosomes are automatically created during the `Species.from_dict()` construction process, or you can create them and add loci using the chain API:

```python
chr1 = sp.add("chr1")
chr1.add("A").add_alleles(["A1", "A2"])
chr1.add("B").add_alleles(["B1", "B2"])
chr1.add("C").add_alleles(["C1", "C2"])
```

You can retrieve a chromosome instance by name using the `Species.get_chromosome()` method:

```python
# Get chromosome by name
chr1 = sp.get_chromosome("chr1")
chr_x = sp.get_chromosome("ChrX")
```

You can delete a chromosome from a `Species`:

```python
sp.remove_chromosome("chr1")
```

After deletion, the chromosome will be removed from the species' genetic architecture, but the `Chromosome` instance itself will continue to exist.

#### Recombination Rate and Recombination Map

The recombination rate defines the probability of crossover events between loci on a chromosome, producing recombinant gametes during the simulation of meiosis and gamete formation.

Recombination rates are managed through a `RecombinationMap`, which stores recombination rates between adjacent loci.

You can set recombination rates between adjacent loci using the following methods. For recombination between multiple loci, no interference is assumed, meaning recombination between each pair of loci is independent.

```python
# Method 1: Set recombination rates between adjacent loci one by one
chr1.set_recombination("A", "B", 0.1)  # 10% recombination rate between A and B
chr1.set_recombination("B", "C", 0.2)  # 20% recombination rate between B and C

# Method 2: Batch set recombination rates
chr1.set_recombination_bulk({
    ("A", "B"): 0.1,
    ("B", "C"): 0.2
})

# Method 3: Access the recombination map using Locus names as indices
chr1.recombination_map["A", "B"] = 0.1
chr1.recombination_map["B", "C"] = 0.2

# Method 4: Set all adjacent interval recombination rates at once using a slice
# (the list length must equal number of loci - 1)
chr1.recombination_map[:] = [0.1, 0.2]
```

Recombination rates between adjacent loci should be in the range $[0.0, 0.5]$, where $0.0$ indicates complete linkage (no crossing over) and $0.5$ indicates crossing over always occurs, approximating independent assortment.

When no recombination rate is specified between adjacent loci, the default value is $0.0$, i.e., complete linkage.

> **Note**: Recombination rate settings depend on locus order, which is controlled by the `position` parameter. See the [About the `position` Parameter](#about-the-position-parameter) section under Locus.

### Locus: Genetic Locus

The locus (`Locus` class) defines the set of alleles at a specific position.

Like chromosomes, loci are automatically created during the `Species.from_dict()` construction process, or you can create them and add alleles using the chain API.

```python
chr1 = sp.get_chromosome("chr1")
chr1.add("A").add_alleles(["A1", "A2"])
chr1.add("B").add_alleles(["B1", "B2"])
chr1.add("C").add_alleles(["C1", "C2"])
```

When creating a `Locus` using the chain API, the following parameters can be specified:

- `position`: Indicates the relative position of the locus on the chromosome. If the `position` parameter is not specified, it defaults to `max(existing locus positions) + 1`.
- `recombination_rate_with_previous`: Indicates the recombination rate between this locus and the previous locus. If not specified, it defaults to $0.0$, i.e., complete linkage. **If it is the first locus**, this indicates the recombination rate between this locus and the next locus.

```python
chr1.add("A", position=0.0)
chr1.add("B", position=50.0)
chr1.add("C", position=100.0, recombination_rate_with_previous=0.05)
```

You can retrieve a `Locus` instance using the following methods:

```python
# Get locus by name
locus_A = chr1.get_locus("A")
locus_B = chr1.get_locus("B")

# Get locus across the entire Species scope
locus = sp.get_locus("A")
```

You can delete a `Locus` instance from a chromosome:

```python
chr1.remove_locus("A")
```

After deletion, the locus will be removed from the chromosome, but the `Locus` instance itself will continue to exist. The loci on either side of the deleted locus become new adjacent loci, and the recombination rate between them is automatically set to the sum of the recombination rates on both sides of the original locus.

#### About the `position` Parameter

The `position` parameter is used to define the relative position of a locus on a chromosome, **serving only as a sorting label**; its absolute magnitude is unrelated to the recombination rate.

If `position` is not specified, the system automatically sets it to `max(existing locus positions) + 1`.

Please avoid modifying the `position` parameter after creation, as this may lead to unexpected results. It is recommended to set the `position` parameter once during creation.

> **Note**: If the `position` parameter is modified after creation and the change alters the order of loci, the system will update the recombination rates. The behavior is equivalent to removing the locus and re-adding it to the specified position, with a recombination rate of $0.0$ to the previous locus.

## Entity Layer in Detail

### Gene: Allele Instance

The `Gene` class is the interface between the structure layer and the entity layer, representing a concrete allele instance.

**The identifier `name` of a `Gene` must be unique within the `Species` scope.**

The `Species.get_gene` method can be used to quickly retrieve a `Gene` instance, but direct manipulation of `Gene` instances is generally not required. In situations where a specific allele needs to be specified, the string `name` can usually be used directly to reference the allele instance.

```python
# Get genes across the entire Species scope
gene_wt = sp.get_gene("WT")
gene_drive = sp.get_gene("Drive")
```

### Haplotype: Haplotype

`Haplotype` represents the combination of alleles across all loci on a single chromosome. For a chromosome containing $N$ loci, the number of possible haplotypes is the product of the number of alleles at each locus: $\prod_{i=1}^N \text{number of alleles at each locus}$.

Manual retrieval of `Haplotype` instances is generally not required.

```python
# Get all possible haplotypes on a chromosome
chr1 = sp.get_chromosome("chr1")  # Get chromosome object
all_haplotypes = chr1.get_all_haplotypes()

# Iterate over all haplotypes
for hap in all_haplotypes:
    print(f"Haplotype: {hap}")
    # Access the allele at each locus
    for locus in chr1.loci:
        gene = hap.get_gene_at_locus(locus)
        print(f"  {locus.name}: {gene.name}")
```

### HaploidGenotype: Haploid Genotype

`HaploidGenotype` represents a complete haploid genotype, containing the combination of haplotypes from all chromosomes of the species.

#### Retrieving Haploid Genotypes from Formatted Strings

**String parsing is the most flexible approach**, supporting direct retrieval of haploid genotypes from strings. When printing a haploid genotype, it is also automatically converted to string format, consistent with the input string format.

```python
sp = nt.Species.from_dict(
    name="TestDrive",
    structure={
        "chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
        "chr2": {"X": ["X", "x"], "Y": ["Y", "y"]}
    }
)

# Retrieve haploid genotype directly from string
hg1 = sp.get_haploid_genotype_from_str("ABC;XY")
hg2 = sp.get_haploid_genotype_from_str("a/b/c;x/y")  # Equivalent notation

print(f"Haploid genotype: {hg1}")  # Output: ABC;XY
```

#### String Parsing Syntax Rules

String parsing follows these syntax rules:

- **Semicolon (;) separates different chromosomes**: Each semicolon separates the gene combination of one chromosome
- **Slash (/) separates genes within the same chromosome**: Each slash separates the allele at one locus
- **Single-character genes may omit the slash**: If all genes are single characters, the slash delimiter can be omitted
- **Multi-character genes must use the slash**: If gene names contain multiple characters, they must be separated by slashes

```python
# Example 1: Single-character genes, slash can be omitted
hg1 = sp.get_haploid_genotype_from_str("ABC;XY")
# Equivalent to: hg1 = sp.get_haploid_genotype_from_str("A/B/C;X/Y")

# Example 2: Multi-character genes, must use slash
hg2 = sp.get_haploid_genotype_from_str("WT/Drive/R2;X/Y")

# Example 3: Mix of single-character and multi-character genes
hg3 = sp.get_haploid_genotype_from_str("A/WT/Drive;X/Y")
```

#### Caching Mechanism

`HaploidGenotype` uses a **Species-scoped caching mechanism**, using reversible, ordered strings as keys to ensure performance and consistency.

```python
# String parsing is automatically cached
hg1 = sp.get_haploid_genotype_from_str("ABC;XY")
hg2 = sp.get_haploid_genotype_from_str("ABC;XY")

print(hg1 is hg2)  # Output: True (same instance)
```

### Genotype: Diploid Genotype (Core Concept)

`Genotype` is the most central genetic object in NATAL, representing the complete diploid genotype of an individual, composed of maternal and paternal haploid genotypes.

#### Retrieving Genotypes from Formatted Strings

Like `HaploidGenotype`, `Genotype` also supports direct retrieval from strings and automatically outputs string format when printing a genotype.

```python
sp = nt.Species.from_dict(
    name="TestDrive",
    structure={"chr1": {"loc": ["WT", "Drive"]}}
)

# Retrieve genotype directly from string
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
drive_drive = sp.get_genotype_from_str("Drive|Drive")

print(f"Genotype: {wt_drive}")  # Output: WT|Drive (strictly preserves maternal|paternal order)
```

#### String Parsing Syntax Rules

The string parsing syntax for `Genotype` is essentially the same as for `HaploidGenotype`, with the addition of maternal and paternal separation:

- **Pipe (|) separates maternal and paternal**: The left side of the pipe is the maternal haploid genotype, and the right side is the paternal haploid genotype
- **Other rules are the same as HaploidGenotype**: Including semicolons to separate chromosomes, slashes to separate genes, and the ability to omit slashes for single-character genes

```python
# Example 1: Single-character genes, slash can be omitted
gt1 = sp.get_genotype_from_str("ABC|abc")
# Equivalent to: gt1 = sp.get_genotype_from_str("A/B/C|a/b/c")

# Example 2: Multi-character genes, must use slash
gt2 = sp.get_genotype_from_str("WT/Drive/R2|WT/Drive/R2")

# Example 3: Mix of single-character and multi-character genes
gt3 = sp.get_genotype_from_str("A/WT/Drive|a/WT/Drive")
```

#### Caching Mechanism

Like `HaploidGenotype`, `Genotype` uses a **Species-scoped caching mechanism**, using reversible, ordered strings as keys to ensure performance and consistency.

#### Pattern: Natural Extension of String Format

Building on the precise string format, NATAL provides **Pattern matching** as a natural extension of the string format, supporting wildcards and advanced matching capabilities.

**Pattern is a superset of the precise string format**: It adds the following capabilities to the precise string format:
- `*` wildcard: matches any allele
- `{A,B,C}` set matching: matches any allele in the set
- `!A` exclusion matching: matches any allele except A
- `()` grouping: explicitly groups loci on a chromosome
- `::` unordered matching: indicates that maternal and paternal order is irrelevant

For detailed rules and examples, please refer to [Genotype Pattern Matching](2_genotype_patterns.md).

**Relevant methods in `Species`**:
- `parse_genotype_pattern(pattern: str)`: Parse a diploid genotype pattern
- `parse_haploid_genome_pattern(pattern: str)`: Parse a haploid genotype pattern
- `enumerate_genotypes_matching_pattern(pattern: str)`: Enumerate genotypes matching the pattern
- `enumerate_haploid_genomes_matching_pattern(pattern: str)`: Enumerate haploid genotypes matching the pattern

Pattern syntax maintains compatibility with the precise string format; all precise strings can be correctly matched by Pattern.

## Complete Example

```python
# 1. Define genetic architecture
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

# 2. Check scale
all_haploid = sp.get_all_haploid_genotypes()
all_genotypes = sp.get_all_genotypes()
print(f"Haploid genotypes: {len(all_haploid)}")  # 2*3*2 = 12
print(f"Diploid genotypes: {len(all_genotypes)}")  # 12*12 = 144

# 3. Work with specific genotypes
gt = sp.get_genotype_from_str("A1|A2")
print(f"Maternal haplotype: {gt.maternal}")
print(f"Paternal haplotype: {gt.paternal}")
```

## Collaborative Workflow Between Entity Layer and IndexRegistry

### Entity Layer Generation Timing and IndexRegistry Management

During the **simulation initialization phase**, each `Genotype` and `HaploidGenotype` is generated and registered with the IndexRegistry for management. The specific workflow is as follows:

```
String "A1|A2"
    ↓ Species.get_genotype_from_str()
Global cache Species.genotype_cache
    ↓ [cache hit]
Genotype object (unique instance)
    ↓ IndexRegistry.register_genotype()
Integer index (e.g., 5)
    ↓
numpy array access individual_count[:, :, 5]
```

### Genotype Object and IndexRegistry Coordination

```python
pop = nt.AgeStructuredPopulation(species=sp, ...)

# Get IndexRegistry
registry = pop.registry  # or pop._index_registry

# Genotype → Integer index
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = registry.genotype_index(gt)
print(f"Genotype index: {gt_idx}")

# Reverse: Integer index → Genotype
gt_back = registry.index_to_genotype[gt_idx]
print(f"Genotype: {gt_back}")

# Use in numpy arrays
individual_count = pop.state.individual_count  # shape: (n_sexes, n_ages, n_genotypes)
female_count_of_gt = individual_count[1, :, gt_idx]  # Female count across all ages for a genotype
```

> For more details on IndexRegistry, see [IndexRegistry Indexing Mechanism](4_index_registry.md)

***

## Chapter Summary

| Layer     | Object         | Purpose          | Creation/Access Method                       |
| --------- | -------------- | ---------------- | -------------------------------------------- |
| **Structure** | `Species`    | Define species genetic architecture | `from_dict()` or chain API |
| **Structure** | `Chromosome` | Define chromosomes and recombination | `species.add()` |
| **Structure** | `Locus`      | Define genetic loci | `chromosome.add()` |
| **Entity** | `Gene`       | Allele instance | `sp.get_gene()` |
| **Entity** | `HaploidGenotype` | Haploid genotype | `sp.get_haploid_genotype_from_str()` |
| **Entity** | `Genotype`   | Diploid genotype | `sp.get_genotype_from_str()` |

### Key Features

1. **Global Caching Mechanism**: `Genotype` uses string caching to ensure performance and consistency
2. **Bidirectional Conversion**: Strings and objects can be converted to each other, supporting flexible manipulation
3. **Index Mapping**: Coordinates with `IndexRegistry` to achieve efficient mapping between objects and indices
4. **Layered Design**: Structure layer and entity layer are separated, supporting complex genetic architecture modeling

### Application Value

The genetic architecture system provides powerful modeling capabilities for population genetic simulation. Understanding the genetic architecture system is the foundation for using advanced NATAL features such as the `Modifier` mechanism.

## Related Sections

- [Quick Start: Get Started with NATAL in 15 Minutes](1_quickstart.md) - Basic usage examples
- [Population Initialization](2_population_initialization.md) - Chain-based construction from Species to a runnable population
- [IndexRegistry Indexing Mechanism](4_index_registry.md) - Detailed mechanism of object indexing
- [Genotype Pattern Syntax and Matching](2_genotype_patterns.md) - Genotype pattern expression, `|`/`::` order rules, and matching examples
- [Modifier Mechanism](3_modifiers.md) - How to define genetic rules based on Genotype
