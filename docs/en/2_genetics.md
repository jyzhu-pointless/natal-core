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
    gamete_labels=["default", "Cas9_deposited"],  # Optional: gamete labels for maternal effects (e.g. Cas9 deposition)
    somatic_labels=["wildtype", "Cas9_high"],     # Optional: somatic labels for individual state (e.g. Cas9 expression)
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
chr_x = sp.get_chromosome("chrX")
if chr_x.is_sex_chromosome:
    print(f"Sex chromosome type: {chr_x.sex_type}")  # Output: "X"
    print(f"Sex chromosome system: {chr_x.sex_system}")  # Output: "XY"
```

#### Gamete Labels and Somatic Labels

`gamete_labels` and `somatic_labels` introduce label dimensions to the genetic system, marking extra information carried by gametes and individuals.

**Gamete labels** (pre-existing) define the marker types a gamete can carry. The `Species.gamete_labels` attribute defaults to an empty list `[]`; when none are declared the registry uses the single `"default"` label (no distinction). A common use is marking gametes with deposited Cas9 protein (`"Cas9_deposited"`), together with presets such as HomingDrive.

**Somatic labels** (new) are the symmetric counterpart — they define the somatic markers an individual can carry. `Species.somatic_labels` also defaults to `[]`; when none are declared the registry uses `"default"`. They can mark states such as an individual's Cas9 expression level or toxin load. Once declared, the registry uses exactly the declared labels and does not add `"default"` automatically.

Both are declared when the `Species` is constructed and are shared by every population of that species:

```python
sp = nt.Species("Mosquito",
    gamete_labels=["default", "Cas9_deposited"],
    somatic_labels=["wildtype", "Cas9_high"],
)
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
chr_x.add("white").add_alleles(["wp", "w"])

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
removed_species = nt.Species("RemovedChromosomeExample")
removed_chr = removed_species.add("removed_chr")
removed_species.remove_chromosome("removed_chr")
assert removed_species.get_chromosome("removed_chr") is None
removed_chr.add("D", position=150.0)  # Existing object remains usable.
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
# Get all possible haplotypes on a chromosome: take each haploid genotype's
# haplotype for this chromosome, then deduplicate
chr1 = sp.get_chromosome("chr1")  # Get chromosome object
all_haplotypes = list(dict.fromkeys(
    hg.get_haplotype_for_chromosome(chr1)
    for hg in sp.get_all_haploid_genotypes()
))

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

**String parsing is the most flexible approach**, supporting direct retrieval of haploid genotypes from strings. Printing a haploid genotype also converts it to string format automatically, but the output uses the canonical slash-separated form and is not necessarily character-for-character identical to the input (for example, input `"ABC;XY"` prints as `"A/B/C;X/Y"`).

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
hg2 = sp.get_haploid_genotype_from_str("A/B/C;X/Y")  # Equivalent notation

print(f"Haploid genotype: {hg1}")  # Output: A/B/C;X/Y
print(hg1 is hg2)  # Output: True (both spellings resolve to the same instance)
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

# Example 2: Slashes written explicitly yield the same haploid genotype
hg2 = sp.get_haploid_genotype_from_str("A/B/C;X/Y")
print(hg1 is hg2)  # Output: True

# Example 3: All recessive alleles
hg3 = sp.get_haploid_genotype_from_str("abc;xy")
print(hg3)  # Output: a/b/c;x/y
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
    name="TestGenotype",
    structure={
        "chr1": {"A": ["A", "a"], "B": ["B", "b"], "C": ["C", "c"]},
        "chr2": {"X": ["WT", "Drive"], "Y": ["R1", "R2"]},
    }
)

# Retrieve genotype directly from string
gt1 = sp.get_genotype_from_str("ABC|abc; WT/R1|Drive/R2")
gt2 = sp.get_genotype_from_str("A/B/C|a/b/c; WT/R1|Drive/R2")
gt3 = sp.get_genotype_from_str("abc|ABC; Drive/R2|WT/R1")

print(f"Genotype: {gt1}")  # Output: A/B/C|a/b/c;WT/R1|Drive/R2
print(gt1 is gt2, gt1 is gt3)  # Output: True True (equivalent spellings; unordered=True also canonicalizes order)
```

#### String Parsing Syntax Rules

The string parsing syntax for `Genotype` is essentially the same as for `HaploidGenotype`, with the addition of maternal and paternal separation:

- **Pipe (|) separates maternal and paternal**: within each chromosome segment, the left side of the pipe is that chromosome's maternal haplotype and the right side is the paternal haplotype; segments are still separated by semicolons, e.g. `A/B/C|a/b/c; WT/R1|Drive/R2`. **Note:** by default (`Species.unordered=True`), the system canonicalizes the pair so `A|a` and `a|A` resolve to the same genotype — tracking parent-of-origin explicitly requires `unordered=False`.
- **Other rules are the same as HaploidGenotype**: Including semicolons to separate chromosomes, slashes to separate genes, and the ability to omit slashes for single-character genes

```python
# Example 1: Single-character genes, slash can be omitted (chr1 segment)
gt1 = sp.get_genotype_from_str("ABC|abc; WT/R1|Drive/R2")
# Equivalent to: gt1 = sp.get_genotype_from_str("A/B/C|a/b/c; WT/R1|Drive/R2")

# Example 2: Multi-character genes must use slash (the chr2 segment's WT/Drive and R1/R2 cannot omit slashes)
gt2 = sp.get_genotype_from_str("A/B/C|a/b/c; WT/R1|Drive/R2")

# Example 3: Swap maternal/paternal and use different alleles
gt3 = sp.get_genotype_from_str("abc|ABC; Drive/R2|WT/R1")
print(gt1 is gt2, gt1 is gt3)  # Output: True True
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
- `::` unordered matching: indicates that maternal and paternal order is irrelevant (for pattern matching; string genotype representation always uses `|` and canonicalizes automatically when `unordered=True`)

For detailed rules and examples, please refer to [Genotype Pattern Matching](2_genotype_patterns.md).

**Relevant methods in `Species`**:
- `parse_genotype_pattern(pattern: str)`: Parse a diploid genotype pattern
- `parse_haploid_genome_pattern(pattern: str)`: Parse a haploid genotype pattern
- `enumerate_genotypes_matching_pattern(pattern: str)`: Enumerate genotypes matching the pattern
- `enumerate_haploid_genomes_matching_pattern(pattern: str)`: Enumerate haploid genotypes matching the pattern

Pattern syntax maintains compatibility with the precise string format; all precise strings can be correctly matched by Pattern.

#### Genotype Canonicalization

**What is canonicalization?**

When `Species.unordered=True` (the default), the system normalizes maternal/paternal order so that `A|a` and `a|A` resolve to the same `Genotype` instance.

The algorithm performs a per-locus allele index comparison. At each locus, the maternal and paternal allele indices are compared; if the maternal index is greater than the paternal index, the two haplotypes for that chromosome are swapped. This logic lives in `canonical_haploid_pair()` in `_helpers.py`.

**When does it happen?**

Canonicalization occurs at `Genotype.__new__` time, via a call to `canonical_haploid_pair()`. Combined with the instance cache, the canonical form hits the cache first, so `A|a` and `a|A` return the **same `Genotype` object**.

**Sex chromosome special handling**

Sex chromosomes with different types (`X|Y`, `Z|W`) preserve their maternal/paternal ordering — the swap logic is skipped. Same-type sex chromosomes (`X|X`, `Z|Z`) are canonicalized per-locus like autosomes.

**When to use `unordered=False`**

- Parent-of-origin effects (genomic imprinting)
- Tracking which parent contributed which allele
- Any scenario where `A|a ≠ a|A` matters

**Pattern matching implications**

- `|` in pattern: strict order (but `Species.unordered` canonicalization means both `A|a` and `a|A` resolve to the same `Genotype` in the registry)
- `::` in pattern: explicitly unordered match, works regardless of order
- With `unordered=True` (default): `|` in patterns is auto-promoted to `::` for resolution (because canonicalization already normalized the stored genotype)

**Migration note**

- In v0.1.x, `unordered` defaulted to `False`. v0.2.0 changed it to `True`.
- Code that relied on the `A|a ≠ a|A` distinction needs explicit `unordered=False`.

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
gt = sp.get_genotype_from_str("A1/B1|A2/B2; C1|C2")
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
pop = (nt.AgeStructuredPopulation
    .setup(species=sp, name="IndexDemo")
    .age_structure(n_ages=4, new_adult_age=2)
    .initial_state({
        "female": {"A1/B1|A2/B2; C1|C2": [0, 0, 100, 0]},
        "male": {"A1/B1|A2/B2; C1|C2": [0, 0, 100, 0]},
    })
    .build()
)

# Get IndexRegistry
registry = pop.registry  # or pop.index_registry

# Genotype → integer index (a ZType index also needs the somatic label; this species declares no somatic_labels, so the default label is "default")
gt = sp.get_genotype_from_str("A1/B1|A2/B2; C1|C2")
gt_idx = registry.ztype_index(gt, "default")
print(f"ZType index: {gt_idx}")

# Reverse: integer index → (Genotype, slab label)
gt_back, slab_back = registry.index_to_ztype[gt_idx]
print(f"ZType: {gt_back} @{slab_back}")

# Use in numpy arrays
individual_count = pop.state.individual_count  # shape: (n_sexes, n_ages, n_ztypes)
female_count_of_gt = individual_count[0, :, gt_idx]  # Female count across all ages for a genotype
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
