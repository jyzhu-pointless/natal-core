# Genetic Structures and Entities

This chapter provides an in‑depth explanation of the genetic object system in NATAL: the structural layer (Species/Chromosome/Locus), the entity layer (Gene/Genotype), and the crucial stringification and global caching mechanisms.

## Concept Overview Migrated from the Quick Start

In the Quick Start we kept only the minimal mental model of “structure layer vs entity layer”. Here is the complete explanation:

- **Structure layer (static template)** : Describes the genetic space allowed by the model; does not directly represent any individual.
    - `Species`: Species‑level container, manages chromosomes and global indexing.
    - `Chromosome`: Chromosome level, organises loci and recombination information.
    - `Locus`: Locus level, defines the set of alleles possible at that position.
- **Entity layer (dynamic instance)** : Represents genetic objects that actually appear and evolve during the simulation.
    - `Gene`: A concrete allele at a locus.
    - `Haplotype`: The combination of alleles on a single chromosome across multiple loci.
    - `HaploidGenotype`: The haploid genome across all chromosomes.
    - `Genotype`: The diploid genotype obtained by combining the maternal and paternal haploid genomes.

Why separate the layers?

- The structure layer is defined once during modelling, reusable and stable.
- The entity layer constantly appears, transforms, and is counted during the simulation.
- Separation keeps the API clean and facilitates low‑level indexing and high‑performance computation.

## Core Concepts

NATAL divides genetic objects into two layers:

### 🔧 Structural Layer

**Static**, model‑level blueprints that define the topology of the genetic architecture:

| Class | Description | Example |
|-------|-------------|---------|
| `Species` | Root node of a species, contains all chromosomes | `Species("Mosquito")` |
| `Chromosome` | Chromosome (linkage group), contains loci | Autosomes, X/Y/W sex chromosomes |
| `Locus` | Genetic locus, contains alleles | "A", "B", "white" |

### 🧬 Entity Layer

**Concrete**, individual‑level genetic objects that represent actual genetic material:

| Class | Description | Bound to | Example |
|-------|-------------|----------|---------|
| `Gene` | An allele | Locus | "w+", "w", "Drive" |
| `Haplotype` | A haplotype (allele combination on one chromosome) | Chromosome | Possibly hundreds |
| `HaploidGenotype` | A haploid genome (haplotypes of all chromosomes) | Species | $\sim 2^\text{number of loci}$ possibilities |
| `Genotype` | A diploid genotype | Species | $\sim \text{(number of haploid genotypes)}^2$ possibilities |

## Detailed Structural Layer

### Species – Species and Genetic Architecture

#### Creation method 1: from_dict (recommended for quick definition)

```python
from natal.genetic_structures import Species

sp = Species.from_dict(
    name="Mosquito",
    structure={
        "chr1": {
            "A": ["WT", "Drive", "Resistance"],  # locus A, 3 alleles
            "B": ["B1", "B2"],                   # locus B, 2 alleles
        },
        "chr2": {
            "C": ["C1", "C2"],
        },
    }
)
```

Advantage: concise syntax, easy to load from configuration files.

#### Creation method 2: chained API (more flexible)

```python
sp = Species("Mosquito")

# Autosome
chr1 = sp.add("chr1")
chr1.add("A").add_alleles(["WT", "Drive", "Resistance"])
chr1.add("B").add_alleles(["B1", "B2"])

# X‑linked
chr_x = sp.add("ChrX", sex_type="X")
chr_x.add("white").add_alleles(["w+", "w"])

# Y‑linked (males only)
chr_y = sp.add("ChrY", sex_type="Y")
chr_y.add("Ymarker").add_alleles(["Y"])

# W‑linked (females only, in some birds/butterflies)
# chr_w = sp.add("ChrW", sex_type="W")
```

### Sex Chromosomes

```python
# Inspect properties
if chr_x.is_sex_chromosome:
    print(f"Sex type: {chr_x.sex_type}")  # "X"
    print(f"Sex system: {chr_x.sex_system}")  # "XY"
```

### Chromosome – Chromosomes and Recombination

```python
chr1 = sp.add("chr1")
locus_A = chr1.add("A", position=0.0)      # position 0
locus_B = chr1.add("B", position=50.0)     # position 50

# Set recombination rate (between A and B)
chr1.recombination_map[0] = 0.1  # 10% recombination

# Or via factory method
from natal.genetic_structures import Chromosome
chr2 = Chromosome(
    name="chr2",
    recombination_rates=[0.05, 0.1, 0.05]  # rates between multiple loci
)
```

### Locus – Genetic Locus

```python
# Method 1: add after creation
locus = chr1.add("A")
locus.add_alleles(["A1", "A2", "A3"])

# Method 2: factory method
from natal.genetic_structures import Locus
locus = Locus.with_alleles("A", ["A1", "A2"])

# List alleles
for allele in locus.alleles:
    print(allele.name)
```

## Detailed Entity Layer

### Gene – Allele Instance

An allele is the boundary between “structure” and “entity”.

```python
# Obtain an allele (via Species)
sp = Species.from_dict(
    name="Test",
    structure={"chr1": {"A": ["a1", "a2"]}}
)

a1 = sp.get_gene("a1")
a2 = sp.get_gene("a2")

print(f"Gene name: {a1.name}")
print(f"Locus: {a1.locus}")  # Locus object
print(f"Allele index: {a1.allele_index}")  # 0 or 1
```

### Haplotype

The combination of alleles on a single chromosome. For N loci, there are ∏(number of alleles per locus) possibilities.

```python
# Obtain all possible haplotypes for a chromosome
chr1 = sp["chr1"]  # get the chromosome
all_haplotypes = chr1.get_all_haplotypes()

for hap in all_haplotypes:
    print(f"Haplotype: {hap}")
    # Access alleles at each locus
    for locus in chr1.loci:
        gene = hap[locus]
        print(f"  {locus.name}: {gene.name}")
```

### HaploidGenotype

The complete haploid genome, containing haplotypes for all chromosomes.

```python
# Obtain all possible haploid genomes
all_haploid_gts = sp.get_all_haploid_genotypes()
print(f"Total haploid genotypes: {len(all_haploid_gts)}")

# Access the haplotype for a specific chromosome
hg = all_haploid_gts[0]
hap_chr1 = hg["chr1"]  # Haplotype object
```

### Genotype (diploid, most important!)

Represents the genotype of an individual, composed of two haploid genomes (maternal and paternal).

#### Ways to obtain a Genotype

**Method 1: String parsing (recommended)**

```python
# The most flexible way – create directly from a string
wt_wt = sp.get_genotype_from_str("WT|WT")
wt_drive = sp.get_genotype_from_str("WT|Drive")
drive_drive = sp.get_genotype_from_str("Drive|Drive")

print(f"Genotype: {wt_drive}")  # prints: WT|Drive or Drive|WT (depending on ordering)
```

**Method 2: From haploid genomes**

```python
hg1 = all_haploid_gts[0]
hg2 = all_haploid_gts[1]
genotype = sp.make_genotype(hg1, hg2)
```

**Method 3: Obtain all possible genotypes**

```python
all_genotypes = sp.get_all_genotypes()
print(f"Total diploid genotypes: {len(all_genotypes)}")

for gt in all_genotypes:
    print(f"  {gt}")
```

#### Stringifying Genotype (global cache)

**This is a key feature of NATAL**: Genotype uses a global cache with strings as normalised keys.

```python
# Key point 1: string normalisation
wt_drive = sp.get_genotype_from_str("WT|Drive")
drive_wt = sp.get_genotype_from_str("Drive|WT")

# At the Species level, they are normalised to the same object
print(wt_drive == drive_wt)  # True
print(str(wt_drive) == str(drive_wt))  # True

# Key point 2: caching means string consistency
gt1 = sp.get_genotype_from_str("WT|Drive")
gt2 = sp.get_genotype_from_str("WT|Drive")
print(gt1 is gt2)  # True (same object)

# Key point 3: can be used as dictionary keys
genotype_map = {
    wt_drive: 100,
    drive_drive: 50,
}
```

#### Accessing the internal structure of a Genotype

```python
gt = sp.get_genotype_from_str("WT|Drive")

# Obtain the two haploid genomes
mat = gt.maternal  # HaploidGenotype
pat = gt.paternal  # HaploidGenotype

# Obtain the haplotype for a specific chromosome
mat_chr1_hap = mat["chr1"]  # Haplotype
pat_chr1_hap = pat["chr1"]

# Obtain alleles at a specific locus
mat_A = mat_chr1_hap["A"]  # Gene
pat_A = pat_chr1_hap["A"]

print(f"Maternal A: {mat_A.name}")  # "WT" or "Drive"
print(f"Paternal A: {pat_A.name}")

# Check heterozygosity
is_het_A = mat_A != pat_A
print(f"Heterozygous at A: {is_het_A}")

# Or use built‑in methods
print(f"Overall heterozygous: {gt.is_heterozygous()}")
print(f"Heterozygous at locus A: {gt.is_heterozygous(sp['chr1']['A'])}")
```

## Global Caching Mechanism

### Why is global caching needed?

1. **Memory efficiency**: avoid creating duplicate Genotype objects.
2. **Hashing consistency**: string normalisation ensures `"WT|Drive"` and `"Drive|WT"` map to the same object.
3. **Index stability**: together with IndexRegistry, each Genotype gets a unique integer index.

### How the cache works

```
String "WT|Drive"
    ↓ [normalisation] → sort alphabetically
"Drive|WT"
    ↓ [check cache]
Species.genotype_cache["Drive|WT"]
    ↓ [hit or create]
Genotype object (globally unique)
    ↓ [register with IndexRegistry]
Integer index (e.g., idx=5)
```

### Using the cache

```python
# Cache is managed automatically; the user does not need explicit operations
gt1 = sp.get_genotype_from_str("WT|Drive")
gt2 = sp.get_genotype_from_str("WT|Drive")
# gt1 and gt2 are the same object, no extra memory overhead

# Use as dictionary keys
fitness_map = {
    sp.get_genotype_from_str("WT|WT"): 1.0,
    sp.get_genotype_from_str("WT|Drive"): 0.95,
}
# Dictionary lookup automatically uses Genotype.__hash__ and __eq__

# Referencing during initialisation
initial_individual_count = {
    "female": {
        "WT|WT": [600, 500, 400, ...],
        "WT|Drive": [100, 80, 60, ...],
    }
}
# Strings are parsed at initialisation time and automatically converted to Genotype objects by Species
```

## Sex‑specific Genotypes

In some genetic architectures, the genotype is associated with sex:

```python
# Example: X‑linked inheritance
sp = Species.from_dict(
    name="XLinked",
    structure={
        "ChrX": {"white": ["w+", "w"]},
        "ChrY": {"Ymarker": ["Y"]},
    }
)

# Genotypes at the X‑linked locus
# Female (XX): w+|w+, w+|w, w|w (diploid)
# Male (XY): w+|Y, w|Y (effectively only one allele)

all_gts = sp.get_all_genotypes()
for gt in all_gts:
    print(f"{gt} (Sex: {gt.sex_type})")
    # Output:
    # w+|w+ (Sex: both)
    # w+|w (Sex: both)
    # w|w (Sex: both)
    # w+|Y (Sex: male)
    # w|Y (Sex: male)
```

## Complete Example with Genetic Entities

```python
from natal.genetic_structures import Species

# 1. Define the genetic architecture
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

# 2. Inspect the scale
all_haploid = sp.get_all_haploid_genotypes()
all_genotypes = sp.get_all_genotypes()
print(f"Haploid genotypes: {len(all_haploid)}")  # 2*3*2 = 12
print(f"Diploid genotypes: {len(all_genotypes)}")  # (12+1)*12/2 = 78

# 3. Operate on a specific genotype
gt = sp.get_genotype_from_str("A1|A2")
print(f"Maternal haplotype: {gt.maternal}")
print(f"Paternal haplotype: {gt.paternal}")

# 4. Iterate over all genotypes
for gt in all_genotypes:
    # Use in a model
    pop.initial_individual_count["female"][str(gt)] = 0
```

## Relationship with IndexRegistry

Genotype objects work closely with the IndexRegistry (indexing mechanism):

```python
from natal.nonWF_population import AgeStructuredPopulation

pop = AgeStructuredPopulation(species=sp, ...)

# Obtain the IndexRegistry
ic = pop.registry  # or pop._index_registry

# Genotype → integer index
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = ic.genotype_index(gt)
print(f"Genotype index: {gt_idx}")

# Reverse: integer index → Genotype
gt_back = ic.index_to_genotype[gt_idx]
print(f"Genotype: {gt_back}")

# Use in numpy arrays
individual_count = pop.state.individual_count  # shape: (n_sexes, n_ages, n_genotypes)
female_count_of_gt = individual_count[1, :, gt_idx]  # female count for that genotype across all ages
```

> For more details on IndexRegistry, see [IndexRegistry Indexing Mechanism](index_registry.md)

## Quick Reference for Common Operations

### Creating and querying Genotypes

```python
# Obtain from a string
gt = sp.get_genotype_from_str("WT|Drive")

# All possible genotypes
all_gts = sp.get_all_genotypes()

# Check whether a specific genotype exists
try:
    gt = sp.get_genotype_from_str("Invalid|Invalid")
except KeyError:
    print("Genotype does not exist")
```

### Accessing the internals of a Genotype

```python
gt = sp.get_genotype_from_str("A1|A2")

# The two haploid genomes
mat = gt.maternal
pat = gt.paternal

# Haplotypes for a specific chromosome
mat_hap = mat["chr1"]
pat_hap = pat["chr1"]

# Alleles at a specific locus
mat_gene = mat_hap["A"]
pat_gene = pat_hap["A"]

# Gene name and properties
print(mat_gene.name)  # "A1"
print(mat_gene.locus)  # Locus object
```

### Inspecting and operating

```python
gt = sp.get_genotype_from_str("A1|A2")
locus = sp["chr1"]["A"]

# Check heterozygosity
is_het = gt.is_heterozygous(locus)

# Obtain alleles at that locus
alleles_at_locus = gt.get_alleles_at_locus(locus)  # (mat_gene, pat_gene)
```

---

## 🎯 Chapter Summary

| Layer | Object | Purpose | Creation |
|-------|--------|---------|----------|
| **Structure** | Species | Defines the genetic architecture of a species | `from_dict()` or chained API |
| **Structure** | Chromosome | Defines a chromosome and recombination | `species.add()` |
| **Structure** | Locus | Defines a genetic locus | `chromosome.add()` |
| **Entity** | Gene | An allele instance | `sp.get_gene()` |
| **Entity** | Genotype | Diploid genotype (most commonly used) | `sp.get_genotype_from_str()` |

**Key points**:
1. Genotype uses global caching and string normalisation.
2. Strings and objects can be converted back and forth.
3. Genotype works together with IndexRegistry to provide object↔index mapping.
4. Understanding these is the foundation for using Modifiers and advanced features.

---

## 📚 Related Chapters

- [Quick Start: 15 Minutes to NATAL](quickstart.md) – basic usage examples
- [Builder System Explained](builder_system.md) – chained construction from Species to a runnable population
- [IndexRegistry Indexing Mechanism](index_registry.md) – detailed indexing mechanism
- [Modifier Mechanism](modifiers.md) – how to define genetic rules based on Genotype

---

**Ready to start building a population?** [Proceed to the next chapter: Builder System Explained →](builder_system.md)
