# `IndexRegistry` Indexing Mechanism

This chapter explains how NATAL associates genetic objects (Genotype, HaploidGenotype, etc.) with integer indices. This is the key that connects the “high‑level object world” to the “low‑level numerical computation world”.

## Core Concept

The world of genetic objects ↔ The world of integer indices

```
Genotype("Drive|WT")      ↔  integer index 2
HaploidGenotype("R1")     ↔  integer index 3
"default" (glab)          ↔  integer index 0
"Cas9_deposited" (glab)   ↔  integer index 1
```

**Purpose**: NumPy arrays use integer indices (efficient), but users want to work with genetic objects (intuitive). `IndexRegistry` translates between them.

## `IndexRegistry` Class

```python
class IndexRegistry:
    """Stable object → integer index registry"""

    # Mapping dictionaries
    genotype_to_index: Dict[Genotype, int] = {}
    index_to_genotype: List[Genotype] = []

    haplo_to_index: Dict[HaploidGenotype, int] = {}
    index_to_haplo: List[HaploidGenotype] = []

    glab_to_index: Dict[str, int] = {}
    index_to_glab: List[str] = []
```

## Three Registration Dimensions

### 1. Genotype Index

```python
ic = pop.registry  # IndexRegistry instance

# Register a genotype
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = ic.register_genotype(gt)
# Returns: integer index, e.g., 5

# Look up index
gt_idx = ic.genotype_index(gt)  # Get index of a registered genotype

# Reverse lookup
gt_back = ic.index_to_genotype[gt_idx]  # Get the corresponding Genotype object

# Count
n_gt = ic.num_genotypes()  # Total number of genotypes
```

### 2. Haploid Genotype Index

```python
# Haploid genotype (haploid genome)
hg = all_haploid_genotypes[0]

# Register
hg_idx = ic.register_haplogenotype(hg)

# Look up
hg_idx = ic.haplo_index(hg)
hg_back = ic.index_to_haplo[hg_idx]

# Count
n_hg = ic.num_haplogenotypes()
```

### 3. Gamete Label Index

```python
# Gamete labels (e.g., "default", "Cas9_deposited")

# Register
label_idx = ic.register_gamete_label("Cas9_deposited")

# Look up
label_idx = ic.gamete_label_index("Cas9_deposited")
label_back = ic.index_to_glab[label_idx]

# Count
n_glabs = ic.num_gamete_labels()
```

## Practical Examples

### Example 1: Access individual counts

```python
from natal.type_def import Sex

state = pop.state
ic = pop.registry

# Get index of a specific genotype
gt = sp.get_genotype_from_str("A1|A2")
gt_idx = ic.genotype_index(gt)

# Number of females, age 3, genotype A1|A2
count = state.individual_count[Sex.FEMALE, 3, gt_idx]

# Total number of females of genotype A1|A2 across all ages
total = state.individual_count[Sex.FEMALE, :, gt_idx].sum()
```

### Example 2: Modify sperm storage

```python
female_gt = sp.get_genotype_from_str("A1|A1")
male_gt = sp.get_genotype_from_str("A1|A2")

female_gt_idx = ic.genotype_index(female_gt)
male_gt_idx = ic.genotype_index(male_gt)

# Sperm stored in females of genotype A1|A1 at age 3, coming from males of genotype A1|A2
state.sperm_storage[3, female_gt_idx, male_gt_idx] = 500.0
```

### Example 3: Iterate over all genotypes

```python
ic = pop.registry

for gt_idx in range(ic.num_genotypes()):
    gt = ic.index_to_genotype[gt_idx]

    # Total number of individuals with this genotype
    total = state.individual_count[:, :, gt_idx].sum()

    print(f"{gt}: {total:.0f} individuals")
```

## Compressed Index (Advanced)

When dealing with combinations of gametes and labels, we need to handle composite indices:

```python
# A gamete can be a combination of (haploid_idx, label_idx)
# e.g., (A1, default) and (A1, Cas9) are two different “labelled gametes”

def compress_hg_glab(haploid_idx: int, label_idx: int, n_glabs: int) -> int:
    """Compress (haploid, label) into a single index"""
    return haploid_idx * n_glabs + label_idx

def decompress_hg_glab(compressed: int, n_glabs: int) -> Tuple[int, int]:
    """Decompress a single index into (haploid, label)"""
    return compressed // n_glabs, compressed % n_glabs

# Usage example
n_glabs = 2
haploid_idx = 3
label_idx = 1

compressed = compress_hg_glab(haploid_idx, label_idx, n_glabs)
# compressed = 3 * 2 + 1 = 7

haploid_back, label_back = decompress_hg_glab(compressed, n_glabs)
# (3, 1)
```

**Use**: In mapping matrices, gametes are often compressed into a single dimension to save space.

## Usage in Modifiers

When writing Modifiers, you need to use IndexRegistry to obtain genotype indices:

```python
def gene_drive_modifier(pop):
    """
    Gene drive modifier: increase the proportion of drive allele gametes from Drive|WT heterozygotes
    """
    ic = pop.registry

    # Find the index of the Drive|WT genotype
    drive_wt = pop.species.get_genotype_from_str("Drive|WT")
    drive_wt_idx = ic.genotype_index(drive_wt)

    # Return mapping: genotype → {(gamete, label): frequency}
    return {
        drive_wt: {
            ("Drive", "Cas9_deposited"): 0.95,  # 95% drive gametes
            ("WT", "Cas9_deposited"): 0.05,
        }
    }

pop.set_gamete_modifier(gene_drive_modifier, hook_name="drive")
```

## Usage in Hooks

Numba Hooks must work with indices, not objects:

```python
from numba import njit

@njit
def my_hook(ind_count, tick):
    """
    Numba‑compatible Hook – must use integer indices

    Note: Hooks cannot access IndexRegistry at compile time,
    so you need to determine indices at the Python level and hard‑code them.
    """
    if tick == 10:
        # Suppose we already know that the index of genotype A1|A2 is 5
        gt_idx = 5

        # Kill all females of that genotype at ages 2‑4
        ind_count[1, 2:5, gt_idx] = 0

    return 0  # continue

# Use selector patterns to avoid hard‑coding (see Hook System chapter)
```

## Interaction with Modifiers

Modifiers return dictionaries that can use objects as keys; the framework automatically converts them to indices:

```python
# User‑written Modifier (high‑level)
def my_modifier(pop):
    return {
        sp.get_genotype_from_str("A1|A2"): {
            ("A1", "default"): 0.6,
            ("A2", "default"): 0.4,
        }
    }

# Framework processing (automatic)
for gt_key, gamete_dict in my_modifier(pop).items():
    gt_idx = pop.registry.genotype_index(gt_key)

    for (allele_name, label), freq in gamete_dict.items():
        # Look up the haploid genotype index corresponding to allele_name
        # Look up the label index for label
        # Write to the mapping matrix
        pass
```

## Performance Tips

### Cache index lookups

```python
# ❌ Inefficient: repeated lookups
for tick in range(100):
    gt_idx = pop.registry.genotype_index(gt)  # dictionary lookup each time
    # ...

# ✅ Efficient: look up once
gt_idx = pop.registry.genotype_index(gt)
for tick in range(100):
    # reuse gt_idx
    # ...
```

### Batch operations

```python
# ❌ Inefficient: modify one by one
for gt in all_genotypes:
    gt_idx = ic.genotype_index(gt)
    config.viability_fitness[FEMALE, 3, gt_idx] = 0.9

# ✅ Efficient: vectorized
gt_indices = [ic.genotype_index(gt) for gt in all_genotypes]
config.viability_fitness[FEMALE, 3, gt_indices] = 0.9
```

## Quick Reference

| Operation | Code | Returns |
|-----------|------|---------|
| Get genotype index | `ic.genotype_index(gt)` | int |
| Get genotype from index | `ic.index_to_genotype[idx]` | Genotype |
| Get haploid genotype index | `ic.haplo_index(hg)` | int |
| Get label index | `ic.gamete_label_index("label")` | int |
| Total number of genotypes | `ic.num_genotypes()` | int |
| Total number of haploid genotypes | `ic.num_haplogenotypes()` | int |
| Total number of labels | `ic.num_gamete_labels()` | int |

## Relationship with Global Caching

Genotype uses a global cache, while IndexRegistry maintains the object → index registration:

```
String "A1|A2"
    ↓ Species.get_genotype_from_str()
Global cache Species.genotype_cache
    ↓ [hit]
Genotype object (unique)
    ↓ IndexRegistry.register_genotype()
Integer index (e.g., 5)
    ↓
numpy array access individual_count[:, :, 5]
```

The two mechanisms work together:
1. **Global cache** guarantees object uniqueness.
2. **IndexRegistry** builds the mapping from objects to indices.

---

## 🎯 Chapter Summary

| Concept | Purpose | Frequency of use |
|---------|---------|------------------|
| **IndexRegistry** | object ↔ index registry | Used in Modifiers, Hooks |
| **genotype_index()** | get genotype index | common |
| **haplo_index()** | get haploid genotype index | rare (advanced) |
| **gamete_label_index()** | get label index | moderate |
| **compress/decompress** | compress gamete index | low‑level functions |

**Key points**:
1. IndexRegistry is maintained automatically; users usually access it via `pop.registry`.
2. Modifiers return object → frequency dictionaries; the framework automatically converts them to indices.
3. Numba Hooks work directly with indices (object information is lost).
4. Global cache + IndexRegistry form the complete object → index system.

---

## 📚 Related Chapters

- [Genetic Structures and Entities](genetic_structures.md) – creating Genotype objects
- [PopulationState & PopulationConfig](population_state_config.md) – index usage in configuration
- [Modifier Mechanism](modifiers.md) – using IndexRegistry in Modifiers
- [Hook System](hooks.md) – advanced selector patterns for Hooks

---

**Ready to dive into configuration compilation details?** [Proceed to the next chapter: PopulationState & PopulationConfig →](population_state_config.md)
