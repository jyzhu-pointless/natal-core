# `IndexRegistry` Indexing Mechanism

`IndexRegistry` is a core component in the NATAL framework responsible for associating genetic objects (such as Genotype, HaploidGenotype, etc.) with integer indices. It serves as a key bridge connecting the "high-level object world" with the "low-level numerical computation world," ensuring that users can work with intuitive genetic objects while the underlying computation efficiently handles integer indices.

## Core Concepts

The object-to-index mapping maintained by `IndexRegistry`:

```
Genotype("Drive|WT")      ↔  integer index 2
HaploidGenotype("R1")     ↔  integer index 3
"default" (glab)          ↔  integer index 0
"Cas9_deposited" (glab)   ↔  integer index 1
```

**Design purpose**: `IndexRegistry` resolves the tension between numpy arrays using integer indices (efficient) and users' preference for genetic objects (intuitive), handling efficient conversion between the two.

## `IndexRegistry` Class

```python
class IndexRegistry:
    """Stable object→integer index registry"""

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
- Maintains the mapping between Genotype objects and integer indices
- Supports genotype registration and lookup
- Provides total genotype count statistics

### 2. HaploidGenotype Index
- Maintains the mapping between HaploidGenotype objects and integer indices
- Supports haploid genotype registration and lookup
- Provides total haploid genotype count statistics

### 3. Gamete Label Index
- Maintains the mapping between gamete label strings and integer indices
- Supports label registration and lookup
- Provides total label count statistics

## User Interface Notes

**Important**: `IndexRegistry` is a low-level data table; users typically do not need to call its methods directly. When accessing genotypes, haploid genotypes, or gamete labels, users should use the following high-level interfaces:

### Direct String Access
```python
# Operate directly with strings without worrying about underlying indices
pop.state.individual_count[Sex.FEMALE, 3, "A1|A2"]
```

### Pattern Matching with Pattern
```python
# Use Pattern for pattern-based operations
from natal.pattern import Pattern
pattern = Pattern("A1|*")
# Match all genotypes containing A1
```

## Internal Framework Usage

Within the NATAL framework, `IndexRegistry` is used for:

### 1. State Data Storage
- The individual count matrix uses integer indices for efficient storage
- The sperm storage matrix is managed using genotype indices
- All state data is accessed via indices

### 2. Modifier System
- Object dictionaries returned by Modifiers are automatically converted to indices
- The framework handles the object-to-index conversion process
- Users only need to work with high-level objects

### 3. Hook System
- Numba Hooks use precomputed indices for efficient operation
- Avoids accessing dynamic registries at compile time
- Avoids hardcoded indices through the selector pattern

## Performance Optimization

Although users do not need to directly manipulate indices, understanding the indexing mechanism helps in writing efficient code:

### Caching Index Lookups
In scenarios where the same genotype is used repeatedly, caching indices can improve performance.

### Batch Operations
For operations involving multiple genotypes, using vectorized approaches is more efficient than processing them one by one.

## Relationship with Global Cache

`IndexRegistry` works in coordination with the Genotype global cache:

```
String "A1|A2"
    ↓ Species.get_genotype_from_str()
Global Cache Species.genotype_cache
    ↓ [hit]
Genotype object (unique)
    ↓ IndexRegistry.register_genotype()
Integer index (e.g., 5)
```

---

## Related Sections

- [Genetic Structures and Entities](2_genetics.md) - Genotype object creation
- [PopulationState & PopulationConfig](4_population_state_config.md) - Index application in configuration
- [Modifier Mechanism](3_modifiers.md) - IndexRegistry usage in Modifiers
- [Hook System](2_hooks.md) - Advanced Hook selector patterns

---

**Ready to dive into configuration compilation details?** [Continue to next chapter: PopulationState & PopulationConfig →](4_population_state_config.md)
