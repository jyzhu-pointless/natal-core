# `IndexRegistry` Indexing Mechanism

`IndexRegistry` is a core component in the NATAL framework responsible for associating genetic objects (such as Genotype, HaploidGenotype, etc.) with integer indices. It serves as a key bridge connecting the "high-level object world" with the "low-level numerical computation world," ensuring that users can work with intuitive genetic objects while the underlying computation efficiently handles integer indices.

## Core Concepts

The object-to-index mapping maintained by `IndexRegistry` operates in two layers:

### ZType (Zygote Type) — Diploid Individual Space

Each individual in the population is identified by a ZType:

```
ZType = (Genotype, slab_label)

(Genotype("Drive|WT"), "default")       ↔  ZType index 0
(Genotype("Drive|WT"), "infected")      ↔  ZType index 1
(Genotype("WT|WT"),   "default")        ↔  ZType index 2
```

The first component is the diploid genotype (e.g., `"Drive|WT"`). The second component is a **slab** (somatic label), which models non-heritable individual traits such as infection status or transgenic marker expression.

**Design purpose**: Flat ZType indexing allows independent pruning of individual (genotype, slab) pairs during compression. Unlike a formula `g * n_slabs + s`, each pair has its own dict entry and can be removed independently.

### GType (Gamete Type) — Haploid Gamete Space

Each gamete (sperm/egg) produced by the population is identified by a GType:

```
GType = (HaploidGenotype, glab_label)

(HaploidGenotype("Drive"), "default")        ↔  GType index 0
(HaploidGenotype("Drive"), "cas9_deposited") ↔  GType index 1
(HaploidGenotype("WT"),    "default")         ↔  GType index 2
```

The first component is the haploid genotype. The second component is a **glab** (gamete label), which classifies gametes by origin mechanism (e.g., which maternal genotype produced them, whether Cas9 was deposited).

**Design purpose**: GType space allows the engine to track gamete subpopulations that behave differently during fertilisation, without expanding the diploid state space.

### Slab and Glab — Symmetric Label Systems

| Layer | Label | Meaning | Use Case |
|-------|-------|---------|----------|
| Diploid (ZType) | **slab** (somatic label) | Non-heritable individual state | Infection status, transgenic background |
| Haploid (GType) | **glab** (gamete label) | Gamete origin classification | Cas9 deposition, maternal effect |

Both systems are **symmetric** in design:
- Labels are defined on the `Species` object via `somatic_labels` / `gamete_labels`.
- If unspecified, a single `"default"` label is created automatically.
- The engine cross-products every genotype/haplotype with every label, producing the full ZType/GType space.

Slabs are used by concrete Presets such as **Wolbachia** (cytoplasmic incompatibility modelled via a `"wolbachia_infected"` slab) and **TransgenicBackground** (marker expression tracked per individual). Without these Presets, most simulations have a single `"default"` slab and the slab system is invisible.

### Index Registry Structure

The old registry stored flat lists of genotypes and haplotypes. The modern registry uses flat lists of (entity, label) pairs:

```python
class IndexRegistry:
    """Stable object→integer index registry"""

    # ZType space (diploid layer primary index)
    _ztype_to_index: Dict[Tuple[Genotype, str], int] = {}
    _index_to_ztype: List[Tuple[Genotype, str]] = []

    # GType space (gamete layer primary index)
    _gtype_to_index: Dict[Tuple[HaploidGenotype, str], int] = {}
    _index_to_gtype: List[Tuple[HaploidGenotype, str]] = []

    # Label metadata (ordered lists)
    slab_labels: List[str] = []
    glab_labels: List[str] = []
```

### Relationship with the Old API

Backward-compatible properties reconstruct the flat lists from the ZType/GType spaces:

```python
# Old-style: unique genotypes only (deduplicated from ZType space)
registry.index_to_genotype  # [Genotype("A|A"), Genotype("A|a"), ...]
registry.haplo_to_index     # {HaploidGenotype("A"): 0, HaploidGenotype("a"): 1, ...}

# New-style: includes label dimension
registry.index_to_ztype     # [(Genotype("A|A"), "default"), (Genotype("A|A"), "infected"), ...]
registry.index_to_gtype     # [(HaploidGenotype("A"), "default"), (HaploidGenotype("A"), "cas9_deposited"), ...]
```

The computed `N_ztype` is the length of `_index_to_ztype` — this is the value consumed as the last axis of the engine's `individual_count` array.

## Registration Process

### Build-Time Registration

During `build_registry()`, the registry is populated from the Species:

```python
String "A1|A2"
    ↓ Species.get_genotype_from_str()
Genotype object (unique)
    ↓ IndexRegistry.register_genotype()   # auto-cross-products with ALL slab_labels
ZType entries: (A1|A2, "default"), (A1|A2, "infected"), ...
```

Labels are registered first so that the auto-cross-product covers all slab/glab combinations:

1. Register `glab_labels` → `GType = Haplotype × all_glabs`
2. Register `slab_labels` → `ZType = Genotype × all_slabs`
3. Register genotypes → each becomes `n_slab` ZType entries
4. Register haplotypes → each becomes `n_glab` GType entries

### Registration API

```python
# Low-level: register a single (genotype, slab) pair
registry.register_ztype(Genotype("A|a"), "default")       # returns ZType index

# High-level: register a genotype + auto-cross-product with all slabs
registry.register_genotype(Genotype("A|a"))                # returns list of ZType indices

# Similar for GType space
registry.register_gtype(HaploidGenotype("A"), "default")   # returns GType index
registry.register_haplogenotype(HaploidGenotype("A"))      # returns list of GType indices

# Label registration
registry.register_somatic_label("infected")                # returns slab index
registry.register_gamete_label("cas9_deposited")           # returns glab index
```

## The `@slab` Syntax

When a genotype string includes an `@slab` suffix, it specifies both the genotype pattern and the slab constraint:

```
"A|a@infected"      → Genotype("A|a") with slab = "infected"
"WT|Dr@default"     → Genotype("WT|Dr") with slab = "default"
"Drive|WT"          → Genotype("Drive|WT") with no slab constraint (see below)
```

The `@slab` suffix is parsed by `ZygoteTypePattern` (defined in `natal.patterns.elements.diploid`). The base genotype pattern is everything before the last `@`.

### Naming Convention: `genotypes` Accepts ZType Strings

User-facing parameters are named `genotypes` for familiarity, but they actually accept ZType pattern strings that may include an optional `@slab` suffix. This is intentional — most users only need genotype selection, and slab is an advanced feature only needed when using cytoplasmic Presets.

```python
# Typical usage — no slab, just genotype
Op.add(genotypes="Drive|WT", delta=500)

# Advanced usage — slab-constrained
Op.add(genotypes="Drive|WT@infected", delta=500)
```

### `@`-Absence Behavior: Two Different Rules

The meaning of "no `@slab`" depends on which API function you are calling:

| Context | Resolution Method | No `@slab` Means |
|---------|------------------|-------------------|
| Hooks (`Op.add`, `Op.kill`, etc.) and `fitness()` | `resolve_ztype_indices()` | **All slabs** — matches every registered slab variant |
| `initial_state()` | `resolve_default_ztype_index()` | **Only `@default` slab** — returns the first matching ZType |

#### Why This Difference?

- **Hooks and `fitness()`** use `resolve_ztype_indices()` which returns every ZType whose pattern matches. When the pattern has no slab constraint (`slab is None`), `ZygoteTypePattern.matches()` returns `True` for *any* slab label, so all slab variants of matching genotypes are selected. This is the safe default — you don't want a hook to miss individuals just because they carry a different slab.

- **`initial_state()`** uses `resolve_default_ztype_index()` which returns the *first* matching ZType. When no `@slab` is specified, this means only the `@default` slab variant is populated. This is intentional: initial state is a precise specification of where individuals start, and the `@default` slab is the "unmarked" state. To place individuals in a non-default slab, explicitly use the `@slab` suffix or tuple syntax:

```python
# Places individuals in @default slab (first matching ZType)
.initial_state(individual_count={"male": {"Drive|WT": 500}})

# Places individuals explicitly in @infected slab
.initial_state(individual_count={"male": {"Drive|WT@infected": 500}})

# Tuple syntax also works
.initial_state(individual_count={"male": {("Drive|WT", "infected"): 500}})
```

#### Important: `initial_state` Keys Must Be Exact

Keys passed to `initial_state()` must be exact genotype strings — fuzzy patterns like `"*|*"` or `"Drive|*"` will not behave as expected. Such patterns may silently only match the *first* ZType in registration order, which is almost certainly wrong. If you need pattern-style matching for initial state, use a `first`-event hook with `Op.set_count()` instead.

## Index Compression (Reachability BFS)

### Motivation

The full combinatorial space `(Genotypes × slabs) × (Haplotypes × glabs)` can be large. Most genotypes and haplotypes are never reachable from the initial conditions — they have zero individuals and no genetic modifiers produce them. Index compression prunes these unreachable entries, reducing array sizes and computation.

### BFS Algorithm

Compression uses a fixed-point BFS (implemented in `build_compression_mask` in `natal.genetics.structures._helpers`). The algorithm is symmetric for the GType and ZType layers:

```
1. Seeds: collect reachable genotypes
   a. initial_individual_count > 0  (genotypes that start with individuals)
   b. declared genotypes             (seeds from .declare(), see below)

2. From reachable genotypes, derive reachable haplotypes:
   for each reachable genotype g:
       reachable_haplotypes += gametes_produced_by(g)

3. Fixed-point iteration:
   for each pair (hl1, hl2) of reachable haplotypes:
       for each genotype g that (hl1, hl2) can form:
           if g is new:
               reachable_genotypes += g
               reachable_haplotypes += gametes_produced_by(g)
               → continue iteration

4. When no new genotypes/haplotypes are discovered → fixed point reached.

5. Build compression masks:
   - GType mask: -1 for pruned (haplotype, glab) pairs, ≥0 for survivors
   - ZType mask: -1 for pruned (genotype, slab) pairs, ≥0 for survivors
```

The key insight: once the reachable set stabilises, the compression mask maps old indices to new compressed indices. Pruned entries are permanently removed from the registry via `registry.compress(mask)`.

### Declare Semantics

The `declare` mechanism (`setup(compress=True, declared_zygote_types={"AA"})` or the deprecated `compress_genotypes(True).declare("AA")`) adds **seeds** to the BFS, not just final entries in the genotype list.

Example: If the initial state only has `aa` individuals, and a hook releases `AA` individuals at tick 100:

1. Without declare: the BFS starts with only `aa`. Reachable haplotypes are `{a}`. The fixed point is reached immediately — `A` is never discovered. When the hook tries to release `AA` at runtime, its ZType index is -1 (pruned), causing an error.

2. With `declare("AA")`: `AA` is a seed. Reachable haplotypes become `{a, A}`. The BFS combines `A` + `a` → `Aa` is discovered, which produces `{A, a}` again. Fixed point: `{AA, Aa, aa}` are all reachable, and compression preserves all three.

```
Initial seeds: {aa}
  + declare("AA")
  → reachable_genotypes = {aa, AA}
  → reachable_haplotypes = {a, A}
  → combine A + a → Aa discovered (new!)
  → reachable_genotypes = {aa, AA, Aa}
  → combine A + A → AA (already known)
  → combine a + a → aa (already known)
  → fixed point
  → all 3 genotypes survive compression
```

Without the `declare`, `A` would never enter the reachable haplotype set, and `AA` (and `Aa`) would be pruned.

#### Key Points

- `declared_zygote_types` is set on `.setup(compress=True, declared_zygote_types=...)`.
- The BFS is **symmetrical** — declaring a genotype also brings in all haplotypes it produces, which may combine to form additional genotypes not explicitly declared.
- Declared genotypes are expanded to **all slab variants** for the BFS (internally, they are treated as reachable ZTypes across all slabs).
- The deprecated `.compress_genotypes(True).declare("AA")` chain method still works but the preferred API is `.setup(compress=True, declared_zygote_types={"AA"})`.

## User Interface Notes

**Important**: `IndexRegistry` is a low-level data table; users typically do not need to call its methods directly. When accessing genotypes, haploid genotypes, or gamete labels, users should use the following high-level interfaces:

### Access via Index

```python
# Look up the genotype index via IndexRegistry, then access
idx = pop.index_registry.genotype_to_index["A1|A2"]
pop.state.individual_count[0, 3, idx]
```

### Pattern Matching with GenotypeSelector

```python
# Use GenotypeSelector for pattern-based operations
from natal.patterns import GenotypeSelector
selector = GenotypeSelector("A1|*", pop.index_registry)
indices = selector.select()  # Returns matching integer index array
```

**Note**: The old import path `from natal.genetic_patterns import GenotypeSelector` has been updated to `from natal.patterns import GenotypeSelector`.

## Internal Framework Usage

Within the NATAL framework, `IndexRegistry` is used for:

### 1. State Data Storage

- The individual count matrix uses ZType indices for efficient storage (the last axis is `n_ztypes`).
- The sperm storage matrix is managed using ZType indices.
- Engine arrays for gamete dynamics use GType indices.
- All state data is accessed via indices.

### 2. Modifier System

- Object dictionaries returned by Modifiers are converted to indices via the registry.
- The framework handles the object-to-index conversion process.
- Users only need to work with high-level objects.

### 3. Hook System

- Numba Hooks use precomputed indices for efficient operation.
- Avoids accessing dynamic registries at compile time.
- Avoids hardcoded indices through the selector pattern.

### 4. Index Compression

- `rebuild_config_maps()` (in `natal.configurator._registry_builder`) runs the BFS.
- The resulting masks are applied via `registry.compress(ztype_mask, gtype_mask)`.
- After compression, all registry properties reflect only the surviving entries.

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
ZType entries (one per slab)
```

---

## Related Sections

- [Genetic Structures and Entities](2_genetics.md) — Genotype and HaploidGenotype creation
- [PopulationState & PopulationConfig](4_population_state_config.md) — Index application in configuration
- [Modifier Mechanism](3_modifiers.md) — IndexRegistry usage in Modifiers
- [Hook System](2_hooks.md) — Advanced Hook selector patterns

---

**Ready to dive into configuration compilation details?** [Continue to next chapter: PopulationState & PopulationConfig →](4_population_state_config.md)
