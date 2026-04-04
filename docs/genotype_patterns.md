# Genotype Pattern Matching

This chapter solves a key problem:

**How to describe and filter genotypes in an efficient, readable, and maintainable way?**

We use string‑based genotype pattern matching to solve this problem. Genotype pattern matching has a simple regular‑expression‑like syntax, used both for defining groups in observations (aggregation) and for `genotype_filter` in preset rules.

## 1. Why Pattern Matching Is Needed

When the model becomes complex enough, hard‑coding genotype lists quickly gets out of hand:

- Multiple chromosomes, multiple loci.
- A large number of allele combinations.
- The need to define rules or observation groups in batches for “certain classes of genotypes”.

Pattern matching upgrades “explicit enumeration of genotype lists” to “semantic expressions”, with the benefits:

1. Fewer rules.
2. Higher readability.
3. Easier binding to experimental configurations (YAML/JSON).

## 2. Syntax Overview (Read This Section First)

A pattern string is parsed in three layers from outside to inside:

1. **Chromosome layer**: separate chromosome segments with `;`.
2. **Homologous chromosome layer**: each segment must contain `|` or `::`.
3. **Locus layer**: within each copy of a homologous chromosome, separate locus patterns with `/`.

Keep the skeleton in mind:

`<chr1_hap1>/<...>|<chr1_hap2>/<...>; <chr2_hap1>/<...>|<chr2_hap2>/<...>`

### 2.1 Delimiter Meanings

| Syntax element | Meaning | Example |
|---|---|---|
| `;` | Separates different chromosome segments | `A/B|C/D; E/F|G/H` |
| `|` | Ordered match: `Maternal|Paternal` | `A/B|C/D` |
| `::` | Unordered match: the two copies of a homologous chromosome can be swapped | `A/B::C/D` |
| `/` | Separates locus patterns within one copy of a homologous chromosome | `A/B/C` |

### 2.2 Locus Atomic Patterns

| Pattern | Meaning | Example |
|---|---|---|
| `X` | Exact match of allele `X` | `A1` |
| `*` | Wildcard (any allele) | `*` |
| `{A,B,C}` | One of the enumerated set | `{A1,A2}` |
| `!X` | Not `X` | `!A1` |

### 2.3 Combined Examples

1. Exact match: `A1/B1|A2/B2; C1/D1|C2/D2`
2. Mixed wildcards: `A1/*|A2/B2; */D1|C2/*`
3. Set match: `{A1,A2}/B1|A3/B2; C1/D1|C2/D2`
4. Negation match: `!A1/B1|A2/B2; C1/D1|C2/D2`

### 2.4 Common Errors and Fixes

1. **Error**: `Chromosome pattern must contain '|' or '::'`
   - **Reason**: A chromosome segment does not specify the two copies of the homologous chromosome.
   - **Fix**: Do not write `C1/C1`; use the full `...|...` or `...::...` form.

2. **Error**: Chromosome segment count mismatch.
   - **Reason**: The number of `;`‑separated segments differs from the number of chromosomes in the species.
   - **Fix**: Add segments to match the species definition.

3. **Error**: Locus count mismatch.
   - **Reason**: The number of `/`‑separated locus patterns within a copy differs from the number of loci on that chromosome.
   - **Fix**: Fill in each locus, or use `*` as a placeholder.

## 3. Integration with Observation

The `groups["genotype"]` in the Observation chapter already directly uses `GeneticPattern` parsing. It is recommended to pass pattern strings directly rather than writing long lists by hand.

Recommended workflow:

1. Maintain pattern strings in the experiment configuration.
2. Put the pattern string directly into `groups["genotype"]`.
3. Let the Observation internals handle parsing and matching uniformly.

```python
groups = {
    "target_group": {
        # Ordered match: Maternal|Paternal
        "genotype": "A1/B1|A2/B2; C1/D1|C2/D2",
        "sex": "female",
    },
    "target_group_unordered": {
        # Unordered match: the two copies of a homologous chromosome can be swapped
        "genotype": "A1/B1::A2/B2; C1/D1::C2/D2",
        "sex": "female",
    }
}
```

If you need to debug the matched set, use `species.enumerate_genotypes_matching_pattern(...)` to expand offline.

This ensures that Observation and Preset reuse the same semantic source.

## 4. Integration with Preset

It is recommended to place “pattern parsing” inside the Preset, rather than scattering it in scripts.

A practical structure:

1. The Preset receives a pattern parameter.
2. During initialisation/binding, compile the pattern into a filter function.
3. Reuse it in `add_convert(..., genotype_filter=...)`.

## 5. Example: Parameterized Preset

```python
class PatternDrivenPreset(GeneticPreset):
    def __init__(self, target_pattern: str, conversion_rate: float):
        super().__init__(name="PatternDrivenPreset")
        self.target_pattern = target_pattern
        self.conversion_rate = conversion_rate

    def _build_filter(self, species):
        # Returns Callable[[Genotype], bool]
        return species.parse_genotype_pattern(self.target_pattern)

    def gamete_modifier(self, population):
        ruleset = GameteConversionRuleSet("pattern_rules")
        pattern_filter = self._build_filter(population.species)

        ruleset.add_convert(
            from_allele="W",
            to_allele="D",
            rate=self.conversion_rate,
            genotype_filter=pattern_filter,
        )
        return ruleset.to_gamete_modifier(population)
```

This structure upgrades a Preset from “fixed rules” to “configurable rule templates”.

---

## Looking Back and Beyond

- [Population Observation Rules](observation_rules.md)
- [Design Your Own Preset (1): Starting from Allele Conversion Rules](allele_conversion_rules.md)
- [Design Your Own Preset (2): Using `genotype_filter` to Control Rule Scope](genotype_filter.md)
- [Design Your Own Preset (3): Encapsulation, Validation, and Pre‑release Checks](preset_encapsulation_and_validation.md)
- [Genetic Presets Usage Guide](genetic_presets.md)
