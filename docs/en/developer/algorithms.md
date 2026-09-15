# Numerical algorithms (navigation)

This chapter used to collect the numerical loops in one place. That material now lives in the topic chapters, so this page is a navigation entry: it points to where each algorithm landed, restates the numerical constraints that do not belong to a single topic, and keeps the code and test links.

## Where each algorithm went

| Topic | Entry point now | Question answered |
| --- | --- | --- |
| The full reproduction derivation (M, F, P, pairing, eggs, probability loss) | [How one reproduction stage is calculated](reproduction.md) | How the 200 offspring are computed step by step |
| Survival, density ordering, generation replacement | [How survival and generation replacement are calculated](survival.md) | How juveniles are filtered and when old adults disappear |
| Age weights, long-term sperm storage, age advancement | [Age structure and long-term sperm storage](age_structure.md) | How overlapping generations and stored sperm breed |
| Density curves, carrying capacity, equilibrium quantities | [How density regulation and equilibrium are computed](density_regulation.md) | What `x`, `g(x)`, `C*`, and `s*` each are |
| Sampling points, random streams, reproducibility scope | [Random sampling and reproducibility](randomness.md) | Which boundaries skip sampling, and what a seed promises |
| The fused Wright-Fisher step | [The fused Wright-Fisher execution path](wright_fisher.md) | Which stages are merged and which hooks stop running |
| How genetic maps are generated from a declaration | [How genetic presets and conversion rules compile](genetic_compilation.md) | Baseline and rule application order |
| The coordinate change at publication | [Reachability, index compression, and publication](publication.md) | How arrays are rebuilt on the final axes |

## Numerical constraints still worth remembering

These constraints do not fit one topic, but any algorithm change runs into them:

- **The offspring-tensor accumulation order is fixed.** `compute_offspring_tensor_flat()` in [offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs) nests `(gf, gm, go, hf, hm)` to compute `P[gf, gm, go] = Σ meiosis_f[gf, hf] · meiosis_m[gm, hm] · fusion[hf, hm, go]`, skips zero terms, and preserves the term-by-term accumulation order. The source comment forbids reassociation, vectorisation, or fused multiply-adds: the Python and native sides must agree bit for bit.
- **Probability loss is not renormalised.** A parental slice of P may sum to less than 1 as an expressed loss; only the stochastic path first draws the surviving egg count from that sum and then allocates types with the conditional probabilities.
- **Zero-weight rows never divide.** A mating-probability row is normalised only when its sum is finite and above the threshold; otherwise the whole row is zeroed, so "no available males" never becomes NaN or uniform mating.
- **Density regulation precedes survival.** Changing the order changes the result for the same parameters; see [how density regulation and equilibrium are computed](density_regulation.md).
- **Discrete sampling rounds first.** Discrete stochastic mode rounds counts to integers before drawing, while continuous sampling keeps fractional mass. The two are not comparable draw by draw.

## Hand-checkable examples

Each has full context in its topic chapter; these are the results that can be done in your head:

| Example | Result |
| --- | --- |
| Both parents produce A or a gametes with probability 1/2 | one parental P slice is `(1/4, 1/2, 1/4)` |
| Two males with weights 20 and 10 | mating-object probabilities 2/3 and 1/3, not "every female mates" |
| 100 A\|a adults, two eggs each, juvenile survival 0.5 | `early` per sex `[100, 100]`, `late` per sex `[50, 100]` |

## Code and test entry points

- [offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs): offspring-tensor derivation.
- [discrete_generation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs): discrete stages and the fused step.
- [age_structured.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/age_structured.rs): age-structured stages and sperm storage.
- [density_regulation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/density_regulation.rs) and [equilibrium.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/equilibrium.rs): curves and equilibrium quantities.
- [rng.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/rng.rs): sampling distributions and the random stream.
- [test_offspring_alignment.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_alignment.py), [test_offspring_single_derivation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_single_derivation.py), [test_rust_discrete_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_rust_discrete_lifecycle.py): numerical and lifecycle contracts.

Start the numerical main line at [how one reproduction stage is calculated](reproduction.md), or rebuild the overall picture from [architecture and responsibility boundaries](architecture.md).
