# Numerical algorithms

For the full reproduction derivation, see [How one reproduction stage is calculated](reproduction.md), including partner weights, pair counts, inheritance loss, and stage-specific sex and fitness effects.

This chapter connects numerical meaning to concrete loops. The formulas describe local computations, not an entire lifecycle. Fitness, mating rates, egg production, survival, and density regulation act at different points and cannot be combined into one multiplier without specifying order.

## Deriving offspring probabilities from inheritance maps

In [offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs), `compute_offspring_tensor_flat()` combines meiosis table M and fusion table F into P. Let i, j, k identify female parent, male parent, and offspring ZTypes, and u, v identify female and male gamete GTypes:

\[
P_{ijk}=\sum_{u=0}^{G-1}\sum_{v=0}^{G-1}M_{0iu}M_{1jv}F_{uvk}.
\]

M has shape `(2, Z, G)`, F has `(G, G, Z)`, and P has `(Z, Z, Z)`. Rust uses flat row-major arrays: P's offset is `(i * Z + j) * Z + k`; the male meiosis offset is `(Z + j) * G + v`.

The loop order is i, j, k, u, v, skipping zero entries of M. The source deliberately preserves term-by-term accumulation to prevent reassociation or fused multiply-add from changing bit-level results. Check numerical equivalence requirements before substituting a mathematically equivalent tensor contraction.

For a hand calculation, let both parents produce A or a gametes with probability 1/2, and let fusion map AA, Aa/aA, and aa to three offspring types. The parental P slice is `(1/4, 1/2, 1/4)`. This is a neutral inheritance example, before offspring counts or zygote fitness are applied.

## Mating probabilities and counts are separate

In [discrete_generation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs), `compute_mating_probability()` constructs weights and normalizes each female-type row. With sexual-selection fitness S and the male count vector m supplied to the function:

\[
w_{ij}=S_{ij}m_j,\qquad q_{ij}=w_{ij}/\sum_j w_{ij}.
\]

Division occurs only when the row total is finite and greater than `EPS`; otherwise the row becomes zero. An unavailable male pool therefore produces neither NaNs nor uniform mating. Weights 20 and 10 give partner probabilities 2/3 and 1/3, but do not mean every female mates.

`mate_discrete()` then computes or samples pair counts using female counts and mating rates. `fertilize_discrete()` handles egg production, parental fecundity, and offspring allocation. The age-structured `compute_mating_probability_matrix()`, `sample_mating()`, and `fertilize()` additionally handle age weights and sperm storage; a discrete pair buffer cannot replace that state.

## From pair counts to the next generation

In the deterministic discrete path, let pair count be Cᵢⱼ, reproduction probability b, eggs per female e, and parental fecundities fᵢ and fⱼ. Expected eggs for this cross are Cᵢⱼ·b·e·fᵢ·fⱼ. `fertilize_discrete()` multiplies this by Pᵢⱼₖ and accumulates offspring types.

A parental slice of P can sum to less than 1, representing mass lost through inheritance fusion paths. Unconditional normalization would erase that loss. The stochastic path first thins egg counts by the slice sum and then allocates viable offspring with conditionally normalized type probabilities. Sex assignment follows sex-chromosome compatibility or the global sex ratio; males receive the total minus the female count.

`reproduction()` then applies `zygote_viability_fitness`. `survival()` first calls `recruit_juveniles()` for density regulation, then retains individuals using age-0 baseline survival times the corresponding sex/age/type viability. Finally, `aging()` makes them adults. Zygote fitness and ordinary viability therefore act at different positions.

For a calculation spanning these stages, take 100 already mated neutral A|a parental pairs, b=1, e=2, both parental fecundities equal to 1, sex ratio 1/2, no density regulation, zygote fitness 1, age-0 survival 1/2 for both sexes, and ordinary viability 1. Inheritance yields `(50, 100, 50)` AA, Aa, aa offspring in total. Sex assignment gives `(25, 50, 25)` per sex; survival leaves `(12.5, 25, 12.5)` per sex, which aging moves into age 1. Fractions are deterministic expected counts, not promises about integer random outcomes.

## Density curves differ from applied scaling

[density_regulation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/density_regulation.rs) separates curves from kernel inputs. Let x be actual competition strength divided by equilibrium competition strength, and r the low-density growth rate:

| Mode | Curve g(x) |
| --- | --- |
| 0: no regulation | 1 |
| 1: fixed | min(1, 1/x), or 1 for x ≤ 0 |
| 2: linear / logistic | max(0, r − (r − 1)x) |
| 3: Beverton–Holt | r / (1 + (r − 1)x) |
| 4: Ricker | r^(1 − x) |

`scaling_factor()` dispatches curves; lifecycle kernels use `regulation_scaling()`. For modes 2–4, the latter also multiplies by equilibrium survival and returns zero for nonpositive or NaN equilibrium competition strength. Fixed mode computes `equilibrium / actual` directly, avoiding the extra rounding of `1 / (actual / equilibrium)`.

At r=2 and x=2, linear, Beverton–Holt, and Ricker curve values are 0, 2/3, and 1/2. Final recruitment scaling also depends on equilibrium survival. Age-structured competition involves age weights, so x cannot universally mean total population divided by K. Modes 5 and above currently have no custom-curve registry.

## Random paths require boundary checks

[rng.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/rng.rs) centralizes `binomial()`, `poisson()`, `multinomial()`, and continuous counterparts. Deterministic paths use expected counts, discrete random paths sample, and continuous sampling follows different numerical rules. Continuous sampling does not make the entire simulation automatically differentiable.

For example, `continuous_binomial()` returns boundary values for probabilities near zero or one and n·p for n ≤ 1 + EPS. Otherwise it uses two Gamma draws in a fixed order to form a Beta proportion and multiplies by n. Reordering draws changes the downstream random stream even if the marginal distribution remains the same.

Age-structured `sample_survival_with_sperm()` maintains female counts and sperm storage together. Tests should check more than total population: stored mated females must not exceed female counts, and their association must remain correct after aging.

## Verification entry points

- [Offspring unit tests](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/tests/unit/kernels/offspring.rs) and [test_offspring_alignment.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_alignment.py): probability derivation and axis alignment.
- [Density-curve unit tests](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/tests/unit/kernels/density_regulation.rs) and [test_density_zero_equilibrium.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_density_zero_equilibrium.py): curves and zero-equilibrium behavior.
- [test_mgdrive1_compatible_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_mgdrive1_compatible_lifecycle.py): reference lifecycle scenarios.

Algorithm changes need justified expected values, constraints such as nonnegativity or conservation, statistical properties, and required reproducibility. Comparing two implementations of the same formula can preserve the same mistake in both.
