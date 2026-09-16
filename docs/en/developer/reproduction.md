# How one reproduction stage is calculated

In the [previous chapter](model_journey.md), 100 A|a females and 100 A|a males produced 200 offspring with eggs per female set to 2. This is not population size multiplied by a growth rate. It follows partner selection, pairing, egg production, inheritance allocation, sex assignment, and zygote fitness.

This chapter unpacks that process. Formulas and tables explain numerical behavior rather than provide executable code. The main path is an **ordinary staged discrete-generation model**: deterministic, autosomal, and without additional label conversions. Random, age-structured, and fused Wright–Fisher paths are distinguished later.

## Separate rules, current counts, and temporary results

`reproduction()` reads age-1 adults and writes age-0 offspring. It neither makes offspring adults nor removes old adults; subsequent survival and aging do that.

Compressed individual types are AA, Aa, aa, giving Z=3; gametes are A, a, giving G=2. The prose omits separators and default labels; actual type names remain A|A@default and so on.

| Data | Shape or indexing | Lifetime |
| --- | --- | --- |
| Individual counts `ind` | `(sex, age, Z)`, here `(2, 2, 3)` | Session-owned, updated by stages |
| Meiosis M | `(sex, Z, G)` | Produced by construction or genetic recompilation |
| Gamete fusion F | `(G, G, Z)` | Produced by construction or genetic recompilation |
| Offspring P | `(female Z, male Z, offspring Z)` | Derived from M and F, reused in reproduction |
| Sexual-selection fitness S | `(female Z, male Z)` | Relative partner weights |
| Effective males | `(Z,)` | Temporary for this reproduction stage |
| Mating probabilities Q and pair counts C | Both `(Z, Z)` | Temporary for this reproduction stage |
| Offspring and sex-specific buffers | Each `(Z,)` | Written back to age 0 at stage completion |

P describes inheritance for a parental combination; C describes how much of that combination occurs now. Two runs can use the same P but produce different C and offspring counts because their parental abundances differ.

## Preparing inheritance: two gametes become one offspring

### M describes gametes produced by one parent

With Mendelian segregation and no conversions, both sexes use this table:

| Parent | A gamete probability | a gamete probability |
| --- | --- | --- |
| AA | 1 | 0 |
| Aa | 1/2 | 1/2 |
| aa | 0 | 1 |

M retains a sex axis even though its two slices match here. Sex-dependent mechanisms can give different female and male mappings.

An M row contains probabilities, not gamete counts. `(1/2, 1/2)` for an Aa parent neither means it produces only one gamete nor determines egg production.

### F describes fusion of two gametes

F's first two axes preserve female and male gamete origin; its last axis lists offspring types:

| Female gamete | Male gamete | Offspring `(AA, Aa, aa)` |
| --- | --- | --- |
| A | A | `(1, 0, 0)` |
| A | a | `(0, 1, 0)` |
| a | A | `(0, 1, 0)` |
| a | a | `(0, 0, 1)` |

A/a and a/A are separate origin combinations even though both produce unordered genotype Aa. Counting only one halves the Aa probability.

### P combines the gamete process in advance

Let i, j, k be female-parent, male-parent, and offspring ZType indices, and u, v female and male GType indices:

\[
P_{ijk}=\sum_{u=0}^{G-1}\sum_{v=0}^{G-1}M_{0iu}M_{1jv}F_{uvk}.
\]

Each term multiplies the probability of the female producing u, the male producing v, and that pair producing k. Summing all possible gamete combinations gives the parental cross's probability of offspring k.

For Aa × Aa:

| Gamete combination | Contribution | Offspring |
| --- | --- | --- |
| A × A | 1/2 × 1/2 = 1/4 | AA |
| A × a | 1/2 × 1/2 = 1/4 | Aa |
| a × A | 1/2 × 1/2 = 1/4 | Aa |
| a × a | 1/2 × 1/2 = 1/4 | aa |

Thus `P[1, 1, :]` is `(1/4, 1/2, 1/4)`. Other crosses have their own slices: AA × AA gives `(1, 0, 0)`, and AA × aa gives `(0, 1, 0)`. P stores these rules independently of whether current AA abundance is zero.

Python `recompute_offspring_tensor()` normalizes inputs to contiguous float64 arrays and invokes Rust `compute_offspring_tensor_flat()`. Rust stores P at row-major offset `(i × Z + j) × Z + k`, looping in i, j, k, u, v order and skipping zero terms. Values are simple here, but accumulation order affects floating-point results in general; the source explicitly preserves operation order for bit-level consistency.

The tradeoff is concrete: P uses Z³ storage, but reproduction no longer enumerates all gamete combinations for each parental cross. Compressing before deriving P reduces that derived storage.

## Step one: which males enter the partner pool?

`reproduction()` reads age-1 male counts and multiplies by the male adult mating rate. If nⱼᵐ is the count of male type j and rₘ the male adult mating rate:

\[
m_j=n_j^m r_m.
\]

Here raw male counts are `(0, 100, 0)` and rₘ=1, leaving the same effective vector.

For each female type i, `compute_mating_probability()` constructs and normalizes partner weights:

\[
w_{ij}=S_{ij}m_j,\qquad
Q_{ij}=\frac{w_{ij}}{\sum_j w_{ij}}.
\]

S rows are female types and columns are male types. With all S entries equal to 1, effective male composition determines selection; preferred combinations receive larger relative weights. Division occurs only for finite row sums exceeding EPS=10⁻¹⁰; otherwise that Q row is set to zero.

For example, consider 80 AA and 20 aa males, with S values 1 and 2 for a particular female type. Weights are 80 and 40, giving partner probabilities 2/3 and 1/3 rather than 80% and 20%. Sexual selection changes allocation among partners; the number of participating females is calculated next.

### Males are not consumed as one-use slots

Only Aa males exist in the main example, so their partner probability is 1. Reducing male count from 100 to 1 still assigns all participating females to Aa in deterministic execution, provided effective weights remain above numerical thresholds. Pair counts are not capped at `min(females, males)`, and pairing does not subtract from male counts.

Likewise, a common positive rₘ applied to every male type in a deme cancels under ordinary normalization. Changing rₘ from 1 to 0.5 does not necessarily halve offspring; setting it to zero empties the effective male pool. Extremely small values near numerical thresholds are outside this cancellation argument.

This is an important model-semantic question when discussing “mating rates” with an agent. Male scarcity that reduces mating opportunities, or a cap on matings per male, requires an explicit mechanism. Partner weights alone do not implement those limits.

## Step two: how many females mate, and with which male type?

Let nᶠᵢ be adult female count of type i and p the female adult mating rate clamped to [0,1]. Deterministic `mate_discrete()` computes:

\[
C_{ij}=n_i^f p Q_{ij}.
\]

A cell of C counts females of type i paired with male type j. Here nᶠ=(0,100,0) and p=1, so only `C[1,1]=100` is nonzero.

Reducing only female mating rate to 0.5 makes that cell 50. This differs from the male-rate effect: female rate directly controls participation, while male rate first contributes to partner weights.

Discrete C is a temporary per-step buffer, not a persistent sperm bank. The next step regenerates it from the new adults.

## Step three: pair counts become expected eggs

For each nonzero parental combination, `fertilize_discrete()` reads eggs per female e, reproductive participation b, female fecundity fᶠᵢ, and male fecundity fᵐⱼ. Fecundity here is a multiplier on egg contribution; it is not partner preference or survival probability.

Deterministic egg counts are:

\[
E_{ij}=C_{ij}\,b\,e\,f_i^f\,f_j^m.
\]

The example gives 100 × 1 × 2 × 1 × 1 = 200. Male fecundity of 0.5 would yield 100 expected eggs from this cross, reducing its reproductive contribution even if it is still chosen as a partner.

The kernel reads b from the adult reproduction-rate array. The discrete lifecycle here fixes it at 1; age-structured models additionally handle participation and fertility by age. Several rates appearing in one product does not make them interchangeable parameters.

## Step four: allocate offspring types without erasing inheritance loss

Accumulate every parental contribution to offspring type k:

\[
O_k=\sum_i\sum_j E_{ij}P_{ijk}.
\]

Only Aa × Aa contributes here, giving 200 × `(1/4, 1/2, 1/4)` = `(50, 100, 50)`.

Crucially, **a parental P slice need not sum to 1**. When inheritance maps contain loss, its sum h is the retained probability mass after those mappings. It does not include the separate `zygote_viability_fitness` applied later.

For an illustrative loss variant, multiply every probability in F by 1/2. The Aa × Aa P slice becomes `(1/8, 1/4, 1/8)`, with h=1/2. Two hundred expected eggs contribute only `(25, 50, 25)`, totaling 100. Renormalizing the slice to `(1/4, 1/2, 1/4)` to “make the probabilities valid” would incorrectly restore the lost half.

The deterministic path multiplies by P directly. The stochastic path first thins eggs by h and then allocates retained offspring using P/h. These steps separate retention from conditional type assignment. Only the second needs conditional normalization; stored P must not be overwritten for that purpose.

## Step five: divide type counts by sex

There are no sex chromosomes here, so `sex_ratio` is the female fraction q=0.5. Deterministic assignment gives females Oₖq and males the residual Oₖ minus females. Each sex receives `(25, 50, 25)`.

Sex-chromosome models take another path: female-only or male-only types are assigned directly, while other types use the female-to-total compatibility ratio. Insufficient total compatibility falls back to 0.5. The example's global ratio cannot be applied unconditionally to every species.

Even with global q=0.5, final survivors need not be equally divided: sex-specific zygote fitness and survival can act afterward.

## Step six: zygote fitness writes age 0

Only after sex assignment does `reproduction()` apply `zygote_viability_fitness[sex, type]`. Deterministic execution multiplies by it; stochastic execution uses corresponding binomial or continuous sampling.

Its value is 1 here, leaving `(25, 50, 25)` per sex at early. If all female zygote fitness values are instead 0.5, early females are `(12.5, 25, 12.5)` and males `(25, 50, 25)`: 50 females and 100 males. This changes survival after sex assignment, not the original assignment probability.

Reproduction is now complete. Old adults remain at age 1; survival next performs density regulation and ordinary viability, and aging replaces generations. Early counts include zygote fitness but not survival.

### Why survival multipliers cannot all be combined

Zygote fitness acts before density regulation; ordinary viability acts afterward. Without density feedback, some uniform multipliers can happen to produce the same final counts. That does not establish general equivalence when moving them between stages.

Start with the example's 200 offspring, switch to fixed density regulation with K=100, and use baseline juvenile survival 1:

| Scenario | After zygote fitness | After fixed regulation | After ordinary viability |
| --- | --- | --- | --- |
| Zygote fitness=0.5; ordinary viability=1 | 100 | 100 | 100 |
| Zygote fitness=1; ordinary viability=0.5 | 200 | 100 | 50 |

Both contain a 0.5 multiplier, yet differ twofold because regulation reads different juvenile counts. An agent proposing to combine fitness into one multiplier must establish the conditions under which that is equivalent.

## What changes when randomness is enabled?

The ordinary discrete integer-stochastic path preserves these stage relationships but obtains intermediate counts through conditional draws:

| Position | Sampling meaning |
| --- | --- |
| Female participation | Binomial draw with p from rounded female count |
| Partner type | Multinomial allocation using valid Q rows |
| Reproductive participation | Retain paired females with b; near-one branches can retain directly |
| Eggs | Poisson with mean participating pairs times eggs contributed per pair |
| Inheritance loss | Binomial retention with h; near-one values can skip the draw |
| Offspring type | Multinomial allocation with conditional P/h |
| Sex | Binomial female assignment; males receive the residual |
| Zygote fitness | Separate binomial retention for each sex |

Thus integer-stochastic execution does not enforce exactly 50 AA, 100 Aa, and 50 aa offspring. Total egg production itself fluctuates. A deterministic example checks formulas, not distributions, variances, or multistep mean trajectories. With density feedback especially, the mean of random trajectories must not be equated to a deterministic trajectory.

The discrete `fertilize_discrete()` examined here directly uses Poisson or its continuous counterpart on its stochastic egg path. It does not contain the `fixed_egg_count` branch present in age-structured `fertilize()`. A shared Blueprint field does not guarantee identical use by both implementations.

`continuous_sampling` is not an integer sample converted to float. Continuous binomial uses a Beta proportion but returns n·p directly for n≤1+EPS; continuous Poisson uses Gamma sampling; continuous multinomial normalizes Gamma draws. These are separate numerical rules. Fractional results alone establish neither deterministic equivalence nor automatic differentiability.

Draw order is also behavior. Multinomial sampling uses sequential conditional binomial draws with the final category receiving the residual. Empty contributions and boundary probabilities can consume fewer draws. Changing traversal order or making an additional random call can alter downstream fixed-seed trajectories even if the target distribution is unchanged.

## Boundaries that need separate attention

| Condition | Current handling | Unsupported inference |
| --- | --- | --- |
| Effective male total or adult female total is exactly zero | `reproduction()` returns early with state unchanged | Age 0 is not unconditionally cleared; externally introduced juveniles can remain |
| A Q row has invalid total weight or total ≤ EPS | Q row becomes zero; deterministic pair allocation for it is zero | Normalization alone does not establish every stochastic downstream boundary behavior |
| A pair cell is zero or negative | Fertilization skips it | Entry points do not thereby accept arbitrary negative counts; validation has separate duties |
| Inheritance sum h ≤ EPS | Stochastic fertilization skips the cross | Forced normalization must not generate offspring |
| Eggs or a type contribution ≤ EPS | Relevant branches skip the contribution | Real-number algebraic equivalence cannot be assumed unconditionally near thresholds |

The zero-parent check here deliberately includes retained age-0 counts to establish actual early-return semantics. Special stochastic inputs such as all-zero Q require following the sampler and caller together; this chapter does not promote deterministic observations into verified stochastic guarantees.

## What age structure adds

Age-structured `sample_mating()` updates sperm storage indexed by `(female age, female type, male type)`. A cell counts mated females carrying sperm of that male type, not sperm cells or independently surviving males.

`fertilize()` iterates reproductive ages and stored parental combinations. Egg contribution also includes that age's fertility, and participation uses its reproduction rate. Females can reproduce using stored sperm, so no new pairings this step does not automatically imply no offspring. Survival and aging must then preserve the relationship among female counts, sperm storage, and age.

Replacing persistent storage with a temporary `(Z,Z)` pairing table changes dependence on model history. Fused Wright–Fisher is another separate update path that does not execute this complete stage combination. Neither model should be accepted using only the discrete example in this chapter.

## Accepting an agent's reproduction change

First ask which quantity changes: partner weights Q, pair counts C, eggs E, inheritance P, sex fraction, or survival at a particular stage. “Adjust reproduction rate” does not specify an implementation position.

Then request evidence distinguishing easily confused alternatives:

| Acceptance question | Distinguishing scenario |
| --- | --- |
| Was one gamete-origin combination omitted? | Both middle paths of Aa × Aa must contribute to Aa |
| Were males incorrectly capped or consumed? | Compare deterministic results for 1 versus 100 males of a single type |
| Were the two mating rates confused? | Lower female rate alone, then a common positive male rate alone |
| Was inheritance loss erased? | A P slice summing to 1/2 must retain the corresponding loss |
| Was fitness moved incorrectly? | Compare the two fixed K=100 scenarios with a 0.5 multiplier |
| Was only final total checked? | Inspect type, sex, early/late stages, and final age slots |
| Is random equivalence claimed? | Specify whether evidence concerns distribution, mean, variance, or seed replay, and its scope |

This is a behavioral acceptance route, not a requirement to read Rust loops yourself. The agent should supply inputs, justified expectations, and actual outcomes rather than only report passing tests.

## Implementation and verification entry points

- [genetics/matrices.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/genetics/matrices.py): `recompute_offspring_tensor()` and cross-language array normalization.
- [kernels/offspring.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/offspring.rs): P loops, offsets, and accumulation order.
- [kernels/discrete_generation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs): `compute_mating_probability()`, `mate_discrete()`, `fertilize_discrete()`, and `reproduction()`; survival and aging follow in the same file.
- [kernels/rng.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/rng.rs): sampling, rounding, and probability boundaries; [kernels/age_structured.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/age_structured.rs): persistent sperm storage.
- [test_offspring_single_derivation.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_single_derivation.py) and [test_offspring_alignment.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_offspring_alignment.py): derivation and coordinate consistency.
- [test_rust_discrete_lifecycle.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_rust_discrete_lifecycle.py) and [Rust RNG unit tests](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/tests/unit/kernels/rng.rs): existing discrete and sampling checks, not exhaustive evidence for every random variant above.

Neutral inheritance, mating-rate comparisons, inheritance loss, sex-specific zygote fitness, losses before/after regulation, and zero-parent boundaries received targeted numerical checks. Their scope is the concrete examples and current implementation, not a new scientific model or recertification of every random distribution.
