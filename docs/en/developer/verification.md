# How to verify numerical, state, and cross-language behaviour

[The previous chapter](development.md) covered turning a request into a change; this one covers proving the change is right: where expected values come from, what counts as evidence, and how "the implementation is correct" differs from "one run looked fine". Thresholds, coverage, and gate commands live in the repository's [quality_checks_spec.md](https://github.com/jyzhu-pointless/natal-core/blob/main/quality_checks_spec.md); this chapter is about method.

## Four sources of expected values

| Source | Where it applies | Example |
| --- | --- | --- |
| Hand calculation | stage counts, probabilities, small single-locus models | 100 heterozygotes × 2 eggs = 200 offspring, Mendelian split 50/100/50 |
| Invariants | conservation, symmetry, boundaries | the stacked total is unchanged by migration; `g(1) = 1`; every row sums to 1 |
| Analytic relations | a closed form between parameters and results | `s* = K / (egg production × mean age-0 survival)` |
| Existing tests | pre-existing contracts | projection, contract mapping, restore paths |

Every number in this guide states which source it came from; when a source is unclear, fall back to an invariant or a hand calculation instead of "run it and see".

## What does not count as evidence

| Observation | Why it is not enough |
| --- | --- |
| The program ran without errors | Equal shapes with different meanings also run |
| The totals are right | Order, grouping, and index errors all preserve totals |
| One stochastic run is close to the expectation | One realisation is not a distribution; compare distributions or switch to deterministic mode |
| Deterministic and stochastic results look "about the same" | They are different declarations and need their own criteria |
| All tests are green | They cover only what they assert |

## Five behaviours that need dedicated checks

1. **Numbers and stages**: check the intermediate boundaries (`early`, `late`) and the final state separately; equal totals do not prove the stages are right.
2. **State and ownership**: writing a snapshot does nothing, the session holds the authority, shared arrays are not writable — verify these by testing whether a change takes effect, not by reading totals.
3. **Atomic failure**: an illegal candidate produces no partial write, and a failing callback keeps earlier commits (see [callback transactions](transactions.md)).
4. **Index consistency**: compare compression results **by type identity**, never by index position.
5. **Cross-language bit equality**: for one input, the Python and native sides must agree bit for bit; changes touching accumulation order (the offspring tensor, for example) must preserve it (see [numerical algorithms](algorithms.md)).

## Writing statistical checks

When the result is itself random, state the criterion as a statistical claim with a tolerance:

| Check | Formulation |
| --- | --- |
| Expectation | the mean over many seeds falls near the expectation, with a stated tolerance and its justification (for example: 60 seeds give a mean adult total of 102.0 against 100.0, tolerance 5%) |
| Reproducibility | the same seed matches bit for bit; a different seed does not |
| Sampling type | discrete mode yields integral counts; continuous mode keeps fractional mass |
| Degenerate branches | a zero total or a zero-sum probability vector skips sampling and never divides by zero |

There is no need to pile up repetitions for the look of rigour; justifying the tolerance matters more than enlarging the sample.

## Reading the evidence an agent reports

Check each item:

1. **Does the expected value have a source**: hand calculation, an invariant, or "same as last time"?
2. **Is the assertion about observable behaviour**: a private field being reshuffled, or something a user can see?
3. **Is there a distinguishing scenario**: if a correct and a wrong implementation behave identically under these inputs, the inputs are not evidence.
4. **Is the unverified scope stated**: which model combinations, paths, or modes were not covered.
5. **Are the commands and their outputs shown**: rather than "this should pass".

A bare "verified" is not a delivery; what is needed is a reproducible command, the numbers, and what failure would look like.

## How this guide verifies itself

The numbers in this guide come from per-batch verification scripts (full initialisation, native runs, assertions), and each batch records:

- the script path and its actual result;
- the result of both strict documentation builds and their warning count;
- the documentation checks (heading levels, code-block and diagram counts, links, table cell counts, diagrams in the built pages);
- an explicit split between self-testing and independent review.

Scripts and temporary tooling can be cleaned up by the system, so conclusions rest on commands that can be repeated from the repository together with the existing tests; this guide never treats a one-off script's success as the project's full gate passing.

## Change type versus minimum verification

| Change type | Minimum |
| --- | --- |
| Documentation | accuracy, bilingual sync, links, and a build |
| Local code change | targeted tests, then the full gate before delivery |
| High-risk code (formulas, random distributions, state restoration, shared mutable data, public API, cross-language exchange) | independent review, an independently run full gate, and test reinforcement where needed |

Classification and collaboration follow the repository `AGENTS.md`; this chapter sets no separate gate.

Return to the [reading route](index.md) for another entry point, or read [from a development need to an acceptable change](development.md) to see how these practices land on a concrete request.
