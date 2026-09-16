# How density regulation and equilibrium are computed

The first action of the [survival](survival.md) stage is density regulation: it decides how much the juvenile cohort is scaled up or down before being filtered per type. This chapter explains that curve's input and output, how the carrying capacity relates to the equilibrium quantities, and why the order matters.

## One variable, one curve

Every mode shares one signature:

- `x` = current juvenile competition strength ÷ equilibrium competition strength;
- `g(x)` = the scaling factor applied on top of the equilibrium survival rate.

The built-in curves (`r` is the low-density growth rate):

| Mode | id | `g(x)` | `g(0)` | `g(2)` at r = 2 |
| --- | --- | --- | --- | --- |
| `no_competition` | 0 | 1 | 1 | 1 |
| `fixed` | 1 | `min(1, 1/x)` | 1 | 0.5 |
| `linear` (alias `logistic`) | 2 | `max(0, r - (r - 1)x)` | `r` | 0 |
| `beverton_holt` | 3 | `r / (1 + (r - 1)x)` | `r` | 2/3 |
| `ricker` | 4 | `r^(1 - x)` | `r` | 0.5 |

Three shared properties are checked explicitly in the implementation: `g(1) == 1` exactly, non-increasing in `x`, and finite, non-negative and bounded. The values at `r = 2` were verified: `fixed` gives 0.5 at `x = 2` while `beverton_holt` gives 2/3; `linear` can reach zero under high competition (the whole cohort disappears), whereas `fixed` only culls proportionally.

Note that `fixed` is a **ceiling**, not compensation: below equilibrium it does not magnify (`g = 1`), it only removes the excess above the carrying capacity proportionally. The compensatory curves (`linear`, `beverton_holt`, `ricker`) magnify to `r` at low density.

## Where the equilibrium quantities come from

The equilibrium competition strength `C*` and equilibrium survival rate `s*` are derived on demand from the current parameters and are never stored as derived state:

```text
distribution: age-1 total = K; females = K x sex_ratio x s_f / (sex_ratio x s_f + (1 - sex_ratio) x s_m)
              (the surviving sex ratio -- equal age-0 survival reduces it to sex_ratio);
              older ages decay by the previous age's survival
production:   produced = sum over breeding ages of (females x reproduction rate x fertility x eggs_per_female)
C*  = produced x juvenile competition weight + sum over juvenile ages of (counts x weight)
s_0 = sex ratio x female age-0 survival + (1 - sex ratio) x male age-0 survival
s*  = K / (produced x s_0)
```

Verified: with K = 100, two eggs per female, sex ratio 0.5 and age-0 survival 0.5 for both sexes, the derived equilibrium distribution is 50 females and 50 males, production is 100, so `C* = 100` and `s* = 100 / (100 × 0.5) = 2`. Raising the clutch to four eggs makes production and `C*` 200 while `s*` becomes 1.

Two easy misreadings:

- `s*` may exceed 1. It is the curve's reference value, not a probability, and it is never sampled.
- A declared `equilibrium_individual_distribution` is used as-is instead of being derived from K; `external_expected_eggs` replaces only the production term inside `s*` and never affects `C*`.

## Order: density regulation before survival

```mermaid
flowchart LR
    J["age-0 juveniles 200"] --> D["density regulation: x = 200 / C*, multiply by g(x)"]
    D --> S["per (sex, ZType): base survival x viability"]
    S --> O["written back to age-0"]
```

The verified comparison (unit age-0 survival, so only density acts):

| Configuration | `late` juvenile total |
| --- | --- |
| `no_competition` | 200 |
| `fixed`, K = 100 | 100 |
| `beverton_holt`, x = 2 | 133.3 = 200 × 2/3 |
| `beverton_holt`, x = 1 | 100 (`g(1) = 1`) |
| `beverton_holt`, x = 0.5 | 66.7 = 50 × 4/3 |

The same parameters under a fixed K give 200 → 100 → 50 (scale first, then multiply by the 0.5 survival rate); "survive first, then scale" would give 200 × 0.5 = 100 → 100. Those two happen to produce the same number here but mean different things, and changing K or the survival rate separates them immediately. That is what makes the order a distinguishable behaviour: a single result is not enough to identify it.

## Zero equilibrium, thresholds, and degenerate branches

| Case | Behaviour |
| --- | --- |
| `x ≤ 0` (no juveniles) | `fixed` returns 1 directly and never divides |
| `linear` driving `g ≤ 0` under high competition | clamped to 0: the whole cohort is cleared, a legitimate model outcome |
| production or `s_0` near zero | there is no scale to solve for, so `s*` degenerates to 1 instead of dividing by zero |
| an unknown `growth_mode` | refused at declaration; there is no "unknown mode behaves like no competition" fallback |

## What a change affects

| Agent proposal | Test to apply |
| --- | --- |
| Use `fixed` as a hard ceiling | It is a ceiling, but it does not magnify below equilibrium; use a compensatory curve for low-density growth |
| Raise `r` in `beverton_holt` | `r` changes both `g(0)` and the curve shape; the fixed point `g(1) = 1` is unaffected |
| Move density regulation after survival | The order changes observable results; compare `fixed` plus K to separate them |
| Use `s*` as a survival probability | It can exceed 1; it is a reference scale |
| Let an unknown mode fall back to no competition | Unknown modes are refused at declaration; do not rely on a silent fallback |

## Implementation and verification entry points

| Entry point | Responsibility in this chapter |
| --- | --- |
| [kernels/density_regulation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/density_regulation.rs) | The curves, `regulation_scaling()`, and the curve-property checks |
| [kernels/equilibrium.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/equilibrium.rs) | `equilibrium_metrics()`: deriving `C*` and `s*` |
| [kernels/discrete_generation.rs](https://github.com/jyzhu-pointless/natal-core/blob/main/rust/src/kernels/discrete_generation.rs): `scaling_factor()`, `recruit_juveniles()` | Competition strength, scaling, resampling |
| [model/ecology.py](https://github.com/jyzhu-pointless/natal-core/blob/main/src/natal/frontend/model/ecology.py) | Declaration forms for the equilibrium distribution and external expected eggs |

The five curve values, both `C*`/`s*` derivations, the order comparison, and the degenerate branches were all verified from one set of inputs. Among the existing tests, [test_density_zero_equilibrium.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_density_zero_equilibrium.py) and [test_default_growth_mode.py](https://github.com/jyzhu-pointless/natal-core/blob/main/tests/test_default_growth_mode.py) protect curve properties and the default mode.

Next, read [Random sampling and reproducibility](randomness.md).
