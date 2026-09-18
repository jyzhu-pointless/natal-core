"""P14: fixed_egg_count on the discrete-generation stochastic path.

Executed failing regression target (QC 2026-09-15).

Contract (builder docstring + docs/zh/4_simulation_engine.md 3.3 +
docs/zh/3_runtime_modification.md "both"): ``fixed_egg_count=True``
disables Poisson noise on egg counts, for both model types.  Observed:
``rust/src/kernels/age_structured.rs:336`` is the only reader of
``Blueprint::fixed_egg_count``; the discrete kernel
(``kernels/discrete_generation.rs::fertilize_discrete``) always draws
``poisson(rng, total_lambda)`` in stochastic mode, so the flag is a
silent no-op for discrete populations: one-egg females produce Poisson(1)
clutch totals (e.g. 891 and 863 offspring instead of exactly 900).
"""

from __future__ import annotations

import numpy as np

from _helpers import qc_species_2, neutral_population


def test_fixed_egg_count_disables_poisson_noise_on_discrete_path() -> None:
    totals = []
    for seed in (1, 2, 3, 4, 5, 6, 7, 8):
        pop = neutral_population(
            qc_species_2(f"QC0915_p14_{seed}"),
            f"QC0915_p14_{seed}",
            female={"W|W": 900},
            male={"W|W": 300},
            eggs_per_female=1.0,
            sex_ratio=0.5,
            survival=1.0,
            stochastic=True,
            extra_steps=lambda b: b.reproduction(fixed_egg_count=True),
        )
        pop._initialize_session(seed=seed)
        pop._rust_backend_seed = seed
        pop.run(1)
        totals.append(float(np.asarray(pop.state.individual_count[:, 1, :]).sum()))
    # With the flag honored, every female produces exactly one egg and the
    # next generation total is exactly 900 in every replicate.
    assert totals == [900.0] * len(totals), (
        f"fixed_egg_count=True still shows Poisson noise: {totals}"
    )
