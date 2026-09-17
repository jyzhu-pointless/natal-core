"""Verify the discrete model's inline equilibrium computation: population response after a hook changes K.

Demonstrates the slice-4 in-hook parameter write: a single-parameter hook
(``def hook(pop) -> int``) changes K directly through ``pop.params``.  The
write takes effect immediately (the next run uses the new value) and is
recorded in the ``pop.params_log`` snapshot.
"""

import natal as nt
from natal.frontend.hooks.tick_context import TickContext

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT"]}})

# Change K inside a hook: one-shot halving once tick >= 7 (a closure flag
# guarantees it fires only once).
# The hook is declared in the build chain: its plan is compiled and injected
# once at build() time; nothing is registered afterwards.
_fired = {"done": False}


@nt.hook(event="first")
def halve_carrying_capacity(pop: TickContext) -> int:
    if not _fired["done"] and pop.tick >= 7:
        pop.params.carrying_capacity = float(pop.params.carrying_capacity) * 0.5  # type: ignore[arg-type]  # ParamsView read is statically object; the route resolves a float scalar
        _fired["done"] = True
    return 0


pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=sp, name="demo", stochastic=False)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                 juvenile_growth_mode="beverton_holt")
    .hooks(halve_carrying_capacity)
    .build()
)

# Run 5 ticks to reach equilibrium.
pop.run(5)

# Run 10 more ticks to observe the response: the hook fires exactly once
# (at tick 7, guarded by the closure flag) and K stays halved afterwards.
print(f"{'tick':>4}  {'total':>10}  {'K':>10}")
for i in range(10):
    pop.run(1)
    total = pop.state.individual_count.sum()
    K = pop.params.carrying_capacity
    print(f"{5+i+1:>4}  {total:>10.0f}  {K:>10.0f}")

print("\nParameter snapshots (written inside the hook, (tick, parameter, old value -> new value)):")
for row in pop.params_log:
    print(f"  tick={row[0]}  {row[1]}  {row[2]:.0f} → {row[3]:.0f}")
