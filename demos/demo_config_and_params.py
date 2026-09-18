"""Demonstrate every parameter-configuration route: build time, runtime, and inside hooks.

Scenario: the second population (``pop2``) runs ticks 1-10 while hooks
modify the environment:
  - tick 5:  ``hook_degrade`` fires once — carrying capacity K is halved
             (10000 -> 5000) and eggs_per_female drops from 50 to 35.
  - tick 8:  ``hook_recover`` fires — K restored to 10000,
             eggs_per_female back to 50, sex_ratio back to 0.5.
  - the loop ends at tick 10.  Neither hook touches the custom
             ``temperature`` field.

Configuration routes covered:
  1. Build time — PopulationBuilder chained API -> build()
  2. Runtime — pop.update().method(...)
  3. Python side — set_param(config, "name", v)
  4. In-hook parameter writes — pop.params.xxx = v (the first-class form
     for single-parameter hooks; writes take effect immediately and are
     recorded in the pop.params_log snapshot)
"""

from __future__ import annotations

import natal as nt
from natal.frontend.builder import set_param
from natal.frontend.hooks.tick_context import TickContext
from natal.frontend.model import BEVERTON_HOLT

# ═══════════════════════════════════════════════════════════════════════════════
# 0. Prepare the species
# ═══════════════════════════════════════════════════════════════════════════════

sp = nt.Species.from_dict(
    name="demo_params",
    structure={"auto": {"A": ["WT", "Var"]}},
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Build-time configuration — PopulationBuilder chained API
# ═══════════════════════════════════════════════════════════════════════════════
# New route: PopulationBuilder.from_species() -> chained methods -> build()
# Each chained method calls set_param() internally and writes config
# immediately; no freeze/build step.
#
# PopulationBuilder wraps a ModelDraft (the unified build draft) and exposes
# these domain methods:
#   .setup(...)             — simulation flags (stochastic, continuous_sampling, ...)
#   .age_structure(...)     — age dimension (age-structured models only)
#   .initial_state(...)     — initial population distribution (dict -> 3-D array)
#   .reproduction(...)      — reproduction parameters (eggs_per_female, sex_ratio, ...)
#   .competition(...)       — competition parameters (carrying_capacity, low_density_growth_rate, ...)
#   .survival(...)          — survival rates (female_age0_survival, male=, ...)
#   .custom(...)            — custom fields (stored in the config.custom structured array)
#   .hooks(...)             — register hooks (passed to the Population constructor)
#
# Terminal methods:
#   .apply()  — run deferred operations (presets/modifiers/fitness) + sync equilibrium
#   .build()  — create the Population object after apply()

pop = (
    nt.DiscreteGenerationPopulation
    .setup(sp)                                        # (1) PopulationBuilder entry
    .setup(stochastic=False)                          # (2) deterministic simulation
    .initial_state({                                   # (3) initial population
        "female": {"WT|WT": 5000, "WT|Var": 1000},
        "male":   {"WT|WT": 5000, "WT|Var": 1000},
    })
    .reproduction(                                     # (4) reproduction parameters
        eggs_per_female=50,   # → config.eggs_per_female
        sex_ratio=0.5,        # → config.sex_ratio
    )
    .competition(                                     # (5) competition parameters
        carrying_capacity=10000,          # → config.carrying_capacity
        low_density_growth_rate=6.0,      # → config.low_density_growth_rate
        juvenile_growth_mode=BEVERTON_HOLT,   # → config.juvenile_growth_mode
    )
    .custom(temperature=25.0, debug=False)             # (6) custom fields
    .build(name="demo_params")                         # (7) terminal: apply() + create the Population
)

print("=" * 60)
print("初始配置")
print("=" * 60)
print(f"  K          = {pop.config.carrying_capacity}")
print(f"  eggs       = {pop.config.eggs_per_female}")
print(f"  sex_ratio  = {pop.config.sex_ratio}")
print(f"  growth_r   = {pop.config.low_density_growth_rate}")
print(f"  temperature = {pop.config.custom['temperature']}")
print(f"  debug      = {bool(pop.config.custom['debug'])}")
print(f"  population = {pop.state.individual_count.sum():.0f} individuals")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Runtime modification — the pop.update() chained API
# ═══════════════════════════════════════════════════════════════════════════════
# pop.update() returns a RuntimeUpdater handle;
# subsequent chained methods commit in place through the same low-level
# write and take effect immediately.
#
# The chained syntax matches the build-time PopulationBuilder: build and
# runtime share one set of domain methods.

# ── 2a. Single-parameter change ──
pop.update().competition(carrying_capacity=5000)
print("\npop.update().competition(K=5000)")
print(f"  K = {pop.config.carrying_capacity}  ← 立即生效")

# ── 2b. Chained multi-parameter change ──
pop.update().reproduction(eggs_per_female=30, sex_ratio=0.4).competition(
    low_density_growth_rate=3.0
)
print(f"  eggs = {pop.config.eggs_per_female}")
print(f"  sr   = {pop.config.sex_ratio}")
print(f"  r    = {pop.config.low_density_growth_rate}")

# ── 2c. custom fields: pop.update().custom() resets the whole block ──
pop.update().custom(temperature=35.0, debug=True)
print(f"  temperature = {pop.config.custom['temperature']}  (升高!)")
print(f"  debug       = {bool(pop.config.custom['debug'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Python side — the low-level set_param() interface
# ═══════════════════════════════════════════════════════════════════════════════
# set_param(config, "name", v) is the low-level implementation behind every
# high-level API:
#   1. Look the parameter name up in the parameters.py registry (full names,
#      short names, and aliases)
#   2. Locate the right config field and array index
#   3. Write in place (arrays via field[idx] = v, custom slots via a dict write)
#   4. Equilibrium-sensitive parameters (K/eggs/sr) automatically call
#      sync_equilibrium_metrics()
#
# Intended for Python-side scripts, objmode hooks, and interactive notebooks.

set_param(pop.config, "carrying_capacity", 8000)
set_param(pop.config, "low_density_growth_rate", 4.0)
print(f"\nset_param() 修改后: K={pop.config.carrying_capacity}, r={pop.config.low_density_growth_rate}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. In-hook modification — single-parameter hook (the finalized slice-4 form)
# ═══════════════════════════════════════════════════════════════════════════════
# The only signature for a custom hook: def hook(pop) -> int
# pop is the live population the engine lends you (TickContext): params are
# writable, state is writable, metrics are computed on demand, stop()
# terminates, and rng is this deme's stream.
# Writing parameters inside a hook is first-class: writes take effect
# immediately and are recorded in pop.params_log.


@nt.hook(event="early")
def hook_degrade(pop: TickContext) -> int:
    # Fires once at tick=5: environment degrades (K halved, eggs reduced).
    if pop.tick == 5:
        pop.params.carrying_capacity = float(pop.params.carrying_capacity) * 0.5  # type: ignore[arg-type]  # ParamsView read is statically object; the route resolves a float scalar   # K halved
        pop.update().reproduction(eggs_per_female=35.0)                     # eggs cut to 70%
    return 0


@nt.hook(event="early")
def hook_recover(pop: TickContext) -> int:
    # Fires at tick=8: environment recovers (string-name routing via pop.update())
    if pop.tick == 8:
        pop.update().competition(carrying_capacity=10000.0)
        pop.update().reproduction(eggs_per_female=50.0, sex_ratio=0.5)
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Build with hooks + full run
# ═══════════════════════════════════════════════════════════════════════════════
# Note: the @hook-decorated functions above must be registered before build().
# A second population carrying the full hook set is built here to demonstrate.

# Note: the hook-carrying population is built through PopulationBuilder.
pop2 = (
    nt.DiscreteGenerationPopulation
    .setup(sp, name="demo_hooks", stochastic=False)
    .initial_state({
        "female": {"WT|WT": 5000},
        "male":   {"WT|WT": 5000},
    })
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(
        carrying_capacity=10000,
        low_density_growth_rate=6.0,
        juvenile_growth_mode=BEVERTON_HOLT,
    )
    .hooks(hook_degrade, hook_recover)
    .build()
)

print(f"\n{'=' * 60}")
print("带 Hook 的完整运行")
print("=" * 60)
print(f"初始: total={pop2.state.individual_count.sum():.0f}, K={pop2.config.carrying_capacity}")

# Run 10 ticks; the hooks fire at ticks 5 and 8 and modify K/eggs.
for t in range(1, 11):
    pop2.run(1)
    total = pop2.state.individual_count.sum()
    k = pop2.config.carrying_capacity
    eggs = pop2.config.eggs_per_female
    marker = " <-- hook 触发!" if t in (5, 8) else ""
    print(f"  tick={t:>2}  total={total:>6.0f}  K={k:>6.0f}  eggs={eggs:>5.0f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Summary: parameter-modification routes at a glance
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("参数修改方式速查")
print("=" * 60)
print(f"  参数快照 log 行数: {len(pop2.params_log)}")
print("""
  方式                      | 位置         | 说明
  ──────────────────────────┼──────────────┼────────────────────────────
  pop.update().method(...)  | 运行时       | 链式修改，最易用
  pop.params.xxx = v        | hook 内/运行时 | 一等写法，记入 params_log
  set_param(config,n,v)     | Python 侧    | 底层接口，字符串名路由
  ctx.update().method(...)  | hook 内      | 与构建链同语法
""")

print("演示完成 ✅")
