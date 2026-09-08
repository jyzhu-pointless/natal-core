"""Run a real AgeStructuredPopulation through the Rust lifecycle backend.

The demo builds one deterministic population, registers a declarative CSR
hook, enables the engine session, and runs 10 recorded ticks.  Between
ticks the trajectory stays a deterministic function of the seed, so the
final state and the recorded history are printed as the run summary.
"""

from __future__ import annotations

import natal as nt
from natal.backends.rust.rust_backend import rust_backend_available

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 检查原生扩展
# ═══════════════════════════════════════════════════════════════════════════════

if not rust_backend_available():
    print("natal._engine_rs is not built.")
    print("Build it first with: maturin develop --skip-install")
    raise SystemExit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 准备 Species 与 Population
# ═══════════════════════════════════════════════════════════════════════════════

sp = nt.Species.from_dict(
    name="rust_backend_demo_species",
    structure={"chr1": {"loc": ["A", "B"]}},
    gamete_labels=["default"],
)

pop = (
    nt.AgeStructuredPopulation.setup(sp, stochastic=False, name="rust_demo_pop")
    .initial_state(
        individual_count={
            "female": {"A|A": 200, "A|B": 100},
            "male": {"A|A": 150, "A|B": 150},
        }
    )
    .reproduction(
        eggs_per_female=10.0,
        sex_ratio=0.5,
        female_age_based_mating_rate=1.0,
        male_age_based_mating_rate=1.0,
        age_based_reproduction_rate=1.0,
        female_age_based_fertility=1.0,
        fixed_egg_count=True,
    )
    .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
    .competition(juvenile_growth_mode=1, carrying_capacity=500)
    .build()
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 注册一个 CSR declarative hook
# ═══════════════════════════════════════════════════════════════════════════════

control_ops = [
    nt.Op.scale(genotypes="*", ages="*", sex="both", factor=0.98),
    nt.Op.add(genotypes="A|A", ages=1, sex="female", delta=5.0, when="tick >= 2"),
]
pop.register_hooks(control_ops, event="early", name="demo_control")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 启用引擎会话并运行
# ═══════════════════════════════════════════════════════════════════════════════

pop.enable_rust_backend(seed=2026)

n_steps = 10
pop.run(n_steps, record_every=1, clear_history_on_start=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 输出结果
# ═══════════════════════════════════════════════════════════════════════════════

sperm = getattr(pop.state, "sperm_storage", None)

print("=" * 64)
print("Deterministic engine run (10 recorded ticks)")
print("=" * 64)
print(f"  final tick               : {pop.tick}")
print(f"  total population         : {pop.get_total_count():.1f}")
print(f"  history rows             : {pop.history.individual_count.shape[0]}")
if sperm is not None:
    print(f"  sperm storage total      : {float(sum(sperm.ravel())):.1f}")

if pop.tick != n_steps:
    raise RuntimeError("The run stopped before the requested tick count.")

print("\nDemo finished successfully.")
