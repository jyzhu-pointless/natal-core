"""Run a real AgeStructuredPopulation through the Rust lifecycle backend.

The demo builds two identical deterministic populations:

1. ``reference`` keeps the default Numba lifecycle path.
2. ``rust_pop`` calls ``enable_rust_backend()`` before running.

After 10 recorded ticks the two states and histories are compared
element-by-element.  A declarative CSR hook is registered on both
populations first, so the comparison also covers hook execution inside Rust.
"""

from __future__ import annotations

import numpy as np

import natal as nt
from natal.engine.backends.rust_backend import rust_backend_available

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 检查原生扩展
# ═══════════════════════════════════════════════════════════════════════════════

if not rust_backend_available():
    print("natal._engine_rs is not built.")
    print("Build it first with: maturin develop --skip-install")
    raise SystemExit(0)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 准备 Species 与 Population 构建函数
# ═══════════════════════════════════════════════════════════════════════════════

sp = nt.Species.from_dict(
    name="rust_backend_demo_species",
    structure={"chr1": {"loc": ["A", "B"]}},
    gamete_labels=["default"],
)


def build_population(name: str):
    """Build one deterministic age-structured population."""
    return (
        nt.AgeStructuredPopulation.setup(sp, stochastic=False, name=name)
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


reference = build_population("rust_demo_reference")
rust_pop = build_population("rust_demo_pop")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 注册同一个 CSR declarative hook（两个种群保持一致）
# ═══════════════════════════════════════════════════════════════════════════════

control_ops = [
    nt.Op.scale(genotypes="*", ages="*", sex="both", factor=0.98),
    nt.Op.add(genotypes="A|A", ages=1, sex="female", delta=5.0, when="tick >= 2"),
]

for pop in (reference, rust_pop):
    pop.register_declarative_hook("early", control_ops, name="demo_control")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 启用 Rust 后端并运行
# ═══════════════════════════════════════════════════════════════════════════════

rust_pop.enable_rust_backend(seed=2026)
print("Rust backend enabled:", rust_pop.using_rust_backend)

n_steps = 10
reference.run(n_steps, record_every=1, clear_history_on_start=True)
rust_pop.run(n_steps, record_every=1, clear_history_on_start=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 对比结果
# ═══════════════════════════════════════════════════════════════════════════════

ind_equal = np.array_equal(
    rust_pop.state.individual_count,
    reference.state.individual_count,
)
sperm_equal = np.array_equal(
    rust_pop.state.sperm_storage,
    reference.state.sperm_storage,
)
history_equal = np.array_equal(
    rust_pop.history.individual_count,
    reference.history.individual_count,
)

print("=" * 64)
print("Rust backend vs Numba reference (deterministic, 10 ticks)")
print("=" * 64)
print(f"  final tick                     : {rust_pop.tick}")
print(f"  total population               : {rust_pop.get_total_count():.1f}")
print(f"  individual_count identical     : {ind_equal}")
print(f"  sperm_storage identical        : {sperm_equal}")
print(f"  recorded history identical     : {history_equal}")

if not (ind_equal and sperm_equal and history_equal):
    raise RuntimeError("Rust backend diverged from the Numba reference.")

print("\nDemo finished successfully.")
