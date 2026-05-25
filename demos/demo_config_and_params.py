"""演示所有参数配置方式：构建时、运行时、hook 内。

场景：模拟一个种群经历环境突变。
  - tick 0-5:  承载力 K=10000，温度 T=25°C（适宜）
  - tick 5-10: 环境退化，K 减半、T 升高（通过 hook 内修改）
  - tick 10-15: 环境恢复，K 回升、T 恢复正常（通过 pop.update() 修改）

覆盖的修改方式：
  1. 构建时 — Configurator 链式 API → build()
  2. 运行时 — pop.update().method(...)
  3. Python 侧 — set_param(config, "name", v)
  4. Hook 内直接写 — config.field[()] = v  （nopython，最快）
  5. Hook 内 hook_set_param — hook_set_param(config, "name", v)  （objmode 封装）
"""

from __future__ import annotations

import natal as nt
from natal.configurator import hook_set_param, set_param
from natal.population_config import CONCAVE, DiscretePopulationConfig
from natal.population_state import DiscretePopulationState

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 准备 Species
# ═══════════════════════════════════════════════════════════════════════════════

sp = nt.Species.from_dict(
    name="demo_params",
    structure={"auto": {"A": ["WT", "Var"]}},
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 构建时配置 — Configurator 链式 API
# ═══════════════════════════════════════════════════════════════════════════════
# 新路径：Configurator.from_species() → 链式方法 → build()
# 每个链式方法内部调用 set_param() 立即写入 config，不需 freeze/build。
#
# Configurator 包装了一个 PopulationConfig，提供以下领域方法：
#   .setup(...)           — 模拟标志（stochastic、continuous_sampling 等）
#   .age_structure(...)   — 年龄维度（n_ages, new_adult_age, generation_time）
#   .initial_state(...)   — 初始种群分布（字典 → 3-D 数组）
#   .reproduction(...)    — 繁殖参数（eggs_per_female, sex_ratio 等）
#   .competition(...)     — 竞争参数（carrying_capacity, low_density_growth_rate 等）
#   .survival(...)        — 存活率（female_age0_survival, male= 等）
#   .custom(...)          — 自定义字段（存储到 config.custom structured array）
#   .hooks(...)           — 注册 hook（传给 Population 构造函数）
#
# 终端方法：
#   .apply()  — 执行 deferred 操作（presets/modifiers/fitness）+ sync equilibrium
#   .build()  — apply() 后创建 Population 对象

pop = (
    nt.DiscreteGenerationPopulation
    .setup(sp, legacy_path=False)                     # ① 新路径入口
    .setup(stochastic=False)                          # ② 确定性模拟
    .age_structure(n_ages=2, new_adult_age=1)          # ③ Discrete 固定 n_ages=2
    .initial_state({                                   # ④ 初始种群
        "female": {"WT|WT": 5000, "WT|Var": 1000},
        "male":   {"WT|WT": 5000, "WT|Var": 1000},
    })
    .reproduction(                                     # ⑤ 繁殖参数
        eggs_per_female=50,   # → config.expected_eggs_per_female[()]
        sex_ratio=0.5,        # → config.sex_ratio[()]
    )
    .competition(                                     # ⑥ 竞争参数
        carrying_capacity=10000,          # → config.carrying_capacity[()]
        low_density_growth_rate=6.0,      # → config.low_density_growth_rate[()]
        juvenile_growth_mode=CONCAVE,   # → config.juvenile_growth_mode[()]
    )
    .custom(temperature=25.0, debug=False)             # ⑦ 自定义字段
    .build(name="demo_params")                         # ⑧ 终端：apply() + 创建 Population
)

print("=" * 60)
print("初始配置")
print("=" * 60)
print(f"  K          = {pop.config.carrying_capacity[()]}")
print(f"  eggs       = {pop.config.expected_eggs_per_female[()]}")
print(f"  sex_ratio  = {pop.config.sex_ratio[()]}")
print(f"  growth_r   = {pop.config.low_density_growth_rate[()]}")
print(f"  temperature = {pop.config.custom['temperature'][()]}")
print(f"  debug      = {bool(pop.config.custom['debug'][()])}")
print(f"  population = {pop.state.individual_count.sum():.0f} individuals")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 运行时修改 — pop.update() 链式 API
# ═══════════════════════════════════════════════════════════════════════════════
# pop.update() 拿当前 config 包一个 Configurator 返回，
# 后续链式方法同样通过 set_param() 原地写入，立即生效。
#
# 与构建时的 Configurator 是同一个类——构建和运行时都用同一套 API。

# ── 2a. 单个参数修改 ──
pop.update().competition(carrying_capacity=5000)
print("\npop.update().competition(K=5000)")
print(f"  K = {pop.config.carrying_capacity[()]}  ← 立即生效")

# ── 2b. 链式多参数修改 ──
pop.update().reproduction(eggs_per_female=30, sex_ratio=0.4).competition(
    low_density_growth_rate=3.0
)
print(f"  eggs = {pop.config.expected_eggs_per_female[()]}")
print(f"  sr   = {pop.config.sex_ratio[()]}")
print(f"  r    = {pop.config.low_density_growth_rate[()]}")

# ── 2c. custom 字段通过 pop.update().custom() 重设整个 custom ──
pop.update().custom(temperature=35.0, debug=True)
print(f"  temperature = {pop.config.custom['temperature'][()]}  (升高!)")
print(f"  debug       = {bool(pop.config.custom['debug'][()])}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Python 侧 — set_param() 底层接口
# ═══════════════════════════════════════════════════════════════════════════════
# set_param(config, "name", v) 是所有高层 API 的底层实现：
#   1. 在 parameters.py 注册表中查找参数名（支持全名、短名、别名）
#   2. 定位到正确的 config 字段和数组索引
#   3. 原地写入（0-d ndarray 用 field[()] = v，数组索引用 field[idx] = v）
#   4. equilibrium-sensitive 参数（K/eggs/sr）自动调用 sync_equilibrium_metrics()
#
# 适用场景：Python 侧脚本、objmode hook 内、notebook 交互式分析

set_param(pop.config, "carrying_capacity", 8000)
set_param(pop.config, "low_density_growth_rate", 4.0)
print(f"\nset_param() 修改后: K={pop.config.carrying_capacity[()]}, r={pop.config.low_density_growth_rate[()]}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Hook 内修改 — 三种写法
# ═══════════════════════════════════════════════════════════════════════════════
# Hook 签名：(state, config, deme_id) → int
# config 作为参数直接传入，可以原地修改。
# 修改在 hook return 后立即对后续 hook 和当前 tick 的剩余流程可见。

# ── 写法 A：直接改 config 字段（推荐，nopython 最快）──
# 适用：你知道字段名，且参数是 0-d ndarray。


@nt.hook(event="early", custom=True)
def hook_direct(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    _deme_id: int,
) -> int:
    # 只在 tick=5 触发一次：高温导致环境退化
    if state.n_tick == 5:
        config.carrying_capacity[()] *= 0.5       # K 减半
        config.expected_eggs_per_female[()] *= 0.7 # 繁殖率降至 70%
    return 0


# ── 写法 B：hook_set_param（字符串参数名，objmode 封装在内部）──
# 适用：需要字符串名路由，或 hook 内需要混合 Python 逻辑。
# 代价：每次调用有 objmode 边界开销（~微秒级）。


@nt.hook(event="early", custom=True)
def hook_objmode(
    state: DiscretePopulationState,
    config: DiscretePopulationConfig,
    _deme_id: int,
) -> int:
    if state.n_tick == 8:
        hook_set_param(config, "carrying_capacity", 10000.0)
        hook_set_param(config, "reproduction.eggs_per_female", 50.0)
        hook_set_param(config, "reproduction.sex_ratio", 0.5)
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 带 Hook 的构建 + 完整运行
# ═══════════════════════════════════════════════════════════════════════════════
# 注意：上面 @hook 装饰器的函数需要在 build() 前注册。
# 这里重新构建一个带完整 hooks 的种群来演示。

# 注意：带 hook 的种群目前通过 legacy_path 构建以确保 hook 编译路径正确。
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
        juvenile_growth_mode=CONCAVE,
    )
    .hooks(hook_direct, hook_objmode)
    .build()
)

print(f"\n{'=' * 60}")
print("带 Hook 的完整运行")
print("=" * 60)
print(f"初始: total={pop2.state.individual_count.sum():.0f}, K={pop2.config.carrying_capacity[()]}")

# 运行 5 ticks → hook 触发条件满足 → K/eggs 被修改
for t in range(1, 11):
    pop2.run(1)
    total = pop2.state.individual_count.sum()
    k = pop2.config.carrying_capacity[()]
    eggs = pop2.config.expected_eggs_per_female[()]
    marker = " <-- hook 触发!" if t in (5, 8) else ""
    print(f"  tick={t:>2}  total={total:>6.0f}  K={k:>6.0f}  eggs={eggs:>5.0f}{marker}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 总结：参数修改方式速查
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 60}")
print("参数修改方式速查")
print("=" * 60)
print("""
  方式                    | 位置         | 性能   | 说明
  ────────────────────────┼──────────────┼────────┼──────────────────────────
  pop.update().method(...) | between-tick | Python | 运行时链式修改，最易用
  set_param(config,n,v)   | between-tick | Python | 底层接口，字符串名路由
  config.field[()] = v    | hook nopython| 最快   | 直接写 0-d ndarray
  hook_set_param(c,n,v)   | hook 内      | 同 objmode | objmode 封装，单次调用便捷"
""")

print("演示完成 ✅")
