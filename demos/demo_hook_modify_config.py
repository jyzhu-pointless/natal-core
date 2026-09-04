"""验证 discrete 模型 inline 均衡计算：hook 内改 K 后种群响应。

演示切片④的 hook 内参数写入：单参数 hook（def hook(pop) -> int）
通过 pop.params 直接改 K，写入立即生效（下一次 run 使用新值），
并记录到 pop.params_log 参数快照。
"""

import natal as nt
from natal.frontend.hooks.tick_context import TickContext

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT"]}})

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=sp, name="demo", stochastic=False)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                 juvenile_growth_mode="beverton_holt")
    .build()
)

# 跑 5 ticks 到平衡
pop.run(5)


# hook 内改 K：tick=7 时一次性把 K 减半（用闭包标志保证只触发一次）
_fired = {"done": False}


@nt.hook(event="first")
def halve_carrying_capacity(pop: TickContext) -> int:
    if not _fired["done"] and pop.tick >= 7:
        pop.params.carrying_capacity = float(pop.params.carrying_capacity) * 0.5  # type: ignore[arg-type]  # ParamsView read is statically object; the route resolves a float scalar
        _fired["done"] = True
    return 0


pop.update().hooks(halve_carrying_capacity)

# 再跑 10 ticks 观察：hook 每次触发都会检查并收紧 K
print(f"{'tick':>4}  {'total':>10}  {'K':>10}")
for i in range(10):
    pop.run(1)
    total = pop.state.individual_count.sum()
    K = pop.params.carrying_capacity
    print(f"{5+i+1:>4}  {total:>10.0f}  {K:>10.0f}")

print("\n参数快照（hook 内写入，(tick, 参数, 旧值 → 新值)）：")
for row in pop.params_log:
    print(f"  tick={row[0]}  {row[1]}  {row[2]:.0f} → {row[3]:.0f}")
