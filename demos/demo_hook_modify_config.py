"""验证 discrete 模型 inline 均衡计算：改 K 后种群响应。"""

import natal as nt

sp = nt.Species.from_dict(name="demo", structure={"auto": {"A": ["WT"]}})

pop = (
    nt.DiscreteGenerationPopulation
    .setup(species=sp, name="demo", stochastic=False)
    .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .competition(carrying_capacity=10000, low_density_growth_rate=6.0,
                 juvenile_growth_mode="concave")
    .build()
)

# 跑 5 ticks 到平衡
pop.run(5)

# 改 K
pop.config.carrying_capacity[()] = 5000.0
print(f"K 减半 → {pop.config.carrying_capacity[()]}")

# 再跑 10 ticks 观察
print(f"{'tick':>4}  {'total':>10}  {'K':>10}")
for i in range(10):
    pop.run(1)
    total = pop.state.individual_count.sum()
    K = pop.config.carrying_capacity[()]
    print(f"{5+i+1:>4}  {total:>10.0f}  {K:>10.0f}")
