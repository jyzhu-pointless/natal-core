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
print(f"{'tick':>4}  {'total':>10}  {'K':>10}  expected_comp  expected_surv")
for i in range(10):
    pop.run(1)
    state = pop.state
    total = state.individual_count[0, 1, :].sum() + state.individual_count[1, 1, :].sum()
    K = pop.config.carrying_capacity[()]
    # compute equilibrium inline for debug
    from natal.engine.simulation.age_structured import compute_equilibrium_metrics
    ec, es = compute_equilibrium_metrics(
        carrying_capacity=float(K),
        expected_eggs_per_female=float(pop.config.expected_eggs_per_female),
        age_based_survival_rates=pop.config.age_based_survival_rates,
        age_based_mating_rates=pop.config.age_based_mating_rates,
        age_based_reproduction_rates=pop.config.age_based_reproduction_rates,
        female_age_based_relative_fertility=pop.config.female_age_based_relative_fertility,
        relative_competition_strength=pop.config.age_based_relative_competition_strength,
        sex_ratio=float(pop.config.sex_ratio),
        new_adult_age=pop.config.new_adult_age,
        n_ages=pop.config.n_ages,
    )
    print(f"{5+i+1:>4}  {total:>10.0f}  {K:>10.0f}  {ec:>12.0f}  {es:>12.4f}")
