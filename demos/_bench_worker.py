"""Numba vs pure-Python benchmark worker — invoked by bench_numba.py."""
import sys
import time

sys.path.insert(0, ".")

from natal.genetics import Species
from natal.population.age_structured import AgeStructuredPopulation

warmup = int(sys.argv[1])
bench = int(sys.argv[2])

sp = Species.from_dict(
    name="bench_sub",
    structure={"chr1": {"loc": ["A1", "A2", "A3", "A4", "A5"]}},
    gamete_labels=["default"],
)
genos = [str(g) for g in sp.get_all_genotypes()]
dist = {g: [0, 1000] for g in genos}
pop = (
    AgeStructuredPopulation.setup(sp, stochastic=False)
    .age_structure(n_ages=8, new_adult_age=2)
    .initial_state(individual_count={"female": dist, "male": dist})
    .reproduction(eggs_per_female=50, sex_ratio=0.5)
    .survival(female_age_based_survival=0.9, male_age_based_survival=0.9)
    .competition(carrying_capacity=50_000, juvenile_growth_mode="logistic")
    .build()
)

pop.run(warmup, finish=False)
t0 = time.perf_counter()
pop.run(bench, finish=False)
elapsed = time.perf_counter() - t0
print(f"ELAPSED={elapsed:.6f}")
