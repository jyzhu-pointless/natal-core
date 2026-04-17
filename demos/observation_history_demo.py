"""Simple discrete demo for readable observation output.

Linear script version for slides: build -> run -> export a compact report.
"""


from __future__ import annotations

import json

import natal as nt
from natal.state_translation import output_current_state, output_history

nt.disable_numba()

species = nt.Species.from_dict(
    name="ObservationHistoryDemoSpecies",
    structure={"chr1": {"loc": ["WT", "Dr"]}},
)

population = (
    nt.DiscreteGenerationPopulation
    .setup(species=species, name="ObservationHistoryDemo", stochastic=False)
    .initial_state(
        individual_count={
            "female": {
                "WT|WT": 180.0,
                "WT|Dr": 20.0,
            },
            "male": {
                "WT|WT": 180.0,
                "WT|Dr": 20.0,
            },
        }
    )
    .survival(
        female_age0_survival=1.0,
        male_age0_survival=1.0,
    )
    .reproduction(
        eggs_per_female=50.0,
    )
    .competition(
        juvenile_growth_mode="concave",
        low_density_growth_rate=6.0,
        carrying_capacity=400,
    )
    .build()
)

population.run(n_steps=5, record_every=1)

groups: dict[str, dict[str, object]] = {
    "drive_carriers": {
        "genotype": ["WT::Dr", "Dr|Dr"],
        "age": [1],
    },
    "wildtype": {
        "genotype": "WT|WT",
        "age": [1],
    },
}

current_state = output_current_state(
    population=population,
    groups=groups,
    collapse_age=True,
    include_zero_counts=False,
)

history_report = output_history(
    population=population,
    groups=groups,
    collapse_age=True,
    include_zero_counts=False,
)

report = {
    "current_state": current_state,
    "history": history_report,
}

print(json.dumps(report, ensure_ascii=False, indent=2))

