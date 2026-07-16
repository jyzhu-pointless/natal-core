"""Simple discrete demo for observation output and history tracking.

Linear script version for slides: build -> run -> export a compact report.
"""

from __future__ import annotations

import json

import natal as nt
from natal.patterns import IndividualSelector

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
    .with_observation(
        groups={
            "drive_carriers": IndividualSelector(ztype="WT|Dr") | IndividualSelector(ztype="Dr|Dr"),
            "wildtype": IndividualSelector(ztype="WT|WT"),
        },
    )
    .record_history(mode="raw")
    .build()
)

population.run(n_steps=5, record_every=1)

# Current state via canonical observation
current_result = population.observe()
current_state = {
    "tick": current_result.tick,
    "axes": list(current_result.axes),
    "labels": {k: list(v) for k, v in current_result.labels.items()},
    "values": current_result.values.tolist(),
}

# History via post-hoc observation
obs_hist = population.history.observe(population.observation)
history_report = {
    "ticks": list(obs_hist.ticks),
    "axes": list(obs_hist.axes),
    "labels": {"group": list(obs_hist.schema.observation.labels)},
    "values": obs_hist.values.tolist(),
}

report = {
    "current_state": current_state,
    "history": history_report,
}

print(json.dumps(report, ensure_ascii=False, indent=2))
