from __future__ import annotations

from natal.numba_utils import njit_switch
from natal.population_config import PopulationConfig


@njit_switch(cache=True)
def set_config_param(config: PopulationConfig, param_id: int, value: float) -> None:
    if param_id == 11:
        config.generation_time[()] = value
        return
    if param_id == 14:
        config.age_based_survival_rates[0] = value
        return
    if param_id == 15:
        config.age_based_survival_rates[1] = value
        return
    if param_id == 16:
        config.age_based_survival_rates[0][0] = value
        return
    if param_id == 17:
        config.age_based_survival_rates[1][0] = value
        return
    if param_id == 21:
        config.expected_eggs_per_female[()] = value
        return
    if param_id == 22:
        config.sex_ratio[()] = value
        return
    if param_id == 23:
        config.age_based_mating_rates[0] = value
        return
    if param_id == 24:
        config.age_based_mating_rates[1] = value
        return
    if param_id == 25:
        config.age_based_mating_rates[0][1] = value
        return
    if param_id == 26:
        config.age_based_mating_rates[1][1] = value
        return
    if param_id == 27:
        config.age_based_reproduction_rates[1] = value
        return
    if param_id == 28:
        config.sperm_displacement_rate[()] = value
        return
    if param_id == 29:
        config.female_age_based_relative_fertility[0] = value
        return
    if param_id == 34:
        config.age_based_relative_competition_strength[1] = value
        return
    if param_id == 35:
        config.juvenile_growth_mode[()] = value
        return
    if param_id == 36:
        config.low_density_growth_rate[()] = value
        return
    if param_id == 37:
        config.carrying_capacity[()] = value
        return
    if param_id == 40:
        config.expected_competition_strength[()] = value
        return
    if param_id == 41:
        config.expected_survival_rate[()] = value
        return
    pass  # unknown param_id or tensor — no-op
