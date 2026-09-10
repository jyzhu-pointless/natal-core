"""Ecology bridging: equilibrium metric derivation from draft values.

The equilibrium calibration is the ecology computation of the model
assembly side: the sensitive-parameter sync path and the build-time map
computation share one dispatch point so the kernel choice cannot drift
apart, and the ``pop.params`` query surface reads the same derivation.
The numeric kernel lives in Rust.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .draft import ModelDraft

__all__ = [
    "derive_equilibrium_metrics_from_draft",
    "equilibrium_metrics_dispatch",
]


def equilibrium_metrics_dispatch(
    carrying_capacity: float,
    eggs_per_female: float,
    sex_ratio: float,
    survival_rates: NDArray[np.float64],
    reproduction_rates: NDArray[np.float64],
    fertility: NDArray[np.float64],
    competition_weights: NDArray[np.float64],
    new_adult_age: int,
    n_ages: int,
    declared_distribution: NDArray[np.float64] | None,
    external_expected_eggs: float | None,
) -> tuple[float, float]:
    """Run the Rust equilibrium kernel.

    Single dispatch point for the equilibrium calibration:
    the sensitive-parameter sync path and the build-time map computation
    both funnel through here so the kernel choice cannot drift apart.
    Callers feed already-resolved reproduction vectors (the None-fallback
    to the female mating row is caller policy).  The Rust engine is the
    only execution backend: a missing extension propagates.

    Args:
        carrying_capacity: Carrying capacity K (age-1 total).
        eggs_per_female: Baseline offspring count per female.
        sex_ratio: Female proportion.
        survival_rates: ``(2, n_ages)`` survival matrix.
        reproduction_rates: Resolved ``(n_ages,)`` participation vector.
        fertility: ``(n_ages,)`` relative female fertility.
        competition_weights: ``(n_ages,)`` juvenile competition weights.
        new_adult_age: First adult age index.
        n_ages: Total age classes.
        declared_distribution: ``None`` or empty means derive mode.
        external_expected_eggs: Champer egg override (``None`` = unused).

    Returns:
        ``(expected_competition_strength, expected_survival_rate)`` from
        the Rust kernel.
    """
    from natal._engine_rs import equilibrium_metrics_flat as rust_metrics

    declared = (
        np.ascontiguousarray(declared_distribution, dtype=np.float64)
        if declared_distribution is not None and declared_distribution.size > 0
        else None
    )
    return rust_metrics(
        float(carrying_capacity),
        float(eggs_per_female),
        float(sex_ratio),
        np.ascontiguousarray(survival_rates, dtype=np.float64),
        np.ascontiguousarray(reproduction_rates, dtype=np.float64),
        np.ascontiguousarray(fertility, dtype=np.float64),
        np.ascontiguousarray(competition_weights, dtype=np.float64),
        int(new_adult_age),
        int(n_ages),
        declared,
        external_expected_eggs,
    )


def derive_equilibrium_metrics_from_draft(
    draft: ModelDraft,
) -> tuple[float, float]:
    """Derive the equilibrium metrics from a draft's current values.

    Single read-side derivation shared by the sensitive-write sync and
    the ``pop.params`` query surface (one numeric source; the draft's
    stored copies are retired).  The declared
    distribution and Champer override are read from the draft itself,
    and the reproduction fallback (female mating row) is resolved here.

    Args:
        draft: The draft whose ecology drives the derivation.

    Returns:
        ``(expected_competition_strength, expected_survival_rate)`` —
        always freshly computed, never a cached copy.
    """
    reproduction = (
        draft.age_based_reproduction_rates
        if draft.age_based_reproduction_rates is not None
        else draft.age_based_mating_rates[0]
    )
    return equilibrium_metrics_dispatch(
        draft.carrying_capacity,
        draft.eggs_per_female,
        draft.sex_ratio,
        draft.age_based_survival_rates,
        reproduction,
        draft.female_age_based_fertility,
        draft.age_based_relative_competition_strength,
        int(draft.new_adult_age),
        int(draft.n_ages),
        draft.equilibrium_individual_distribution,
        draft.external_expected_eggs,
    )
