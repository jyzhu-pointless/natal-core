"""P10 completion-audit contracts (whole-program cross-phase scenarios).

Independent evaluator tests for the two 验证计划 scenarios whose
combined coverage was missing after the P0–P9 approvals:

* **Scenario 5 (stochastic complement)** — two demes share genetics, one
  deme is updated, and the sibling demes' *stochastic* trajectories stay
  bitwise-isolated.  The existing isolation tests
  (``test_forked_deme_trajectory_diverges``,
  ``test_k_write_changes_trajectory_only_at_target``) run deterministic
  builds; this file adds the same contract under ``stochastic=True`` so
  a fork that perturbed a sibling's per-deme RNG consumption order would
  be caught as a bitwise divergence from an untouched same-seed
  baseline.
* **Scenario 6 (call-observation evidence)** — the spatial container's
  aggregate reads must answer from the native session without
  refreshing per-deme state caches (no "refresh-all then restack"), and
  the recommended scalar update entry on a deme
  (``deme(i).update().competition(...)``) must not materialize the full
  contract.  The panmictic scalar-batch contract is already pinned in
  ``tests/test_routes_slice3.py::TestScalarShape``; these tests extend
  the same call-observation method to the spatial surfaces.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.population import SpatialPopulation


def _species(name: str) -> nt.Species:
    """Two-allele unordered species: genotypes WT|WT, WT|A, A|A."""
    return nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "A"]}}, gamete_labels=["default"]
    )


def _build_stochastic_ring(name: str, *, n_demes: int = 4, seed: int = 11) -> SpatialPopulation:
    """Build a stochastic discrete spatial population sharing one genetics variant.

    All demes start from the same heterozygote seed and share the
    homogeneity-group genetics tables, so a per-deme fork is the only
    way one deme's tables diverge.  No migration is declared: demes are
    demographically independent, so any cross-deme trajectory change
    after a single-deme write can only come from shared RNG state.  The
    session is initialized with the explicit *seed* so two builds with
    the same seed replay bitwise.
    """
    species = _species(f"{name}_species")
    per_deme = {"female": {"WT|A": 100}, "male": {"WT|A": 100}}
    pop = (
        nt.SpatialPopulation.builder(
            species, n_demes=n_demes, pop_type="discrete_generation"
        )
        .setup(name=name, stochastic=True)
        .initial_state(individual_count=nt.batch_setting([per_deme] * n_demes))
        .survival(female_age0_survival=0.9, male_age0_survival=0.9)
        .reproduction(eggs_per_female=4.0, sex_ratio=0.5)
        .competition(carrying_capacity=600.0, low_density_growth_rate=2.0)
        .build()
    )
    pop._initialize_session(seed=seed)  # pyright: ignore[reportPrivateUsage]  # explicit same-seed initialization, as in test_spatial_session_ownership
    return pop


# ============================================================================
# Scenario 5: shared genetics, one-deme update, stochastic trajectory isolation
# ============================================================================


def test_shared_genetics_fork_isolates_sibling_stochastic_trajectories() -> None:
    """A deme genetics fork leaves siblings' stochastic runs bitwise intact.

    Requirement (验证计划, cross-phase scenario 5): two demes sharing the
    genetics tables; after updating one, the other's numerics and random
    trajectory still satisfy the existing isolation contract.  Both
    populations run stochastic with the same seed; if forking or the
    forked deme's execution perturbed a sibling's per-deme RNG stream
    (consumption order or count), the sibling states would diverge from
    the untouched baseline bitwise.  Catches: fork touching shared
    streams, fork re-seeding siblings, or the forked deme's changed
    table sizes shifting sibling draws.
    """
    modified = _build_stochastic_ring("P10ForkStochastic")
    baseline = _build_stochastic_ring("P10ForkStochastic")

    # Sanity: the demes share one genetics variant before the fork.
    backend = modified._rust_spatial_session()  # pyright: ignore[reportPrivateUsage]  # bank identity is the sharing contract
    assert backend is not None and backend.n_variants == 1

    forked = np.full_like(
        np.asarray(modified.deme(2).config.viability_fitness), 0.5
    )
    modified.deme(2).write_genetics("viability_fitness", forked)
    # The fork created a second bank entry for the written deme only.
    assert backend.n_variants == 2

    modified.run(4, record_every=0)
    baseline.run(4, record_every=0)

    # The written deme diverges (numerics changed)...
    assert not np.array_equal(
        modified.deme(2).state.individual_count,
        baseline.deme(2).state.individual_count,
    )
    # ...while every sibling tracks the same-seed baseline bitwise
    # (numerics AND random trajectory).
    for i in (0, 1, 3):
        np.testing.assert_array_equal(
            modified.deme(i).state.individual_count,
            baseline.deme(i).state.individual_count,
            err_msg=f"stochastic trajectory of untouched deme {i} diverged",
        )


def test_shared_ecology_write_isolates_sibling_stochastic_trajectories() -> None:
    """A deme ecology scalar write leaves siblings' stochastic runs bitwise intact.

    Same contract as above through the ecology channel: the write touches
    only the target deme's column entry and must not re-seed, re-order,
    or consume sibling RNG streams.
    """
    modified = _build_stochastic_ring("P10KStochastic")
    baseline = _build_stochastic_ring("P10KStochastic")

    modified.deme(1).write_ecology("carrying_capacity", 40.0)

    modified.run(4, record_every=0)
    baseline.run(4, record_every=0)

    assert not np.array_equal(
        modified.deme(1).state.individual_count,
        baseline.deme(1).state.individual_count,
    )
    for i in (0, 2, 3):
        np.testing.assert_array_equal(
            modified.deme(i).state.individual_count,
            baseline.deme(i).state.individual_count,
            err_msg=f"stochastic trajectory of untouched deme {i} diverged",
        )


# ============================================================================
# Scenario 6: call-observation evidence for spatial reads and scalar updates
# ============================================================================


def test_container_aggregate_reads_never_refresh_deme_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate reads answer from the session without per-deme refreshes.

    Requirement (验证计划, cross-phase scenario 6): the spatial
    whole-population read must not depend on refreshing every internal
    deme snapshot and re-stacking them.  Call observation: ``run`` marks
    every deme's state cache stale, then ``get_total_count`` /
    ``get_female_count`` / ``get_male_count`` /
    ``aggregate_individual_count`` must execute with **zero** calls to
    ``_refresh_deme_state`` and zero per-deme ``state`` property reads;
    a nonzero count means the refresh-all-then-restack regression came
    back.  The native stacked state is still consulted (the counts
    match the stacked sums), so the zero assertions cannot pass
    vacuously.
    """
    pop = _build_stochastic_ring("P10AggregateSpy")
    pop.run(2, record_every=0)  # every deme state cache is now stale
    assert pop._rust_spatial_session() is not None  # pyright: ignore[reportPrivateUsage]  # native path is the one under test

    calls = {"refresh": 0, "deme_state": 0}
    original_refresh = SpatialPopulation._refresh_deme_state
    original_state = type(pop.deme(0)).state

    def spy_refresh(self: SpatialPopulation, index: int) -> None:
        calls["refresh"] += 1
        original_refresh(self, index)

    def spy_state(self: Any) -> Any:
        calls["deme_state"] += 1
        return original_state.__get__(self)

    monkeypatch.setattr(SpatialPopulation, "_refresh_deme_state", spy_refresh)
    monkeypatch.setattr(type(pop.deme(0)), "state", property(spy_state))

    total = pop.get_total_count()
    female = pop.get_female_count()
    male = pop.get_male_count()
    aggregate = pop.aggregate_individual_count()

    assert calls == {"refresh": 0, "deme_state": 0}
    assert total == female + male
    # The values still come from the authoritative stacked session state.
    stacked = pop._native_stacked_state()  # pyright: ignore[reportPrivateUsage]  # authority identity check
    assert stacked is not None
    np.testing.assert_array_equal(aggregate, np.sum(stacked[1], axis=0))
    assert int(aggregate.sum()) == total


def test_deme_scalar_update_avoids_full_contract_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deme's recommended scalar update commits through the scalar batch.

    Requirement (计划「更新」章 + cross-phase scenario 6): scalar updates
    read only the needed scalars and must not copy the full genetics for
    one parameter.  Call observation on the spatial surface: while
    ``deme(i).update().competition(carrying_capacity=...)`` runs, the
    deme's native channel must receive the atomic scalar ``apply`` batch
    and must never see ``refresh_params`` (the full-contract path); the
    ``materialize`` helper is likewise never called.  Mirrors
    ``test_routes_slice3.py::TestScalarShape::test_scalar_marks_contract_dirty``
    for the spatial deme channel.
    """
    import importlib

    from natal.backends.rust import rust_backend

    materialize_module = importlib.import_module("natal.contracts.materialize")

    pop = _build_stochastic_ring("P10DemeScalarSpy")
    pop.run(1, record_every=0)

    calls = {"refresh_params": 0, "materialize": 0, "apply": 0}
    channel_type = rust_backend.RustDemeParameters
    original_apply = channel_type.apply
    original_refresh = channel_type.refresh_params
    original_materialize = materialize_module.materialize

    def spy_apply(self: Any, writes: dict[str, float]) -> None:
        calls["apply"] += 1
        original_apply(self, writes)

    def spy_refresh(self: Any, fields: list[str], params_obj: Any) -> None:
        calls["refresh_params"] += 1
        original_refresh(self, fields, params_obj)

    def spy_materialize(*args: Any, **kwargs: Any) -> Any:
        calls["materialize"] += 1
        return original_materialize(*args, **kwargs)

    monkeypatch.setattr(channel_type, "apply", spy_apply)
    monkeypatch.setattr(channel_type, "refresh_params", spy_refresh)
    monkeypatch.setattr(materialize_module, "materialize", spy_materialize)

    pop.deme(1).update().competition(carrying_capacity=321.0)

    assert calls["apply"] == 1
    assert calls["refresh_params"] == 0
    assert calls["materialize"] == 0
    assert float(pop.params.carrying_capacity[1]) == 321.0
