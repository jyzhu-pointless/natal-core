"""Check removal of per-deme import bypasses and ordered native parameter commits."""

from typing import Literal, Protocol

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt
from natal._engine_rs import HeterogeneousSpatialEngineSession
from natal.backends.rust.rust_backend import RustHeterogeneousSpatialLifecycleBackend
from natal.frontend.hooks import Op


class NativeCandidate(Protocol):
    """The native transaction channels exercised by the direct callback."""

    def apply(self, writes: dict[str, float]) -> None: ...

    def tensor_write(self, name: str, values: NDArray[np.float64]) -> None: ...


def _session(model: Literal["age_structured", "discrete_generation"]) -> HeterogeneousSpatialEngineSession:
    """Build two neutral demes with a late declaration overriding callback ecology."""
    species = nt.Species.from_dict(name=f"NativeSpatialHandoff_{model}", structure={"chr": {"loc": ["WT", "Dr"]}})
    builder = nt.SpatialPopulation.builder(species, n_demes=2, pop_type=model).setup(stochastic=False)
    if model == "age_structured":
        builder = builder.age_structure(n_ages=2, new_adult_age=1)
    pop = (
        builder.initial_state(individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}})
        .reproduction(eggs_per_female=0, sex_ratio=0.5)
        .competition(carrying_capacity=100000, low_density_growth_rate=2)
        .hooks(Op.set_param("carrying_capacity", 900.0, event="late"))
        .build()
    )
    backend = pop._rust_spatial_backend
    assert backend is not None
    return backend._session


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
def test_native_deme_import_bridge_is_removed(
    model: Literal["age_structured", "discrete_generation"],
) -> None:
    """Managed deme edits use validated hook transactions, with no second import bridge."""
    session = _session(model)
    assert not hasattr(session, "set_deme_state")
    assert not hasattr(RustHeterogeneousSpatialLifecycleBackend, "set_deme_state")


def test_discrete_spatial_callback_queue_preserves_later_declarative_ecology() -> None:
    """An early transaction's genetics commit must retain the later K assignment per deme."""
    session = _session("discrete_generation")
    visits: list[tuple[int, int]] = []

    class EarlyNativeCallback:
        """Request the private transaction ABI rather than borrowed state arguments."""

        __natal_transaction__ = True

        def __call__(self, ind: None, sperm: None, tick: int, deme: int, candidate: NativeCandidate) -> int:
            """Stage distinct fitness variants and an ecology write before the late event."""
            assert ind is None and sperm is None
            visits.append((tick, deme))
            candidate.apply({"carrying_capacity": 100.0 + deme})
            candidate.tensor_write("viability_fitness", np.full(12, 0.25 + 0.5 * deme))
            return 0

    session.set_python_callbacks([], [EarlyNativeCallback()], [])
    for expected_tick in (1, 2):
        assert session.run_tick() == expected_tick
        for deme in range(2):
            assert session.get_deme_scalar(deme, "carrying_capacity") == 900.0
            np.testing.assert_array_equal(
                session.get_deme_tensor(deme, "viability_fitness"),
                np.full(12, 0.25 + 0.5 * deme),
            )
    assert visits == [(0, 0), (0, 1), (1, 0), (1, 1)]


@pytest.mark.parametrize("model", ["age_structured", "discrete_generation"])
@pytest.mark.parametrize("entry", ["constructor", "set_state"])
@pytest.mark.parametrize("invalid", ["negative_ind", "nan_ind", "inf_ind", "negative_sperm", "nan_sperm", "inf_sperm", "negative_tick"])
def test_native_stacked_handoff_rejects_invalid_values_without_mutation(
    model: Literal["age_structured", "discrete_generation"], entry: str, invalid: str,
) -> None:
    """Both native handoffs reject nonphysical payloads before replacing state or RNG."""
    from natal.backends.rust.rust_backend import (
        ecology_columns_from_drafts,
        genetics_variant_bank,
    )
    from natal.contracts.materialize import SpatialMigration, materialize
    from tests.test_review_runtime_regressions import _population

    draft = _population("StackedHandoff", "age" if model == "age_structured" else "discrete").config
    columns = ecology_columns_from_drafts([draft, draft])
    bank, ids = genetics_variant_bank([draft, draft])
    migration = SpatialMigration(
        indptr=np.zeros(3, dtype=np.int64), dest_idx=np.zeros(0, dtype=np.int64),
        weights=np.zeros(0), rate=np.zeros((2, 2, draft.n_ages)),
    )
    blueprint = materialize(draft, migration).blueprint
    individuals = np.stack([draft.initial_individual_count] * 2)
    sperm = np.stack([draft.initial_sperm_storage] * 2)

    def construct(ind: NDArray[np.float64], storage: NDArray[np.float64], tick: int) -> HeterogeneousSpatialEngineSession:
        return HeterogeneousSpatialEngineSession(blueprint, columns, bank, ids, ind, storage, tick, model=model, seed=191)

    subject = construct(individuals, sperm, 0)
    control = construct(individuals, sperm, 0)
    assert subject.run_tick() == control.run_tick() == 1
    before = subject.state_snapshot()
    ind_shape, sperm_shape = individuals.shape, sperm.shape
    individuals = before[1].reshape(ind_shape).copy()
    sperm = before[2].reshape(sperm_shape).copy()
    tick = before[0]
    if invalid == "negative_tick":
        tick = -1
    else:
        target = individuals if invalid.endswith("ind") else sperm
        target.flat[0] = -1.0 if invalid.startswith("negative") else (np.nan if invalid.startswith("nan") else np.inf)
    subject.stop()
    with pytest.raises(ValueError, match="nonnegative" if invalid == "negative_tick" else "finite nonnegative"):
        if entry == "constructor":
            construct(individuals, sperm, tick)
        else:
            subject.set_state(individuals, sperm, tick)
    np.testing.assert_equal(subject.state_snapshot(), before)
    assert subject.execution_state() == ("Stopped", 0)
    # Restoring only the unchanged arrays leaves the RNG untouched; a stochastic
    # next tick must therefore match a session that never saw the invalid import.
    subject.set_state(before[1].reshape(ind_shape), before[2].reshape(sperm_shape), before[0])
    assert subject.run_tick() == control.run_tick() == 2
    np.testing.assert_equal(subject.state_snapshot(), control.state_snapshot())
