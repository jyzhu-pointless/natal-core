"""Compiler ownership boundaries and sampling distribution edge contracts."""
from __future__ import annotations

import numpy as np
import pytest

from tests._config_assertions import assert_config_equal
from tests.test_review_runtime_regressions import ModelKind, _population


@pytest.mark.parametrize("model", ["age", "discrete"])
def test_config_import_rejects_wrong_type_and_changed_layout_atomically(model: ModelKind) -> None:
    """Invalid imports cannot partially replace current native ecology or shape."""
    pop = _population(f"ImportBoundary_{model}", model)
    before = pop.config
    with pytest.raises(TypeError):
        pop.import_config(object())
    with pytest.raises(ValueError, match="layout"):
        pop.import_config(before._replace(n_ztypes=before.n_ztypes + 1))
    assert_config_equal(pop.config, before)


def test_deferred_zygote_modifier_registration_compiles_once() -> None:
    """Batch registration leaves live products unchanged until explicit refresh."""
    pop = _population("DeferredZygoteBoundary")
    calls = []

    def modifier() -> dict[str, float]:
        calls.append(1)
        return {}

    before = pop.config.offspring_tensor
    pop.add_zygote_modifier(modifier, refresh=False)
    assert calls == []
    np.testing.assert_array_equal(pop.config.offspring_tensor, before)
    pop.refresh_modifiers(rebuild_maps=False)
    assert calls == [1]
    np.testing.assert_array_equal(pop.config.offspring_tensor, before)


@pytest.mark.parametrize("declared", [None, [0], ["WT|WT"]])
def test_compression_preserves_closed_wild_type_population(declared: list[str] | list[int] | None) -> None:
    """A wild-type-only Mendelian model needs exactly one active type per axis."""
    import natal as nt
    from natal.frontend.builder import PopulationBuilder

    species = nt.Species.from_dict(
        name=f"CompressedBoundary_{declared}",
        structure={"chr": {"locus": ["WT", "Dr"]}},
        gamete_labels=["default", "deposited"],
    )
    pop = (PopulationBuilder.for_discrete(species)
           .setup(stochastic=False, compress=True, declared_zygote_types=declared)
           .initial_state({"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
           .build())
    assert pop.config.n_ztypes == 1
    assert pop.config.n_gtypes == 1
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, np.ones((2, 1, 1)))
    np.testing.assert_array_equal(pop.config.gametes_to_zygotes_map, np.ones((1, 1, 1)))
    pop.refresh_modifiers()
    np.testing.assert_array_equal(pop.config.offspring_tensor, np.ones((1, 1, 1)))


def test_native_age_tick_snapshot_advances_session_once() -> None:
    """Native session advancement is observed through an isolated snapshot."""
    pop = _population("NativeTickSnapshot", "age", stochastic=False)
    backend = pop._rust_lifecycle_backend
    state = pop._live_state()
    backend.set_state(state)
    original = state.individual_count.copy()
    _, _, stopped = backend.run(n_steps=1, record_every=0)
    tick, counts, _ = backend.state_snapshot()
    assert not stopped and tick == 1
    np.testing.assert_array_equal(state.individual_count, original)
    counts[0] += 1.0
    _, fresh_counts, _ = backend.state_snapshot()
    assert not np.array_equal(counts, fresh_counts)


def test_legacy_backend_tick_adapters_are_removed() -> None:
    """Backends expose only session-owned state advancement."""
    import natal.backends.rust.rust_backend as rust_backend

    assert not hasattr(rust_backend.RustLifecycleBackend, "run_tick")
    assert not hasattr(rust_backend.RustLifecycleBackend, "run_tick_inplace")
    assert not hasattr(rust_backend.RustDiscreteLifecycleBackend, "run_tick")
    assert not hasattr(rust_backend.RustDiscreteLifecycleBackend, "run_tick_inplace")
    assert not hasattr(rust_backend, "rust_run_age_structured_aging")
    assert not hasattr(rust_backend, "rust_run_discrete_aging")
    import natal._engine_rs as engine

    assert not hasattr(engine, "age_structured_aging")
    assert not hasattr(engine, "discrete_aging")


def test_legacy_scalar_log_ignores_no_change_and_retains_event_metadata() -> None:
    """The scalar compatibility adapter follows successful-change log semantics."""
    pop = _population("LegacyScalarLogBoundary")
    pop.log_param_change("temperature", 2.5, 3.0)
    pop.log_param_change("temperature", 3.0, 3.0)
    assert pop.params_log == ((0, "temperature", 2.5, 3.0),)
    assert pop.params_log_details[0][1:4] == ("update", 0, "temperature")


def test_dashboard_equilibrium_metrics_follow_current_configuration() -> None:
    """The UI read helper must agree with current native ecology after updates."""
    from natal.frontend.ui.dashboard_population import _derive_metrics

    pop = _population("DashboardMetricBoundary", "age", stochastic=False)
    pop.update().competition(carrying_capacity=1234)
    c_star, s_star = _derive_metrics(pop.config)
    assert c_star == pop.params.expected_competition_strength
    assert s_star == pop.params.expected_survival_rate


def test_legacy_preset_apply_preserves_manual_zygote_registration() -> None:
    """The deprecated apply spelling still installs zygote-only manual recipes."""
    import natal as nt
    from natal.frontend.genetics.compile import RecipeHost
    from natal.frontend.modifiers.module import ZygoteModifier

    class ZygoteOnly(nt.GeneticPreset):
        """A neutral zygote-only recipe exercises the legacy registration contract."""

        def gamete_modifier(self, host: RecipeHost) -> None:
            """Declare no meiosis change."""
            return None

        def zygote_modifier(self, host: RecipeHost) -> ZygoteModifier:
            """Declare an empty, probability-preserving fertilization patch."""
            return lambda: {}

        def fitness_patch(self) -> None:
            """Declare no fitness change."""
            return None

    pop = _population("LegacyZygotePresetBoundary")
    before = pop.config.offspring_tensor
    ZygoteOnly(name="zygote-only").apply(pop)
    assert [name for _, name, _ in pop.zygote_modifiers] == ["zygote-only/zygote"]
    assert pop.presets == []
    np.testing.assert_array_equal(pop.config.offspring_tensor, before)
