"""Compiler ownership boundaries and sampling distribution edge contracts."""
from __future__ import annotations

import numpy as np
import pytest

from natal.frontend.hooks.runtime import sampling
from tests._config_assertions import assert_config_equal
from tests.test_review_runtime_regressions import ModelKind, _population


@pytest.mark.parametrize("shape", [0.2, 1.0, 2.0, 1e9])
def test_gamma_sampler_has_analytic_mean_and_variance(shape: float, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gamma(shape,1) has mean=variance=shape across both rejection regimes.

    20,000 independent draws use eight standard errors per moment. The Gamma
    fourth central moment is 3*k*k+6*k, giving the variance-estimator bound.
    These wide family-wise margins detect lost variance without seed selection.
    """
    rng = np.random.default_rng(91304)
    monkeypatch.setattr(np.random, "random", rng.random)
    monkeypatch.setattr(np.random, "normal", rng.normal)
    n = 20000
    draws = np.array([sampling._bounded_gamma(shape) for _ in range(n)])
    assert abs(draws.mean() - shape) < 8 * np.sqrt(shape / n)
    assert abs(draws.var() - shape) < 8 * np.sqrt((2 * shape**2 + 6 * shape) / n)
    assert np.all(draws >= 0)


def test_gamma_sampler_rejection_budget_and_precision_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Degenerate random sources terminate, and sub-ULP noise is not fabricated."""
    monkeypatch.setattr(np.random, "random", lambda: 0.0)
    assert sampling._bounded_gamma(2.0) == 2.0
    monkeypatch.setattr(np.random, "random", lambda: 0.999999999)
    assert sampling._bounded_gamma(0.2) == 0.2
    limit = float(2**104)
    assert sampling._bounded_gamma(limit) == limit


@pytest.mark.parametrize("n,p,expected", [(5, 0, 0), (5, 1, 5), (0.5, 0.25, 0.125)])
def test_continuous_binomial_exact_degenerate_limits(n: float, p: float, expected: float) -> None:
    """Probability endpoints and the declared fractional-n rule are exact."""
    assert sampling.continuous_binomial(n, p) == expected


@pytest.mark.parametrize("n,p", [(np.inf, 0.5), (10, np.nan)])
def test_continuous_binomial_rejects_nonfinite_inputs(n: float, p: float) -> None:
    """Undefined parameters fail explicitly instead of entering rejection loops."""
    with pytest.raises(ValueError, match="finite"):
        sampling.continuous_binomial(n, p)


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
    from natal.frontend.configurator import Configurator

    species = nt.Species.from_dict(
        name=f"CompressedBoundary_{declared}",
        structure={"chr": {"locus": ["WT", "Dr"]}},
        gamete_labels=["default", "deposited"],
    )
    pop = (Configurator.for_discrete(species)
           .setup(stochastic=False, compress=True, declared_zygote_types=declared)
           .initial_state({"female": {"WT|WT": 10}, "male": {"WT|WT": 10}})
           .build())
    assert pop.config.n_ztypes == 1
    assert pop.config.n_gtypes == 1
    np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, np.ones((2, 1, 1)))
    np.testing.assert_array_equal(pop.config.gametes_to_zygotes_map, np.ones((1, 1, 1)))
    pop.refresh_modifiers()
    np.testing.assert_array_equal(pop.config.offspring_tensor, np.ones((1, 1, 1)))


@pytest.mark.parametrize("inplace", [False, True])
def test_legacy_age_tick_adapter_matches_session_and_preserves_ownership(inplace: bool) -> None:
    """Both retained tick adapters advance the same native state exactly once."""
    pop = _population(f"LegacyTickAdapter_{inplace}", "age", stochastic=False)
    backend = pop._rust_lifecycle_backend
    state = pop._live_state()
    original = state.individual_count.copy()
    next_state, result = (backend.run_tick_inplace(state) if inplace else backend.run_tick(state))
    tick, counts, _ = backend.state_snapshot()
    assert result == 0 and tick == next_state.n_tick == 1
    np.testing.assert_array_equal(next_state.individual_count.ravel(), counts)
    if inplace:
        assert np.shares_memory(next_state.individual_count, state.individual_count)
    else:
        np.testing.assert_array_equal(state.individual_count, original)
        assert not np.shares_memory(next_state.individual_count, state.individual_count)


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


def test_legacy_csr_interpreter_rejects_unknown_mutation_code() -> None:
    """Malformed low-level plans cannot silently pretend an operation executed."""
    from dataclasses import replace
    import natal as nt
    from tests.test_ops_addendum_adversarial import _compile_plan, _invoke_plan

    pop = _population("UnknownMutationBoundary", "age", stochastic=False)
    plan = _compile_plan(pop, [nt.Op.scale(genotypes="*", factor=0.5)])
    malformed = replace(plan, op_types=np.full_like(plan.op_types, -1))
    counts = pop._live_state().individual_count.copy()
    before = counts.copy()
    with pytest.raises(ValueError, match="Unknown mutation opcode"):
        _invoke_plan(malformed, counts, None, tick=0, stochastic=False)
    np.testing.assert_array_equal(counts, before)


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


def test_legacy_csr_female_scaling_preserves_sperm_and_virgin_ratio() -> None:
    """Halving 10 mothers with six mated leaves five mothers and three mated."""
    import natal as nt
    from tests.test_ops_addendum_adversarial import _compile_plan, _invoke_plan

    pop = _population("FemaleSpermScaleBoundary", "age", stochastic=False)
    plan = _compile_plan(pop, [nt.Op.scale(genotypes="WT|WT", ages=[1], sex="female", factor=0.5)])
    state = pop._live_state()
    counts = np.zeros_like(state.individual_count)
    sperm = np.zeros_like(state.sperm_storage)
    counts[0, 1, 0] = 10
    sperm[1, 0, :] = [2, 4, 0]
    _invoke_plan(plan, counts, sperm, tick=0, stochastic=False)
    assert counts[0, 1, 0] == 5
    np.testing.assert_array_equal(sperm[1, 0], [1, 2, 0])
    assert counts[0, 1, 0] - sperm[1, 0].sum() == 2
