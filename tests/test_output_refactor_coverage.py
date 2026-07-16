"""Coverage contracts for typed output configuration and dashboard consumers."""

from __future__ import annotations

import json

import numpy as np
import pytest

import natal as nt
import natal.ui.dashboard_population as dashboard_module
from natal.data import DiscretePopulationState
from natal.patterns import IndividualSelector
from natal.spatial.configurator import SpatialConfigurator
from natal.ui.dashboard_population import Dashboard


class _FakeChart:
    """Minimal chart boundary used by Dashboard update tests."""

    options: dict[str, list[dict[str, str | list[list[float]]]]]
    update_count: int

    def __init__(self, series: list[dict[str, str | list[list[float]]]]) -> None:
        """Initialize the chart with its serialized series.

        Args:
            series: Serialized chart series data.
        """
        self.options = {"series": series}
        self.update_count = 0

    def update(self) -> None:
        """Record one chart refresh."""
        self.update_count += 1


class _FakeTabs:
    """Minimal tab boundary used by history inspection tests."""

    def __init__(self) -> None:
        """Initialize an empty tab-selection log."""
        self.values: list[str] = []

    def set_value(self, value: str) -> None:
        """Record the selected tab value.

        Args:
            value: Tab identifier to log.
        """
        self.values.append(value)


class _FakeUI:
    """Capture download and notification effects without starting NiceGUI."""

    def __init__(self) -> None:
        """Initialize empty download and notification logs."""
        self.downloads: list[bytes] = []
        self.notifications: list[str] = []

    def download(
        self,
        payload: bytes,
        *,
        filename: str,
        media_type: str,
    ) -> None:
        """Capture a serialized export payload.

        Args:
            payload: Raw bytes written to the download.
            filename: Expected file name (asserted to end with ``.json``).
            media_type: Expected content type (asserted to be ``application/json``).
        """
        assert filename.endswith(".json")
        assert media_type == "application/json"
        self.downloads.append(payload)

    def notify(self, message: str, **_kwargs: str) -> None:
        """Capture a notification message.

        Args:
            message: Notification text to log.
            **_kwargs: Additional keyword arguments (ignored).
        """
        self.notifications.append(message)


def _species(name: str) -> nt.Species:
    """Create a two-allele species with three active ZTypes.

    Args:
        name: Species identifier.

    Returns:
        A species with one biallelic locus.
    """
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr"]}},
    )


def _discrete_population(
    name: str,
    *,
    history_mode: str = "raw",
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic discrete population for typed History tests.

    Args:
        name: Base identifier for species and population.
        history_mode: ``"raw"`` (default) or ``"observation"``.

    Returns:
        A built discrete-generation population.
    """
    configurator = (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(f"{name}_species"),
            name=name,
            stochastic=False,
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 30.0, "WT|Dr": 20.0},
                "male": {"WT|WT": 10.0, "Dr|Dr": 40.0},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=10.0)
        .competition(
            juvenile_growth_mode="concave",
            low_density_growth_rate=2.0,
            carrying_capacity=100,
        )
    )
    if history_mode == "observation":
        configurator.record_history(mode="observation")
    return configurator.build()


def _dashboard(population: nt.DiscreteGenerationPopulation) -> Dashboard:
    """Construct Dashboard state without invoking the NiceGUI constructor.

    Args:
        population: The population to attach to the dashboard.

    Returns:
        A Dashboard instance with pre-initialized chart buffers.
    """
    dashboard = object.__new__(Dashboard)
    dashboard.pop = population
    dashboard._chart_history = []
    dashboard._allele_freq_history = {}
    dashboard._history_ticks = []
    dashboard._last_chart_tick = -1
    dashboard.view_min = None
    dashboard.view_max = None
    dashboard.max_chart_points = 500
    dashboard.chart_pop = _FakeChart([{"name": "population", "data": []}])
    dashboard.chart_allele = _FakeChart([])
    return dashboard


def test_base_population_requires_installed_history_and_observation() -> None:
    """Uninitialized public output properties fail with explicit errors."""
    population = object.__new__(nt.AgeStructuredPopulation)
    population._history_obj = None  # type: ignore[reportPrivateUsage]  # construct pre-build state
    population._observation = None  # type: ignore[reportPrivateUsage]  # construct pre-build state

    with pytest.raises(RuntimeError, match="History has not been initialized"):
        _ = population.history
    with pytest.raises(RuntimeError, match="Observation has not been initialized"):
        _ = population.observation


def test_spatial_runtime_rejects_output_schema_mutation() -> None:
    """Runtime SpatialConfigurator cannot replace frozen output policies."""
    configurator = SpatialConfigurator(_species("spatial_runtime"), n_demes=1)
    configurator._pop_ref = object()  # type: ignore[reportPrivateUsage]  # emulate runtime binding

    with pytest.raises(RuntimeError, match="build phase"):
        configurator.with_observation(groups={"all": IndividualSelector()})
    with pytest.raises(RuntimeError, match="build phase"):
        configurator.record_history(mode="observation")


def test_spatial_configurator_rejects_invalid_groups_and_mode() -> None:
    """Spatial output configuration validates public boundary values."""
    configurator = SpatialConfigurator(_species("spatial_invalid"), n_demes=1)

    with pytest.raises(TypeError, match="mapping"):
        configurator.with_observation(groups=[])  # type: ignore[arg-type]  # invalid runtime input
    with pytest.raises(ValueError, match="non-empty"):
        configurator.with_observation(groups={})
    with pytest.raises(ValueError, match="mode must be"):
        configurator.record_history(mode="invalid")  # type: ignore[arg-type]  # invalid runtime input


@pytest.mark.parametrize("max_rows", [0, -1])
def test_spatial_configurator_rejects_invalid_max_rows(max_rows: int) -> None:
    """History capacity must be None or a positive row count.

    Args:
        max_rows: An invalid capacity value supplied via parametrize.
    """
    configurator = SpatialConfigurator(_species(f"spatial_rows_{max_rows}"), n_demes=1)
    with pytest.raises(ValueError, match="max_rows"):
        configurator.record_history(max_rows=max_rows)


def test_output_capacity_validation_and_spatial_success_path() -> None:
    """Both configurators validate capacity and retain valid frozen settings."""
    species = _species("capacity_validation")
    with pytest.raises(ValueError, match="max_rows"):
        nt.DiscreteGenerationPopulation.setup(species).record_history(max_rows=0)

    configurator = SpatialConfigurator(species, n_demes=1)
    groups = {"wild": IndividualSelector(ztype="WT|WT")}
    result = configurator.with_observation(groups=groups).record_history(
        mode="observation", max_rows=3
    )
    assert result is configurator
    assert configurator._observation_groups == groups  # type: ignore[reportPrivateUsage]  # verify frozen builder input
    assert configurator._record_history_mode == "observation"  # type: ignore[reportPrivateUsage]  # verify frozen builder input
    assert configurator._record_history_max_rows == 3  # type: ignore[reportPrivateUsage]  # verify frozen builder input


def test_dashboard_metrics_preserve_exact_counts_and_allele_frequencies() -> None:
    """Typed counts produce exact totals and allele frequencies."""
    population = _discrete_population("dashboard_metrics")
    dashboard = _dashboard(population)
    counts = np.zeros_like(population.state.individual_count)
    counts[0, 1] = np.array([3.0, 5.0, 7.0])
    counts[1, 1] = np.array([11.0, 13.0, 17.0])

    total, frequencies = dashboard._compute_metrics_from_counts(counts)
    assert total == 56.0
    assert frequencies["WT"] == pytest.approx(46.0 / 112.0)
    assert frequencies["Dr"] == pytest.approx(66.0 / 112.0)
    assert frequencies["WT"] + frequencies["Dr"] == pytest.approx(1.0)


def test_dashboard_raw_state_supports_discrete_and_rejects_observation() -> None:
    """Typed state reconstruction accepts raw records and rejects aggregates."""
    raw_population = _discrete_population("dashboard_raw_state")
    raw_population.run(n_steps=1, record_every=1)
    raw_dashboard = _dashboard(raw_population)

    state = raw_dashboard._raw_history_state(0)
    assert isinstance(state, DiscretePopulationState)
    assert state.n_tick == raw_population.history.ticks[0]
    np.testing.assert_array_equal(
        state.individual_count,
        raw_population.history.individual_count[0],
    )

    observed_population = _discrete_population(
        "dashboard_observation_state", history_mode="observation"
    )
    observed_dashboard = _dashboard(observed_population)
    with pytest.raises(ValueError, match="requires raw history"):
        observed_dashboard._raw_history_state(0)


def test_dashboard_rebuild_and_update_charts_use_typed_history() -> None:
    """Chart buffers exactly mirror typed raw History records."""
    population = _discrete_population("dashboard_charts")
    population.run(n_steps=1, record_every=1)
    dashboard = _dashboard(population)

    dashboard._rebuild_chart_history()
    expected_totals = [float(row.sum()) for row in population.history.individual_count]
    assert dashboard._history_ticks == list(population.history.ticks)
    assert dashboard._chart_history == [
        [tick, total]
        for tick, total in zip(population.history.ticks, expected_totals)
    ]

    dashboard._chart_history = []
    dashboard._allele_freq_history = {}
    dashboard._history_ticks = []
    dashboard._last_chart_tick = -1
    dashboard._update_charts()
    assert dashboard.chart_pop.options["series"][0]["data"] == dashboard._chart_history
    assert dashboard.chart_pop.update_count == 1
    assert dashboard.chart_allele.update_count == 1


def test_dashboard_observation_history_and_live_update_skip_raw_projection() -> None:
    """Aggregate History is skipped while the current live state remains chartable."""
    population = _discrete_population(
        "dashboard_observation_charts", history_mode="observation"
    )
    dashboard = _dashboard(population)

    dashboard._rebuild_chart_history()
    assert dashboard._chart_history == []
    assert dashboard._history_ticks == []

    dashboard._update_charts()
    assert dashboard._chart_history == [[0, float(population.get_total_count())]]
    assert dashboard._history_ticks == [0]
    assert dashboard.chart_pop.options["series"][0]["data"] == dashboard._chart_history


def test_dashboard_export_and_inspection_use_typed_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export and inspection consume typed records without flat parsing.

    Args:
        monkeypatch: Pytest monkeypatch fixture for UI replacement.
    """
    population = _discrete_population("dashboard_export")
    population.run(n_steps=1, record_every=1)
    dashboard = _dashboard(population)
    fake_ui = _FakeUI()
    monkeypatch.setattr(dashboard_module, "ui", fake_ui)

    dashboard._do_export_logic(
        include_config=False,
        include_history=True,
        include_hooks=False,
    )
    payload = json.loads(fake_ui.downloads[0].decode("utf-8"))
    assert [entry["tick"] for entry in payload["history"]] == list(
        population.history.ticks
    )

    inspected: list[tuple[int, bool]] = []
    dashboard._update_inspection_view = (  # type: ignore[method-assign]  # replace UI renderer
        lambda _state, tick, *, is_history: inspected.append((tick, is_history))
    )
    dashboard.tabs_main = _FakeTabs()
    dashboard.inspection_mode = False
    dashboard.inspected_tick = None
    history_tick = population.history.ticks[0]
    dashboard.inspect_tick(history_tick)
    assert inspected == [(history_tick, True)]
    assert dashboard.tabs_main.values == ["inspection"]

    dashboard.inspect_tick(999)
    assert fake_ui.notifications[-1] == "No history found for Tick 999"


def test_dashboard_observation_export_uses_only_current_typed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observation History export never attempts to reconstruct raw snapshots.

    Args:
        monkeypatch: Pytest monkeypatch fixture for UI replacement.
    """
    population = _discrete_population(
        "dashboard_observation_export", history_mode="observation"
    )
    dashboard = _dashboard(population)
    fake_ui = _FakeUI()
    monkeypatch.setattr(dashboard_module, "ui", fake_ui)

    dashboard._do_export_logic(
        include_config=False,
        include_history=True,
        include_hooks=False,
    )
    payload = json.loads(fake_ui.downloads[0].decode("utf-8"))
    assert len(payload["history"]) == 1
    assert payload["history"][0]["tick"] == population.tick == 0
