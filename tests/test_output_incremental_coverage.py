"""Focused public-seam coverage for History and Observation integrations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pytest

import natal as nt
import natal.ui.spatial_dashboard as spatial_dashboard_module
from natal.output import (
    population_observation_history_to_readable_dict,
    spatial_population_history_to_readable_dict,
    spatial_population_observation_history_to_readable_dict,
)
from natal.output.observation import ObservationFilter
from natal.patterns import IndividualSelector
from natal.spatial.configurator import batch_setting
from natal.ui.spatial_dashboard import SpatialDashboard


def _species(name: str) -> nt.Species:
    """Build the biallelic species shared by focused output tests.

    Args:
        name: Unique species identifier.

    Returns:
        A species with three diploid genotypes.
    """
    return nt.Species.from_dict(
        name,
        {"Chr1": {"L1": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _groups() -> dict[str, IndividualSelector]:
    """Return one exhaustive observation group.

    Returns:
        A single group selecting every ZType coordinate.
    """
    return {"all": IndividualSelector()}


def _build_discrete(
    name: str,
    *,
    history_mode: Literal["raw", "observation"] = "raw",
    collapse_age: bool = False,
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic panmictic population with exact total 100.

    Args:
        name: Unique population identifier.
        history_mode: History storage mode.
        collapse_age: Whether observation removes the age axis.

    Returns:
        A configured discrete-generation population.
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
        .reproduction(eggs_per_female=2.0)
        .competition(
            juvenile_growth_mode="concave",
            low_density_growth_rate=2.0,
            carrying_capacity=1000.0,
        )
    )
    if history_mode == "observation":
        configurator.with_observation(
            groups=_groups(),
            collapse_age=collapse_age,
        )
    return configurator.record_history(mode=history_mode).build()


def _build_age(name: str) -> nt.AgeStructuredPopulation:
    """Build an age-structured population with nonzero sperm storage.

    Args:
        name: Unique population identifier.

    Returns:
        A deterministic two-age population.
    """
    return (
        nt.AgeStructuredPopulation.setup(
            species=_species(f"{name}_species"),
            name=name,
            stochastic=False,
        )
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": [1.0, 2.0]},
                "male": {"WT|Dr": [3.0, 4.0]},
            },
            sperm_storage={"WT|WT": {"WT|Dr": {1: 0.5}}},
        )
        .survival(
            female_age_based_survival=[1.0],
            male_age_based_survival=[1.0],
        )
        .reproduction(
            eggs_per_female=0.0,
            female_age_based_mating_rate=[0.0, 0.0],
            male_age_based_mating_rate=[0.0, 0.0],
        )
        .competition(juvenile_growth_mode=nt.NO_COMPETITION)
        .record_history(mode="raw")
        .build()
    )


def _build_spatial(
    name: str,
    *,
    pop_type: Literal["age_structured", "discrete_generation"] = "discrete_generation",
    history_mode: Literal["raw", "observation"] = "raw",
    collapse_age: bool = False,
    deme_mode: Literal["preserve", "aggregate"] = "preserve",
) -> nt.SpatialPopulation:
    """Build a deterministic two-deme population for output contracts.

    Args:
        name: Unique population identifier.
        pop_type: Underlying deme lifecycle model.
        history_mode: Spatial History storage mode.
        collapse_age: Whether observation removes the age axis.
        deme_mode: Whether observation preserves or aggregates demes.

    Returns:
        A two-deme spatial population.
    """
    configurator = (
        nt.SpatialPopulation.builder(
            _species(f"{name}_species"),
            n_demes=2,
            topology=None,
            pop_type=pop_type,
        )
        .setup(name=name, stochastic=False)
    )
    if pop_type == "age_structured":
        configurator = (
            configurator.age_structure(n_ages=2, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"WT|WT": [1.0, 2.0]},
                    "male": {"WT|Dr": [3.0, 4.0]},
                },
                sperm_storage={"WT|WT": {"WT|Dr": {1: 0.5}}},
            )
            .survival(
                female_age_based_survival=[1.0],
                male_age_based_survival=[1.0],
            )
            .reproduction(
                eggs_per_female=0.0,
                female_age_based_mating_rate=[0.0, 0.0],
                male_age_based_mating_rate=[0.0, 0.0],
            )
            .competition(juvenile_growth_mode=nt.NO_COMPETITION)
        )
    else:
        configurator = (
            configurator.initial_state(
                individual_count={
                    "female": {"WT|WT": 30.0, "WT|Dr": 20.0},
                    "male": {"WT|WT": 10.0, "Dr|Dr": 40.0},
                }
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2.0)
            .competition(
                juvenile_growth_mode="concave",
                low_density_growth_rate=2.0,
                carrying_capacity=1000.0,
            )
        )
    if history_mode == "observation":
        configurator.with_observation(
            groups=_groups(),
            collapse_age=collapse_age,
            demes=[0, 1],
            deme_mode=deme_mode,
        )
    return configurator.record_history(mode=history_mode).build()


def _numeric_leaves(value: object) -> list[float]:
    """Collect numeric leaves from a JSON-compatible output payload.

    Args:
        value: Nested mapping, sequence, scalar, or metadata value.

    Returns:
        Floating-point leaves in traversal order.
    """
    if isinstance(value, dict):
        leaves: list[float] = []
        for nested in value.values():
            leaves.extend(_numeric_leaves(nested))
        return leaves
    if isinstance(value, list):
        leaves = []
        for nested in value:
            leaves.extend(_numeric_leaves(nested))
        return leaves
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    return []


@dataclass(frozen=True)
class _LegacyGroupSpec:
    """Duck-typed legacy observation group used at the public compiler seam."""

    genotype: list[str]
    age: list[int]
    sex: str


@pytest.mark.parametrize("mapping", [False, True])
def test_legacy_group_objects_compile_exact_coordinates(mapping: bool) -> None:
    """Legacy object groups preserve exact genotype, age, and sex selection.

    Args:
        mapping: Whether to pass the object through the mapping input form.
    """
    population = _build_discrete(f"legacy_group_{mapping}")
    compiler = ObservationFilter(population.index_registry)
    group = _LegacyGroupSpec(genotype=["WT|WT"], age=[1], sex="female")
    groups: object = {"legacy": group} if mapping else [group]
    observation = compiler.build_filter(
        diploid_genotypes=population.species,
        groups=groups,  # type: ignore[arg-type]  # exercise supported legacy duck type
        n_sexes=2,
        n_ages=population.config.n_ages,
        n_ztypes=population.config.n_ztypes,
    )

    result = observation.apply(population.state.individual_count)
    assert result.shape == (1, 2, population.config.n_ages)
    np.testing.assert_array_equal(result[0, 0], np.array([0.0, 30.0]))
    np.testing.assert_array_equal(result[0, 1], np.array([0.0, 0.0]))


def test_haploid_genotype_resolves_to_complete_diploid_space() -> None:
    """A haploid genotype resolves through its species without losing genotypes."""
    species = _species("haploid_resolution")
    haploid = next(iter(species.iter_haploid_genotypes()))
    resolved = ObservationFilter.resolve_diploid_genotypes(haploid)
    expected = list(species.iter_genotypes(unordered=species.unordered))

    assert resolved is not None
    assert list(resolved) == expected


def test_raw_age_history_to_dict_preserves_counts_and_sperm() -> None:
    """Panmictic raw serialization preserves both state tensors exactly."""
    population = _build_age("raw_age_dict")
    population.record_snapshot()

    payload = population.history.to_dict(include_zero_counts=True)
    snapshot = payload["snapshots"][0]
    individual_values = _numeric_leaves(snapshot["individual_count"])
    sperm_values = _numeric_leaves(snapshot["sperm_storage"])

    assert payload["n_snapshots"] == 1
    assert sum(individual_values) == float(population.state.individual_count.sum()) == 10.0
    assert sum(sperm_values) == pytest.approx(
        float(population.state.sperm_storage.sum())
    )


def test_raw_spatial_history_to_dict_preserves_each_deme_tensor() -> None:
    """Spatial raw serialization keeps exact per-deme count and sperm buffers."""
    population = _build_spatial(
        "raw_spatial_dict",
        pop_type="age_structured",
        history_mode="raw",
    )
    population.record_snapshot()

    payload = population.history.to_dict(include_zero_counts=True)
    snapshot = payload["snapshots"][0]
    expected_counts = [deme.state.individual_count.ravel().tolist() for deme in population.demes]
    expected_sperm = [deme.state.sperm_storage.ravel().tolist() for deme in population.demes]

    assert payload["state_type"] == "SpatialPopulation"
    assert payload["n_snapshots"] == 1
    assert snapshot["individual_count_per_deme"] == expected_counts
    assert snapshot["sperm_storage_per_deme"] == expected_sperm


@pytest.mark.parametrize("spatial", [False, True])
@pytest.mark.parametrize("collapse_age", [False, True])
@pytest.mark.parametrize("deme_mode", ["preserve", "aggregate"])
def test_observation_history_to_dict_conserves_selected_total(
    spatial: bool,
    collapse_age: bool,
    deme_mode: Literal["preserve", "aggregate"],
) -> None:
    """Every observation axis combination preserves the exhaustive-group total.

    Args:
        spatial: Whether to use a two-deme population.
        collapse_age: Whether to remove age from the result.
        deme_mode: Whether to preserve or aggregate the spatial axis.
    """
    if spatial:
        population = _build_spatial(
            f"obs_dict_{collapse_age}_{deme_mode}",
            history_mode="observation",
            collapse_age=collapse_age,
            deme_mode=deme_mode,
        )
        expected_total = sum(
            float(deme.state.individual_count.sum()) for deme in population.demes
        )
    else:
        population = _build_discrete(
            f"obs_dict_nonspatial_{collapse_age}_{deme_mode}",
            history_mode="observation",
            collapse_age=collapse_age,
        )
        expected_total = float(population.state.individual_count.sum())
    population.record_snapshot()

    payload = population.history.to_dict(include_zero_counts=True)
    observed = payload["snapshots"][0]["observed"]
    observed_values = _numeric_leaves(observed)

    assert payload["labels"] == ["all"]
    assert payload["n_snapshots"] == 1
    assert sum(observed_values) == expected_total


def test_panmictic_observation_history_translation_uses_typed_history() -> None:
    """The public translator reads the canonical History when no array is supplied."""
    population = _build_discrete(
        "translation_panmicitic",
        history_mode="observation",
    )
    population.record_snapshot()

    payload = population_observation_history_to_readable_dict(population)
    observed = payload["snapshots"][0]["observed"]

    assert payload["n_snapshots"] == 1
    assert payload["labels"] == ["all"]
    assert sum(_numeric_leaves(observed)) == 100.0


def test_raw_spatial_history_translation_uses_typed_history() -> None:
    """Raw spatial translation decodes every recorded deme without an override."""
    population = _build_spatial("translation_raw_spatial", history_mode="raw")
    population.record_snapshot()

    payload = spatial_population_history_to_readable_dict(population)
    snapshot = payload["snapshots"][0]
    translated_total = sum(
        sum(_numeric_leaves(deme_payload["individual_count"]))
        for deme_payload in snapshot["demes"].values()
    )

    assert payload["n_demes"] == 2
    assert payload["n_snapshots"] == 1
    assert translated_total == 200.0


@pytest.mark.parametrize("collapse_age", [False, True])
@pytest.mark.parametrize("deme_mode", ["preserve", "aggregate"])
def test_spatial_observation_history_translation_conserves_total(
    collapse_age: bool,
    deme_mode: Literal["preserve", "aggregate"],
) -> None:
    """Spatial observation translation preserves totals for every output shape.

    Args:
        collapse_age: Whether observation removes the age axis.
        deme_mode: Whether observation preserves or aggregates demes.
    """
    population = _build_spatial(
        f"translation_spatial_{collapse_age}_{deme_mode}",
        history_mode="observation",
        collapse_age=collapse_age,
        deme_mode=deme_mode,
    )
    population.record_snapshot()

    payload = spatial_population_observation_history_to_readable_dict(population)
    snapshot = payload["snapshots"][0]

    assert payload["n_snapshots"] == 1
    assert snapshot["labels"] == ["all"]
    assert sum(_numeric_leaves(snapshot["aggregate"])) == 200.0


def test_population_record_and_restore_error_paths_are_atomic() -> None:
    """Recording lifecycle errors leave the typed timeline unchanged."""
    population = _build_discrete("record_restore_errors")
    population._running = True  # type: ignore[reportPrivateUsage]  # adversarial lifecycle state
    with pytest.raises(RuntimeError, match="running"):
        population.record_snapshot()
    population._running = False  # type: ignore[reportPrivateUsage]  # restore stable lifecycle state
    assert population.history.ticks == ()

    with pytest.raises(ValueError, match="No history"):
        population.restore_checkpoint(0)
    assert population.history.ticks == ()

    observed = _build_discrete(
        "observation_restore_error",
        history_mode="observation",
    )
    observed.record_snapshot()
    original = observed.history._to_numpy().copy()
    with pytest.raises(ValueError, match="observation-mode"):
        observed.restore_checkpoint(0)
    np.testing.assert_array_equal(observed.history._to_numpy(), original)


def test_age_import_rejects_bad_tuple_and_count_shape_atomically() -> None:
    """All age-state validation completes before arrays or History mutate."""
    population = _build_age("age_import_atomic_extra")
    population.record_snapshot()
    initial_count = population.state.individual_count.copy()
    initial_sperm = population.state.sperm_storage.copy()
    initial_history = population.history._to_numpy().copy()

    with pytest.raises(ValueError, match="length 2"):
        population.import_state((initial_count, initial_sperm, initial_sperm))  # type: ignore[arg-type]  # invalid tuple arity
    with pytest.raises(ValueError, match="individual_count shape"):
        population.import_state((np.zeros((1,), dtype=np.float64), initial_sperm))

    np.testing.assert_array_equal(population.state.individual_count, initial_count)
    np.testing.assert_array_equal(population.state.sperm_storage, initial_sperm)
    np.testing.assert_array_equal(population.history._to_numpy(), initial_history)
    assert population.tick == 0


def test_discrete_reset_and_finish_preserve_lifecycle_invariants() -> None:
    """Reset clears History exactly and zero-step finish closes the population."""
    population = _build_discrete("discrete_reset_finish")
    population.record_snapshot()
    initial_count = population.state.individual_count.copy()

    population.reset()
    assert population.history.ticks == ()
    assert population.tick == 0
    np.testing.assert_array_equal(population.state.individual_count, initial_count)

    object.__setattr__(
        population,
        "_config",
        population.config._replace(extreme_speed_mode=1),
    )
    population.run(0, finish=True)
    assert population.is_finished
    assert population.tick == 0


def test_discrete_import_commits_exact_state_and_clears_history() -> None:
    """A valid discrete import atomically replaces state and resets its timeline."""
    population = _build_discrete("discrete_import_success")
    population.record_snapshot()
    replacement = np.full_like(population.state.individual_count, 7.0)
    imported = population.state._replace(
        n_tick=9,
        individual_count=replacement,
    )

    population.import_state(imported)

    assert population.tick == 9
    assert population.history.ticks == ()
    np.testing.assert_array_equal(population.state.individual_count, replacement)
    np.testing.assert_array_equal(population.export_state(), imported.flatten_all())


def test_spatial_lifecycle_errors_clear_reset_and_finish_are_exact() -> None:
    """Spatial lifecycle boundaries preserve one synchronized typed timeline."""
    population = _build_spatial("spatial_lifecycle", history_mode="raw")
    population._running = True  # type: ignore[reportPrivateUsage]  # adversarial lifecycle state
    with pytest.raises(RuntimeError, match="running"):
        population.record_snapshot()
    population._running = False  # type: ignore[reportPrivateUsage]  # restore stable lifecycle state

    with pytest.raises(ValueError, match="No history"):
        population.restore_checkpoint(0)

    observed = _build_spatial(
        "spatial_observation_restore_error",
        history_mode="observation",
    )
    observed.record_snapshot()
    observed_rows = observed.history._to_numpy().copy()
    with pytest.raises(ValueError, match="observation-mode"):
        observed.restore_checkpoint(0)
    np.testing.assert_array_equal(observed.history._to_numpy(), observed_rows)

    population.record_snapshot()
    assert population.history.ticks == (0,)
    population.clear_history()
    assert population.history.ticks == ()

    population.record_snapshot()
    population.reset()
    assert population.history.ticks == ()
    assert population.tick == 0
    assert all(deme.tick == 0 for deme in population.demes)

    population.run(0, finish=True, clear_history_on_start=True)
    assert population.history.ticks == (0,)
    assert all(deme.is_finished for deme in population.demes)

    runnable = _build_spatial("spatial_negative_steps", history_mode="raw")
    initial_counts = np.stack(
        [deme.state.individual_count.copy() for deme in runnable.demes]
    )
    with pytest.raises(ValueError, match="n_steps must be >= 0"):
        runnable.run(-1)
    assert runnable.history.ticks == ()
    np.testing.assert_array_equal(
        np.stack([deme.state.individual_count for deme in runnable.demes]),
        initial_counts,
    )


def test_spatial_runtime_batch_updates_assign_exact_per_deme_values() -> None:
    """The public spatial updater expands batch values by deme."""
    population = _build_spatial("spatial_batch_update", history_mode="raw")

    population.update().competition(
        carrying_capacity=batch_setting([111.0, 222.0])
    )
    assert [float(deme.config.carrying_capacity) for deme in population.demes] == [
        111.0,
        222.0,
    ]

class _FakeUI:
    """Capture spatial dashboard downloads without starting NiceGUI."""

    def __init__(self) -> None:
        """Initialize empty download and notification logs."""
        self.downloads: list[bytes] = []
        self.notifications: list[str] = []

    def download(self, payload: bytes, *, filename: str, media_type: str) -> None:
        """Capture one exact JSON download.

        Args:
            payload: Encoded JSON bytes.
            filename: Requested download filename.
            media_type: Requested MIME type.
        """
        assert filename.endswith(".json")
        assert media_type == "application/json"
        self.downloads.append(payload)

    def notify(self, message: str, **_kwargs: str) -> None:
        """Capture an unexpected UI notification.

        Args:
            message: Notification text.
            **_kwargs: NiceGUI notification options.
        """
        self.notifications.append(message)


def test_spatial_dashboard_history_export_uses_public_translation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dashboard history export emits the exact typed spatial snapshot.

    Args:
        monkeypatch: Fixture replacing the NiceGUI boundary.
    """
    population = _build_spatial("spatial_dashboard_export", history_mode="raw")
    population.record_snapshot()
    dashboard = object.__new__(SpatialDashboard)
    dashboard.pop = population
    fake_ui = _FakeUI()
    monkeypatch.setattr(spatial_dashboard_module, "ui", fake_ui)

    dashboard._do_export_logic(  # type: ignore[reportPrivateUsage]  # UI action seam
        include_config=False,
        include_history=True,
        include_hooks=False,
    )

    payload = json.loads(fake_ui.downloads[0].decode("utf-8"))
    assert fake_ui.notifications == []
    assert payload["population_name"] == population.name
    assert payload["history"]["n_snapshots"] == 1
    assert payload["history"]["snapshots"][0]["tick"] == 0
