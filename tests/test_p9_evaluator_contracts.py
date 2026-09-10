"""Evaluator contracts for P9 (output selectors, derived layout, relocation).

Independent review tests for the P9 unification.  Two groups:

* **Repair targets** — executed failing regressions against confirmed
  requirements (legacy spellings the pre-P9 compiler accepted, the
  documented boundary exception types, and the web UI's error mapping
  convention).  Each docstring cites the requirement and the observed
  pre-fix failure.
* **Negative / preservation contracts** — absence tests for interfaces P9
  removed and pins for legacy semantics the new tests did not cover
  (numeric-string sex, compressed indices over slabs, mixed selector /
  dict mappings, sperm-storage row derivation).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict

import numpy as np
import pytest
from fastapi.testclient import TestClient

import natal as nt
from natal.frontend.output.history import (
    HistorySchema,
    ObservationMetadata,
    PopulationLayout,
    SpatialHistoryLayout,
)
from natal.frontend.output.observation import Observation, ObservationFilter
from natal.frontend.patterns import IndividualSelector
from natal.frontend.registry.index import IndexRegistry
from natal.frontend.utils.types import Sex
from natal.frontend.webui.app import create_app
from tests.test_webui_lifecycle import _build_age_population


def _species(name: str) -> nt.Species:
    """Two-allele unordered species: genotypes WT|WT, WT|A, A|A."""
    return nt.Species.from_dict(
        name=name, structure={"chr1": {"loc": ["WT", "A"]}}
    )


def _registry(name: str, slabs: tuple[str, ...] = ("default",)) -> IndexRegistry:
    """Registry over every genotype, each registered under every slab."""
    species = _species(name)
    registry = IndexRegistry()
    for genotype in species.iter_genotypes(unordered=species.unordered):
        for slab in slabs:
            registry.register_ztype(genotype, slab)
    return registry


def _mask(registry: IndexRegistry, groups: Any, n_ages: int = 3) -> np.ndarray:
    """Compile *groups* through the legacy boundary and return the 4-D mask."""
    observation = ObservationFilter(registry).build_filter(
        groups=groups, n_sexes=2, n_ages=n_ages, n_ztypes=registry.n_ztypes
    )
    return observation.build_mask(2, n_ages, registry.n_ztypes)


# ═════════════════════════════════════════════════════════════════════════════
# Repair targets (executed failing regressions)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {"groups": [{"genotype": ["WT::WT"], "age_start": 2, "age_end": 1}]},
            id="inverted-age-window",
        ),
        pytest.param(
            {"groups": [{"genotype": ["XX::XX"]}]},
            id="pattern-matches-nothing",
        ),
        pytest.param({"groups": []}, id="no-groups"),
        pytest.param(
            {"groups": [{"age_start": 7, "age_end": 9}]},
            id="age-window-outside-layout",
        ),
    ],
)
def test_webui_observation_user_errors_are_client_errors_not_500(
    body: Dict[str, Any],
) -> None:
    """User-input observation errors must not surface as HTTP 500.

    Requirement: P9 migrates the UI together with the unified selector
    boundary (plan "旧 observation 字典与 selector … UI 和 translation 一起
    迁移").  The boundary now raises ``ValueError`` for selections that
    match nothing; ``rest.py`` already maps input ``ValueError`` to 422
    (see ``_int_query`` and ``_get_diff``).  Before P9 every one of these
    bodies returned 200 with zero rows; after P9 they raise an unhandled
    ``ValueError`` inside ``_post_observation`` and FastAPI answers 500.
    """
    app = create_app(_build_age_population("p9_eval_webui_errors"), title="p9")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/observation", json={**body, "collapse_age": True}
        )
    assert response.status_code == 422, response.text
    assert "detail" in response.json()


def test_string_age_selector_raises_type_error_not_recursion_error() -> None:
    """``_age_selector_values`` promises ``TypeError`` for unsupported types.

    A ``str`` is an ``Iterable`` of one-character strings, so the current
    boundary recurses forever (``RecursionError``) instead of raising the
    documented ``TypeError``.  The pre-P9 compiler treated ``"2"`` as a
    (never-matching) literal; either the documented ``TypeError`` or a
    ``ValueError`` is acceptable — an interpreter recursion limit is not.
    """
    registry = _registry("p9_eval_age_str")
    with pytest.raises((TypeError, ValueError)):
        ObservationFilter(registry).build_filter(
            groups={"bad": {"age": "2"}}, n_sexes=2, n_ages=3, n_ztypes=3
        )


@pytest.mark.parametrize(
    ("age_spec", "expected_ages"),
    [
        pytest.param(np.arange(0, 2), [0, 1], id="ndarray"),
        pytest.param([np.int64(1)], [1], id="list-of-numpy-int"),
        pytest.param((np.int32(0), np.int32(2)), [0, 1, 2], id="numpy-int-pair"),
    ],
)
def test_numpy_integer_ages_keep_legacy_acceptance(
    age_spec: Any, expected_ages: list[int]
) -> None:
    """NumPy integer ages were accepted by the pre-P9 compiler.

    ``_make_age_predicate`` iterated any iterable and compared with ``==``,
    so ``np.arange``/``np.int64`` ages selected the expected classes.  The
    unified boundary tests ``isinstance(value, int)``, which NumPy integer
    scalars fail, and raises ``TypeError`` — a regression for the common
    scientific spelling ``{"age": np.arange(a, b)}``.
    """
    registry = _registry("p9_eval_np_ages")
    mask = _mask(registry, {"g": {"age": age_spec}})
    selected = sorted(int(a) for a in np.nonzero(mask[0, 0, :, 0])[0])
    assert selected == expected_ages


def test_genotype_object_entries_keep_legacy_acceptance() -> None:
    """``Genotype`` objects in a legacy genotype list resolved via ``str()``.

    The pre-P9 compiler parsed ``str(entry)`` for every non-int, non-``"*"``
    entry, so ``{"genotype": [registry.index_to_genotype[i]]}`` selected the
    slabs of that genotype.  The unified boundary only duck-types objects
    exposing ``.genotype`` (patterns) and raises ``TypeError`` for a
    ``Genotype`` instance.
    """
    registry = _registry("p9_eval_geno_obj", slabs=("default", "infected"))
    genotype = registry.index_to_genotype[1]
    by_object = _mask(registry, {"g": {"genotype": [genotype]}})
    by_pattern = _mask(registry, {"g": {"genotype": [str(genotype)]}})
    np.testing.assert_array_equal(by_object, by_pattern)


def test_dashboard_inverted_age_window_does_not_select_every_age() -> None:
    """An inverted panel age window must not silently become "all ages".

    ``ObservationPanel._selector_from_panel_state`` builds
    ``range(start, end + 1)``; for ``start > end`` the range is empty and
    ``IndividualSelector`` treats an empty age tuple as a wildcard, so the
    group silently counts the whole population.  Pre-P9 the same panel
    state produced an empty group; the unified boundary raises for the
    equivalent dict spelling.  Either raising or selecting nothing is
    acceptable; compiling to the full-population mask is not.
    """
    from natal.frontend.ui.dashboard_helpers import ObservationPanel

    registry = _registry("p9_eval_dashboard_inverted")
    panel = object.__new__(ObservationPanel)
    with pytest.raises(ValueError):
        selector = panel._selector_from_panel_state(  # pyright: ignore[reportPrivateUsage]  # migrated UI seam under review
            {"genotype": None, "sex": "both", "age_start": 2, "age_end": 1}
        )
        ObservationFilter(registry).build_mask_from_selectors(
            n_sexes=2, n_ages=3, n_ztypes=registry.n_ztypes,
            selectors=(selector,), collapse_age=False,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Negative contracts: interfaces retired by P9
# ═════════════════════════════════════════════════════════════════════════════


def test_spatial_history_layout_left_the_public_surface() -> None:
    """``SpatialHistoryLayout`` is no longer exported (已定稿取舍 #4).

    It stays an internal derived type on ``natal.frontend.output.history``.
    """
    import natal.frontend.output as output_pkg

    assert not hasattr(nt, "SpatialHistoryLayout")
    assert "SpatialHistoryLayout" not in nt.__all__
    assert "SpatialHistoryLayout" not in output_pkg.__all__
    assert not hasattr(output_pkg, "SpatialHistoryLayout")
    assert "SpatialHistoryLayout" not in nt._lazy_map  # pyright: ignore[reportPrivateUsage]  # export index under review


def test_legacy_spec_compiler_surface_is_gone() -> None:
    """The dict-based mask compiler and its storage no longer exist.

    Only the selector compiler remains; ``Observation`` carries selectors,
    not a parallel ``specs`` tuple.
    """
    assert not hasattr(ObservationFilter, "build_mask_from_specs")
    assert not hasattr(ObservationFilter, "_normalize_group_specs")
    assert not hasattr(ObservationFilter, "_make_age_predicate")
    assert not hasattr(ObservationFilter, "_resolve_sexes")
    field_names = {field.name for field in dataclasses.fields(Observation)}
    assert "specs" not in field_names
    assert "_selectors" in field_names


def test_derived_schema_fields_reject_stored_spellings() -> None:
    """Derived fields are not constructor inputs any more.

    ``HistorySchema.spatial_layout`` and ``ObservationMetadata.n_groups``
    are read-only derivations; the pre-P9 keyword spellings are rejected
    rather than silently accepted and ignored.  No persisted schema format
    exists (schemas are rebuilt from the live population), so this is a
    constructor contract, not an on-disk migration.
    """
    layout = PopulationLayout(
        kind="spatial_age_structured", n_demes=2, n_sexes=2, n_ages=3,
        n_ztypes=4, has_sperm_storage=True, sex_labels=("female", "male"),
        ztype_labels=tuple(f"z{i}" for i in range(4)),
    )
    with pytest.raises(TypeError):
        HistorySchema(  # type: ignore[call-arg]  # negative contract
            mode="raw", population=layout, row_size=1,
            spatial_layout=SpatialHistoryLayout(2, 24, 48),
        )
    with pytest.raises(TypeError):
        ObservationMetadata(labels=("g",), collapse_age=False, n_groups=1)  # type: ignore[call-arg]  # negative contract


# ═════════════════════════════════════════════════════════════════════════════
# Preservation: legacy semantics the P9 tests did not pin
# ═════════════════════════════════════════════════════════════════════════════


def test_numeric_string_sex_keeps_integer_semantics() -> None:
    """``"0"``/``"1"`` sex strings select by index exactly like ints.

    The legacy compiler fell back to ``int(value)`` for non-label strings;
    the boundary must keep that, and label spellings must stay
    case-insensitive with the single-letter aliases.
    """
    registry = _registry("p9_eval_sex_numeric")
    by_index_str = _mask(registry, {"g": {"sex": "1"}})
    by_int = _mask(registry, {"g": {"sex": 1}})
    by_enum = _mask(registry, {"g": {"sex": Sex.MALE}})
    by_label = _mask(registry, {"g": {"sex": "male"}})
    np.testing.assert_array_equal(by_index_str, by_int)
    np.testing.assert_array_equal(by_index_str, by_enum)
    np.testing.assert_array_equal(by_index_str, by_label)
    assert by_index_str[0, int(Sex.MALE)].all()
    assert not by_index_str[0, int(Sex.FEMALE)].any()

    female_zero = _mask(registry, {"g": {"sex": "0"}})
    assert female_zero[0, int(Sex.FEMALE)].all()
    assert not female_zero[0, int(Sex.MALE)].any()

    for spelling in ("Male", "MALE", "M", "m"):
        np.testing.assert_array_equal(_mask(registry, {"g": {"sex": spelling}}), by_label)
    both = _mask(registry, {"g": {"sex": ["f", 1]}})
    assert both[0].all()


def test_compressed_genotype_index_spans_every_slab() -> None:
    """``genotype: [i]`` selects all slab columns of registry genotype *i*.

    With two slabs each genotype owns two ZType columns; the index
    spelling must select exactly those two, match the bare pattern string,
    and differ from the slab-pinned pattern that selects one column.
    """
    registry = _registry("p9_eval_slabs", slabs=("default", "infected"))
    assert registry.n_ztypes == 6
    target = registry.index_to_genotype[1]
    expected_columns = sorted(
        position
        for position, (genotype, _slab) in enumerate(registry.index_to_ztype)
        if genotype == target
    )
    assert len(expected_columns) == 2

    by_index = _mask(registry, {"g": {"genotype": [1]}})
    selected = sorted(int(z) for z in np.nonzero(by_index[0, 0, 0, :])[0])
    assert selected == expected_columns
    np.testing.assert_array_equal(
        by_index, _mask(registry, {"g": {"genotype": [str(target)]}})
    )

    pinned = _mask(registry, {"g": {"genotype": [f"{target}@infected"]}})
    pinned_columns = [int(z) for z in np.nonzero(pinned[0, 0, 0, :])[0]]
    assert len(pinned_columns) == 1
    assert registry.index_to_ztype[pinned_columns[0]] == (target, "infected")

    # ``"*"`` anywhere in the list keeps its legacy short-circuit meaning.
    wildcard = _mask(registry, {"g": {"genotype": [str(target), "*"]}})
    assert wildcard[0].all()


def test_mixed_selector_and_dict_mapping_dispatches_per_value() -> None:
    """A mapping may hold selectors and legacy dicts side by side.

    ``GroupsInput`` now admits ``Mapping[str, IndividualSelector]``; the
    per-value dispatch must pass selector values through by identity and
    normalize dict values, without confusing one for the other.
    """
    registry = _registry("p9_eval_mixed")
    selector = IndividualSelector(ztype=str(registry.index_to_genotype[0]))
    observation = ObservationFilter(registry).build_filter(
        groups={"sel": selector, "dict": {"genotype": [0]}},
        n_sexes=2, n_ages=3, n_ztypes=registry.n_ztypes,
    )
    stored = observation._selectors  # pyright: ignore[reportPrivateUsage]  # boundary storage under review
    assert stored is not None
    assert stored[0] is selector
    assert stored[1] is not selector
    mask = observation.build_mask(2, 3, registry.n_ztypes)
    np.testing.assert_array_equal(mask[0], mask[1])
    assert observation.labels == ("sel", "dict")

    as_list = ObservationFilter(registry).build_filter(
        groups=[selector, {"genotype": [0]}],
        n_sexes=2, n_ages=3, n_ztypes=registry.n_ztypes,
    )
    assert as_list.labels == ("group_0", "group_1")
    np.testing.assert_array_equal(as_list.build_mask(2, 3, registry.n_ztypes), mask)


def test_spatial_layout_derivation_includes_sperm_storage() -> None:
    """``sperm_per_deme`` derives as ``n_ages * n_ztypes²`` when stored.

    The moved P9 test only exercised ``has_sperm_storage=False``; the raw
    row width the History reads back must agree with the derived layout.
    """
    layout = PopulationLayout(
        kind="spatial_age_structured", n_demes=3, n_sexes=2, n_ages=4,
        n_ztypes=5, has_sperm_storage=True, sex_labels=("female", "male"),
        ztype_labels=tuple(f"z{i}" for i in range(5)),
    )
    derived = SpatialHistoryLayout.from_population(layout)
    assert derived == SpatialHistoryLayout(n_demes=3, ind_per_deme=40, sperm_per_deme=100)
    schema = HistorySchema(
        mode="raw", population=layout,
        row_size=1 + 3 * (derived.ind_per_deme + derived.sperm_per_deme),
    )
    assert schema.spatial_layout == derived
    # Observation-mode schemas over a spatial layout never expose a raw layout.
    observed = HistorySchema(
        mode="observation", population=layout, row_size=1 + 2 * 3 * 2,
        observation=ObservationMetadata(
            labels=("a", "b"), collapse_age=True,
            deme_indices=(0, 1, 2), deme_mode="preserve",
        ),
    )
    assert observed.spatial_layout is None
    assert observed.observation is not None
    assert observed.observation.n_groups == 2


# ═════════════════════════════════════════════════════════════════════════════
# Boundary error contracts (documented ``Raises`` of the normalization helpers)
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("groups", "match"),
    [
        pytest.param({"g": {"genotype": [True]}}, "genotype selector", id="bool-genotype"),
        pytest.param({"g": {"sex": 1.5}}, "sex selector", id="float-sex"),
        pytest.param({"g": {"age": True}}, "age selector", id="bool-age"),
        pytest.param({"g": {"age": object()}}, "age selector", id="opaque-age"),
        pytest.param({"g": 7}, "group entry", id="int-group-entry"),
        pytest.param(7, "groups input", id="int-groups"),
    ],
)
def test_unsupported_spellings_raise_type_error_at_the_boundary(
    groups: Any, match: str
) -> None:
    """Each documented ``TypeError`` branch of the boundary is reachable.

    Booleans are rejected on the genotype and age axes (they would silently
    alias index 0/1), non-iterable non-label sex values, opaque age objects,
    non-mapping group entries, and a non-sequence non-mapping groups input
    all fail loudly with ``TypeError`` rather than a downstream
    ``AttributeError`` or a silent wildcard.
    """
    registry = _registry("p9_eval_type_errors")
    with pytest.raises(TypeError, match=match):
        ObservationFilter(registry).build_filter(
            groups=groups, n_sexes=2, n_ages=3, n_ztypes=registry.n_ztypes
        )


def test_observation_without_selectors_cannot_rebuild_mask() -> None:
    """A mask-less observation needs stored selectors to rebuild.

    ``specs`` no longer exists as a fallback source, so an observation
    constructed without ``_selectors`` reports the missing representation
    instead of silently compiling an empty mask.
    """
    registry = _registry("p9_eval_no_selectors")
    orphan = Observation(labels=("g",), collapse_age=False, _registry=registry)
    with pytest.raises(ValueError, match="no selectors stored"):
        orphan.build_mask(2, 3, registry.n_ztypes)
    detached = Observation(labels=("g",), collapse_age=False)
    with pytest.raises(ValueError, match="no registry reference"):
        detached.build_mask(2, 3, registry.n_ztypes)


# ═════════════════════════════════════════════════════════════════════════════
# Dashboard migration: panel state → unified selectors
# ═════════════════════════════════════════════════════════════════════════════


class _FakeUI:
    """Minimal stand-in for ``nicegui.ui`` and its element containers."""

    def __init__(self) -> None:
        self.labels: list[str] = []
        self.cleared = 0

    def __getattr__(self, _name: str) -> _FakeUI:
        return self

    def __call__(self, *args: object, **kwargs: object) -> _FakeUI:
        if args and isinstance(args[0], str):
            self.labels.append(args[0])
        return self

    def __enter__(self) -> _FakeUI:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def clear(self) -> None:
        self.cleared += 1


def test_dashboard_panel_patterns_or_combine_like_the_legacy_dict() -> None:
    """Two panel genotype patterns with one sex/age window ≡ legacy dict.

    The panel builds one selector per pattern and unions them; the result
    must equal the legacy ``{"genotype": [p1, p2], "sex": ..., "age": (a, b)}``
    spelling compiled through the boundary, so dashboard results are
    unchanged for valid panel state.
    """
    from natal.frontend.ui.dashboard_helpers import ObservationPanel

    registry = _registry("p9_eval_dashboard_merge")
    panel = object.__new__(ObservationPanel)
    selector = panel._selector_from_panel_state(  # pyright: ignore[reportPrivateUsage]  # migrated UI seam under review
        {"genotype": ["WT|WT", "A|A"], "sex": "female", "age_start": 1, "age_end": 2}
    )
    assert selector.n_atoms == 2
    panel_mask = ObservationFilter(registry).build_mask_from_selectors(
        n_sexes=2, n_ages=3, n_ztypes=registry.n_ztypes,
        selectors=(selector,), collapse_age=False,
    )
    legacy_mask = _mask(
        registry,
        {"g": {"genotype": ["WT|WT", "A|A"], "sex": "female", "age": (1, 2)}},
    )
    np.testing.assert_array_equal(panel_mask, legacy_mask)
    assert panel_mask[0, int(Sex.FEMALE), 1:3].sum() == 2 * 2
    assert not panel_mask[0, int(Sex.MALE)].any()


def test_dashboard_apply_compiles_panel_groups_into_an_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ObservationPanel._apply`` stores a selector-compiled observation.

    The migrated apply path must produce ``group_{i}`` labels and a mask
    identical to the legacy dict spelling for the same panel state, and
    must render an error label (not raise) when a group matches nothing.
    """
    from natal.frontend.ui import dashboard_helpers

    registry = _registry("p9_eval_dashboard_apply")
    fake_ui = _FakeUI()
    rendered: list[tuple[Observation, object]] = []
    monkeypatch.setattr(dashboard_helpers, "ui", fake_ui)

    def _record_render(obs: Observation, state: object) -> None:
        # The real renderer projects the state, which compiles the lazily
        # built mask; mirror that so compile-time errors surface in _apply.
        obs.build_mask(2, 3, registry.n_ztypes)
        rendered.append((obs, state))

    monkeypatch.setattr(dashboard_helpers, "render_observation_results", _record_render)

    panel = object.__new__(dashboard_helpers.ObservationPanel)
    panel._results_container = fake_ui  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]  # test double for the NiceGUI column
    panel._collapse_age = None  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    panel._get_registry = lambda: registry  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    panel._get_state = lambda: "state-token"  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    panel._observation = None  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    panel._group_specs = [  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
        {"genotype": ["A|A"], "sex": "male"},
        {"genotype": None, "sex": "both", "age_start": 0, "age_end": 1},
    ]

    panel._apply()  # pyright: ignore[reportPrivateUsage]

    observation = panel._observation  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    assert observation is not None
    assert observation.labels == ("group_0", "group_1")
    assert rendered and rendered[0][0] is observation and rendered[0][1] == "state-token"
    legacy = ObservationFilter(registry).build_filter(
        groups={
            "group_0": {"genotype": ["A|A"], "sex": "male"},
            "group_1": {"age": (0, 1)},
        },
        n_sexes=2, n_ages=3, n_ztypes=registry.n_ztypes,
    )
    np.testing.assert_array_equal(
        observation.build_mask(2, 3, registry.n_ztypes),
        legacy.build_mask(2, 3, registry.n_ztypes),
    )

    # A no-match group is reported as an error label, never as an exception.
    panel._group_specs = [{"genotype": ["A|A|A"], "sex": "both"}]  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    panel._apply()  # pyright: ignore[reportPrivateUsage]
    assert any(label.startswith("Error:") for label in fake_ui.labels)

    panel._group_specs = []  # pyright: ignore[reportPrivateUsage,reportAttributeAccessIssue]
    panel._apply()  # pyright: ignore[reportPrivateUsage]
    assert "No observation groups defined." in fake_ui.labels


def test_blueprint_view_stays_read_only() -> None:
    """The dataclass conversion keeps ``BlueprintView`` immutable.

    P9 replaced the hand-written getters with a frozen dataclass; the
    read-only contract (assignment raises ``AttributeError``) and the field
    surface must be unchanged.
    """
    from natal.frontend.hooks.tick_context import BlueprintView

    view = BlueprintView(
        n_sexes=2, n_ages=3, n_ztypes=4, discrete=False, stochastic=True,
        continuous_sampling=False, extreme_speed_mode=0,
        ztype_names=("a", "b", "c", "d"), gtype_names=("x", "y"),
    )
    with pytest.raises(AttributeError):
        view.n_sexes = 3  # type: ignore[misc]  # read-only contract
    assert {f.name for f in dataclasses.fields(view)} == {
        "n_sexes", "n_ages", "n_ztypes", "discrete", "stochastic",
        "continuous_sampling", "extreme_speed_mode", "ztype_names", "gtype_names",
    }
