"""Spatial runtime modification after the stage-3 write-plane change.

The ``_SpatialUpdate`` configurator chain (``pop.update()`` /
``pop.update_deme()``) is deleted.  Runtime modification of a spatial
population goes through the params data plane instead:

- ``pop.deme(i)`` returns a :class:`DemeSlice` compat view; reads delegate
  to the underlying deme (``.config`` / ``.state`` / ``.registry`` deep
  interfaces stay fully compatible for the UI and tests).
- ``DemeSlice.write_ecology(field, value)`` writes the deme's ecology
  column entry AND its draft (per-field clone-on-write, equilibrium
  metrics re-synced) so every backend sees the same value.
- ``DemeSlice.write_genetics(field, values)`` forks the deme's genetics
  variant (Rust bank) and detaches the draft tables, so demes that shared
  the genetics keep their numerics bitwise unchanged.
- ``pop.params`` exposes the ecology columns: read-protected views and a
  validated whole-column ``tensor_write``.
- Inside hooks, the ``TickContext`` lends the same writable params
  surface (tested in the hook suites).

Each behavioral assertion below proves a numerical invariant (bitwise
array equality or an exact population-dynamics ordering), not an
implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.spatial.population import DemeSlice, SpatialPopulation


@pytest.fixture(scope="module")
def species():
    """Build the minimal species shared by the spatial write tests."""
    return nt.Species.from_dict(
        name="__test_spatial_write_plane__",
        structure={"auto": {"A": ["WT"]}},
    )


def _build_discrete(species, *, n_demes: int = 4, k: float = 500.0):
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=n_demes, topology=topo, pop_type="discrete_generation")
        .setup(name="write_plane", stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=k, low_density_growth_rate=6.0,
                     juvenile_growth_mode="beverton_holt")
        .build()
    )


def _build_two_allele_discrete(name: str):
    """A no-migration two-allele discrete spatial population (WT/Dr)."""
    species = nt.Species.from_dict(
        name="__test_spatial_meiosis_plane__",
        structure={"auto": {"A": ["WT", "Dr"]}},
    )
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=4, topology=nt.SquareGrid(2, 2),
                 pop_type="discrete_generation")
        .setup(name=name, stochastic=False)
        .initial_state(individual_count={
            "female": {"WT|WT": 100}, "male": {"WT|WT": 100},
        })
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0,
                     juvenile_growth_mode="beverton_holt")
        .build()
    )


def _build_age(species, *, n_demes: int = 4, k: float = 500.0):
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation
        .builder(species, n_demes=n_demes, topology=topo, pop_type="age_structured")
        .setup(name="write_plane_age", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(individual_count={
            "female": {"WT|WT": {1: 100}}, "male": {"WT|WT": {1: 100}},
        })
        .reproduction(eggs_per_female=10)
        .competition(carrying_capacity=k, low_density_growth_rate=6.0,
                     juvenile_growth_mode="beverton_holt")
        .build()
    )


@pytest.fixture
def homogeneous_pop(species):
    """2x2 homogeneous discrete spatial population."""
    return _build_discrete(species)


@pytest.fixture
def homogeneous_age_pop(species):
    """2x2 homogeneous age-structured spatial population."""
    return _build_age(species)


# ══════════════════════════════════════════════════════════════════════════
# Negative contracts: the update() configurator chain is gone
# ══════════════════════════════════════════════════════════════════════════


class TestSpatialUpdateRemoved:
    """The deleted runtime configurator chain must stay inaccessible."""

    def test_spatial_population_update_removed(self, homogeneous_pop) -> None:
        """SpatialPopulation.update must not exist."""
        assert not hasattr(homogeneous_pop, "update"), (
            "SpatialPopulation.update must be deleted; runtime writes go "
            "through pop.params / deme(i) slices"
        )

    def test_spatial_population_update_deme_removed(self, homogeneous_pop) -> None:
        """SpatialPopulation.update_deme must not exist."""
        assert not hasattr(homogeneous_pop, "update_deme"), (
            "SpatialPopulation.update_deme must be deleted"
        )

    def test_spatial_update_class_removed(self) -> None:
        """The _SpatialUpdate wrapper must not be importable."""
        import natal.frontend.spatial.population as spatial_population

        assert not hasattr(spatial_population, "_SpatialUpdate")

    def test_spatial_configurator_batchable_constant_removed(
        self, homogeneous_pop,
    ) -> None:
        """Runtime dispatch must not depend on a method-name allowlist."""
        updater = homogeneous_pop.deme(0)
        assert not hasattr(updater, "_BATCHABLE_METHODS")

    def test_detach_fields_constant_removed(self) -> None:
        """The per-field detach constant is private to the deleted chain."""
        import natal.frontend.spatial.population as spatial_population

        assert not hasattr(spatial_population, "_DETACH_FIELDS")

    def test_deme_slice_is_the_deme_type(self, homogeneous_pop) -> None:
        """deme(i) hands out the compat view, not the raw deme object."""
        assert isinstance(homogeneous_pop.deme(0), DemeSlice)


# ══════════════════════════════════════════════════════════════════════════
# DemeSlice read compatibility (UI / test deep-interface audit)
# ══════════════════════════════════════════════════════════════════════════


class TestDemeSliceReadCompat:
    """Reads through deme(i) must behave exactly like the raw deme."""

    def test_deep_interfaces_delegate(self, homogeneous_pop) -> None:
        """config/state/registry/name hooks expose the deme's own objects."""
        pop = homogeneous_pop
        deme0 = pop.deme(0)
        assert deme0.config is pop._demes[0].config  # pyright: ignore[reportPrivateUsage]  # compat contract: same object
        assert deme0.state is pop._demes[0].state  # pyright: ignore[reportPrivateUsage]  # compat contract
        assert deme0.name == pop._demes[0].name  # pyright: ignore[reportPrivateUsage]  # compat contract
        assert deme0.registry is pop._demes[0].registry  # pyright: ignore[reportPrivateUsage]  # compat contract (UI reads)
        assert deme0.export_config().n_ages == pop._demes[0].export_config().n_ages  # pyright: ignore[reportPrivateUsage]  # compat contract

    def test_state_writes_through_slice_are_live(
        self, homogeneous_pop,
    ) -> None:
        """Counts written through the slice land on the live state array."""
        pop = homogeneous_pop
        pop.deme(2).state.individual_count[0, 0, 0] = 77.0
        assert float(pop._demes[2].state.individual_count[0, 0, 0]) == 77.0  # pyright: ignore[reportPrivateUsage]  # compat contract

    def test_attribute_writes_forward_to_deme(self, homogeneous_pop) -> None:
        """Attribute assignment through the slice reaches the deme."""
        pop = homogeneous_pop
        marker = object()
        pop.deme(1).custom_marker = marker  # type: ignore[attr-defined]  # dynamic delegation contract
        assert pop._demes[1].custom_marker is marker  # pyright: ignore[reportAttributeAccessIssue]  # dynamic delegation contract

    def test_demes_property_yields_slices(self, homogeneous_pop) -> None:
        """The demes sequence holds one slice per deme in order."""
        slices = homogeneous_pop.demes
        assert len(slices) == 4
        assert [s.index for s in slices] == [0, 1, 2, 3]
        # Same underlying config objects as the raw demes (sharing contract).
        assert slices[3].config is homogeneous_pop._demes[3].config  # pyright: ignore[reportPrivateUsage]  # compat contract


# ══════════════════════════════════════════════════════════════════════════
# DemeSlice ecology write path
# ══════════════════════════════════════════════════════════════════════════


class TestDemeSliceEcologyWrite:
    """write_ecology updates column, draft, and derived metrics atomically."""

    def test_deme0_k_write_does_not_affect_deme1(self, homogeneous_pop) -> None:
        """ deme 0 gets a private K; deme 1's array stays untouched."""
        pop = homogeneous_pop
        k_before = pop.deme(0).config.carrying_capacity
        pop.deme(0).write_ecology("carrying_capacity", 999.0)
        # deme 1 keeps the shared array and the original value.
        assert pop.deme(1).config.carrying_capacity is k_before
        assert float(pop.deme(1).config.carrying_capacity) == 500.0
        # deme 0 has a private array with the new value.
        assert pop.deme(0).config.carrying_capacity is not k_before
        assert float(pop.deme(0).config.carrying_capacity) == 999.0

    def test_k_write_resyncs_equilibrium_metrics(self, homogeneous_pop) -> None:
        """A sensitive write recomputes the equilibrium metric caches."""
        pop = homogeneous_pop
        before = float(pop.deme(0).config.expected_competition_strength)
        pop.deme(0).write_ecology("carrying_capacity", 100.0)
        after = float(pop.deme(0).config.expected_competition_strength)
        assert after != before, "equilibrium metric must follow the K write"

    def test_vector_write_lands_per_deme(self, homogeneous_age_pop) -> None:
        """A survival vector write touches only the target deme."""
        pop = homogeneous_age_pop
        saved = [
            pop.deme(i).config.age_based_survival_rates.copy() for i in range(4)
        ]
        pop.deme(1).write_ecology("survival_rates", np.array([[0.5, 0.6], [0.5, 0.6]]))
        np.testing.assert_array_equal(
            pop.deme(1).config.age_based_survival_rates,
            [[0.5, 0.6], [0.5, 0.6]],
        )
        for i in (0, 2, 3):
            np.testing.assert_array_equal(
                pop.deme(i).config.age_based_survival_rates, saved[i],
                err_msg=f"deme {i} survival leaked from a deme-1 write",
            )

    def test_ecology_column_read_view(self, homogeneous_pop) -> None:
        """pop.params exposes the ecology column as a protected view."""
        pop = homogeneous_pop
        column = pop.params.carrying_capacity
        assert column.shape == (4,)
        assert column.tolist() == [500.0, 500.0, 500.0, 500.0]
        with pytest.raises(ValueError):
            column[0] = 1.0  # type: ignore[index]  # write-protection contract

    def test_whole_column_tensor_write_broadcasts(
        self, homogeneous_pop,
    ) -> None:
        """A scalar column write broadcasts and syncs every deme draft."""
        pop = homogeneous_pop
        pop.params.tensor_write("carrying_capacity", 250.0)
        assert pop.params.carrying_capacity.tolist() == [250.0] * 4
        for i in range(4):
            assert float(pop.deme(i).config.carrying_capacity) == 250.0

    def test_k_write_changes_trajectory_only_at_target(
        self, homogeneous_pop,
    ) -> None:
        """A K drop shrinks only the written deme after a run."""
        pop = homogeneous_pop
        baseline = _build_discrete(pop.species)
        pop.deme(1).write_ecology("carrying_capacity", 10.0)
        baseline.run(4, record_every=0)
        pop.run(4, record_every=0)
        assert pop.deme(0).get_total_count() > 5 * pop.deme(1).get_total_count()
        # Untouched demes track the baseline bitwise.
        for i in (0, 2, 3):
            np.testing.assert_array_equal(
                pop.deme(i).state.individual_count,
                baseline.deme(i).state.individual_count,
            )

    def test_unknown_ecology_field_raises(self, homogeneous_pop) -> None:
        """Unknown field names fail without any write."""
        pop = homogeneous_pop
        with pytest.raises(KeyError):
            pop.deme(0).write_ecology("no_such_field", 1.0)


# ══════════════════════════════════════════════════════════════════════════
# DemeSlice genetics fork write path
# ══════════════════════════════════════════════════════════════════════════


class TestDemeSliceGeneticsFork:
    """Genetics writes fork the deme's tables; sharing demes are isolated."""

    def test_fitness_replace_isolated_to_target(self, homogeneous_pop) -> None:
        """fitness on deme 0 leaves demes 1-3 bitwise unchanged."""
        pop = homogeneous_pop
        saved = [
            pop.deme(i).config.viability_fitness.copy() for i in range(4)
        ]
        pop.deme(0).write_genetics(
            "viability_fitness", np.full_like(saved[0], 0.5)
        )
        assert float(pop.deme(0).config.viability_fitness[0, 0, 0]) == 0.5
        for i in range(1, 4):
            np.testing.assert_array_equal(
                pop.deme(i).config.viability_fitness, saved[i],
                err_msg=f"deme {i} viability changed by a deme-0 genetics write",
            )

    def test_forked_deme_trajectory_diverges(self, homogeneous_pop) -> None:
        """A 0.1 viability fork drops only the written deme after a run."""
        pop = homogeneous_pop
        baseline = _build_discrete(pop.species)
        pop.deme(2).write_genetics(
            "viability_fitness",
            np.full_like(pop.deme(2).config.viability_fitness, 0.1),
        )
        baseline.run(3, record_every=0)
        pop.run(3, record_every=0)
        assert pop.deme(2).get_total_count() < baseline.deme(2).get_total_count()
        for i in (0, 1, 3):
            np.testing.assert_array_equal(
                pop.deme(i).state.individual_count,
                baseline.deme(i).state.individual_count,
            )

    def test_unknown_genetics_field_raises(self, homogeneous_pop) -> None:
        """Unknown tensor names fail without any write."""
        pop = homogeneous_pop
        with pytest.raises(KeyError):
            pop.deme(0).write_genetics("survival_rates", np.zeros(1))

    def test_meiosis_fork_recomputes_derived_tensor_and_isolates(self) -> None:
        """A meiosis fork recomputes the derived offspring tensor (C3 fix).

        Forcing WT|WT individuals to transmit only ``Dr`` gametes must
        recompute the deme's offspring tensor — the engines only consume
        the derived table — while demes still sharing the original
        tables stay bitwise unchanged, and the biased deme's next
        generation is entirely Dr|Dr at the same total census (a meiosis
        bias redistributes genotypes; it does not change counts).
        """
        pop = _build_two_allele_discrete("meiosis_fork")
        baseline = _build_two_allele_discrete("meiosis_fork_base")

        biased = np.asarray(
            pop.deme(0).config.zygotes_to_gametes_map, dtype=np.float64
        ).copy()
        biased[:, 0, :] = [0.0, 1.0]
        pop.deme(0).write_genetics("meiosis_map", biased)

        # The fork's derived tensor follows the biased meiosis...
        np.testing.assert_allclose(
            pop.deme(0).config.offspring_tensor[0, 0, :], [0.0, 0.0, 1.0]
        )
        # ...while sharing demes keep the Mendelian original bitwise.
        np.testing.assert_allclose(
            pop.deme(1).config.offspring_tensor[0, 0, :], [1.0, 0.0, 0.0]
        )

        pop.run(1, record_every=0)
        baseline.run(1, record_every=0)

        biased_counts = pop.deme(0).state.individual_count
        np.testing.assert_allclose(biased_counts[:, 1, 0], 0.0)  # WT|WT gone
        np.testing.assert_allclose(biased_counts[:, 1, 1], 0.0)  # WT|Dr gone
        assert biased_counts[:, 1, 2].min() > 0.0  # all mass on Dr|Dr
        # Same total census as the unbiased baseline deme.
        np.testing.assert_allclose(
            biased_counts.sum(), baseline.deme(0).state.individual_count.sum()
        )
        # Untouched demes track the baseline bitwise.
        for i in (1, 2, 3):
            np.testing.assert_array_equal(
                pop.deme(i).state.individual_count,
                baseline.deme(i).state.individual_count,
            )

    def test_meiosis_fork_rejects_invalid_tables_atomically(self) -> None:
        """Non-distribution meiosis rows are refused on the fork channel too."""
        pop = _build_two_allele_discrete("meiosis_fork_bad")
        saved = np.asarray(pop.deme(0).config.zygotes_to_gametes_map).copy()
        saved_offspring = np.asarray(pop.deme(0).config.offspring_tensor).copy()

        wrong_sum = saved.copy()
        wrong_sum[0, 1, :] = [0.6, 0.6]
        with pytest.raises(ValueError, match="must be probability distributions"):
            pop.deme(0).write_genetics("meiosis_map", wrong_sum)

        negative = saved.copy()
        negative[0, 1, :] = [-0.5, 1.5]
        with pytest.raises(ValueError, match="must be non-negative"):
            pop.deme(0).write_genetics("meiosis_map", negative)

        # Zero writes: both demes' tables are bitwise unchanged.
        np.testing.assert_array_equal(
            pop.deme(0).config.zygotes_to_gametes_map, saved
        )
        np.testing.assert_array_equal(
            pop.deme(0).config.offspring_tensor, saved_offspring
        )
        np.testing.assert_array_equal(
            pop.deme(1).config.zygotes_to_gametes_map, saved
        )

    def test_deme_params_tensor_write_refuses_genetics_fields(
        self, homogeneous_pop,
    ) -> None:
        """Per-deme ``params.tensor_write`` refuses every genetics tensor.

        Spatial demes start with shared draft tables; an in-place genetics
        write would leak into all other demes.  The refusal routes the
        caller to ``write_genetics`` (the forking channel); ecology
        vectors keep working through ``tensor_write``.
        """
        from natal.frontend.population._params_view import _GENETICS_TENSORS

        pop = homogeneous_pop
        for field in sorted(_GENETICS_TENSORS):
            with pytest.raises(RuntimeError, match="write_genetics"):
                pop.deme(0).params.tensor_write(
                    field, np.zeros(1, dtype=np.float64)
                )

        # Ecology vectors stay writable through the same surface.
        rates = np.asarray(pop.deme(0).params.survival_rates, dtype=np.float64)
        pop.deme(0).params.tensor_write("survival_rates", rates)
        np.testing.assert_array_equal(
            pop.deme(0).params.survival_rates, rates
        )


# ══════════════════════════════════════════════════════════════════════════
# Build semantics preserved by the merged single-entry build
# ══════════════════════════════════════════════════════════════════════════


class TestBuildSemanticsPreserved:
    """The single-entry build keeps the sharing and heterogeneity rules."""

    def test_shared_config_after_homogeneous_build(
        self, homogeneous_pop, homogeneous_age_pop,
    ) -> None:
        """All demes point to the same config object after a homogeneous build."""
        for pop in (homogeneous_pop, homogeneous_age_pop):
            config0 = pop.deme(0).config
            for i in range(1, 4):
                assert pop.deme(i).config is config0

    def test_homogeneous_pop_is_spatial_population(
        self, homogeneous_pop,
    ) -> None:
        """build() still returns a SpatialPopulation (single entry point)."""
        assert isinstance(homogeneous_pop, SpatialPopulation)


class TestVariantEquilibriumDeclaration:
    """Variant recompute must honor declared distributions/egg overrides.

    Regression: the variant rebuild used to drop
    ``equilibrium_individual_distribution`` and ``external_expected_eggs``
    from the recompute, silently switching heterogeneous demes back into
    derivation mode.

    Heterogeneity is driven by ``eggs_per_female`` on purpose: its batch
    parameter name is a draft field, so the builder takes the
    ``_build_variant_config`` fast path this class exercises.  (A batched
    ``carrying_capacity`` registers as ``age_1_carrying_capacity``, which
    is not a draft field — that configuration replays the whole template
    and computes its metrics through the sync path instead.)
    """

    def _eggs_heterogeneous_population(
        self, name: str, *, external_females: float | None = None
    ):
        """Two demes with eggs 10/20, a declared distribution, optional override.

        Declared distribution: 50 females + 50 males at age 1, both adult
        ages reproducing at rate 1 with fertility 1.

        Returns:
            The built population and the declared (2, 4) array.
        """
        species = nt.Species.from_dict(
            name="__test_spatial_eq_decl__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        declared = np.zeros((2, 4))
        declared[0, 1] = 50.0
        declared[1, 1] = 50.0
        competition_kwargs: dict[str, object] = {
            "carrying_capacity": 100.0,
            "juvenile_growth_mode": 3,
            "equilibrium_distribution": declared,
        }
        if external_females is not None:
            competition_kwargs["expected_num_new_adult_females"] = external_females
        return (
            nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
            .setup(name=name, stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 50, 0, 0]},
                    "male": {"A|A": [0, 50, 0, 0]},
                }
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                eggs_per_female=nt.batch_setting([10.0, 20.0]),
            )
            .survival(
                female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
                male_age_based_survival=[1.0, 0.9, 0.7, 0.0],
            )
            .competition(**competition_kwargs)
            .build()
        ), declared

    def test_variant_recompute_honors_declared_distribution(self) -> None:
        """A variant egg count keeps every deme on the declared distribution.

        Hand computation: 50 declared females at age 1, reproduction rate
        1, fertility 1 → produced eggs = 50 * 1 * 1 * eggs_per_female.
        Competition weight[0] = 1 → C* = produced.  s* = 100 / produced
        (s_0_avg = 1).  Deme 0 (eggs 10): (500, 0.2).  Deme 1 (eggs 20):
        (1000, 0.1).  The pre-fix variant recomputed in derivation mode
        and produced (2530, 0.0395...) on deme 1 instead.
        """
        pop, _declared = self._eggs_heterogeneous_population("eq_declared_variant")

        d0 = pop.demes[0].config
        d1 = pop.demes[1].config
        assert float(d0.eggs_per_female) == 10.0
        assert float(d1.eggs_per_female) == 20.0
        assert float(d0.expected_competition_strength) == 500.0
        assert float(d0.expected_survival_rate) == 0.2
        assert float(d1.expected_competition_strength) == 1000.0
        assert float(d1.expected_survival_rate) == 0.1

    def test_variant_external_eggs_drive_survival_not_competition(self) -> None:
        """The variant's survival rate uses the persisted egg override.

        With ``expected_num_new_adult_females=100`` the override is
        100 * eggs * (1 + 0.9 + 0.7 * 0.63...) — computed on the template
        — and only feeds s*: C* stays declared-driven (500 / 1000), s*
        becomes 100 / external per deme.
        """
        pop, _declared = self._eggs_heterogeneous_population(
            "eq_external_variant", external_females=100.0
        )
        ext0 = float(pop.demes[0].config.external_expected_eggs)
        ext1 = float(pop.demes[1].config.external_expected_eggs)
        assert ext0 == ext1  # the override is declared, not per-deme
        # External eggs only change s*, never the competition strength.
        assert float(pop.demes[0].config.expected_competition_strength) == 500.0
        assert float(pop.demes[1].config.expected_competition_strength) == 1000.0
        assert (
            float(pop.demes[0].config.expected_survival_rate)
            == 100.0 / ext0
        )
        assert (
            float(pop.demes[1].config.expected_survival_rate)
            == 100.0 / ext1
        )

    def test_variant_python_fallback_carries_the_two_params(self, monkeypatch) -> None:
        """The forced Python fallback recompute carries declared + external.

        The assertion is scoped to the *variant* call (eggs = 20), so a
        regression that drops the two kwargs again cannot hide behind the
        template/sync calls that legitimately carry them.
        """
        from natal.frontend.data import _engine as engine_module

        calls: list[dict[str, float | None]] = []

        def recording_dispatch(
            carrying_capacity: float,
            eggs_per_female: float,
            sex_ratio: float,
            survival_rates: object,
            reproduction_rates: object,
            fertility: object,
            competition_weights: object,
            new_adult_age: int,
            n_ages: int,
            declared_distribution: object,
            external_expected_eggs: float | None,
        ) -> None:
            calls.append({"eggs": eggs_per_female})
            return None  # force the python fallback everywhere

        monkeypatch.setattr(
            engine_module, "equilibrium_metrics_dispatch", recording_dispatch
        )
        pop, declared = self._eggs_heterogeneous_population(
            "eq_fallback_variant", external_females=100.0
        )

        variant_calls = [c for c in calls if c["eggs"] == 20.0]
        assert variant_calls, "the variant path never dispatched"
        # And the fallback results honor the declared distribution, i.e.
        # the python branch carried the two kwargs too.
        assert float(pop.demes[1].config.expected_competition_strength) == 1000.0
        assert (
            float(pop.demes[1].config.expected_survival_rate)
            == 100.0 / float(pop.demes[1].config.external_expected_eggs)
        )

    def test_variant_metrics_isolated_between_demes(self) -> None:
        """A runtime ecology write on deme 0 leaves deme 1's metrics alone."""
        pop, _declared = self._eggs_heterogeneous_population("eq_isolation_variant")
        pop.run(1, record_every=0)

        pop.demes[0].write_ecology("carrying_capacity", 999.0)

        assert float(pop.demes[1].config.expected_competition_strength) == 1000.0
        assert float(pop.demes[1].config.expected_survival_rate) == 0.1

    def test_bad_declared_shape_raises_value_error(self) -> None:
        """A declared distribution of the wrong shape fails loudly."""
        species = nt.Species.from_dict(
            name="__test_spatial_eq_badshape__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        with pytest.raises(ValueError, match=r"\(2, 4\)"):
            (
                nt.SpatialPopulation.builder(
                    species, n_demes=2, pop_type="age_structured"
                )
                .setup(name="eq_badshape", stochastic=False)
                .age_structure(n_ages=4, new_adult_age=1)
                .initial_state(
                    individual_count={
                        "female": {"A|A": [0, 50, 0, 0]},
                        "male": {"A|A": [0, 50, 0, 0]},
                    }
                )
                .reproduction(
                    female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                    male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                    eggs_per_female=nt.batch_setting([10.0, 20.0]),
                )
                .survival(
                    female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
                    male_age_based_survival=[1.0, 0.9, 0.7, 0.0],
                )
                .competition(
                    carrying_capacity=100.0,
                    juvenile_growth_mode=3,
                    equilibrium_distribution=np.zeros((2, 3)),
                )
                .build()
            )
