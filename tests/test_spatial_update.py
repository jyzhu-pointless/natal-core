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
from natal.frontend.data._engine import (
    derive_equilibrium_metrics_from_draft,
)
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
        nt.SpatialPopulation.builder(
            species, n_demes=n_demes, topology=topo, pop_type="discrete_generation"
        )
        .setup(name="write_plane", stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 100},
                "male": {"WT|WT": 100},
            }
        )
        .reproduction(eggs_per_female=10)
        .competition(
            carrying_capacity=k,
            low_density_growth_rate=6.0,
            juvenile_growth_mode="beverton_holt",
        )
        .build()
    )


def _build_two_allele_discrete(name: str):
    """A no-migration two-allele discrete spatial population (WT/Dr)."""
    species = nt.Species.from_dict(
        name="__test_spatial_meiosis_plane__",
        structure={"auto": {"A": ["WT", "Dr"]}},
    )
    return (
        nt.SpatialPopulation.builder(
            species,
            n_demes=4,
            topology=nt.SquareGrid(2, 2),
            pop_type="discrete_generation",
        )
        .setup(name=name, stochastic=False)
        .initial_state(
            individual_count={
                "female": {"WT|WT": 100},
                "male": {"WT|WT": 100},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(
            carrying_capacity=100000.0,
            low_density_growth_rate=2.0,
            juvenile_growth_mode="beverton_holt",
        )
        .build()
    )


def _build_age(species, *, n_demes: int = 4, k: float = 500.0):
    topo = nt.SquareGrid(2, 2)
    return (
        nt.SpatialPopulation.builder(
            species, n_demes=n_demes, topology=topo, pop_type="age_structured"
        )
        .setup(name="write_plane_age", stochastic=False)
        .age_structure(n_ages=2, new_adult_age=1)
        .initial_state(
            individual_count={
                "female": {"WT|WT": {1: 100}},
                "male": {"WT|WT": {1: 100}},
            }
        )
        .reproduction(eggs_per_female=10)
        .competition(
            carrying_capacity=k,
            low_density_growth_rate=6.0,
            juvenile_growth_mode="beverton_holt",
        )
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
        self,
        homogeneous_pop,
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
        np.testing.assert_array_equal(deme0.config.viability_fitness, pop._demes[0].config.viability_fitness)
        # DemeSlice.state returns an independent snapshot of the deme's
        # live container: equal by value, never the same object
        # or buffer (the public population-level state has been a
        # snapshot since R5; spatial slices now follow the same rule).
        snapshot = deme0.state
        live = pop._demes[0]._state  # pyright: ignore[reportPrivateUsage]  # engine truth
        assert snapshot is not live  # compat contract: snapshot, not the live container
        assert snapshot.n_tick == live.n_tick
        np.testing.assert_array_equal(snapshot.individual_count, live.individual_count)
        assert not np.may_share_memory(snapshot.individual_count, live.individual_count)
        assert deme0.name == pop._demes[0].name  # pyright: ignore[reportPrivateUsage]  # compat contract
        assert deme0.registry is pop._demes[0].registry  # pyright: ignore[reportPrivateUsage]  # compat contract (UI reads)
        assert deme0.export_config().n_ages == pop._demes[0].export_config().n_ages  # pyright: ignore[reportPrivateUsage]  # compat contract

    def test_state_reads_are_snapshots_and_scoped_transaction_writes(
        self,
        homogeneous_pop,
    ) -> None:
        """Slice reads hand out snapshots; scoped hook transactions write state.

        Inversion: a write through a retained ``deme.state``
        snapshot is inert (it cannot reach the deme's live array), while
        a scoped hook transaction commits its candidate to the live state
        and every subsequent read.
        """
        pop = homogeneous_pop
        live = pop._demes[2]._state  # pyright: ignore[reportPrivateUsage]  # engine truth
        snapshot = pop.deme(2).state
        # Snapshot read: independent copy, equal values.
        assert snapshot is not live
        assert not np.may_share_memory(snapshot.individual_count, live.individual_count)
        snapshot.individual_count[0, 0, 0] = 77.0
        assert (
            float(pop.deme(2).state.individual_count[0, 0, 0]) == 0.0
        )  # write through the snapshot is inert
        assert (
            float(live.individual_count[0, 0, 0]) == 0.0
        )  # the live array never moved
        # A scoped event writes state without creating a separate deme clock.
        changed = pop.deme(2).state.individual_count.copy()
        changed[0, 0, 0] = 77.0
        from tests.spatial_test_state import set_deme_state
        set_deme_state(pop, 2, {"n_tick": pop.tick, "individual_count": changed})
        assert float(pop._demes[2].state.individual_count[0, 0, 0]) == 77.0  # pyright: ignore[reportPrivateUsage]  # live write landed
        assert (
            float(pop.deme(2).state.individual_count[0, 0, 0]) == 77.0
        )  # reads see it

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
        np.testing.assert_array_equal(slices[3].config.viability_fitness, homogeneous_pop._demes[3].config.viability_fitness)


# ══════════════════════════════════════════════════════════════════════════
# DemeSlice ecology write path
# ══════════════════════════════════════════════════════════════════════════


class TestDemeSliceEcologyWrite:
    """write_ecology updates column, draft, and derived metrics atomically."""

    def test_deme0_k_write_does_not_affect_deme1(self, homogeneous_pop) -> None:
        """deme 0 gets a private K; deme 1's array stays untouched."""
        pop = homogeneous_pop
        k_before = pop.deme(0).config.carrying_capacity
        pop.deme(0).write_ecology("carrying_capacity", 999.0)
        # deme 1 keeps the shared array and the original value.
        assert pop.deme(1).config.carrying_capacity == k_before
        assert float(pop.deme(1).config.carrying_capacity) == 500.0
        # deme 0 has a private array with the new value.
        assert pop.deme(0).config.carrying_capacity != k_before
        assert float(pop.deme(0).config.carrying_capacity) == 999.0

    def test_k_write_resyncs_equilibrium_metrics(self, homogeneous_pop) -> None:
        """A sensitive write recomputes the equilibrium metric caches."""
        pop = homogeneous_pop
        before = derive_equilibrium_metrics_from_draft(pop.deme(0).config)[0]
        pop.deme(0).write_ecology("carrying_capacity", 100.0)
        after = derive_equilibrium_metrics_from_draft(pop.deme(0).config)[0]
        assert after != before, "equilibrium metric must follow the K write"

    def test_vector_write_lands_per_deme(self, homogeneous_age_pop) -> None:
        """A survival vector write touches only the target deme."""
        pop = homogeneous_age_pop
        saved = [pop.deme(i).config.age_based_survival_rates.copy() for i in range(4)]
        pop.deme(1).write_ecology("survival_rates", np.array([[0.5, 0.6], [0.5, 0.6]]))
        np.testing.assert_array_equal(
            pop.deme(1).config.age_based_survival_rates,
            [[0.5, 0.6], [0.5, 0.6]],
        )
        for i in (0, 2, 3):
            np.testing.assert_array_equal(
                pop.deme(i).config.age_based_survival_rates,
                saved[i],
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
        self,
        homogeneous_pop,
    ) -> None:
        """A scalar column write broadcasts and syncs every deme draft."""
        pop = homogeneous_pop
        pop.params.tensor_write("carrying_capacity", 250.0)
        assert pop.params.carrying_capacity.tolist() == [250.0] * 4
        for i in range(4):
            assert float(pop.deme(i).config.carrying_capacity) == 250.0

    def test_k_write_changes_trajectory_only_at_target(
        self,
        homogeneous_pop,
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
        saved = [pop.deme(i).config.viability_fitness.copy() for i in range(4)]
        pop.deme(0).write_genetics("viability_fitness", np.full_like(saved[0], 0.5))
        assert float(pop.deme(0).config.viability_fitness[0, 0, 0]) == 0.5
        for i in range(1, 4):
            np.testing.assert_array_equal(
                pop.deme(i).config.viability_fitness,
                saved[i],
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
        np.testing.assert_array_equal(pop.deme(0).config.zygotes_to_gametes_map, saved)
        np.testing.assert_array_equal(
            pop.deme(0).config.offspring_tensor, saved_offspring
        )
        np.testing.assert_array_equal(pop.deme(1).config.zygotes_to_gametes_map, saved)

    def test_deme_params_tensor_write_validates_shape_and_forks(self, homogeneous_pop) -> None:
        """Native scoped writers reject malformed tensors and isolate valid ones."""
        from natal.frontend.population._params_view import _GENETICS_TENSORS

        pop = homogeneous_pop
        original = pop.deme(1).params.viability_fitness.array
        for field in sorted(_GENETICS_TENSORS):
            with pytest.raises(ValueError, match="expected"):
                pop.deme(0).params.tensor_write(field, np.zeros(0, dtype=np.float64))
        pop.deme(0).params.tensor_write("viability_fitness", np.full_like(original, .5))
        np.testing.assert_array_equal(pop.deme(0).params.viability_fitness.array, .5)
        np.testing.assert_array_equal(pop.deme(1).params.viability_fitness.array, original)
        rates = np.asarray(pop.deme(0).params.survival_rates, dtype=np.float64)
        pop.deme(0).params.tensor_write("survival_rates", rates)
        np.testing.assert_array_equal(pop.deme(0).params.survival_rates, rates)


class TestBuildSemanticsPreserved:
    """Homogeneous construction preserves equal values with isolated snapshots."""

    def test_shared_config_after_homogeneous_build(
        self,
        homogeneous_pop,
        homogeneous_age_pop,
    ) -> None:
        """Homogeneous demes expose equal isolated config snapshots."""
        for pop in (homogeneous_pop, homogeneous_age_pop):
            config0 = pop.deme(0).config
            for i in range(1, 4):
                snapshot = pop.deme(i).config
                assert snapshot is not config0
                np.testing.assert_array_equal(snapshot.viability_fitness, config0.viability_fitness)
                assert not np.shares_memory(snapshot.viability_fitness, config0.viability_fitness)

    def test_homogeneous_pop_is_spatial_population(
        self,
        homogeneous_pop,
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
        assert derive_equilibrium_metrics_from_draft(d0)[0] == 500.0
        assert derive_equilibrium_metrics_from_draft(d0)[1] == 0.2
        assert derive_equilibrium_metrics_from_draft(d1)[0] == 1000.0
        assert derive_equilibrium_metrics_from_draft(d1)[1] == 0.1

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
        assert derive_equilibrium_metrics_from_draft(pop.demes[0].config)[0] == 500.0
        assert derive_equilibrium_metrics_from_draft(pop.demes[1].config)[0] == 1000.0
        assert (
            derive_equilibrium_metrics_from_draft(pop.demes[0].config)[1]
            == 100.0 / ext0
        )
        assert (
            derive_equilibrium_metrics_from_draft(pop.demes[1].config)[1]
            == 100.0 / ext1
        )

    def test_variant_derive_carries_the_two_params(self) -> None:
        """The derive surface honors declared + external bit-for-bit.

        Slice 2 retired the variant recompute (the metrics are derived
        on read): the declared distribution drives the competition mass
        and the egg override drives the expected survival rate.
        """
        from natal.frontend.data._engine import (
            derive_equilibrium_metrics_from_draft,
        )

        pop, _declared = self._eggs_heterogeneous_population(
            "eq_fallback_variant2", external_females=100.0
        )
        cfg = pop.demes[1].config
        metrics = derive_equilibrium_metrics_from_draft(cfg)

        assert metrics[0] == 1000.0
        ext = float(cfg.external_expected_eggs)
        assert metrics[1] == 100.0 / ext

    def test_variant_metrics_isolated_between_demes(self) -> None:
        """A runtime ecology write on deme 0 leaves deme 1's metrics alone."""
        pop, _declared = self._eggs_heterogeneous_population("eq_isolation_variant")
        pop.run(1, record_every=0)

        pop.demes[0].write_ecology("carrying_capacity", 999.0)

        assert derive_equilibrium_metrics_from_draft(pop.demes[1].config)[0] == 1000.0
        assert derive_equilibrium_metrics_from_draft(pop.demes[1].config)[1] == 0.1

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


class TestEquilibriumDistributionChannels:
    """The wrapper-advertised equilibrium_distribution kwargs are live.

    Both the age_structure and survival wrappers advertise an
    ``equilibrium_distribution`` parameter that used to be forwarded to
    template methods rejecting it (TypeError).  Both now route the
    declaration through competition — the working channel.
    """

    def _population(self, name: str, channel: str, *, heterogeneous: bool = False):
        """Build a two-deme population declaring via the given channel."""
        species = nt.Species.from_dict(
            name="__test_spatial_eq_channel__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        declared = np.zeros((2, 4))
        declared[0, 1] = 50.0
        declared[1, 1] = 50.0
        eggs: object = nt.batch_setting([10.0, 20.0]) if heterogeneous else 10.0
        age_kwargs: dict[str, object] = {"n_ages": 4, "new_adult_age": 1}
        survival_kwargs: dict[str, object] = {
            "female_age_based_survival": [1.0, 0.9, 0.7, 0.0],
            "male_age_based_survival": [1.0, 0.9, 0.7, 0.0],
        }
        if channel == "age_structure":
            age_kwargs["equilibrium_distribution"] = declared
        elif channel == "survival":
            survival_kwargs["equilibrium_distribution"] = declared
        else:
            raise ValueError(f"unknown channel {channel!r}")
        return (
            nt.SpatialPopulation.builder(species, n_demes=2, pop_type="age_structured")
            .setup(name=name, stochastic=False)
            .age_structure(**age_kwargs)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 50, 0, 0]},
                    "male": {"A|A": [0, 50, 0, 0]},
                }
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                eggs_per_female=eggs,  # type: ignore[arg-type]  # scalar or BatchSetting by construction
            )
            .survival(**survival_kwargs)
            .competition(carrying_capacity=100.0, juvenile_growth_mode=3)
            .build()
        )

    def test_age_structure_channel_declares_distribution(self) -> None:
        """age_structure(equilibrium_distribution=...) takes effect.

        Declared: 50F+50M at age 1, eggs 10 → C* = 50*1*1*10 = 500 and
        s* = 100/500 = 0.2 (hand-computed anchors).
        """
        pop = self._population("eq_channel_age", "age_structure")
        cfg = pop.demes[0].config
        assert derive_equilibrium_metrics_from_draft(cfg)[0] == 500.0
        assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2

    def test_survival_channel_declares_distribution(self) -> None:
        """survival(equilibrium_distribution=...) takes effect identically."""
        pop = self._population("eq_channel_survival", "survival")
        cfg = pop.demes[0].config
        assert derive_equilibrium_metrics_from_draft(cfg)[0] == 500.0
        assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2

    def test_channel_survives_heterogeneous_replay(self) -> None:
        """The declaration recorded in the replay log reaches variants.

        Heterogeneous eggs drive the variant rebuild path; the variant
        deme must stay on the declared distribution (500 / 1000), not the
        derivation mode the pre-fix variant recompute produced.
        """
        pop = self._population("eq_channel_replay", "age_structure", heterogeneous=True)
        assert derive_equilibrium_metrics_from_draft(pop.demes[0].config)[0] == 500.0
        assert derive_equilibrium_metrics_from_draft(pop.demes[1].config)[0] == 1000.0


class TestEquilibriumChannelAdversarial:
    """Adversarial follow-ups for the wrapper declaration channels.

    Every test here pins a way the routing could silently diverge from
    the direct ``competition(equilibrium_distribution=...)`` channel:
    bit-level draft equivalence, replay-log recording (the contract the
    variant/replay paths depend on), per-deme batch declarations,
    overwrite ordering, None semantics, error paths, and the documented
    ``generation_time`` dead channel.
    """

    AGE_DECLARED = np.array([[0.0, 50.0, 0.0, 0.0], [0.0, 50.0, 0.0, 0.0]])

    @staticmethod
    def _species(tag: str) -> nt.Species:
        return nt.Species.from_dict(
            name=f"__test_spatial_eq_adv_{tag}__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )

    def _population(
        self,
        tag: str,
        channel: str,
        *,
        declared: object = "default",
        eggs: object = 10.0,
        carrying_capacity: object = 100.0,
        extra_survival_decl: object = None,
        tail_competition_decl: object = "unset",
    ):
        """Build a two-deme age-structured population via one channel.

        Args:
            tag: Unique species-name fragment (species cache is a
                singleton per name).
            channel: ``"age_structure"``, ``"survival"`` or
                ``"competition"`` — where the declaration enters.
            declared: The declared distribution (ndarray or BatchSetting);
                ``"default"`` uses the 50F/50M-at-age-1 anchor array.
            eggs: Scalar or ``BatchSetting`` eggs_per_female.
            carrying_capacity: Scalar or ``BatchSetting`` K.
            extra_survival_decl: Optional second declaration through the
                survival channel (last-write-wins probe).
            tail_competition_decl: Value passed as
                ``equilibrium_distribution`` to the final competition
                call; ``"unset"`` omits the kwarg entirely.

        Returns:
            The built SpatialPopulation.
        """
        if isinstance(declared, str) and declared == "default":
            declared = self.AGE_DECLARED
        age_kwargs: dict[str, object] = {"n_ages": 4, "new_adult_age": 1}
        survival_kwargs: dict[str, object] = {
            "female_age_based_survival": [1.0, 0.9, 0.7, 0.0],
            "male_age_based_survival": [1.0, 0.9, 0.7, 0.0],
        }
        competition_kwargs: dict[str, object] = {
            "carrying_capacity": carrying_capacity,
            "juvenile_growth_mode": 3,
        }
        if channel == "age_structure":
            age_kwargs["equilibrium_distribution"] = declared
        elif channel == "survival":
            survival_kwargs["equilibrium_distribution"] = declared
        elif channel == "competition":
            competition_kwargs["equilibrium_distribution"] = declared
        else:
            raise ValueError(f"unknown channel {channel!r}")
        if extra_survival_decl is not None:
            survival_kwargs["equilibrium_distribution"] = extra_survival_decl
        if tail_competition_decl != "unset":
            competition_kwargs["equilibrium_distribution"] = tail_competition_decl
        return (
            nt.SpatialPopulation.builder(
                self._species(tag), n_demes=2, pop_type="age_structured"
            )
            .setup(name=f"eq_adv_{tag}", stochastic=False)
            .age_structure(**age_kwargs)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 50, 0, 0]},
                    "male": {"A|A": [0, 50, 0, 0]},
                }
            )
            .reproduction(
                female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
                eggs_per_female=eggs,  # type: ignore[arg-type]  # scalar or BatchSetting by construction
            )
            .survival(**survival_kwargs)
            .competition(**competition_kwargs)
            .build()
        )

    def test_three_entry_channels_bit_identical(self) -> None:
        """age_structure / survival / competition entries converge bit-exact.

        Channel-independence invariant: the declared array stored on the
        draft and both equilibrium metrics are bit-identical regardless
        of which wrapper the declaration entered through.  A routing bug
        that transposes, re-dtypes, or drops the declaration on one
        channel breaks the byte comparison or the exact metric equality.
        """
        pops = {
            ch: self._population(f"equiv_{ch}", ch)
            for ch in ("age_structure", "survival", "competition")
        }
        ref = pops["competition"].demes[0].config
        for ch, pop in pops.items():
            cfg = pop.demes[0].config
            stored = cfg.equilibrium_individual_distribution
            assert stored is not None, f"{ch}: declaration lost entirely"
            assert stored.tobytes() == self.AGE_DECLARED.tobytes(), (
                f"{ch}: stored distribution bytes differ from the declared array"
            )
            assert derive_equilibrium_metrics_from_draft(cfg)[0] == 500.0, ch
            assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2, ch
            # Bit-level metric equality against the direct channel.
            assert (
                derive_equilibrium_metrics_from_draft(cfg)[0].hex()
                == derive_equilibrium_metrics_from_draft(ref)[0].hex()
            ), ch
            assert (
                derive_equilibrium_metrics_from_draft(cfg)[1].hex()
                == derive_equilibrium_metrics_from_draft(ref)[1].hex()
            ), ch
            # The rest of the ecology anchors are channel-independent too.
            assert float(cfg.carrying_capacity) == 100.0, ch
            assert float(cfg.eggs_per_female) == 10.0, ch

    def test_wrapper_channels_record_competition_replay_entry(self) -> None:
        """Both wrapper channels record the declaration in the replay log.

        The replay log is the sole input to ``_build_template_for_group``
        (full replay) and, through the template it rebuilds, to the
        variant ``_replace`` path.  A routing that touches only
        ``self._template`` directly (never entering the log) keeps the
        homogeneous build green while every heterogeneous rebuild silently
        reverts to derivation mode — this introspection catches it
        without needing a batch setting at all.
        """
        for channel in ("age_structure", "survival"):
            builder = nt.SpatialPopulation.builder(
                self._species(f"log_{channel}"),
                n_demes=2,
                pop_type="age_structured",
            ).setup(name=f"eq_log_{channel}", stochastic=False)
            if channel == "age_structure":
                builder.age_structure(
                    n_ages=4,
                    new_adult_age=1,
                    equilibrium_distribution=self.AGE_DECLARED,
                )
            else:
                builder.age_structure(n_ages=4, new_adult_age=1)
                builder.survival(equilibrium_distribution=self.AGE_DECLARED)
            entries = builder._declaration_log  # pyright: ignore[reportPrivateUsage]  # the log is the replay contract under test; no public accessor exists
            comp_entries = [kw for m, kw in entries if m == "competition"]
            assert any(
                kw.get("equilibrium_distribution") is not None for kw in comp_entries
            ), f"{channel}: no competition entry carries the declaration"

    def test_homogeneous_build_keeps_declaration(self) -> None:
        """Without batch settings the direct-template path keeps the declaration.

        The no-batch build clones the chain-built template, so the routed
        competition call must have reached ``self._template`` as well as
        the log.  Both demes carry the declared array and the exact
        hand-computed metrics (C* = 50*1*1*10 = 500, s* = 100/500 = 0.2).
        """
        pop = self._population("homog", "age_structure")
        for i in range(2):
            cfg = pop.demes[i].config
            stored = cfg.equilibrium_individual_distribution
            assert stored is not None, f"deme{i}: declaration lost on clone"
            assert stored.tobytes() == self.AGE_DECLARED.tobytes()
            assert derive_equilibrium_metrics_from_draft(cfg)[0] == 500.0
            assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2

    def test_carrying_capacity_batch_replay_keeps_declaration(self) -> None:
        """A K batch forces full template replay; the declaration survives it.

        ``carrying_capacity`` registers as the non-draft-field batch name
        ``age_1_carrying_capacity``, so both demes rebuild through
        ``_build_template_for_group`` — the pure replay path.  Under the
        declared distribution both metrics are pinned by the declaration
        itself (total_age_1 = 100 declared, produced = 500), so C* = 500
        and s* = 0.2 on *both* demes even though K differs; only the
        carrying-capacity field tracks the batch.
        """
        pop = self._population(
            "cc_replay",
            "survival",
            carrying_capacity=nt.batch_setting([100.0, 200.0]),
        )
        for i, expected_k in ((0, 100.0), (1, 200.0)):
            cfg = pop.demes[i].config
            stored = cfg.equilibrium_individual_distribution
            assert stored is not None, f"deme{i}: declaration lost in replay"
            assert stored.tobytes() == self.AGE_DECLARED.tobytes()
            assert float(cfg.carrying_capacity) == expected_k
            assert derive_equilibrium_metrics_from_draft(cfg)[0] == 500.0
            assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2

    def test_batch_setting_declaration_applies_per_deme(self) -> None:
        """A BatchSetting of distributions declares one equilibrium per deme.

        Deme 0 (50F/50M at age 1, eggs 10): produced = 500 = C*,
        s* = 100/500 = 0.2.  Deme 1 (30F/30M at age 2): produced =
        30*1*1*10 = 300 = C*, and total_age_1 = 0 (nothing declared at
        age 1) so s* = 0/300 = 0.0.  A routing that collapsed the batch
        to its first value would report 500/0.2 on deme 1 too.
        """
        declared_b = np.zeros((2, 4))
        declared_b[0, 2] = 30.0
        declared_b[1, 2] = 30.0
        pop = self._population(
            "batch_declare",
            "age_structure",
            declared=nt.batch_setting([self.AGE_DECLARED, declared_b]),
        )
        for i, (expected_c, expected_s, expected_bytes) in enumerate(
            (
                (500.0, 0.2, self.AGE_DECLARED.tobytes()),
                (300.0, 0.0, declared_b.tobytes()),
            )
        ):
            cfg = pop.demes[i].config
            stored = cfg.equilibrium_individual_distribution
            assert stored is not None, f"deme{i}: declaration lost"
            assert stored.tobytes() == expected_bytes
            assert derive_equilibrium_metrics_from_draft(cfg)[0] == expected_c
            assert derive_equilibrium_metrics_from_draft(cfg)[1] == expected_s

    def test_double_declaration_last_write_wins(self) -> None:
        """A survival-channel declaration overwrites an earlier age_structure one.

        Both route to the same competition parameter, so the later
        declaration (25F/25M at age 1) wins: produced = 25*1*1*10 = 250
        = C*, s* = 50/250 = 0.2.  First-write-wins or merge semantics
        would produce 500 instead of 250.
        """
        second = np.array([[0.0, 25.0, 0.0, 0.0], [0.0, 25.0, 0.0, 0.0]])
        pop = self._population(
            "double_decl", "age_structure", extra_survival_decl=second
        )
        cfg = pop.demes[0].config
        stored = cfg.equilibrium_individual_distribution
        assert stored is not None
        assert stored.tobytes() == second.tobytes()
        assert derive_equilibrium_metrics_from_draft(cfg)[0] == 250.0
        assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2

    def test_later_none_competition_does_not_clear_declaration(self) -> None:
        """competition(equilibrium_distribution=None) means "don't touch".

        None is the parameter default, not a clear request: a trailing
        competition call that passes None explicitly must leave the
        wrapper's earlier declaration intact (C* stays 500).  A routing
        that forwarded None as a clearing write would flip the draft back
        to derivation mode.
        """
        pop = self._population("none_tail", "age_structure", tail_competition_decl=None)
        cfg = pop.demes[0].config
        stored = cfg.equilibrium_individual_distribution
        assert stored is not None, "explicit None cleared the declaration"
        assert stored.tobytes() == self.AGE_DECLARED.tobytes()
        assert derive_equilibrium_metrics_from_draft(cfg)[0] == 500.0
        assert derive_equilibrium_metrics_from_draft(cfg)[1] == 0.2

    def test_bad_shape_via_wrapper_channel_raises_value_error(self) -> None:
        """A wrong-shaped declaration fails with the expected shape named.

        The routed competition channel validates the (2, n_ages) shape;
        the error must name the expected shape so the wrapper channels
        report it exactly like the direct channel.
        """
        bad = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"\(2, 4\)"):
            self._population("bad_shape", "age_structure", declared=bad)

    def test_survival_generation_time_dead_channel_status_quo(self) -> None:
        """survival(generation_time=...) is the documented dead channel.

        The batch-14 NOTE records that a survival-time structure override
        has no lawful channel until the structure-domain cleanup batch:
        the template's survival has no ``generation_time`` parameter, so
        the wrapper's forward must keep raising TypeError.  The lawful
        channel — age_structure(generation_time=...) — must keep working
        and land the value on the draft (3.0).
        """
        builder = nt.SpatialPopulation.builder(
            self._species("gt_surv"),
            n_demes=2,
            pop_type="age_structured",
        ).setup(name="eq_gt_surv", stochastic=False)
        with pytest.raises(TypeError, match="generation_time"):
            builder.survival(generation_time=3.0)

        # age_structure(generation_time=...) is the live channel: the
        # value must land on the draft verbatim (3.0, vs the derived
        # 0.0 the same demographics produce without the kwarg).
        pop = self._population("gt_age", "age_structure")
        assert float(pop.demes[0].config.generation_time) == 0.0
        builder2 = nt.SpatialPopulation.builder(
            self._species("gt_age2"),
            n_demes=2,
            pop_type="age_structured",
        ).setup(name="eq_gt_age", stochastic=False)
        builder2.age_structure(n_ages=4, new_adult_age=1, generation_time=3.0)
        builder2.initial_state(
            individual_count={
                "female": {"A|A": [0, 50, 0, 0]},
                "male": {"A|A": [0, 50, 0, 0]},
            }
        )
        builder2.reproduction(
            female_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            male_age_based_mating_rate=[0.0, 1.0, 1.0, 0.0],
            eggs_per_female=10.0,
        )
        builder2.survival(
            female_age_based_survival=[1.0, 0.9, 0.7, 0.0],
            male_age_based_survival=[1.0, 0.9, 0.7, 0.0],
        )
        builder2.competition(carrying_capacity=100.0, juvenile_growth_mode=3)
        pop2 = builder2.build()
        assert float(pop2.demes[0].config.generation_time) == 3.0
