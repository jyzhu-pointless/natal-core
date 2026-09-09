"""Tests for base_population.py core methods.

Covers:
- _finalize_hooks() — deferred hook compilation
- _clone() — population cloning
- refresh_modifier_maps() — modifier map refresh
"""

from __future__ import annotations

import numpy as np

import natal as nt
from tests._config_assertions import assert_config_equal

# ══════════════════════════════════════════════════════════════════════════════
# Shared helper
# ══════════════════════════════════════════════════════════════════════════════


def _build_pop(
    species: nt.Species,
    name: str,
    *,
    initial: dict | None = None,
    hooks: list | None = None,
) -> nt.DiscreteGenerationPopulation:
    """Build a minimal DiscreteGenerationPopulation for testing."""
    builder = (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=name, stochastic=False,
        )
        .reproduction(eggs_per_female=50, sex_ratio=0.5)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .competition(carrying_capacity=10000, low_density_growth_rate=5.0)
    )
    if hooks is not None:
        builder = builder.hooks(*hooks)
    if initial is not None:
        builder = builder.initial_state(individual_count=initial)
    return builder.build()


# ══════════════════════════════════════════════════════════════════════════════
# TestFinalizeHooks
# ══════════════════════════════════════════════════════════════════════════════


class TestFinalizeHooks:
    """Tests for ``_finalize_hooks()`` — deferred compilation of @hook functions.

    Hook items passed at build time are queued in ``_pending_hook_items``
    and registered later by ``_finalize_hooks()``.

    ``DiscreteGenerationPopulation.__init__`` calls ``_finalize_hooks()``
    automatically, so these tests verify the post-finalization state.
    """

    def test_pending_hooks_compiled(self, simple_species: nt.Species) -> None:
        """Hook items queued at build time are registered after finalize."""
        @nt.hook(event="early")
        def my_hook(pop):
            return 0

        pop = _build_pop(simple_species, "test_pending", hooks=[my_hook])

        # The deferred item list must be drained after _finalize_hooks()
        assert len(pop._pending_hook_items) == 0

        # The hook should be in compiled hooks
        compiled = pop.get_compiled_hooks()
        assert len(compiled) > 0
        hook_names = [h.name for h in compiled if hasattr(h, "name")]
        assert "my_hook" in hook_names

    def test_plain_function_registered(self, simple_species: nt.Species) -> None:
        """Plain single-parameter callables register as Python callbacks."""
        calls: list[int] = []

        def plain_hook(pop):
            _ = pop
            calls.append(1)
            return 0

        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="test_plain", stochastic=False,
            )
            .hooks(plain_hook, event="early")
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .competition(carrying_capacity=10000, low_density_growth_rate=5.0)
            .build()
        )

        compiled = pop.get_compiled_hooks("early")
        assert [h.name for h in compiled] == ["plain_hook"]
        assert pop.has_python_callbacks()

        # The hook should be executable via trigger_event
        pop.trigger_event("early")
        assert len(calls) == 1

    def test_hook_executor_bookkeeping_removed(self, simple_species: nt.Species) -> None:
        """Native populations no longer carry Python executor bookkeeping."""
        pop = _build_pop(simple_species, "test_executor")
        assert not hasattr(pop, "hook_executor")


# ══════════════════════════════════════════════════════════════════════════════
# TestClone
# ══════════════════════════════════════════════════════════════════════════════


class TestClone:
    """Tests for ``_clone()`` — lightweight functional copy of a population.

    A clone shares compiled state (species, config, registries, hooks) but
    gets an independent state array and history.
    """

    def test_clone_is_different_object(self, simple_species: nt.Species) -> None:
        pop = _build_pop(simple_species, "clone_identity")
        clone = pop._clone("clone_of_identity")
        assert clone is not pop

    def test_clone_shares_species(self, simple_species: nt.Species) -> None:
        pop = _build_pop(simple_species, "clone_species")
        clone = pop._clone("clone_of_species")
        assert clone.species is pop.species

    def test_clone_config_snapshots_are_equal_and_isolated(self, simple_species: nt.Species) -> None:
        pop = _build_pop(simple_species, "clone_config")
        clone = pop._clone("clone_of_config")
        # Public reads return independent snapshots with equal values.
        snapshot = clone.config
        assert_config_equal(snapshot, pop.config)
        snapshot.viability_fitness[...] = 0.0
        assert_config_equal(clone.config, pop.config)

    def test_clone_has_independent_state(self, simple_species: nt.Species) -> None:
        pop = _build_pop(
            simple_species, "clone_state",
            initial={"female": {"WT|WT": 10}, "male": {"WT|WT": 10}},
        )
        clone = pop._clone("clone_of_state")

        # Record original value before modification
        original_val = float(pop.state.individual_count[0, 0, 0])

        # Modify clone's state
        clone.state.individual_count[0, 0, 0] += 100.0

        # Original pop must be unaffected
        assert pop.state.individual_count[0, 0, 0] == original_val

    def test_clone_custom_name(self, simple_species: nt.Species) -> None:
        pop = _build_pop(simple_species, "clone_original")
        clone = pop._clone("my_custom_name")
        assert clone.name == "my_custom_name"

    def test_clone_preserves_tick(self, simple_species: nt.Species) -> None:
        pop = _build_pop(simple_species, "clone_tick")
        pop.import_state(pop.state._replace(n_tick=42))
        clone = pop._clone("clone_of_tick")
        assert clone.tick == 42

    def test_clone_can_run_independently(self, simple_species: nt.Species) -> None:
        """After running the original, a clone can still run independently."""
        pop = _build_pop(
            simple_species, "clone_run",
            initial={"female": {"WT|WT": [0, 1000]}, "male": {"WT|WT": [0, 1000]}},
        )
        pop.run(n_steps=5, record_every=1)
        assert pop.tick == 5

        clone = pop._clone("clone_of_run")

        # Clone can run independently without affecting the original
        clone.run(n_steps=3, record_every=1)
        assert pop.tick == 5


# ══════════════════════════════════════════════════════════════════════════════
# TestRegistryStorage
# ══════════════════════════════════════════════════════════════════════════════


class TestRegistryStorage:
    """The population keeps exactly ONE registry storage field.

    ``registry`` and ``index_registry`` are two public names reading the
    same single ``_index_registry`` field; the legacy duplicate
    ``_registry`` storage must not come back, and clones share the object.
    """

    def test_public_registry_names_read_one_storage_field(
        self, simple_species: nt.Species,
    ) -> None:
        pop = _build_pop(simple_species, "registry_single_storage")
        assert pop.registry is pop.index_registry
        assert pop.registry is pop._index_registry
        assert not hasattr(pop, "_registry"), (
            "legacy duplicate registry storage field came back"
        )

    def test_clone_shares_the_single_registry(
        self, simple_species: nt.Species,
    ) -> None:
        pop = _build_pop(simple_species, "registry_clone_share")
        clone = pop._clone("registry_clone_share_c1")
        assert clone.registry is pop.registry
        assert clone.index_registry is pop.index_registry


# ══════════════════════════════════════════════════════════════════════════════
# TestRefreshModifiers
# ══════════════════════════════════════════════════════════════════════════════


class TestRefreshModifiers:
    """Tests for ``refresh_modifiers()`` — rebuild modifier lists and maps from sources.

    ``refresh_modifiers`` replaces the former ``rebuild_from_presets``.
    With no presets registered the refresh must leave the modifier lists
    empty and rebuild the pure Mendelian probability tables.
    """

    def test_no_presets_no_error(self, simple_species: nt.Species) -> None:
        """Refreshing with no presets keeps both modifier lists empty."""
        pop = _build_pop(simple_species, "refresh_no_presets")
        pop.refresh_modifiers()
        assert pop._gamete_modifiers == []
        assert pop._zygote_modifiers == []

    def test_config_maps_not_none(self, simple_species: nt.Species) -> None:
        """After refresh the three config maps exist with consistent shapes."""
        pop = _build_pop(simple_species, "refresh_maps")
        pop.refresh_modifiers()
        cfg = pop.config
        n_z = int(cfg.n_ztypes)
        n_g = int(cfg.n_gtypes)
        assert cfg.zygotes_to_gametes_map is not None
        assert cfg.zygotes_to_gametes_map.shape == (2, n_z, n_g)
        assert cfg.gametes_to_zygotes_map is not None
        assert cfg.gametes_to_zygotes_map.shape == (n_g, n_g, n_z)
        assert cfg.offspring_tensor is not None
        # Single gamete label ("default"): the fusion collapses the label axis,
        # leaving (maternal, paternal, offspring).
        assert cfg.offspring_tensor.shape == (n_z, n_z, n_z)

    def test_repeated_refresh_bitwise_idempotent(
        self, simple_species: nt.Species
    ) -> None:
        """A second refresh rebuilds byte-identical maps.

        Rebuilding from sources must not compound: the maps are derived
        from the same preset/manual lists each time, so any drift would
        mean state leaks between refreshes.
        """
        pop = _build_pop(simple_species, "refresh_twice")
        pop.refresh_modifiers()
        offspring_first = pop.config.offspring_tensor.copy()
        z2g_first = pop.config.zygotes_to_gametes_map.copy()
        g2z_first = pop.config.gametes_to_zygotes_map.copy()

        pop.refresh_modifiers()

        np.testing.assert_array_equal(pop.config.offspring_tensor, offspring_first)
        np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, z2g_first)
        np.testing.assert_array_equal(pop.config.gametes_to_zygotes_map, g2z_first)

    def test_mendelian_offspring_tensor_values(
        self, simple_species: nt.Species
    ) -> None:
        """No-preset refresh yields textbook Mendelian segregation.

        Genotype order for this species is WT/WT, WT/Dr, WT/R2, Dr/Dr,
        Dr/R2, R2/R2 (one locus, three alleles).
        """
        pop = _build_pop(simple_species, "refresh_mendelian")
        pop.refresh_modifiers()
        tensor = pop.config.offspring_tensor
        names = [str(gt) for gt in pop._index_registry.index_to_genotype]
        idx = {name: i for i, name in enumerate(names)}

        # Heterozygote x same heterozygote: 1:2:1 segregation.
        hd = idx["WT|Dr"]
        np.testing.assert_allclose(
            tensor[hd, hd, :], [0.25, 0.5, 0.0, 0.25, 0.0, 0.0]
        )
        # Homozygote x homozygote: all offspring are parental.
        np.testing.assert_allclose(
            tensor[idx["WT|WT"], idx["WT|WT"], :],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        # Reciprocal crosses agree (autosome, no sex-linked transmission).
        np.testing.assert_allclose(
            tensor[hd, idx["WT|WT"], :], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            tensor[idx["WT|WT"], hd, :], [0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
        )
        # Every parental pair produces a normalized distribution.
        n = len(names)
        np.testing.assert_allclose(tensor.sum(axis=-1), np.ones((n, n)))

    def test_refresh_rebuilds_maps_from_source_not_stale_arrays(
        self, simple_species: nt.Species
    ) -> None:
        """In-place corruption of the config maps is repaired by a refresh.

        The refresh must derive all three probability tables from the
        registry and the modifier source lists, never from the arrays
        currently stored in ``config``.  Otherwise external in-place
        mutation of the published (writeable) arrays would compound
        into every later refresh.
        """
        pop = _build_pop(simple_species, "refresh_repair")
        pop.refresh_modifiers()
        expected_offspring = pop.config.offspring_tensor.copy()
        expected_z2g = pop.config.zygotes_to_gametes_map.copy()
        expected_g2z = pop.config.gametes_to_zygotes_map.copy()

        # Corrupt the published arrays in place (they are writeable today).
        pop.config.offspring_tensor.fill(0.125)
        pop.config.zygotes_to_gametes_map.fill(0.5)
        pop.config.gametes_to_zygotes_map.fill(0.75)

        pop.refresh_modifiers()

        np.testing.assert_array_equal(
            pop.config.offspring_tensor, expected_offspring
        )
        np.testing.assert_array_equal(
            pop.config.zygotes_to_gametes_map, expected_z2g
        )
        np.testing.assert_array_equal(
            pop.config.gametes_to_zygotes_map, expected_g2z
        )

    def test_mendelian_meiosis_and_fusion_map_values(
        self, simple_species: nt.Species
    ) -> None:
        """No-preset refresh yields exact meiosis and fusion maps.

        The offspring tensor is a convolution of these two tables, so
        their values are pinned separately: ``z2g[sex, genotype, gamete]``
        is the segregation distribution of one parent, and
        ``g2z[g1, g2, offspring]`` is one-hot on the unordered genotype
        of the gamete pair.
        """
        pop = _build_pop(simple_species, "refresh_map_values")
        pop.refresh_modifiers()
        z2g = pop.config.zygotes_to_gametes_map
        g2z = pop.config.gametes_to_zygotes_map
        geno = {
            str(gt): i for i, gt in enumerate(pop._index_registry.index_to_genotype)
        }
        haplo = {
            str(h): i for i, h in enumerate(pop._index_registry.index_to_haplo)
        }

        # Heterozygote WT|Dr: Mendelian 1:1 segregation in both sexes.
        np.testing.assert_allclose(z2g[0, geno["WT|Dr"], :], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(z2g[1, geno["WT|Dr"], :], [0.5, 0.5, 0.0])
        # Homozygote WT|WT transmits only the WT gamete.
        np.testing.assert_allclose(z2g[0, geno["WT|WT"], :], [1.0, 0.0, 0.0])
        # Invariant: every genotype's gamete distribution is normalized
        # in both sexes (modifier probability rows sum to 1).
        np.testing.assert_allclose(z2g.sum(axis=-1), np.ones((2, len(geno))))

        # Fusion is one-hot on the unordered genotype of the allele pair,
        # independent of which parent contributed which allele.
        one_hot = np.eye(len(geno))
        for g1, g2, genotype in (
            ("WT", "WT", "WT|WT"),
            ("WT", "Dr", "WT|Dr"),
            ("Dr", "WT", "WT|Dr"),
            ("Dr", "Dr", "Dr|Dr"),
            ("WT", "R2", "WT|R2"),
        ):
            np.testing.assert_array_equal(
                g2z[haplo[g1], haplo[g2], :], one_hot[geno[genotype]]
            )
        # Invariant: fusing any gamete pair produces exactly one genotype.
        np.testing.assert_allclose(
            g2z.sum(axis=-1), np.ones((len(haplo), len(haplo)))
        )


# ══════════════════════════════════════════════════════════════════════════════
# TestRefreshModifierMaps
# ══════════════════════════════════════════════════════════════════════════════


class TestRefreshModifierMaps:
    """Tests for ``refresh_modifier_maps()`` — rebuild modifier maps from derived lists."""

    def test_refresh_modifier_maps_no_error(self, simple_species: nt.Species) -> None:
        """``refresh_modifier_maps()`` should not raise."""
        pop = _build_pop(simple_species, "refresh_noop")
        pop.refresh_modifier_maps()

    def test_refresh_modifier_maps_with_modifiers(self, simple_species: nt.Species) -> None:
        """Adding a modifier and refreshing updates the modifier list correctly."""
        pop = _build_pop(simple_species, "refresh_mods")

        # A no-op modifier (returns empty dict)
        def noop_modifier():
            return {}

        pop.add_gamete_modifier(noop_modifier, name="noop", refresh=True)

        gamete_mods = pop._gamete_modifiers
        names = [name for _, name, _ in gamete_mods]
        assert "noop" in names

        pop.refresh_modifier_maps()
        # After a manual refresh, the modifier should still be present
        names_after = [name for _, name, _ in pop._gamete_modifiers]
        assert "noop" in names_after
