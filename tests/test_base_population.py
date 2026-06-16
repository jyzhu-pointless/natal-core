"""Tests for base_population.py core methods.

Covers:
- _finalize_hooks() — deferred hook compilation
- _clone() — population cloning
- refresh_modifier_maps() — modifier map refresh
"""

from __future__ import annotations

import pytest

import natal as nt

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

    The ``@hook`` decorator attaches metadata (``func.meta``) that
    ``BasePopulation.__init__`` detects.  Hooks with metadata are queued in
    ``_pending_hooks`` and compiled later by ``_finalize_hooks()``.

    ``DiscreteGenerationPopulation.__init__`` calls ``_finalize_hooks()``
    automatically, so these tests verify the post-finalization state.
    """

    def test_pending_hooks_compiled(self, simple_species: nt.Species) -> None:
        """@hook functions queued in ``_pending_hooks`` are compiled after finalize."""
        @nt.hook(event="early", custom=True)
        def my_hook(state, config, deme_id):
            return 0

        pop = _build_pop(simple_species, "test_pending", hooks=[my_hook])

        # _pending_hooks must be cleared after _finalize_hooks()
        assert len(pop._pending_hooks) == 0

        # The hook should be in compiled hooks
        compiled = pop.get_compiled_hooks()
        assert len(compiled) > 0
        hook_names = [h.name for h in compiled if hasattr(h, "name")]
        assert "my_hook" in hook_names

    @pytest.mark.numba_off
    def test_plain_function_registered(self, simple_species: nt.Species) -> None:
        """Plain callable (no @hook decorator) is registered in traditional _hooks.

        Requires Numba disabled because ``set_hook()`` rejects plain Python
        callables when Numba is on.
        """
        calls: list[int] = []

        def plain_hook(population):
            calls.append(1)
            return 0

        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=simple_species, name="test_plain", stochastic=False,
            )
            .hooks({"early": [(plain_hook, "plain_hook", 0)]})
            .reproduction(eggs_per_female=50, sex_ratio=0.5)
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .competition(carrying_capacity=10000, low_density_growth_rate=5.0)
            .build()
        )

        # Plain hooks are registered in _hooks (traditional dict)
        hooks = pop.get_hooks("early")
        hook_names = [name for _, name, _ in hooks]
        assert "plain_hook" in hook_names

        # The hook should also be executable via trigger_event
        pop.trigger_event("early")
        assert len(calls) == 1

    def test_hook_executor_none_after_finalize(self, simple_species: nt.Species) -> None:
        """``_hook_executor`` is None immediately after ``_finalize_hooks()``."""
        pop = _build_pop(simple_species, "test_executor")
        assert pop._hook_executor is None


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

    def test_clone_shares_config(self, simple_species: nt.Species) -> None:
        pop = _build_pop(simple_species, "clone_config")
        clone = pop._clone("clone_of_config")
        # Config is shared by reference (shallow copy)
        assert clone.config is pop.config

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
        pop.tick = 42
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
# TestRefreshModifiers
# ══════════════════════════════════════════════════════════════════════════════


class TestRefreshModifiers:
    """Tests for ``refresh_modifiers()`` — rebuild modifier lists and maps from sources.

    ``refresh_modifiers`` replaces the former ``rebuild_from_presets``.
    These tests require full ``GeneticPreset`` infrastructure and are skipped.
    """

    @pytest.mark.skip(reason="Needs full GeneticPreset infrastructure")
    def test_no_presets_no_error(self, simple_species: nt.Species) -> None:
        ...

    @pytest.mark.skip(reason="Needs full GeneticPreset infrastructure")
    def test_config_maps_not_none(self, simple_species: nt.Species) -> None:
        ...


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
