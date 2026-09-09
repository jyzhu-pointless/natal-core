"""Adversarial tests for the ConfigContext retirement.

Every test in this module is designed around a specific way the retirement
could be violated or the frozen build-path behavior could regress:

  - Negative contracts: the deleted adapter surface (``ConfigContext``,
    ``apply_preset_to_population``, ``_make_ctx``/``_sync_from_ctx``,
    ``fitness._types``) must stay unreachable through every import and
    getattr path, and its definitions must not exist anywhere under
    ``src/natal``.
  - RecipeHost structure: the Configurator itself is now the build-side
    recipe host; its ``species``/``registry``/``index_registry`` surface
    must mirror the population side structurally.
  - State transitions: build-time preset/modifier failures must roll the
    candidate back bit-identically, and the deprecated ``preset.apply``
    path must stay diverged from (but numerically identical to)
    ``pop.apply_preset``.
  - Axis combinations: build granularity x preset set x compression must
    conserve probability mass in every cell of the Cartesian product.
  - Error paths: species-less hosts and haploid-empty registries must
    fail (or early-return) without corrupting state.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast, runtime_checkable

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator
from natal.frontend.configurator import _base as configurator_base
from natal.frontend.configurator import _registry_builder as registry_builder
from natal.frontend.configurator._registry_builder import rebuild_config_maps
from natal.frontend.data import ModelDraft
from natal.frontend.genetics.compile import RecipeHost
from natal.frontend.modifiers.module import GameteModifier
from natal.frontend.registry.index import IndexRegistry

# Root of the installed source tree for the static negative-contract scans.
_SRC_NATAL = Path(__file__).resolve().parent.parent / "src" / "natal"


# ── Shared helpers ────────────────────────────────────────────────────────────


def _fresh_homing() -> nt.HomingDrive:
    """Return an unbound HomingDrive with the frozen test parameters.

    The cascade on the ``WT|Dr`` meiosis row is analytically derivable:
    Mendelian 0.5/0.5, homing WT->Dr at 0.95, then germline resistance
    WT->R2 at 0.03 on the surviving WT pool gives
    ``WT = 0.5*0.05*0.97 = 0.02425``, ``Dr = 0.5 + 0.5*0.95 = 0.975``,
    ``R2 = 0.5*0.05*0.03 = 0.00075``.
    """
    return nt.HomingDrive(
        name="HD",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        drive_conversion_rate=0.95,
        late_germline_resistance_formation_rate=0.03,
        viability_scaling=0.5,
    )


def _fresh_tad() -> nt.ToxinAntidoteDrive:
    """Return an unbound ToxinAntidoteDrive with the frozen test parameters.

    WT->R2 germline disruption at 0.8 in drive carriers maps the ``WT|Dr``
    Mendelian row 0.5/0.5 to ``WT = 0.1``, ``Dr = 0.5``, ``R2 = 0.4``.
    All fitness scalings are pinned to 1.0 so the TAD fitness patch stays
    a no-op and the freeze assertions isolate the gamete-side cascade.
    """
    return nt.ToxinAntidoteDrive(
        name="TAD",
        drive_allele="Dr",
        target_allele="WT",
        disrupted_allele="R2",
        conversion_rate=0.8,
        viability_scaling=1.0,
        fecundity_scaling=1.0,
        zygote_viability_scaling=1.0,
    )


def _fresh_wolbachia() -> nt.Wolbachia:
    """Return an unbound Wolbachia preset scaling infected-slab viability 0.9."""
    return nt.Wolbachia(name="WOLB", viability_scaling=0.9)


@pytest.fixture
def slab_species() -> nt.Species:
    """Species carrying both drive alleles and cytoplasmic slab/glab axes.

    ``normal``/``infected`` somatic slabs plus the ``wolbachia`` gamete
    label let every preset family (homing, toxin-antidote, cytoplasmic)
    run on one architecture, which is what the axis-combination matrix
    requires.  Species are singleton-cached by name, so a function-scoped
    fixture still returns the shared object.
    """
    return nt.Species.from_dict(
        name="ConfigRetiredSlabSpecies",
        structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        gamete_labels=["default", "wolbachia"],
        somatic_labels=["normal", "infected"],
    )


@pytest.fixture
def other_species() -> nt.Species:
    """A second, genetically distinct species for binding-conflict probes."""
    return nt.Species.from_dict(
        name="ConfigRetiredOtherSpecies",
        structure={"c": {"l": ["A", "B"]}},
        gamete_labels=["default"],
    )


def _ztype_index(
    registry: IndexRegistry, species: nt.Species, genotype_str: str, slab: str
) -> int:
    """Resolve a diploid genotype string + slab to the active ZType index."""
    return registry.ztype_index(species.get_genotype_from_str(genotype_str), slab)


def _gtype_index(
    registry: IndexRegistry, species: nt.Species, haplo_str: str, glab: str
) -> int:
    """Resolve a haploid genotype string + glab to the active GType index."""
    return registry.gtype_index(species.get_haploid_genotype_from_str(haplo_str), glab)


def _row_override_modifier(
    sex_idx: int, ztype_idx: int, dist: Mapping[int, float]
) -> GameteModifier:
    """Build a manual gamete modifier pinning one (sex, ztype) meiosis row."""

    def modifier(
        *_args: object, **_kwargs: object
    ) -> Mapping[tuple[int, int], Mapping[int, float]]:
        return {(sex_idx, ztype_idx): dict(dist)}

    return modifier


def _noop_gamete_modifier(
    *_args: object, **_kwargs: object
) -> Mapping[tuple[int, int], Mapping[int, float]]:
    """Return an empty update mapping — touches no meiosis row."""
    return {}


def _broken_gamete_modifier(*_args: object, **_kwargs: object) -> int:
    """Return a non-mapping — deliberately violates the GameteModifier contract."""
    return 42


# ══════════════════════════════════════════════════════════════════════════
# 1. Negative contracts
# ══════════════════════════════════════════════════════════════════════════


class TestNegativeContracts:
    """The retired adapter surface must be unreachable from every path."""

    def test_config_context_class_is_gone(self) -> None:
        """Category: negative contract.

        Invariant: ``ConfigContext`` cannot be imported from, or found as
        an attribute of, its historical home module.
        Attack vector: a half-revert that keeps the class importable
        (e.g. left behind in a submodule or re-exported).
        """
        with pytest.raises(ImportError):
            from natal.frontend.configurator._registry_builder import (  # noqa: F401
                ConfigContext,
            )
        assert getattr(registry_builder, "ConfigContext", None) is None, (
            "ConfigContext is getattr-reachable on _registry_builder"
        )
        assert "ConfigContext" not in dir(registry_builder), (
            "ConfigContext appears in dir(_registry_builder)"
        )

    def test_apply_preset_to_population_is_gone_everywhere(self) -> None:
        """Category: negative contract.

        Invariant: ``apply_preset_to_population`` is unreachable from the
        top-level package, the presets package, and the preset base
        module — the three surfaces that historically carried it.
        Attack vector: an ``__init__`` re-export that survived the
        deletion.
        """
        import natal.frontend.presets as presets_pkg
        import natal.frontend.presets._base as presets_base

        for surface, name in (
            (nt, "natal"),
            (presets_pkg, "natal.frontend.presets"),
            (presets_base, "natal.frontend.presets._base"),
        ):
            assert getattr(surface, "apply_preset_to_population", None) is None, (
                f"apply_preset_to_population is reachable on {name}"
            )
        with pytest.raises(ImportError):
            from natal.frontend.presets import (  # noqa: F401
                apply_preset_to_population,
            )

    def test_source_tree_contains_no_mimicry_definitions(self) -> None:
        """Category: negative contract (static source scan).

        Invariant: the strings ``class ConfigContext``,
        ``def apply_preset_to_population``, and ``class
        FitnessPopulationView`` occur in no source or stub file under
        ``src/natal``.  A definition that exists in source but is merely
        unimported would still count as a revival.
        """
        forbidden = (
            "class ConfigContext",
            "def apply_preset_to_population",
            "class FitnessPopulationView",
        )
        hits: list[str] = []
        for path in sorted(_SRC_NATAL.rglob("*.py*")):
            text = path.read_text(encoding="utf-8", errors="replace")
            for needle in forbidden:
                if needle in text:
                    hits.append(f"{path.relative_to(_SRC_NATAL.parent)}: {needle}")
        assert not hits, f"retired definitions found in source: {hits}"

    def test_configurator_adapter_round_trip_methods_are_gone(
        self, simple_species: nt.Species
    ) -> None:
        """Category: negative contract.

        Invariant: the Configurator class no longer defines the
        ``_make_ctx`` / ``_sync_from_ctx`` adapter pair (neither on the
        class nor on an instance), and the dead
        ``_RUST_GENETICS_TENSORS`` constant is absent from the module
        namespace.
        Attack vector: the adapter round-trip returning as private
        methods, silently re-introducing the population-mimicry path.
        """
        for attr in ("_make_ctx", "_sync_from_ctx"):
            assert getattr(Configurator, attr, None) is None, (
                f"Configurator.{attr} is back — the adapter round-trip returned"
            )
        instance = Configurator.from_species(simple_species)
        assert getattr(instance, "_make_ctx", None) is None
        assert getattr(instance, "_sync_from_ctx", None) is None
        assert not hasattr(configurator_base, "_RUST_GENETICS_TENSORS"), (
            "the dead _RUST_GENETICS_TENSORS constant is reachable again"
        )

    def test_fitness_types_module_is_gone(self) -> None:
        """Category: negative contract.

        Invariant: ``natal.frontend.fitness._types`` is not an importable
        module (``FitnessPopulationView`` and its neighbors were deleted
        with it).
        Attack vector: the module surviving as an orphan file while the
        code paths moved to ``presets._types``.
        """
        assert importlib.util.find_spec("natal.frontend.fitness._types") is None, (
            "natal.frontend.fitness._types is still an importable module"
        )
        with pytest.raises(ImportError):
            importlib.import_module("natal.frontend.fitness._types")
        import natal.frontend.fitness as fitness_pkg

        assert not hasattr(fitness_pkg, "FitnessPopulationView"), (
            "FitnessPopulationView is re-exported from natal.frontend.fitness"
        )


# ══════════════════════════════════════════════════════════════════════════
# 2. RecipeHost protocol structure
# ══════════════════════════════════════════════════════════════════════════


@runtime_checkable
class _RecipeHostMirror(Protocol):
    """Runtime-checkable mirror of RecipeHost's four read members.

    ``RecipeHost`` itself is not ``@runtime_checkable`` (``isinstance``
    against it raises ``TypeError``); this mirror with identical members
    gives a genuine structural conformance check for both hosts.
    """

    @property
    def species(self) -> nt.Species: ...

    @property
    def config(self) -> ModelDraft: ...

    @property
    def registry(self) -> IndexRegistry: ...

    @property
    def index_registry(self) -> IndexRegistry: ...


class TestRecipeHostStructure:
    """The Configurator is the build-side RecipeHost; structure must match."""

    def test_configurator_exposes_stable_lazy_recipe_host_surface(
        self, simple_species: nt.Species
    ) -> None:
        """Category: ownership / structure.

        Invariant: ``registry`` is built lazily exactly once — every
        access returns the same object, ``index_registry`` aliases it, and
        before the first access the cache slot is empty.  All recipes
        compiled against one host must observe a single shared name
        directory, or index lookups would disagree between two registries.
        Attack vector: the property rebuilding the registry per call (each
        recipe would see fresh indices) or handing out a copy.
        """
        cfg = Configurator.from_species(simple_species)
        assert cfg._registry is None, "registry was built eagerly at construction"  # pyright: ignore[reportPrivateUsage]  # laziness is the invariant under test
        first = cfg.registry
        second = cfg.registry
        assert first is second, "registry property returns a fresh object per call"
        assert cfg.index_registry is first, (
            "index_registry does not alias the same registry object"
        )
        assert isinstance(first, IndexRegistry)
        # A 3-allele single-locus species has exactly 6 unordered diploid
        # genotypes and 3 haploid genotypes on the default slab/glab.
        assert first.n_ztypes == 6
        assert first.n_gtypes == 3

    def test_recipe_host_structural_conformance_for_both_hosts(
        self, simple_species: nt.Species
    ) -> None:
        """Category: structure (axis: build-side host vs runtime host).

        Invariant: both hosts that recipes run against — a mid-compile
        Configurator and a built population — satisfy the same structural
        surface with the correct member types, and both spell
        ``registry is index_registry``.
        Attack vector: the Configurator dropping a protocol member
        (recipes would crash only when driven through one host) or
        exposing a differently-typed stand-in.
        """
        cfg = Configurator.from_species(simple_species)
        with pytest.raises(TypeError):
            # Documents why the mirror below is needed: RecipeHost is not
            # runtime_checkable, so a direct isinstance probe is rejected.
            isinstance(cfg, RecipeHost)

        pop = (
            nt.AgeStructuredPopulation.setup(species=simple_species, stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": [0, 100]}, "male": {"WT|WT": [0, 100]}})
            .build()
        )
        for host in (cfg, pop):
            assert isinstance(host, _RecipeHostMirror), (
                f"{type(host).__name__} no longer satisfies the RecipeHost surface"
            )
            assert isinstance(host.species, nt.Species)
            assert isinstance(host.config, ModelDraft)
            assert isinstance(host.registry, IndexRegistry)
            assert host.registry is host.index_registry
        assert cfg.species is simple_species
        assert pop.species is simple_species

    def test_species_less_configurator_properties_raise(
        self, simple_species: nt.Species
    ) -> None:
        """Category: error path.

        Invariant: a raw-constructed Configurator (no species) raises
        ``RuntimeError`` from ``species`` and — because the registry is
        derived from the species — from ``registry``/``index_registry``.
        Attack vector: the lazy registry builder silently registering an
        empty registry instead of failing fast.
        """
        draft = Configurator.from_species(simple_species).config
        bare = Configurator(draft)
        for probe in (
            lambda: bare.species,
            lambda: bare.registry,
            lambda: bare.index_registry,
        ):
            with pytest.raises(RuntimeError, match="require a Species"):
                probe()


# ══════════════════════════════════════════════════════════════════════════
# 3. State transitions / behavior freeze
# ══════════════════════════════════════════════════════════════════════════


class TestBuildPathFreeze:
    """Numerical freeze of the build-side candidate compile."""

    def test_build_freezes_drive_tad_manual_fitness_values(
        self, simple_species: nt.Species
    ) -> None:
        """Category: state transition (freeze) with exact values.

        Invariant: the candidate compile chains modifiers in list order —
        preset-derived first, manual appended last — and a later
        row-writing modifier replaces the whole (sex, ztype) row it
        targets.  Pinned cells on the ``WT|Dr`` meiosis row (columns
        WT/Dr/R2):

          - female row (manual override, chained last): 0.3 / 0.7 / 0.0
          - male row (TAD, the later preset, replaces the homing result):
            0.1 / 0.5 / 0.4
          - both-sex ``WT|R2`` control rows stay Mendelian 0.5 / 0.0 / 0.5
            (no drive allele -> no conversion rule applies)

        Fitness freeze: homing viability_scaling 0.5 (multiplicative over
        the combined Dr+R2 allele class) gives 1/0.5/0.5/0.25/0.25/0.25
        per genotype at the default age, the manual ``fitness()`` replace
        write overwrites WT|WT to 0.7, and the non-default age stays 1.0.
        Attack vector: modifier list re-ordering, per-cell merging instead
        of row replacement, or the manual fitness write composing
        (multiplying) where it must replace.
        """
        cfg = Configurator.from_species(simple_species)
        reg = cfg.registry
        zt_het = _ztype_index(reg, simple_species, "WT|Dr", "default")
        zt_ctrl = _ztype_index(reg, simple_species, "WT|R2", "default")
        wt = _gtype_index(reg, simple_species, "WT", "default")
        dr = _gtype_index(reg, simple_species, "Dr", "default")
        r2 = _gtype_index(reg, simple_species, "R2", "default")

        manual = _row_override_modifier(0, zt_het, {wt: 0.3, dr: 0.7})
        cfg.presets(_fresh_homing(), _fresh_tad())
        cfg.modifiers(gamete_modifiers=[manual])
        cfg.fitness(viability={"WT|WT": 0.7})

        z2g = cfg.config.zygotes_to_gametes_map
        assert z2g.shape == (2, 6, 3), f"unexpected meiosis map shape {z2g.shape}"
        np.testing.assert_allclose(
            z2g[0, zt_het, [wt, dr, r2]], [0.3, 0.7, 0.0], atol=1e-15
        )
        np.testing.assert_allclose(
            z2g[1, zt_het, [wt, dr, r2]], [0.1, 0.5, 0.4], atol=1e-15
        )
        for sex in (0, 1):
            np.testing.assert_allclose(
                z2g[sex, zt_ctrl, [wt, dr, r2]], [0.5, 0.0, 0.5], atol=1e-15
            )
        # Mass conservation: every (sex, ztype) gamete row sums to 1.
        np.testing.assert_allclose(z2g.sum(axis=2), 1.0, atol=1e-12)

        # Fitness: the default age is new_adult_age - 1 = 0 on this draft.
        expected_age0 = np.array([[0.7, 0.5, 0.5, 0.25, 0.25, 0.25]] * 2)
        np.testing.assert_allclose(
            cfg.config.viability_fitness[:, 0, :], expected_age0, atol=1e-15
        )
        np.testing.assert_allclose(
            cfg.config.viability_fitness[:, 1, :], 1.0, atol=1e-15
        )

        # The built population carries the same compiled maps and the
        # modifier-name trail preset-derived-first / manual-last.
        pop = cfg.build()
        names = [name for _, name, _ in pop.gamete_modifiers]
        assert names == ["HD/gamete", "TAD/gamete", None], (
            f"modifier registration order/names drifted: {names}"
        )
        np.testing.assert_array_equal(pop.config.zygotes_to_gametes_map, z2g)
        np.testing.assert_allclose(
            pop.config.offspring_tensor.sum(axis=2), 1.0, atol=1e-12
        )

    def test_failed_preset_batch_rolls_back_and_recovers(
        self, simple_species: nt.Species, other_species: nt.Species
    ) -> None:
        """Category: state transition (build-time rollback sequence).

        Invariant 1 (order: success then failure): when the first preset
        applies fully and a later one is pre-bound to a different species,
        the ``ValueError`` leaves the Configurator bit-identical to its
        pre-call state — same draft object, same maps, same fitness
        tensors, empty modifier/preset lists — and unbinds the first
        preset again (its binding was a transaction side effect).
        Invariant 2 (order: failure first): the failure surfaces before
        any mutation.
        Invariant 3 (recovery): a subsequent ``presets(ok)`` on the same
        Configurator compiles maps bit-identical to a fresh Configurator
        that only ever saw ``ok`` — no residue of the failed batch.
        Attack vector: a rollback that restores the draft identity but not
        the in-place-mutated fitness arrays, or that leaves the first
        preset bound so the recovery double-applies.
        """
        bound = nt.HomingDrive(
            name="BOUND",
            drive_allele="A",
            target_allele="B",
            drive_conversion_rate=0.5,
            species=other_species,
        )

        # -- order A: ok applies fully, then bound fails --------------------
        cfg = Configurator.from_species(simple_species)
        ok = _fresh_homing()
        original_draft = cfg.config
        baseline_z2g = cfg.config.zygotes_to_gametes_map.copy()
        baseline_g2z = cfg.config.gametes_to_zygotes_map.copy()
        baseline_off = cfg.config.offspring_tensor.copy()
        baseline_viab = cfg.config.viability_fitness.copy()
        with pytest.raises(ValueError, match="already bound to species"):
            cfg.presets(ok, bound)
        assert cfg.config is original_draft, "draft identity changed after rollback"
        np.testing.assert_array_equal(cfg.config.zygotes_to_gametes_map, baseline_z2g)
        np.testing.assert_array_equal(cfg.config.gametes_to_zygotes_map, baseline_g2z)
        np.testing.assert_array_equal(cfg.config.offspring_tensor, baseline_off)
        np.testing.assert_array_equal(cfg.config.viability_fitness, baseline_viab)
        assert cfg.gamete_modifiers == []
        assert cfg.zygote_modifiers == []
        assert cfg._presets == []  # pyright: ignore[reportPrivateUsage]  # rollback verification inspects the registration list
        assert ok._bound_species is None, (  # pyright: ignore[reportPrivateUsage]  # binding is the transaction side effect under test
            "the fully-applied preset stayed bound after the batch rolled back"
        )

        # -- order B: bound fails before any mutation -----------------------
        ok2 = _fresh_homing()
        with pytest.raises(ValueError, match="already bound to species"):
            cfg.presets(bound, ok2)
        assert cfg.config is original_draft
        assert cfg.gamete_modifiers == []
        assert ok2._bound_species is None  # pyright: ignore[reportPrivateUsage]  # same probe as above

        # -- recovery: the same Configurator now accepts ok alone ------------
        cfg.presets(ok)
        fresh = Configurator.from_species(simple_species).presets(_fresh_homing())
        np.testing.assert_array_equal(
            cfg.config.zygotes_to_gametes_map,
            fresh.config.zygotes_to_gametes_map,
        )
        np.testing.assert_array_equal(
            cfg.config.gametes_to_zygotes_map,
            fresh.config.gametes_to_zygotes_map,
        )
        np.testing.assert_array_equal(
            cfg.config.offspring_tensor, fresh.config.offspring_tensor
        )
        np.testing.assert_array_equal(
            cfg.config.viability_fitness, fresh.config.viability_fitness
        )
        assert len(cfg.gamete_modifiers) == len(fresh.gamete_modifiers) == 1

    def test_failed_modifier_rebuild_leaves_state_unchanged(
        self, simple_species: nt.Species
    ) -> None:
        """Category: error path / state transition.

        Invariant: a modifier whose callable returns a non-mapping makes
        ``modifiers()`` raise ``TypeError`` at rebuild time, and the
        Configurator's modifier list and draft identity are unchanged
        afterwards — the candidate list is committed only on success.
        Attack vector: the failing modifier appended to
        ``cfg.gamete_modifiers`` anyway, corrupting every later rebuild.
        """
        cfg = Configurator.from_species(simple_species)
        cfg.modifiers(gamete_modifiers=[_noop_gamete_modifier])
        committed = list(cfg.gamete_modifiers)
        draft_before = cfg.config
        with pytest.raises(TypeError, match="must return a mapping"):
            # cast: the callable deliberately violates the GameteModifier
            # contract — that violation is the error path under test.
            cfg.modifiers(
                gamete_modifiers=[cast(GameteModifier, _broken_gamete_modifier)]
            )
        assert cfg.gamete_modifiers == committed, (
            "the failing modifier leaked into the committed list"
        )
        assert cfg.config is draft_before, "draft identity changed after failed rebuild"

    def test_deprecated_apply_vs_apply_preset_divergence_pinned(
        self, simple_species: nt.Species
    ) -> None:
        """Category: state transition (legacy vs modern API divergence).

        Invariant: ``preset.apply(pop)`` registers the modifier into the
        MANUAL list (name ``"HD/gamete"``) and leaves ``pop.presets``
        empty, while ``pop.apply_preset(preset)`` registers the preset and
        derives the same-named modifier — yet for fresh populations the
        two paths produce bit-identical meiosis and offspring tensors.
        Attack vector: the deprecated path silently switching to the
        preset semantics (double-registration risk when both are used) or
        drifting numerically from the modern path.
        """

        def build() -> nt.AgeStructuredPopulation:
            return (
                nt.AgeStructuredPopulation.setup(
                    species=simple_species, stochastic=False
                )
                .age_structure(n_ages=2, new_adult_age=1)
                .initial_state(
                    {"female": {"WT|Dr": [0, 100]}, "male": {"WT|Dr": [0, 100]}}
                )
                .build()
            )

        pop_manual = build()
        pop_preset = build()
        d1 = _fresh_homing()
        d2 = _fresh_homing()
        d1.apply(pop_manual)
        pop_preset.apply_preset(d2)

        manual_entries = pop_manual._manual_gamete  # pyright: ignore[reportPrivateUsage]  # the manual-vs-derived registration split IS the divergence under test
        assert [(mid, name) for mid, name, _ in manual_entries] == [(0, "HD/gamete")]
        assert pop_manual.presets == [], "deprecated apply() registered the preset"
        assert pop_manual._manual_zygote == []  # pyright: ignore[reportPrivateUsage]  # same split probe, zygote side

        assert [p.name for p in pop_preset.presets] == ["HD"]
        assert pop_preset._manual_gamete == []  # pyright: ignore[reportPrivateUsage]  # modern path must not touch the manual list
        assert [name for _, name, _ in pop_preset.gamete_modifiers] == ["HD/gamete"]

        np.testing.assert_array_equal(
            pop_manual.config.zygotes_to_gametes_map,
            pop_preset.config.zygotes_to_gametes_map,
        )
        np.testing.assert_array_equal(
            pop_manual.config.offspring_tensor, pop_preset.config.offspring_tensor
        )
        # And both match the analytic homing row.
        reg = pop_preset.index_registry
        zt_het = _ztype_index(reg, simple_species, "WT|Dr", "default")
        wt = _gtype_index(reg, simple_species, "WT", "default")
        dr = _gtype_index(reg, simple_species, "Dr", "default")
        r2 = _gtype_index(reg, simple_species, "R2", "default")
        for pop in (pop_manual, pop_preset):
            np.testing.assert_allclose(
                pop.config.zygotes_to_gametes_map[1, zt_het, [wt, dr, r2]],
                [0.02425, 0.975, 0.00075],
                atol=1e-15,
            )

    def test_compress_build_subslices_names_and_maps_consistently(
        self, slab_species: nt.Species
    ) -> None:
        """Category: state transition (compression) + structure.

        Invariant: a ``compress=True`` build keeps the three name/map
        surfaces index-aligned — ``n_ztypes == len(ztype_names) ==
        registry.n_ztypes`` and ``n_gtypes == len(gtype_names)`` —
        registry lookups still resolve to the positions the names
        advertise, and the surviving meiosis cells keep their exact
        pre-compression values (compression subslices; it must not
        recompute semantics).
        Attack vector: the registry compressed but the name tuple not
        subsliced (or vice versa), so symbolic lookups land on the wrong
        genotype.
        """
        pop = (
            nt.AgeStructuredPopulation.setup(species=slab_species, stochastic=False)
            .age_structure(n_ages=2, new_adult_age=1)
            .initial_state({"female": {"WT|WT": [0, 100]}, "male": {"WT|Dr": [0, 100]}})
            .presets(_fresh_homing())
            .setup(compress=True)
            .build()
        )
        c = pop.config
        reg = pop.index_registry
        nzt = int(c.n_ztypes)
        ngt = int(c.n_gtypes)
        assert nzt == len(c.ztype_names) == reg.n_ztypes, (
            "ztype name directory disagrees with map width"
        )
        assert ngt == len(c.gtype_names) == reg.n_gtypes
        assert c.zygotes_to_gametes_map.shape == (int(c.n_sexes), nzt, ngt)
        assert c.gametes_to_zygotes_map.shape == (ngt, ngt, nzt)
        assert c.offspring_tensor.shape == (nzt, nzt, nzt)

        # Compression pruned: the 12-ztype full space shrank, and every
        # retained name is unique with the genotype:slab spelling.
        assert nzt < 12
        assert len(set(c.ztype_names)) == nzt
        for name in c.ztype_names:
            assert ":" in name, f"ztype name lost its slab suffix: {name!r}"

        # Symbolic lookups resolve to the index the names advertise.
        zt_het = _ztype_index(reg, slab_species, "WT|Dr", "normal")
        assert c.ztype_names[zt_het] == "WT|Dr:normal"
        wt = _gtype_index(reg, slab_species, "WT", "default")
        dr = _gtype_index(reg, slab_species, "Dr", "default")
        r2 = _gtype_index(reg, slab_species, "R2", "default")
        assert reg.index_to_gtype[wt][1] == "default"
        assert reg.index_to_gtype[dr][1] == "default"

        # Exact surviving values — identical to the uncompressed row.
        np.testing.assert_allclose(
            c.zygotes_to_gametes_map[:, zt_het, [wt, dr, r2]],
            [[0.02425, 0.975, 0.00075]] * 2,
            atol=1e-15,
        )
        np.testing.assert_allclose(c.zygotes_to_gametes_map.sum(axis=2), 1.0, atol=1e-12)
        np.testing.assert_allclose(c.offspring_tensor.sum(axis=2), 1.0, atol=1e-12)

    def test_mendelian_baseline_not_contaminated_by_prior_modifier_build(
        self, simple_species: nt.Species
    ) -> None:
        """Category: state transition / ownership (species blueprint cache).

        Invariant: the species-level Mendelian baseline is cached and read
        only — after a drive build has compiled biased maps from it, a
        fresh no-preset build from the SAME species object still yields
        the pure Mendelian row (0.5/0.5/0.0 on WT|Dr).
        Attack vector: a modifier compile writing through to the cached
        blueprint arrays, permanently contaminating every later build on
        that singleton species.
        """
        drive_cfg = Configurator.from_species(simple_species)
        reg = drive_cfg.registry
        zt_het = _ztype_index(reg, simple_species, "WT|Dr", "default")
        wt = _gtype_index(reg, simple_species, "WT", "default")
        dr = _gtype_index(reg, simple_species, "Dr", "default")
        r2 = _gtype_index(reg, simple_species, "R2", "default")
        drive_cfg.presets(_fresh_homing())
        np.testing.assert_allclose(
            drive_cfg.config.zygotes_to_gametes_map[0, zt_het, [wt, dr, r2]],
            [0.02425, 0.975, 0.00075],
            atol=1e-15,
        )

        clean = Configurator.from_species(simple_species)
        np.testing.assert_allclose(
            clean.config.zygotes_to_gametes_map[0, zt_het, [wt, dr, r2]],
            [0.5, 0.5, 0.0],
            atol=1e-15,
        )


# ══════════════════════════════════════════════════════════════════════════
# 4. Axis combination: build granularity x preset set x compression
# ══════════════════════════════════════════════════════════════════════════


class TestAxisCombinations:
    """Cartesian product: {age, discrete} x {presets} x {compress}."""

    @pytest.mark.parametrize("compress", [False, True], ids=["full", "compressed"])
    @pytest.mark.parametrize(
        "preset_kind", ["none", "homing", "toxin_antidote", "cytoplasmic+homing"]
    )
    @pytest.mark.parametrize("discrete", [False, True], ids=["age", "discrete"])
    def test_build_axis_combinations_conserve_mass(
        self,
        slab_species: nt.Species,
        discrete: bool,
        preset_kind: str,
        compress: bool,
    ) -> None:
        """Category: axis combination.

        Invariant (all 16 combinations of granularity x preset set x
        compression): the built config is dimensionally self-consistent —
        ``offspring_tensor`` is (nzt, nzt, nzt) with
        ``nzt == len(ztype_names) == registry.n_ztypes`` and the meiosis
        map is (n_sexes, nzt, ngt) with ``ngt == n_gtypes ==
        len(gtype_names)`` — and probability mass is conserved through
        every conversion modifier: each (sex, ztype) gamete row, each
        (c1, c2) fusion row, and each (i, j) offspring row sums to 1.
        The WT|Dr normal-slab meiosis row additionally carries the exact
        per-preset expected values (conversion drives only redistribute
        mass within the row).
        Attack vector: a preset family that works uncompressed but
        corrupts shapes or leaks mass under compression, or only on one
        granularity.
        """
        presets: list[nt.GeneticPreset]
        if preset_kind == "none":
            presets = []
        elif preset_kind == "homing":
            presets = [_fresh_homing()]
        elif preset_kind == "toxin_antidote":
            presets = [_fresh_tad()]
        else:
            presets = [_fresh_wolbachia(), _fresh_homing()]

        if discrete:
            chained = nt.DiscreteGenerationPopulation.setup(
                species=slab_species, stochastic=False
            )
        else:
            chained = nt.AgeStructuredPopulation.setup(
                species=slab_species, stochastic=False
            ).age_structure(n_ages=2, new_adult_age=1)
        pop = (
            chained.initial_state(
                {"female": {"WT|WT": [0, 100]}, "male": {"WT|Dr": [0, 100]}}
            )
            .presets(*presets)
            .setup(compress=compress)
            .build()
        )

        c = pop.config
        nzt = int(c.n_ztypes)
        ngt = int(c.n_gtypes)
        assert nzt == len(c.ztype_names) == pop.index_registry.n_ztypes
        assert ngt == len(c.gtype_names) == pop.index_registry.n_gtypes
        assert c.offspring_tensor.shape == (nzt, nzt, nzt)
        assert c.zygotes_to_gametes_map.shape == (int(c.n_sexes), nzt, ngt)
        assert c.gametes_to_zygotes_map.shape == (ngt, ngt, nzt)

        np.testing.assert_allclose(
            c.zygotes_to_gametes_map.sum(axis=2),
            1.0,
            atol=1e-12,
            err_msg=f"meiosis mass leak: {discrete}/{preset_kind}/{compress}",
        )
        np.testing.assert_allclose(
            c.gametes_to_zygotes_map.sum(axis=2),
            1.0,
            atol=1e-12,
            err_msg=f"fusion mass leak: {discrete}/{preset_kind}/{compress}",
        )
        np.testing.assert_allclose(
            c.offspring_tensor.sum(axis=2),
            1.0,
            atol=1e-12,
            err_msg=f"offspring mass leak: {discrete}/{preset_kind}/{compress}",
        )

        # Exact per-preset spot values on the WT|Dr normal-slab row.  With
        # no preset and compression on, the R2 column is pruned (ngt == 2),
        # so the expectation is restricted to the surviving columns.
        reg = pop.index_registry
        zt_het = _ztype_index(reg, slab_species, "WT|Dr", "normal")
        wt = _gtype_index(reg, slab_species, "WT", "default")
        dr = _gtype_index(reg, slab_species, "Dr", "default")
        full_row = {
            "none": [0.5, 0.5, 0.0],
            "homing": [0.02425, 0.975, 0.00075],
            "toxin_antidote": [0.1, 0.5, 0.4],
            "cytoplasmic+homing": [0.02425, 0.975, 0.00075],
        }[preset_kind]
        if ngt == 3:
            cols = [wt, dr, _gtype_index(reg, slab_species, "R2", "default")]
            expected = full_row
        else:
            cols = [wt, dr]
            expected = full_row[:2]
        for sex in (0, 1):
            np.testing.assert_allclose(
                c.zygotes_to_gametes_map[sex, zt_het, cols],
                expected,
                atol=1e-15,
                err_msg=f"spot value drift: {discrete}/{preset_kind}/{compress}",
            )

        # Cytoplasmic axis: the Wolbachia viability patch scales only the
        # infected slab.  Compression prunes the (unreachable) infected
        # slab, so the slab pair is asserted only on the full build.
        if preset_kind == "cytoplasmic+homing" and not compress:
            inf = _ztype_index(reg, slab_species, "WT|WT", "infected")
            nor = _ztype_index(reg, slab_species, "WT|WT", "normal")
            age = int(c.new_adult_age) - 1
            np.testing.assert_allclose(c.viability_fitness[:, age, inf], 0.9, atol=1e-15)
            np.testing.assert_allclose(c.viability_fitness[:, age, nor], 1.0, atol=1e-15)


# ══════════════════════════════════════════════════════════════════════════
# 5. Error paths
# ══════════════════════════════════════════════════════════════════════════


class TestErrorPaths:
    """Invalid hosts and degenerate registries must fail cleanly."""

    def test_species_less_presets_raises_leaving_state_unchanged(
        self, simple_species: nt.Species
    ) -> None:
        """Category: error path.

        Invariant: ``presets()`` on a species-less Configurator raises
        ``RuntimeError`` and the transaction rolls back completely — same
        draft object, empty modifier/preset lists, and the caller-owned
        preset stays unbound (so it can be applied elsewhere afterwards).
        Attack vector: the species guard firing AFTER the candidate lists
        were mutated, leaving half-registered state behind.
        """
        draft = Configurator.from_species(simple_species).config
        bare = Configurator(draft)
        preset = _fresh_homing()
        with pytest.raises(RuntimeError, match="require a Species"):
            bare.presets(preset)
        assert bare.config is draft
        assert bare.gamete_modifiers == []
        assert bare.zygote_modifiers == []
        assert bare._presets == []  # pyright: ignore[reportPrivateUsage]  # rollback verification inspects the registration list
        assert preset._bound_species is None  # pyright: ignore[reportPrivateUsage]  # caller-owned preset must survive the failed transaction unbound

        # The preset is still usable on a properly-hosted Configurator.
        hosted = Configurator.from_species(simple_species).presets(preset)
        assert preset._bound_species is simple_species  # pyright: ignore[reportPrivateUsage]  # confirms the earlier failure bound nothing
        reg = hosted.registry
        zt_het = _ztype_index(reg, simple_species, "WT|Dr", "default")
        wt = _gtype_index(reg, simple_species, "WT", "default")
        dr = _gtype_index(reg, simple_species, "Dr", "default")
        np.testing.assert_allclose(
            hosted.config.zygotes_to_gametes_map[1, zt_het, [wt, dr]],
            [0.02425, 0.975],
            atol=1e-15,
        )

    def test_species_less_modifiers_raises_leaving_state_unchanged(
        self, simple_species: nt.Species
    ) -> None:
        """Category: error path.

        Invariant: ``modifiers()`` on a species-less Configurator raises
        ``RuntimeError`` before the candidate lists are committed.
        Attack vector: the id assignment/list append happening before the
        compile touches ``self.species``.
        """
        draft = Configurator.from_species(simple_species).config
        bare = Configurator(draft)
        with pytest.raises(RuntimeError, match="require a Species"):
            bare.modifiers(gamete_modifiers=[_noop_gamete_modifier])
        assert bare.gamete_modifiers == []
        assert bare.zygote_modifiers == []
        assert bare.config is draft

    def test_rebuild_config_maps_empty_haplo_registry_early_return(
        self, simple_species: nt.Species
    ) -> None:
        """Category: error path (degenerate registry early return).

        Invariant: when the registry has diploid genotypes but NO haploid
        genotypes, ``rebuild_config_maps`` short-circuits — it returns the
        input draft object unchanged (identity, not a copy) with
        ``compression_applied is False``, even with ``compress=True``.
        Attack vector: the early return constructing a replaced draft (a
        caller comparing identity would miss state drift) or reporting
        compression as applied.
        """
        registry = IndexRegistry()
        registry.register_somatic_label("default")
        for genotype in simple_species.get_all_genotypes(
            unordered=simple_species.unordered
        ):
            registry.register_genotype(genotype)
        assert registry.index_to_haplo == []
        assert len(registry.index_to_genotype) == 6

        draft = Configurator.from_species(simple_species).config
        new_draft, applied = rebuild_config_maps(
            simple_species,
            draft,
            registry,
            gamete_modifiers=[],
            zygote_modifiers=[],
            compress=True,
        )
        assert new_draft is draft, "early return replaced the draft object"
        assert applied is False


# ══════════════════════════════════════════════════════════════════════════
# 6. Ownership / isolation
# ══════════════════════════════════════════════════════════════════════════


class TestOwnershipIsolation:
    """Public surfaces must not leak mutable internal state."""

    def test_presets_publishes_isolated_draft_original_untouched(
        self, simple_species: nt.Species
    ) -> None:
        """Category: ownership.

        Invariant: a successful ``presets()`` publishes an isolated draft
        — the caller's previously-held fitness and meiosis arrays are
        neither the same objects as the published ones nor mutated by the
        preset's in-place fitness patch (which ran against a deepcopy).
        Attack vector: the fitness patch writing into the original
        tensors, so a compile that later fails would still have corrupted
        the caller's draft.
        """
        cfg = Configurator.from_species(simple_species)
        original_viab = cfg.config.viability_fitness
        original_z2g = cfg.config.zygotes_to_gametes_map
        cfg.presets(_fresh_homing())
        published = cfg.config
        assert published.viability_fitness is not original_viab
        assert published.zygotes_to_gametes_map is not original_z2g
        np.testing.assert_array_equal(
            original_viab, 1.0
        ), "the preset fitness patch leaked into the caller's fitness arrays"
        reg = cfg.registry
        zt_het = _ztype_index(reg, simple_species, "WT|Dr", "default")
        np.testing.assert_allclose(
            original_z2g[0, zt_het],
            [0.5, 0.5, 0.0],
            atol=1e-15,
            err_msg="the compiled meiosis map leaked into the caller's map",
        )
        # The published draft does carry the patch (0.5 on the heterozygote).
        np.testing.assert_allclose(
            published.viability_fitness[:, 0, zt_het], 0.5, atol=1e-15
        )

    def test_registry_directory_entries_are_defensive_copies(
        self, simple_species: nt.Species
    ) -> None:
        """Category: ownership.

        Invariant: the registry's directory accessors return defensive
        copies — mutating a returned list leaves the registry's indexing
        intact (the next lookup still resolves all 6 ztypes).
        Attack vector: a live internal list handed out, so a hostile
        recipe could corrupt the name directory every later index lookup
        depends on.
        """
        cfg = Configurator.from_species(simple_species)
        reg = cfg.registry
        ztypes = reg.index_to_ztype
        gtypes = reg.index_to_gtype
        ztypes.append(ztypes[0])
        ztypes.clear()
        gtypes.append(gtypes[0])
        assert reg.n_ztypes == 6, "index_to_ztype mutation reached the registry"
        assert reg.n_gtypes == 3, "index_to_gtype mutation reached the registry"
        # Lookups still resolve after the mutation attempt.
        assert _ztype_index(reg, simple_species, "WT|Dr", "default") < 6
        assert _gtype_index(reg, simple_species, "R2", "default") < 3
