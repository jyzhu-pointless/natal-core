"""Cross-path parity safety net for the unified compiler (plan 5.1, slice 5).

The two historical modifier-map rebuild paths — the population-side
``refresh_modifier_maps`` and the build-side ``rebuild_config_maps`` —
must produce bit-identical tables from the same modifier list.  These
tests are the safety net under the compile unification: any divergence
between the paths (baseline source, wrapper ordering, axis projection,
offspring derivation) fails here before it can reach users.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.frontend.configurator import Configurator
from natal.frontend.configurator._base import Configurator as BaseConfigurator


def _species(name: str, *, glabs: int = 1, slabs: int = 1) -> nt.Species:
    """Return a two-allele species with optional gamete/somatic labels."""
    gamete_labels = ["default", "cas9"][:glabs]
    somatic_labels = ["normal", "infected"][:slabs] if slabs > 1 else None
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=gamete_labels,
        somatic_labels=somatic_labels,
    )


def _drive(rate: float = 0.9) -> nt.HomingDrive:
    """Return the shared homing drive for parity probes."""
    return nt.HomingDrive(
        name="__parity_drive__",
        drive_allele="B",
        target_allele="A",
        drive_conversion_rate=rate,
        fecundity_scaling={"female": 0.5},
    )


def _build_configurator(species: nt.Species, drive: nt.HomingDrive,
                        *, compress: bool = False) -> Configurator:
    """Return a fully-chained configurator carrying the drive."""
    setup_kwargs: dict[str, object] = {"stochastic": False}  # object: heterogeneous setup kwarg values
    if compress:
        setup_kwargs["compress"] = True
        setup_kwargs["declared_zygote_types"] = ("A|A", "A|B", "B|B")
    cfg = BaseConfigurator.for_age_structured(species)
    cfg.setup(**setup_kwargs)  # type: ignore[arg-type]  # literal union built above
    cfg.age_structure(n_ages=4, new_adult_age=1)
    cfg.initial_state(
        individual_count={
            "female": {"A|A": [0, 60, 0, 0], "A|B": [0, 10, 0, 0]},
            "male": {"A|A": [0, 40, 0, 0]},
        }
    )
    cfg.reproduction(eggs_per_female=17.0)
    cfg.competition(carrying_capacity=500.0, juvenile_growth_mode=3)
    cfg.presets(drive)
    return cfg


def _manual_modifier() -> object:  # object: no-op probe callable of any modifier shape
    """Return a no-op manual gamete modifier (empty mapping)."""

    def no_op() -> dict[str, dict[int, float]]:
        return {}

    return no_op


class TestCrossPathParity:
    """Same modifier list through both rebuild paths → bit-identical maps."""

    def _parity_check(self, species: nt.Species, cfg: Configurator) -> None:
        pop = cfg.build()
        build_tables = (
            pop.config.zygotes_to_gametes_map.copy(),
            pop.config.gametes_to_zygotes_map.copy(),
            pop.config.offspring_tensor.copy(),
        )
        pop.refresh_modifiers()
        assert np.array_equal(
            build_tables[0], pop.config.zygotes_to_gametes_map
        ), "z2g diverged between build and refresh"
        assert np.array_equal(
            build_tables[1], pop.config.gametes_to_zygotes_map
        ), "g2z diverged between build and refresh"
        assert np.array_equal(
            build_tables[2], pop.config.offspring_tensor
        ), "offspring tensor diverged between build and refresh"

    def test_single_drive_uncompressed(self) -> None:
        """One preset, no compression: the plain-path anchor."""
        species = _species("__parity_plain__")
        cfg = _build_configurator(species, _drive())
        self._parity_check(species, cfg)

    def test_drive_plus_manual_modifier(self) -> None:
        """A preset-derived and a manual modifier coexist on the list."""
        species = _species("__parity_mixed__")
        cfg = _build_configurator(species, _drive())
        pop = cfg.build()
        pop.add_gamete_modifier(_manual_modifier(), name="manual")  # type: ignore[arg-type]  # no-op probe modifier

        build_tables = (
            pop.config.zygotes_to_gametes_map.copy(),
            pop.config.offspring_tensor.copy(),
        )
        pop.refresh_modifiers()
        assert np.array_equal(build_tables[0], pop.config.zygotes_to_gametes_map)
        assert np.array_equal(build_tables[1], pop.config.offspring_tensor)

    def test_multi_glab_species(self) -> None:
        """Two gamete labels: the gtype axis is a product axis."""
        species = _species("__parity_glab__", glabs=2)
        cfg = _build_configurator(species, _drive())
        self._parity_check(species, cfg)

    def test_slab_species(self) -> None:
        """Somatic labels: the ztype axis carries the slab dimension."""
        species = _species("__parity_slab__", glabs=1, slabs=2)
        cfg = _build_configurator(species, _drive())
        self._parity_check(species, cfg)

    def test_compressed_with_declared_types(self) -> None:
        """Compression with all genotypes declared keeps both paths equal."""
        species = _species("__parity_compress__")
        cfg = _build_configurator(species, _drive(), compress=True)
        self._parity_check(species, cfg)

    def test_double_refresh_idempotent(self) -> None:
        """Two consecutive refreshes reproduce the build tables exactly."""
        species = _species("__parity_twice__")
        pop = _build_configurator(species, _drive()).build()
        build_z2g = pop.config.zygotes_to_gametes_map.copy()

        pop.refresh_modifiers()
        pop.refresh_modifiers()

        assert np.array_equal(build_z2g, pop.config.zygotes_to_gametes_map)


class TestRuntimePresetsTransaction:
    """Runtime preset registration compiles once and rolls back cleanly.

    The historical path deep-copied the config and replayed against the
    copy; the converged transaction (plan 5.1) executes the recipes once
    against the live population and restores from snapshots on failure —
    the same shape reconfigure_preset uses.
    """

    def _population(self):
        species = _species("__tx_species__")
        pop = (
            BaseConfigurator.for_age_structured(species)
            .setup(stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 60, 0, 0]},
                    "male": {"A|A": [0, 40, 0, 0]},
                }
            )
            .reproduction(eggs_per_female=17.0)
            .competition(carrying_capacity=500.0, juvenile_growth_mode=3)
            .build()
        )
        return pop

    def test_failed_registration_rolls_back_every_surface(self) -> None:
        """An exploding recipe leaves tables, fitness, and the flag intact."""
        pop = self._population()
        ok = _drive(rate=0.8)
        pop.update().presets(ok)

        class Exploding(nt.HomingDrive):
            def gamete_modifier(  # object: probe override of an arbitrary recipe signature
                self, population: object
            ) -> object:
                raise RuntimeError("boom")

        before = (
            pop.config.zygotes_to_gametes_map.copy(),
            pop.config.gametes_to_zygotes_map.copy(),
            pop.config.offspring_tensor.copy(),
            pop.config.viability_fitness.copy(),
            [p.name for p in pop.presets],
            pop._rust_needs_rebuild,  # noqa: SLF001 — the rebuild flag is the rollback surface under test
        )

        with pytest.raises(RuntimeError, match="boom"):
            pop.update().presets(
                Exploding(name="__tx_bad__", drive_allele="B", target_allele="A")
            )

        assert np.array_equal(
            before[0], pop.config.zygotes_to_gametes_map
        )
        assert np.array_equal(before[1], pop.config.gametes_to_zygotes_map)
        assert np.array_equal(before[2], pop.config.offspring_tensor)
        assert np.array_equal(before[3], pop.config.viability_fitness)
        assert [p.name for p in pop.presets] == before[4]
        assert pop._rust_needs_rebuild == before[5]  # noqa: SLF001  # the flag survived or was restored identically

        pop.run(1)
        assert pop.tick == 1

    def test_successful_registration_moves_the_tables(self) -> None:
        """A committed registration lands the drive's meiosis bias."""
        pop = self._population()
        baseline = pop.config.zygotes_to_gametes_map.copy()

        pop.update().presets(_drive(rate=0.8))

        assert not np.array_equal(
            baseline, pop.config.zygotes_to_gametes_map
        ), "the drive's gamete bias must move the meiosis table"


class TestSingleCompilerSpelling:
    """Both rebuild entry points funnel through one compile function."""

    def test_no_second_modifier_application_spelling(self) -> None:
        """Only genetics/compile.py chains build_modifier_wrappers in src.

        The historical duplicated application loops (population mixin and
        registry builder) are gone; a reintroduced second spelling trips
        this scan, guarding the single-owner contract.
        """
        from pathlib import Path

        src_root = Path(__file__).resolve().parent.parent / "src"
        offenders: list[str] = []
        for path in (src_root / "natal").rglob("*.py"):
            # compile.py owns the application; module.py defines the
            # wrapper builder (its def line is not an application).
            if path.name in ("compile.py", "module.py"):
                continue
            text = path.read_text(encoding="utf-8")
            if "build_modifier_wrappers(" in text:
                offenders.append(str(path.relative_to(src_root)))
        assert offenders == [], (
            f"modifier application re-spelled outside the compiler: {offenders}"
        )
