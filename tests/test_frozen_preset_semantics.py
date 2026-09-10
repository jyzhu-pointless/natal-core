"""Frozen user-surface contract samples: preset registration semantics.

RUST_ONLY_REFACTOR_PLAN.md sections 2.1 and 5.3 freeze the preset
rules: species binding, idempotent registration by object identity,
priority ordering, manual-modifier ordering, fitness composition, and
the existing reconfiguration semantics (including the fact that a
preset reconfiguration rebuilds fitness and thereby overwrites manual
fitness writes — an accepted behavior that must not silently change).

Every probability assertion is the closed-form Mendelian×drive value:
for a heterozygote ``WT|Dr`` with conversion rate ``r``, gametes are
``WT`` with probability ``0.5·(1-r)`` and ``Dr`` with ``0.5 + 0.5·r``.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _species() -> nt.Species:
    """Return the shared two-allele species for preset samples."""
    return nt.Species.from_dict(
        name="FrozenPresetSpecies",
        structure={"chr1": {"loc": ["WT", "Dr"]}},
        gamete_labels=["default"],
    )


def _base_builder(name: str) -> nt.PopulationBuilder:
    """Return a minimal discrete-generation builder chain."""
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=_species(), name=name, stochastic=False
        )
        .initial_state(
            individual_count={
                "female": {"WT|WT": 10},
                "male": {"WT|WT": 10},
            }
        )
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
    )


def _heterozygote_row(pop: nt.DiscreteGenerationPopulation) -> np.ndarray:
    """Return the female meiosis row of the WT|Dr heterozygote.

    Diagnostic read of the public ``pop.config`` snapshot (the frozen
    surface pins the numeric meaning, not this particular accessor).
    """
    names = [str(gt) for gt in pop.index_registry.index_to_genotype]
    return pop.config.zygotes_to_gametes_map[0, names.index("WT|Dr"), :]


# ══════════════════════════════════════════════════════════════════════════════
# Species binding
# ══════════════════════════════════════════════════════════════════════════════


def test_bind_species_then_conflicting_rebind_raises() -> None:
    """A preset binds one species; binding another raises ValueError."""
    first = _species()
    other = nt.Species.from_dict(
        name="FrozenPresetOtherSpecies",
        structure={"chr1": {"loc": ["A", "B"]}},
        gamete_labels=["default"],
    )
    preset = nt.HomingDrive(
        name="FrozenBindDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.5,
    )

    preset.bind_species(first)
    # Rebinding the same species instance is idempotent (no error, no
    # rebinding side effects); the binding is observable only through
    # which species the conflict message names.
    preset.bind_species(first)

    with pytest.raises(ValueError, match="already bound to species 'FrozenPresetSpecies'"):
        preset.bind_species(other)


# ══════════════════════════════════════════════════════════════════════════════
# Registration idempotency and priority
# ══════════════════════════════════════════════════════════════════════════════


def test_registration_idempotent_by_object_identity() -> None:
    """Applying the same preset instance twice applies its effect once."""
    drive = nt.HomingDrive(
        name="FrozenIdempotentDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.95,
    )
    pop = _base_builder("FrozenIdempotentPop").build()

    pop.apply_preset(drive)
    first = _heterozygote_row(pop).copy()
    assert [p.name for p in pop.presets] == ["FrozenIdempotentDrive"]

    pop.apply_preset(drive)
    np.testing.assert_allclose(_heterozygote_row(pop), first)
    # The public snapshots stay single-entry: no duplicate registration.
    assert [p.name for p in pop.presets] == ["FrozenIdempotentDrive"]
    assert [name for _, name, _ in pop.gamete_modifiers] == [
        "FrozenIdempotentDrive/gamete"
    ]


def test_priority_orders_runtime_modifier_application() -> None:
    """Runtime application orders preset modifiers by ascending priority."""
    late = nt.HomingDrive(
        name="FrozenLateDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.1,
        priority=10,
    )
    early = nt.HomingDrive(
        name="FrozenEarlyDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.9,
        priority=0,
    )
    pop = _base_builder("FrozenPriorityPop").build()

    # Register in an order deliberately opposite to priority.
    pop.apply_preset(late)
    pop.apply_preset(early)

    names = [name for _, name, _ in pop.gamete_modifiers]
    assert names == ["FrozenEarlyDrive/gamete", "FrozenLateDrive/gamete"]


def test_manual_modifiers_append_after_preset_modifiers() -> None:
    """Manual modifiers join the list after preset-derived entries."""
    drive = nt.HomingDrive(
        name="FrozenManualOrderDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.5,
    )
    pop = _base_builder("FrozenManualOrderPop").build()
    pop.apply_preset(drive)

    def no_op() -> dict[str, dict[int, float]]:
        return {}

    pop.add_gamete_modifier(no_op, name="manual_entry")

    names = [name for _, name, _ in pop.gamete_modifiers]
    assert names == ["FrozenManualOrderDrive/gamete", "manual_entry"]


# ══════════════════════════════════════════════════════════════════════════════
# Probability tables and reconfiguration
# ══════════════════════════════════════════════════════════════════════════════


def test_homing_conversion_probability_table() -> None:
    """Heterozygote gametes follow ``0.5·(1-r)`` / ``0.5 + 0.5·r``."""
    drive = nt.HomingDrive(
        name="FrozenTableDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.95,
    )
    pop = _base_builder("FrozenTablePop").presets(drive).build()

    row = _heterozygote_row(pop)
    # Genotype order WT|WT, WT|Dr, Dr|Dr; haploid order WT, Dr.
    np.testing.assert_allclose(row, [0.5 * (1 - 0.95), 0.5 + 0.5 * 0.95])
    np.testing.assert_allclose(row.sum(), 1.0)


def test_reconfigure_preset_updates_probability_table() -> None:
    """``reconfigure_preset`` rebuilds the tables with the new rate."""
    drive = nt.HomingDrive(
        name="FrozenReconfDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.95,
    )
    pop = _base_builder("FrozenReconfPop").presets(drive).build()

    pop.update().reconfigure_preset(drive, drive_conversion_rate=0.3)

    row = _heterozygote_row(pop)
    np.testing.assert_allclose(row, [0.5 * (1 - 0.3), 0.5 + 0.5 * 0.3])
    np.testing.assert_allclose(row.sum(), 1.0)


def test_reconfigure_unknown_attribute_raises_and_keeps_tables() -> None:
    """A failing reconfiguration leaves the live tables untouched."""
    drive = nt.HomingDrive(
        name="FrozenBadReconfDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.95,
    )
    pop = _base_builder("FrozenBadReconfPop").presets(drive).build()
    before = _heterozygote_row(pop).copy()

    with pytest.raises(AttributeError):
        pop.update().reconfigure_preset(drive, no_such_knob=0.1)

    np.testing.assert_allclose(_heterozygote_row(pop), before)


# ══════════════════════════════════════════════════════════════════════════════
# Fitness composition
# ══════════════════════════════════════════════════════════════════════════════


def test_preset_fitness_patch_declares_per_allele_and_per_slab_rules() -> None:
    """Fitness patches are declared data, not applied tensors."""
    homing = nt.HomingDrive(
        name="FrozenPatchHoming",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.5,
        fecundity_scaling={"female": 0.6},
    )
    assert homing.fitness_patch() == {
        "viability_per_allele": {("Dr",): (1.0, "multiplicative")},
        "fecundity_per_allele": {("Dr",): ({"female": 0.6}, "multiplicative")},
        "sexual_selection_per_allele": {("Dr",): (1.0, "multiplicative")},
        "zygote_per_allele": {("Dr",): (1.0, "multiplicative")},
    }

    slab_species = nt.Species.from_dict(
        name="FrozenPatchSlabSpecies",
        structure={"c1": {"l1": ["WT", "Dr"]}},
        gamete_labels=["default", "wolbachia"],
        somatic_labels=["normal", "infected"],
    )
    wolbachia = nt.Wolbachia(
        name="FrozenPatchWolbachia",
        infected_slab="infected",
        viability_scaling=0.9,
    )
    wolbachia.bind_species(slab_species)
    assert wolbachia.fitness_patch() == {"viability_per_slab": {"infected": 0.9}}


def test_manual_fitness_write_lands_in_tensor_and_survives_run_start() -> None:
    """``pop.update().fitness(...)`` writes genotype×sex fitness values."""
    pop = _base_builder("FrozenManualFitnessPop").build()

    pop.update().fitness(viability={"female": {"WT|WT": 0.5}, "male": {"WT|WT": 0.25}})

    tensor = pop.params.viability_fitness.array
    names = [str(gt) for gt in pop.index_registry.index_to_genotype]
    wt = names.index("WT|WT")
    # Values land on the age-0 layer, per sex; other cells stay neutral.
    np.testing.assert_allclose(tensor[0, 0, wt], 0.5)
    np.testing.assert_allclose(tensor[1, 0, wt], 0.25)
    np.testing.assert_allclose(tensor[:, 1, :], 1.0)
    np.testing.assert_allclose(tensor[:, 0, names.index("WT|Dr"):], 1.0)


def test_preset_reconfiguration_rebuilds_fitness_over_manual_writes() -> None:
    """A preset reconfiguration resets fitness and re-applies preset patches.

    This pins the accepted behavior called out in plan section 5.3:
    manual fitness writes made before a preset reconfiguration are
    overwritten because preset-derived fitness is rebuilt from scratch.
    """
    drive = nt.HomingDrive(
        name="FrozenOverrideDrive",
        drive_allele="Dr",
        target_allele="WT",
        drive_conversion_rate=0.5,
    )
    pop = _base_builder("FrozenOverridePop").presets(drive).build()
    names = [str(gt) for gt in pop.index_registry.index_to_genotype]
    het = names.index("WT|Dr")

    pop.update().fitness(viability={"WT|Dr": 0.5})
    np.testing.assert_allclose(
        pop.params.viability_fitness.array[:, 0, het], [0.5, 0.5]
    )

    pop.update().reconfigure_preset(drive, drive_conversion_rate=0.4)

    # The rebuild fills 1.0 and the preset's neutral (1.0) per-allele
    # patch composes to 1.0 — the manual 0.5 is gone.
    np.testing.assert_allclose(
        pop.params.viability_fitness.array[:, 0, het], [1.0, 1.0]
    )
