"""QC spot-check 23: Wolbachia slab semantics on the public builder path.

Corrected verdicts after adversarial review.  Maternal transmission
WORKS on the modern ``.presets()`` path when the species declares the
required ``"wolbachia"`` gamete label (documented in the Wolbachia
docstring; see tests/test_slab_integration.py for the canonical setup):
infected mothers then produce exactly 100% infected offspring.

What remains as a real (low-severity) finding: when the required label
is missing, the preset degrades to a SILENT no-op — no gamete/zygote
modifier is registered, no error is raised, and offspring are all
normal.  The docstring requirement is the only guard.  Suggested UX
repair: raise a configuration error when ``_maternal_map`` references an
unregistered gamete label.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _species(name: str, gamete_labels: list[str] | None = None) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"c": {"l": ["A", "B"]}},
        gamete_labels=gamete_labels or ["default", "wolbachia"],
        somatic_labels=["normal", "infected"],
    )


def _slab_masses(pop: nt.DiscreteGenerationPopulation) -> dict[str, float]:
    masses: dict[str, float] = {}
    counts = pop.state.individual_count
    for genotype, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype, slab)
        masses[slab] = masses.get(slab, 0.0) + float(counts[:, :, idx].sum())
    return masses


def _build(name: str, female_counts: dict[str, float], male_counts: dict[str, float],
           wolbachia: nt.Wolbachia, eggs: float, gamete_labels: list[str] | None = None):
    sp = _species(name + "_sp", gamete_labels)
    return (
        nt.DiscreteGenerationPopulation.setup(species=sp, name=name, stochastic=False)
        .initial_state(individual_count={"female": female_counts, "male": male_counts})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=eggs, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                     growth_mode="fixed")
        .presets(wolbachia)
        .build()
    )


class TestWolbachiaTransmission:
    def test_maternal_transmission_complete(self) -> None:
        """Claim: infected mothers -> 100% infected offspring.

        300 infected + 300 normal mothers, uninfected fathers, one egg
        each -> 600 offspring split exactly 300 infected / 300 normal.
        """
        pop = _build(
            "qc_wol_mat",
            {"A|A@infected": 300, "A|A@normal": 300},
            {"A|A@normal": 600},
            nt.Wolbachia(name="wMel", infected_slab="infected"),
            eggs=1,
        )
        assert pop.gamete_modifiers, "preset must register its gamete modifier"
        pop.run(1)
        masses = _slab_masses(pop)
        assert masses.get("infected", 0.0) == pytest.approx(300.0, abs=1e-9)
        assert masses.get("normal", 0.0) == pytest.approx(300.0, abs=1e-9)

    def test_no_paternal_transmission(self) -> None:
        """Claim: infected fathers + normal mothers -> all-normal
        offspring (paternal transmission is correctly absent)."""
        pop = _build(
            "qc_wol_pat",
            {"A|A@normal": 600},
            {"A|A@infected": 600},
            nt.Wolbachia(name="wMel", infected_slab="infected"),
            eggs=1,
        )
        pop.run(1)
        masses = _slab_masses(pop)
        assert masses.get("normal", 0.0) == pytest.approx(600.0, abs=1e-9)
        assert masses.get("infected", 0.0) == 0.0

    def test_missing_required_glabel_is_rejected(self) -> None:
        """Claim (repaired): a species without the required 'wolbachia'
        gamete label fails the build instead of degrading to a silent no-op.

        Previously the preset registered no modifiers, raised nothing, and
        all offspring came out normal — a user who forgot the label got a
        running simulation with no transmission and no diagnostic.  The
        preset now reports the missing label while its modifiers resolve.
        """
        with pytest.raises(ValueError, match="wolbachia"):
            _build(
                "qc_wol_noglab",
                {"A|A@infected": 300, "A|A@normal": 300},
                {"A|A@normal": 600},
                nt.Wolbachia(name="wMel", infected_slab="infected"),
                eggs=1,
                gamete_labels=["default"],
            )

    def test_slab_viability_scaling_registered(self) -> None:
        """Claim: viability_scaling=0.9 lands in the infected-slab
        viability entries of the compiled config."""
        sp = _species("qc_wol_viab_sp")
        pop = (
            nt.DiscreteGenerationPopulation.setup(
                species=sp, name="qc_wol_viab", stochastic=False
            )
            .initial_state(individual_count={
                "female": {"A|A@infected": 300, "A|A@normal": 300},
                "male": {"A|A@normal": 600},
            })
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=1, sex_ratio=0.5)
            .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                         growth_mode="fixed")
            .presets(nt.Wolbachia(name="wMel", infected_slab="infected",
                                  viability_scaling=0.9))
            .build()
        )
        viab = np.asarray(pop.params.viability)
        # Discrete model reads viability at age = new_adult_age - 1 = 0.
        assert viab[0, 0, 0] == pytest.approx(1.0)   # normal female
        assert viab[1, 0, 0] == pytest.approx(1.0)   # normal male
        assert viab[0, 0, 1] == pytest.approx(0.9)   # infected female
        assert viab[1, 0, 1] == pytest.approx(0.9)   # infected male
