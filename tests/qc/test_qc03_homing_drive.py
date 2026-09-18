"""QC spot-checks 7-9: homing-drive conversion and super-Mendelian spread.

Numerical reference for the trajectory check: with germline conversion h
and neutral fitness, the drive allele frequency follows the standard
homing-drive recurrence p' = p + h*p*(1-p) (e.g. Unckless et al. 2017 /
Hammond et al. 2016 deterministic form), because Dr/Dr adults transmit Dr
exclusively and Dr/+ adults transmit Dr with probability (1+h)/2.
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt


def _drive_species(name: str) -> nt.Species:
    return nt.Species.from_dict(
        name=name,
        structure={"chr1": {"loc": ["WT", "Dr", "R2"]}},
        gamete_labels=["default"],
    )


def _drive(rate: float, **kwargs: object) -> nt.HomingDrive:
    return nt.HomingDrive(
        name="QC_Homing",
        drive_allele="Dr",
        target_allele="WT",
        resistance_allele="R2",
        drive_conversion_rate=rate,
        late_germline_resistance_formation_rate=0.0,
        embryo_resistance_formation_rate=0.0,
        functional_resistance_ratio=0.0,
        fecundity_scaling=1.0,
        viability_scaling=1.0,
        **kwargs,  # type: ignore[arg-type]
    )


def _build(pop_name: str, species: nt.Species, drive: nt.HomingDrive,
           female_counts: dict[str, float], male_counts: dict[str, float]):
    return (
        nt.DiscreteGenerationPopulation.setup(
            species=species, name=pop_name, stochastic=False
        )
        .initial_state(individual_count={"female": female_counts, "male": male_counts})
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=1, sex_ratio=0.5)
        .competition(carrying_capacity=1e12, low_density_growth_rate=2.0,
                     growth_mode="no_competition")
        .presets(drive)
        .build()
    )


def _carrier_frequency(pop: nt.DiscreteGenerationPopulation) -> float:
    """Frequency of ztypes that contain the Dr allele (either phase)."""
    total = 0.0
    carrier = 0.0
    counts = pop.state.individual_count
    for genotype, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype, slab)
        mass = float(counts[:, :, idx].sum())
        total += mass
        # Allele names are disjoint here, so the string form is exact.
        if "Dr" in genotype.to_string():
            carrier += mass
    return carrier / total


def _ztype_masses(pop: nt.DiscreteGenerationPopulation) -> dict[str, float]:
    masses: dict[str, float] = {}
    counts = pop.state.individual_count
    for genotype, slab in pop.registry.index_to_ztype:
        idx = pop.registry.ztype_index(genotype, slab)
        masses[genotype.to_string()] = float(counts[:, :, idx].sum())
    return masses


class TestHomingConversionExactness:
    def test_het_x_het_offspring_fractions(self) -> None:
        """Claim: h=0.9 het x het zygotes are 0.9025 / 0.095 / 0.0025.

        Reference: each het parent transmits Dr with (1+h)/2 = 0.95, so
        Dr/Dr = 0.95^2, het = 2*0.95*0.05, WT/WT = 0.05^2.  Rejects
        applying conversion per-zygote (0.9 misread) or per-parent
        double application.
        """
        species = _drive_species("qc_home_exact")
        pop = _build(
            "qc_home_exact_pop", species, _drive(0.9),
            {"WT|Dr": 500}, {"WT|Dr": 500},
        )
        pop.run(1)
        masses = _ztype_masses(pop)
        total = sum(masses.values())
        assert total == pytest.approx(500.0, abs=1e-9)
        by_gtype = {
            "|".join(sorted(key.split("|"))): mass
            for key, mass in masses.items()
        }
        assert by_gtype["Dr|Dr"] == pytest.approx(500.0 * 0.9025, abs=1e-9)
        assert by_gtype["Dr|WT"] == pytest.approx(500.0 * 0.095, abs=1e-9)
        assert by_gtype["WT|WT"] == pytest.approx(500.0 * 0.0025, abs=1e-9)


def _allele_frequency(pop: nt.DiscreteGenerationPopulation) -> float:
    """Dr allele copy frequency: (het + 2*hom) / (2*total)."""
    masses = _ztype_masses(pop)
    total = sum(masses.values())
    het = sum(m for k, m in masses.items() if set(k.split("|")) == {"WT", "Dr"})
    hom = masses.get("Dr|Dr", 0.0)
    return (het + 2.0 * hom) / (2.0 * total)


class TestHomingTrajectory:
    def test_deterministic_recurrence_chain(self) -> None:
        """Claim: the newborn Dr allele frequency equals the parental
        gamete-pool Dr frequency g = f_hom + (1+h)/2 * f_het, computed
        from the ACTUAL parental genotype frequencies; under random
        mating this reproduces the classic p' = p + h*p*(1-p).

        Allele frequency, not carrier frequency, is the recurrence state
        (the two coincide only when every carrier is heterozygote).
        Rejects wrong generation coupling, drive acting on zygotes, or
        density regulation leaking into allele frequencies.
        """
        h = 0.8
        species = _drive_species("qc_home_traj")
        pop = _build(
            "qc_home_traj_pop", species, _drive(h),
            {"WT|WT": 900, "WT|Dr": 100}, {"WT|WT": 900, "WT|Dr": 100},
        )
        for _ in range(8):
            masses = _ztype_masses(pop)
            total = sum(masses.values())
            f_hom = masses.get("Dr|Dr", 0.0) / total
            f_het = sum(
                m for k, m in masses.items() if set(k.split("|")) == {"WT", "Dr"}
            ) / total
            expected_allele = f_hom + (1.0 + h) / 2.0 * f_het
            pop.run(1)
            p_now = _allele_frequency(pop)
            assert p_now == pytest.approx(expected_allele, abs=1e-9), (
                p_now, expected_allele
            )
            # Hardy-Weinberg identity for the neutral two-allele system.
            het = sum(
                m for k, m in _ztype_masses(pop).items()
                if set(k.split("|")) == {"WT", "Dr"}
            )
            hom = _ztype_masses(pop).get("Dr|Dr", 0.0)
            new_total = sum(_ztype_masses(pop).values())
            assert het == pytest.approx(2 * p_now * (1 - p_now) * new_total, abs=1e-6)
            assert hom == pytest.approx(p_now * p_now * new_total, abs=1e-6)

    def test_drive_sweeps_from_low_frequency(self) -> None:
        """Claim: a super-Mendelian drive (h=0.8) starting at allele
        frequency 0.05 rises monotonically toward fixation."""
        species = _drive_species("qc_home_sweep")
        pop = _build(
            "qc_home_sweep_pop", species, _drive(0.8),
            {"WT|WT": 950, "WT|Dr": 50}, {"WT|WT": 950, "WT|Dr": 50},
        )
        previous = _allele_frequency(pop)
        for _ in range(20):
            pop.run(1)
            current = _allele_frequency(pop)
            assert current >= previous - 1e-12, (previous, current)
            previous = current
        assert previous > 0.5


class TestSexSpecificRates:
    def test_female_only_conversion(self) -> None:
        """Claim: {"female": 1.0, "male": 0.0} -> het mother gives 100%."""
        species = _drive_species("qc_home_f")
        pop = _build(
            "qc_home_f_pop", species,
            _drive({"female": 1.0, "male": 0.0}),
            {"WT|Dr": 500}, {"WT|WT": 500},
        )
        pop.run(1)
        assert _carrier_frequency(pop) == pytest.approx(1.0, abs=1e-12)

    def test_male_zero_conversion(self) -> None:
        """Claim: het father with male rate 0 transmits at Mendelian 1/2."""
        species = _drive_species("qc_home_m")
        pop = _build(
            "qc_home_m_pop", species,
            _drive({"female": 1.0, "male": 0.0}),
            {"WT|WT": 500}, {"WT|Dr": 500},
        )
        pop.run(1)
        assert _carrier_frequency(pop) == pytest.approx(0.5, abs=1e-12)

    def test_explicit_zero_dict_is_respected(self) -> None:
        """Claim: explicit {"female": 0.0, "male": 0.9} keeps female at 1/2.

        The rate resolver chains ``or`` over dict keys, so a falsy 0.0
        falls through the chain; the final fallback is also 0.0, so the
        resolved value should still be 0.0 -> het mother transmits Dr at
        Mendelian 1/2.  Guards against any future fallback change.
        """
        species = _drive_species("qc_home_zero")
        pop = _build(
            "qc_home_zero_pop", species,
            _drive({"female": 0.0, "male": 0.9}),
            {"WT|Dr": 500}, {"WT|WT": 500},
        )
        pop.run(1)
        assert _carrier_frequency(pop) == pytest.approx(0.5, abs=1e-12)

    def test_scalar_ninety_symmetry(self) -> None:
        """Claim: scalar h=0.9 is symmetric across the sexes."""
        species = _drive_species("qc_home_sym")
        pop = _build(
            "qc_home_sym_pop", species, _drive(0.9),
            {"WT|Dr": 500}, {"WT|WT": 500},
        )
        pop.run(1)
        assert _carrier_frequency(pop) == pytest.approx(0.95, abs=1e-12)
