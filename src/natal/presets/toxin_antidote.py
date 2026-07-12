"""Toxin-Antidote gene drive preset.

Public module — provides ToxinAntidoteDrive for TARE/TADE gene drive simulations.
"""

from typing import TYPE_CHECKING, Any, Optional

from natal.genetics import Gene, Genotype
from natal.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.modifiers.module import GameteModifier, ZygoteModifier
from natal.modifiers.zygote_conversion import ZygoteConversionRuleSet
from natal.utils.types import Sex

from ._base import GeneticPreset
from ._fitness import make_fitness_patch_given_allele_scaling
from ._types import (
    AlleleScalingMode,
    AlleleSpecifier,
    FecundityScalingConfig,
    PresetFitnessPatch,
    SexSpecificRates,
    SexualSelectionScalingConfig,
    ViabilityScalingConfig,
    ZygoteViabilityScalingConfig,
    count_allele_copies,
)

if TYPE_CHECKING:
    from natal.population.base import BasePopulation


class ToxinAntidoteDrive(GeneticPreset):
    """Toxin-Antidote gene drive (e.g., TARE, TADE).

    This preset implements a toxin-antidote gene drive system where a "drive" allele
    disrupts a "target" allele into a "disrupted" version. The "disrupted" allele
    typically carries a high fitness cost (the toxin effect), while the "drive"
    allele itself often provides a functional rescue (the antidote).

    Key features include germline disruption, embryo disruption, and
    configurable fitness costs for the disrupted allele.

    In a typical TARE (Toxin-Antidote Recessive Embryo lethality) configuration,
    the disrupted allele is set to be recessive lethal (viability_scaling=0.0,
    viability_mode="recessive").

    Attributes:
        conversion_rate (Tuple[float, float]): Female/male germline disruption rates.
        embryo_disruption_rate (Tuple[float, float]): Female/male embryo disruption rates.
        viability_mode (AlleleScalingMode): Scaling mode for viability effects.
        fecundity_mode (AlleleScalingMode): Scaling mode for fecundity effects.
    """

    def __init__(
        self,
        name: str,
        drive_allele: AlleleSpecifier,
        target_allele: AlleleSpecifier,
        disrupted_allele: AlleleSpecifier,
        conversion_rate: SexSpecificRates = 0.8,
        embryo_disruption_rate: SexSpecificRates = 0.0,
        viability_scaling: ViabilityScalingConfig = 1.0,
        fecundity_scaling: FecundityScalingConfig = 1.0,
        sexual_selection_scaling: Optional[SexualSelectionScalingConfig] = None,
        zygote_viability_scaling: ZygoteViabilityScalingConfig = 0.0,
        viability_mode: AlleleScalingMode = "recessive",
        fecundity_mode: AlleleScalingMode = "recessive",
        sexual_selection_mode: AlleleScalingMode = "recessive",
        zygote_viability_mode: AlleleScalingMode = "recessive",
        cas9_deposition_glab: Optional[str] = None,
        species: Optional[Any] = None,
        priority: int = 0,
        use_paternal_deposition: bool = False,
    ):
        """Initialize a toxin-antidote gene drive.

        Args:
            name: Name of the gene drive.
            drive_allele: The allele carrying the antidote and disruption machinery.
            target_allele: The wild-type allele targeted for disruption.
            disrupted_allele: The resulting non-functional/disrupted allele.
            conversion_rate: Probability of target disruption in the germline.
            embryo_disruption_rate: Probability of target disruption in embryos.
            viability_scaling: Fitness multiplier for the disrupted allele.
            fecundity_scaling: Fecundity multiplier for the disrupted allele.
            sexual_selection_scaling: Optional sexual-selection multiplier for the disrupted allele.
                Supports a scalar copy-number effect or a tuple
                ``(default_male, carrier_male)``.
            zygote_viability_scaling: Zygote viability scaling configuration for the disrupted allele.
                Applied during reproduction stage before survival; represents probability that a zygote
                survives to become an individual.
            viability_mode: Scaling mode for viability (default "recessive").
            fecundity_mode: Scaling mode for fecundity (default "recessive").
            sexual_selection_mode: Scaling mode for scalar sexual-selection values.
                Ignored when ``sexual_selection_scaling`` is a tuple.
            zygote_viability_mode: Scaling mode for zygote viability (default "multiplicative").
            cas9_deposition_glab: Gamete label for Cas9 deposition tracking.
            species: Optional species to bind at construction.
            use_paternal_deposition: Whether to enable paternal Cas9 deposition.
        """
        self._str_drive_allele = self._resolve_allele_name(drive_allele)
        self._str_target_allele = self._resolve_allele_name(target_allele)
        self._str_disrupted_allele = self._resolve_allele_name(disrupted_allele)

        self.conversion_rate = self._resolve_rates(conversion_rate)
        self.embryo_disruption_rate = self._resolve_rates(embryo_disruption_rate)

        self.viability_scaling = viability_scaling
        self.fecundity_scaling = fecundity_scaling
        self.sexual_selection_scaling = sexual_selection_scaling
        self.zygote_viability_scaling = zygote_viability_scaling
        self.viability_mode: AlleleScalingMode = viability_mode
        self.fecundity_mode: AlleleScalingMode = fecundity_mode
        self.sexual_selection_mode: AlleleScalingMode = sexual_selection_mode
        self.zygote_viability_mode: AlleleScalingMode = zygote_viability_mode

        self.cas9_deposition_glab = str(cas9_deposition_glab) if cas9_deposition_glab else None
        self.use_paternal_deposition = bool(use_paternal_deposition)

        super().__init__(name=name, species=species, priority=priority)

    def fitness_patch(self) -> PresetFitnessPatch:
        """Return declarative fitness patch for the disrupted allele."""
        return make_fitness_patch_given_allele_scaling(
            self._str_disrupted_allele,
            self.viability_scaling,
            self.fecundity_scaling,
            self.sexual_selection_scaling,
            self.zygote_viability_scaling,  # zygote_viability_scaling
            self.viability_mode,
            self.fecundity_mode,
            self.sexual_selection_mode,
            self.zygote_viability_mode,  # zygote_viability_mode
        )

    @property
    def drive_allele(self) -> Gene:
        """Gene: The drive allele carrying the toxin-antidote construct."""
        return self._resolve_bound_gene(self._str_drive_allele)

    @property
    def target_allele(self) -> Gene:
        """Gene: The wild-type allele targeted for disruption."""
        return self._resolve_bound_gene(self._str_target_allele)

    @property
    def disrupted_allele(self) -> Gene:
        """Gene: The disrupted (cleaved) allele produced by target disruption."""
        return self._resolve_bound_gene(self._str_disrupted_allele)

    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Implement target disruption in the germline of drive carriers."""
        def drive_carrier_filter(gt: Genotype) -> bool:
            """Return True if the genotype carries at least one drive allele."""
            return count_allele_copies(gt, self.drive_allele) > 0

        rule_set = GameteConversionRuleSet(f"{self.name}_GermlineDisruption")
        for sex in (Sex.FEMALE, Sex.MALE):
            rate = self.conversion_rate[sex]
            if rate > 0:
                rule_set.add_allele_convert(
                    from_allele=self.target_allele,
                    to_allele=self.disrupted_allele,
                    rate=rate,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                )

            if self.cas9_deposition_glab and (sex == Sex.FEMALE or self.use_paternal_deposition):
                rule_set.add_glab_convert(
                    from_glab=None,
                    to_glab=self.cas9_deposition_glab,
                    rate=1.0,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                )

        return rule_set.to_gamete_modifier(population) if rule_set.rules else None

    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Implement target disruption in embryos."""
        rule_set = ZygoteConversionRuleSet(f"{self.name}_EmbryoDisruption")

        def zygote_has_drive(gt: Genotype) -> bool:
            """Return True if the zygote carries at least one drive allele."""
            return count_allele_copies(gt, self.drive_allele) > 0

        for sex in (Sex.FEMALE, Sex.MALE):
            rate = self.embryo_disruption_rate[sex]
            if rate > 0:
                m_glab = self.cas9_deposition_glab if sex == Sex.FEMALE else None
                p_glab = self.cas9_deposition_glab if (sex == Sex.MALE and self.use_paternal_deposition) else None
                g_filter = None if (m_glab or p_glab) else zygote_has_drive

                if m_glab or p_glab or g_filter:
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.disrupted_allele,
                        rate=rate,
                        maternal_glab=m_glab,
                        paternal_glab=p_glab,
                        genotype_filter=g_filter,
                    )

        return rule_set.to_zygote_modifier(population) if rule_set.rules else None
