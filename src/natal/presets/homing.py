"""Homing-based gene drive preset.

Public module — provides HomingDrive for CRISPR/Cas9 gene drive simulations.
"""

from typing import TYPE_CHECKING, Any, Optional

from natal.genetics import Gene, Genotype
from natal.modifiers.module import GameteModifier, ZygoteModifier
from natal.utils.types import Sex

from ._base import GeneticPreset
from ._fitness import (
    make_fitness_patch_given_allele_scaling,
)
from ._types import (
    AlleleScalingMode,
    AlleleSpecifier,
    PresetFitnessPatch,
    SexSpecificRates,
    FecundityScalingConfig,
    SexualSelectionScalingConfig,
    ViabilityScalingConfig,
    ZygoteViabilityScalingConfig,
)
from .gamete_conversion import GameteConversionRuleSet
from .zygote_conversion import ZygoteConversionRuleSet

if TYPE_CHECKING:
    from natal.population.base import BasePopulation


class HomingDrive(GeneticPreset):
    """Homing-based gene drive (e.g., CRISPR/Cas9 homing drives).

    This preset implements a homing gene drive that spreads through homology-directed
    repair (HDR) converting wild-type alleles into drive alleles in heterozygotes.
    It can also generate resistance alleles through non-homologous end joining (NHEJ).

    Key features include drive conversion in heterozygotes, germline/embryo
    resistance formation, optional parental Cas9 deposition, and sex-specific
    rate control.

    The drive operates through a sequential cascade:
    1. Homing conversion (WT -> Drive)
    2. Resistance formation in remaining WT alleles
    3. Optional functional resistance split

    Attributes:
        drive_conversion_rate (Tuple[float, float]): Female/male homing rates.
        late_germline_resistance_formation_rate (Tuple[float, float]): Female/male
            late germline resistance rates.
        embryo_resistance_formation_rate (Tuple[float, float]): Female/male embryo
            resistance rates.

    Examples:
        drive = HomingDrive(
            name="MyDrive",
            drive_allele="Drive",
            target_allele="WT",
            resistance_allele="Resistance",
            drive_conversion_rate=0.95,
            late_germline_resistance_formation_rate=0.03
        )
        population.apply_preset(drive)
    """

    def __init__(
        self,
        name: str,
        drive_allele: AlleleSpecifier,
        target_allele: AlleleSpecifier,
        resistance_allele: Optional[AlleleSpecifier] = None,
        functional_resistance_allele: Optional[AlleleSpecifier] = None,
        cas9_allele: Optional[AlleleSpecifier] = None,
        drive_conversion_rate: SexSpecificRates = 0.5,
        late_germline_resistance_formation_rate: SexSpecificRates = 0.0,
        embryo_resistance_formation_rate: SexSpecificRates = 0.0,
        functional_resistance_ratio: float = 0.0,
        fecundity_scaling: FecundityScalingConfig = 1.0,
        viability_scaling: ViabilityScalingConfig = 1.0,
        sexual_selection_scaling: SexualSelectionScalingConfig = 1.0,
        zygote_viability_scaling: ZygoteViabilityScalingConfig = 1.0,
        viability_mode: AlleleScalingMode = "multiplicative",
        fecundity_mode: AlleleScalingMode = "multiplicative",
        sexual_selection_mode: AlleleScalingMode = "multiplicative",
        zygote_viability_mode: AlleleScalingMode = "multiplicative",
        cas9_deposition_glab: Optional[str] = None,
        species: Optional[Any] = None,
        priority: int = 0,
        use_paternal_deposition: bool = False,
    ):
        """Initialize a homing-based gene drive (e.g., CRISPR/Cas9 homing drives).

        This drive spreads via homology-directed repair (HDR) converting wild-type alleles into drive alleles in heterozygotes.
        It can also generate resistance alleles through non-homologous end joining (NHEJ).

        Args:
            name (str): Name of the gene drive.
            drive_allele (str or Gene): The allele carrying the drive cassette.
            target_allele (str or Gene): The wild-type allele targeted by the drive.
            resistance_allele (str or Gene, optional): The non-functional resistance allele formed by NHEJ.
            functional_resistance_allele (str or Gene, optional): The functional resistance allele
                formed by in-frame NHEJ. If not provided, assume no functional resistance.
            cas9_allele (str or Gene, optional): The allele carrying Cas9 for cleavage, used
                when modeling a split drive where Cas9 is separate from the drive locus.
            drive_conversion_rate (float or dict): Probability of drive conversion caused by Cas9 cleavage
                and homology-directed repair in heterozygotes. Can be a single float (applies to both sexes),
                a dict with sex keys, or a tuple (female_rate, male_rate) for sex-specific rates.
            late_germline_resistance_formation_rate (float or dict): Probability of resistance formation
                *after* drive conversion in the germline. Can be a single float (applies to both sexes),
                a dict with sex keys, or a tuple (female_rate, male_rate) for sex-specific rates.
            embryo_resistance_formation_rate (float or dict): Probability of resistance formation
                in embryos due to maternal/paternal Cas9 deposition. Can be a single float, dict, or tuple.
            functional_resistance_ratio (float): Proportion of resistance alleles that are functional
                (in-frame mutations). Range: 0.0 (all non-functional) to 1.0 (all functional).
            fecundity_scaling (float or dict): Fitness multiplier for drive carriers affecting fecundity.
                Applied multiplicatively based on allele copy number.
            viability_scaling (float or dict): Fitness multiplier for drive carriers affecting viability.
                Applied multiplicatively based on allele copy number.
            sexual_selection_scaling (float or tuple): Fitness multiplier affecting sexual selection.
                Can be a single float or tuple (default_selection, carrier_selection).
            zygote_viability_scaling (float or dict): Fitness multiplier affecting survival of zygotes before
                competition takes place. Applied multiplicatively based on allele copy number.
            viability_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            fecundity_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            sexual_selection_mode (str): Scaling mode for scalar sexual_selection_scaling.
                Note: if sexual_selection_scaling is a tuple, mode is ignored.
            zygote_viability_mode (str): Scaling mode: "multiplicative", "dominant", "recessive", or "custom".
                If "custom", scaling values must be tuples (het_val, hom_val).
            cas9_deposition_glab (str, optional): Gamete label for Cas9 deposition tracking.
                Used for maternal/paternal effect modeling.
            species (Species, optional): Species to bind at construction time. If None,
                will be bound when applied to population.
            use_paternal_deposition (bool): Whether to enable paternal Cas9 deposition.
                If True, fathers can deposit Cas9 in embryos.

        Examples:
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     resistance_allele="R2",
            ...     drive_conversion_rate=0.95,
            ...     late_germline_resistance_formation_rate=0.03
            ... )
            >>> population.apply_preset(drive)
        """
        self._str_drive_allele = self._resolve_allele_name(drive_allele)
        self._str_target_allele = self._resolve_allele_name(target_allele)
        self._str_resistance_allele = (self._resolve_allele_name(resistance_allele)
            if resistance_allele else None)
        self._str_functional_resistance_allele = (self._resolve_allele_name(functional_resistance_allele)
            if functional_resistance_allele else None)
        self._str_cas9_allele = self._resolve_allele_name(cas9_allele) if cas9_allele else None

        self.drive_conversion_rate = self._resolve_rates(drive_conversion_rate)
        self.late_germline_resistance_formation_rate = self._resolve_rates(late_germline_resistance_formation_rate)
        self.embryo_resistance_formation_rate = self._resolve_rates(embryo_resistance_formation_rate)
        self.functional_resistance_ratio = float(functional_resistance_ratio)

        # Store declarative fitness scaling configs.
        self.fecundity_scaling = fecundity_scaling
        self.viability_scaling = viability_scaling
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
        """Return declarative fitness patch for homing drive scaling configs."""
        # Combine drive and non-functional resistance alleles into a single group.
        # This ensures that a "Drive|Resistance" genotype is treated as having
        # 2 copies of the "disrupted" allele class, which is crucial for correct
        # dominant/recessive scaling logic.
        alleles = [self._str_drive_allele]
        if self._str_resistance_allele:
            alleles.append(self._str_resistance_allele)

        patch = make_fitness_patch_given_allele_scaling(
            alleles,
            self.viability_scaling,
            self.fecundity_scaling,
            self.sexual_selection_scaling,
            self.zygote_viability_scaling,
            self.viability_mode,
            self.fecundity_mode,
            self.sexual_selection_mode,
            self.zygote_viability_mode,
        )

        return patch

    def _instantiate_allele(self, allele_name: str, population: 'BasePopulation[Any]') -> Gene:
        """Helper to get Gene object for an allele name from the population's species."""
        gene = population.species.gene_index.get(allele_name)
        if gene is None:
            raise ValueError(f"Allele '{allele_name}' not found in species '{population.species.name}'.")
        return gene

    @property
    def drive_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_drive_allele)

    @property
    def target_allele(self) -> Gene:
        return self._resolve_bound_gene(self._str_target_allele)

    @property
    def resistance_genotype(self) -> Gene:
        if self._str_resistance_allele is None:
            raise ValueError(f"Resistance allele not defined in HomingDrive '{self.name}'.")
        return self._resolve_bound_gene(self._str_resistance_allele)

    @property
    def functional_resistance_allele(self) -> Optional[Gene]:
        if self._str_functional_resistance_allele is None:
            return None
        return self._resolve_bound_gene(self._str_functional_resistance_allele)

    @property
    def cas9_allele(self) -> Optional[Gene]:
        if self._str_cas9_allele is None:
            return None
        return self._resolve_bound_gene(self._str_cas9_allele)

    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Implement homing in heterozygous parents, germline resistance, and Cas9 deposition.

        In heterozygotes (drive/wild-type), gametes are biased towards drive.
        """
        def drive_carrier_filter(gt: Genotype) -> bool:
            from natal.presets._types import count_allele_copies

            has_drive = count_allele_copies(gt, self.drive_allele) > 0
            if self.cas9_allele:
                has_cas9 = count_allele_copies(gt, self.cas9_allele) > 0
                return has_drive and has_cas9
            return has_drive

        # RuleSet compiles these rules into a Sequential Cascade.
        # This means the target pool shrinks after every rule.
        # So Rule 2 (Resistance) only acts on the targets that FAILED Rule 1 (Homing).
        rule_set = GameteConversionRuleSet(f"{self.name}_Homing")
        for sex in (Sex.FEMALE, Sex.MALE):
            homing_rate = self.drive_conversion_rate[sex]
            res_rate = self.late_germline_resistance_formation_rate[sex]

            # 1. Homing (Target -> Drive)
            # Examples: If homing_rate is 0.7, 70% of targets become Drive. 30% pass to the next rule.
            if homing_rate > 0:
                rule_set.add_allele_convert(
                    from_allele=self.target_allele,
                    to_allele=self.drive_allele,
                    rate=homing_rate,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                )

            # 2. Germline Resistance (Target -> Resistance)
            # This operates ON THE REMAINDER of the target alleles (e.g. the 30% that survived Homing).
            if res_rate > 0:
                if self.functional_resistance_allele and self.functional_resistance_ratio > 0:
                    # 2a. Functional resistance
                    # Applying absolute `res_rate * func_res_ratio` directly works because GameteAlleleConversionRule
                    # calculates rates against the *current* target pool. So if 30% targets are left, and this
                    # rate is 0.1, it converts 10% of that 30% (overall 3% of origin).
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.functional_resistance_allele,
                        rate=res_rate * self.functional_resistance_ratio,
                        sex_filter=sex,
                        genotype_filter=drive_carrier_filter,
                    )

                    # 2b. Non-functional resistance
                    # The functional rule above removed `res_rate * func_res_ratio` from the available targets.
                    # To hit the correct math for the *remaining* non-functional portion, we divide the
                    # non-functional rate by whatever remains of the target pool after the functional edits.
                    target_remaining = 1.0 - (res_rate * self.functional_resistance_ratio)
                    adjusted_nf_rate = ((res_rate * (1.0 - self.functional_resistance_ratio))
                                        / target_remaining) if target_remaining > 0 else 0.0
                    if adjusted_nf_rate > 0:
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele,
                            to_allele=self.resistance_genotype,
                            rate=adjusted_nf_rate,
                            sex_filter=sex,
                            genotype_filter=drive_carrier_filter,
                        )
                else:
                    # Generic resistance (no functional/non-functional split)
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.resistance_genotype,
                        rate=res_rate,
                        sex_filter=sex,
                        genotype_filter=drive_carrier_filter,
                    )

            # 3. Gamete labeling for maternal Cas9 deposition
            # Instead of editing alleles, this tags the entire output gamete from drive-carrying females
            # with `cas9_deposition_glab`. The zygote modifier will read this tag to apply embryo resistance.
            if sex == Sex.FEMALE or self.use_paternal_deposition:
                rule_set.add_hg_convert(
                    hg_match=lambda hg: True,
                    to_haploid_genotype=lambda hg: hg,
                    rate=1.0,
                    sex_filter=sex,
                    genotype_filter=drive_carrier_filter,
                    target_glab=self.cas9_deposition_glab
                )

        return rule_set.to_gamete_modifier(population) if rule_set.rules else None

    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Implement embryo resistance.

        Cleavage in the embryo (due to deposited Cas9 or zygotic expression)
        converts wild-type alleles into resistance alleles.
        """
        rule_set = ZygoteConversionRuleSet(f"{self.name}_EmbryoResistance")

        def zygote_has_cas9(gt: Genotype) -> bool:
            """Check if the zygote itself carries the Cas9 source (somatic cleavage)."""
            from natal.presets._types import count_allele_copies

            target = self.cas9_allele if self.cas9_allele else self.drive_allele
            return count_allele_copies(gt, target) > 0

        for sex in (Sex.FEMALE, Sex.MALE):
            rate = self.embryo_resistance_formation_rate[sex]
            if rate > 0:
                m_glab = None
                p_glab = None
                g_filter = None

                if self.cas9_deposition_glab:
                    # Label-based deposition (Maternal/Paternal effect)
                    if sex == Sex.FEMALE:
                        m_glab = self.cas9_deposition_glab
                    elif self.use_paternal_deposition:
                        p_glab = self.cas9_deposition_glab
                    else:
                        # Male rate > 0 but no paternal deposition -> somatic/zygotic expression
                        g_filter = zygote_has_cas9
                else:
                    # No labels provided -> cleavage depends on zygote's own Cas9 alleles
                    g_filter = zygote_has_cas9

                # Skip if no filter is active to avoid global mutation bug
                if m_glab is None and p_glab is None and g_filter is None:
                    continue

                func_res_ratio = self.functional_resistance_ratio
                if self.functional_resistance_allele and func_res_ratio > 0:
                    # 1. Functional resistance
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.functional_resistance_allele,
                        rate=rate * func_res_ratio,
                        maternal_glab=m_glab,
                        paternal_glab=p_glab,
                        genotype_filter=g_filter,
                    )
                    # 2. Non-functional resistance on remaining targets
                    target_remaining = 1.0 - (rate * func_res_ratio)
                    nf_rate = (rate * (1.0 - func_res_ratio)) / target_remaining if target_remaining > 0 else 0.0
                    if nf_rate > 0:
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele,
                            to_allele=self.resistance_genotype,
                            rate=nf_rate,
                            maternal_glab=m_glab,
                            paternal_glab=p_glab,
                            genotype_filter=g_filter,
                        )
                else:
                    # Generic resistance (no functional split)
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele,
                        to_allele=self.resistance_genotype,
                        rate=rate,
                        maternal_glab=m_glab,
                        paternal_glab=p_glab,
                        genotype_filter=g_filter,
                    )

        return rule_set.to_zygote_modifier(population) if rule_set.rules else None
