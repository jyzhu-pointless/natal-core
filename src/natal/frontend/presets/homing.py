"""Homing-based gene drive preset.

Public module — provides HomingDrive for CRISPR/Cas9 gene drive simulations.
"""

from typing import TYPE_CHECKING, Any, Optional

from natal.frontend.genetics import Gene
from natal.frontend.modifiers.gamete_conversion import GameteConversionRuleSet
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
from natal.frontend.modifiers.zygote_conversion import ZygoteConversionRuleSet
from natal.frontend.utils.types import Sex

from ._base import GeneticPreset
from ._fitness import (
    make_fitness_patch_given_allele_scaling,
)
from ._types import (
    AlleleScalingMode,
    AlleleSpecifier,
    FecundityScalingConfig,
    PresetFitnessPatch,
    SexSpecificRates,
    SexualSelectionScalingConfig,
    ViabilityScalingConfig,
    ZygoteViabilityScalingConfig,
)

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import RecipeHost


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
        embryo_resistance_formation_rate (Tuple[float, float]): Maternal/paternal
            deposition editing rates per target copy, independent of offspring sex.

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
                in embryos per target copy due to maternal/paternal Cas9 deposition.
                Can be a single float, dict, or tuple (maternal_rate, paternal_rate).
                A scalar sets both rates; the paternal rate is used only when
                use_paternal_deposition is True. Requires cas9_deposition_glab;
                the embryo's own Cas9 genotype never triggers editing.
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
                Must be registered in the species. Without this label, embryo resistance
                is inactive. Tagged gametes can edit embryos that did not inherit drive.
            species (Species, optional): Species to bind at construction time. If None,
                will be bound when applied to population.
            use_paternal_deposition (bool): Whether to enable paternal Cas9 deposition.
                If True, fathers can deposit Cas9 in embryos. If False, the paternal
                embryo resistance rate is inactive.

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
        # Allele specifiers are reduced to names now; binding to Gene objects is
        # deferred until a species is attached.
        self._str_drive_allele = self._resolve_allele_name(drive_allele)
        self._str_target_allele = self._resolve_allele_name(target_allele)
        self._str_resistance_allele = (self._resolve_allele_name(resistance_allele)
            if resistance_allele else None)
        self._str_functional_resistance_allele = (self._resolve_allele_name(functional_resistance_allele)
            if functional_resistance_allele else None)
        self._str_cas9_allele = self._resolve_allele_name(cas9_allele) if cas9_allele else None

        # Scalar/dict/tuple input is normalized to a (female, male) pair; the
        # gamete and zygote modifiers read it per sex.
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

        # The shared per-allele config feeds every fitness channel, each with its
        # own scaling mode; the patch stays declarative and is applied later.
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

    @property
    def drive_allele(self) -> Gene:
        """Gene: The drive allele (e.g. the Cas9/gRNA construct)."""
        return self._resolve_bound_gene(self._str_drive_allele)

    @property
    def target_allele(self) -> Gene:
        """Gene: The wild-type allele targeted for cleavage."""
        return self._resolve_bound_gene(self._str_target_allele)

    @property
    def resistance_genotype(self) -> Gene:
        """Gene: The resistance allele formed by NHEJ repair.

        Raises:
            ValueError: If no resistance allele was configured.
        """
        if self._str_resistance_allele is None:
            raise ValueError(f"Resistance allele not defined in HomingDrive '{self.name}'.")
        return self._resolve_bound_gene(self._str_resistance_allele)

    @property
    def functional_resistance_allele(self) -> Optional[Gene]:
        """Gene or None: The functional resistance allele, if configured."""
        if self._str_functional_resistance_allele is None:
            return None
        return self._resolve_bound_gene(self._str_functional_resistance_allele)

    @property
    def cas9_allele(self) -> Optional[Gene]:
        """Gene or None: The Cas9 source allele, if different from drive_allele."""
        if self._str_cas9_allele is None:
            return None
        return self._resolve_bound_gene(self._str_cas9_allele)

    @staticmethod
    def _rate_at(rate: float | tuple[float, float], sex: Sex) -> float:
        """Return per-sex rate, normalizing a scalar to both sexes.

        ``__init__`` resolves scalar input to a ``(female, male)`` tuple
        via :meth:`_resolve_rates`, but :meth:`reconfigure_preset` writes
        back via ``setattr`` which may restore a plain ``float``.  This
        helper handles both forms transparently.
        """
        if isinstance(rate, (int, float)):
            return float(rate)
        return rate[sex]

    def gamete_modifier(self, host: "RecipeHost") -> Optional[GameteModifier]:
        """Implement homing in heterozygous parents, germline resistance, and Cas9 deposition.

        In heterozygotes (drive/wild-type), gametes are biased towards drive.
        """
        from natal.frontend.presets._types import carrier_pattern

        required = [self.drive_allele.name] + (
            [self.cas9_allele.name] if self.cas9_allele else []
        )
        # A parent is a carrier only when it carries every required allele (drive
        # plus, for split drives, Cas9); this conjunction gates homing.
        carrier = carrier_pattern(host.species, *required)

        # RuleSet compiles these rules into a Sequential Cascade.
        # This means the target pool shrinks after every rule.
        # So Rule 2 (Resistance) only acts on the targets that FAILED Rule 1 (Homing).
        rule_set = GameteConversionRuleSet(f"{self.name}_Homing")
        # One set of rules per sex, because every rule is gated by parent_sex and
        # each sex has its own conversion and resistance rates.
        for sex in (Sex.FEMALE, Sex.MALE):
            homing_rate = HomingDrive._rate_at(self.drive_conversion_rate, sex)
            res_rate = HomingDrive._rate_at(self.late_germline_resistance_formation_rate, sex)

            # 1. Homing (Target -> Drive)
            # Examples: If homing_rate is 0.7, 70% of targets become Drive. 30% pass to the next rule.
            if homing_rate > 0:
                rule_set.add_allele_convert(
                    from_allele=self.target_allele.name,
                    to_allele=self.drive_allele.name,
                    rate=homing_rate,
                    filters={"parent_sex": ("female" if sex == Sex.FEMALE else "male"), "parent": carrier},
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
                        from_allele=self.target_allele.name,
                        to_allele=self.functional_resistance_allele.name,
                        rate=res_rate * self.functional_resistance_ratio,
                        filters={"parent_sex": ("female" if sex == Sex.FEMALE else "male"), "parent": carrier},
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
                            from_allele=self.target_allele.name,
                            to_allele=self.resistance_genotype.name,
                            rate=adjusted_nf_rate,
                            filters={"parent_sex": ("female" if sex == Sex.FEMALE else "male"), "parent": carrier},
                        )
                else:
                    # Generic resistance (no functional/non-functional split)
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele.name,
                        to_allele=self.resistance_genotype.name,
                        rate=res_rate,
                        filters={"parent_sex": ("female" if sex == Sex.FEMALE else "male"), "parent": carrier},
                    )

            # 3. Gamete labeling for maternal Cas9 deposition
            # Tags the entire output gamete from drive-carrying females
            # with `cas9_deposition_glab`. The zygote modifier will read
            # this tag to apply embryo resistance.
            if self.cas9_deposition_glab:
                if sex == Sex.FEMALE or self.use_paternal_deposition:
                    rule_set.add_gtype_convert(
                        to=f"*@{self.cas9_deposition_glab}",
                        rate=1.0,
                        filters={"parent_sex": ("female" if sex == Sex.FEMALE else "male"), "parent": carrier},
                    )

        return rule_set.to_gamete_modifier(host) if rule_set.rules else None  # type: ignore[return-type]  # structurally satisfies the GameteModifier protocol

    def zygote_modifier(self, host: "RecipeHost") -> Optional[ZygoteModifier]:
        """Convert embryonic target copies using parental Cas9 deposition only.

        Maternal deposition edits both inherited target copies regardless of
        the embryo's own Cas9 genotype. Paternal deposition contributes a
        separate sequential conversion only when explicitly enabled.

        Args:
            host: Species and registry used to compile the conversion rules.

        Returns:
            The deposition modifier, or None when no deposition label or
            active parental editing rate is configured.
        """
        if not self.cas9_deposition_glab:
            return None

        rule_set = ZygoteConversionRuleSet(f"{self.name}_EmbryoResistance")
        for sex in (Sex.FEMALE, Sex.MALE):
            if sex == Sex.MALE and not self.use_paternal_deposition:
                continue
            rate = HomingDrive._rate_at(self.embryo_resistance_formation_rate, sex)
            if rate > 0:
                source = "maternal" if sex == Sex.FEMALE else "paternal"
                filters = {source: f"*@{self.cas9_deposition_glab}"}

                func_res_ratio = self.functional_resistance_ratio
                if self.functional_resistance_allele and func_res_ratio > 0:
                    # 1. Functional resistance
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele.name,
                        to_allele=self.functional_resistance_allele.name,
                        rate=rate * func_res_ratio,
                        filters=filters,
                    )
                    # 2. Non-functional resistance on remaining targets
                    # Same remainder rescaling as the germline path: divide by the
                    # target pool left after the functional rule so the
                    # unconditional non-functional mass equals rate*(1-ratio).
                    target_remaining = 1.0 - (rate * func_res_ratio)
                    nf_rate = (rate * (1.0 - func_res_ratio)) / target_remaining if target_remaining > 0 else 0.0
                    if nf_rate > 0:
                        rule_set.add_allele_convert(
                            from_allele=self.target_allele.name,
                            to_allele=self.resistance_genotype.name,
                            rate=nf_rate,
                            filters=filters,
                        )
                else:
                    # Generic resistance (no functional split)
                    rule_set.add_allele_convert(
                        from_allele=self.target_allele.name,
                        to_allele=self.resistance_genotype.name,
                        rate=rate,
                        filters=filters,
                    )

        return rule_set.to_zygote_modifier(host) if rule_set.rules else None  # type: ignore[return-type]  # structurally satisfies the ZygoteModifier protocol
