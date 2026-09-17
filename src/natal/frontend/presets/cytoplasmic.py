"""Cytoplasmic inheritance and slab-based presets.

Public module — provides CytoplasmicPreset, Wolbachia, and TransgenicBackground.
"""

# pyright: reportPrivateUsage=false

from typing import TYPE_CHECKING, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import (
    Species,
)
from natal.frontend.modifiers import (
    GameteConversionRuleSet,
    GameteModifier,
    ZygoteConversionRuleSet,
    ZygoteModifier,
)

from ._base import GeneticPreset
from ._types import PresetFitnessPatch

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import RecipeHost


# ---------------------------------------------------------------------------
# Slab-aware presets
# ---------------------------------------------------------------------------


class CytoplasmicPreset(GeneticPreset):
    """Base class for maternally-inherited cytoplasmic elements.

    Maternal tags redirect offspring that still carry the default slab.
    Existing non-default labels are preserved. The mechanism:
    1. *Gamete tagging* — ``gamete_modifier`` builds a declarative
       ``GameteConversionRuleSet``. Default-glab gametes from specific
       maternal genotypes are reassigned to the cytoplasmic glab.
    2. *Zygote redirect* — ``zygote_modifier`` builds a declarative
       ``ZygoteConversionRuleSet``: tagged maternal gametes redirect
       default-slab offspring to the matching slab, retaining genotype.

    ``default_glab`` and ``default_slab`` explicitly select the source
    labels; they do not change the species' baseline distribution.
    Both rule sets act on the distribution left by preceding modifiers.

    Subclasses must provide ``_maternal_map`` — a dict mapping
    ``{maternal_slab_name: glab_name}``.  Each maternal slab that
    should be heritable gets a unique glab for tagging.

    Example (Wolbachia):
        _maternal_map = {"infected": "wolbachia"}
    """

    _maternal_map: dict[str, str] = {}  # {slab_name: glab_name}

    def __init__(
        self,
        name: str = "",
        species: Optional[Species] = None,
        priority: int = 0,
        *,
        default_glab: str = "default",
        default_slab: str = "default",
    ) -> None:
        """Configure the labels eligible for maternal inheritance.

        Args:
            name: Optional preset name.
            species: Optional species bound at construction time.
            priority: Modifier and fitness application priority.
            default_glab: Source gamete label eligible for maternal tagging.
            default_slab: Source somatic label eligible for offspring relabeling.
                These names select existing labels without changing the
                species' baseline distribution or label order.
        """
        super().__init__(name=name, species=species, priority=priority)
        self.default_glab = default_glab
        self.default_slab = default_slab

    def _active_maternal_map(self, glab_names: List[str] | tuple[str, ...]) -> dict[str, str]:
        """Return the ``{slab: glab}`` pairs whose glab the species declares.

        Filtering silently would disable maternal inheritance without any
        signal: the preset keeps applying its fitness patch, so the
        population still builds and runs while transmission never happens.
        A non-empty ``_maternal_map`` with no matching label is therefore a
        configuration error, reported here.

        Args:
            glab_names: Gamete labels the species actually registers.

        Returns:
            The subset of ``_maternal_map`` whose glab is registered.

        Raises:
            ValueError: If ``_maternal_map`` names a gamete label that the
                species does not declare.
        """
        available = set(glab_names)
        active = {
            slab: glab for slab, glab in self._maternal_map.items()
            if glab in available
        }
        if self._maternal_map and not active:
            missing = sorted(set(self._maternal_map.values()) - available)
            raise ValueError(
                f"Preset '{self.name}' requires gamete label(s) {missing}, but "
                f"the species only declares {sorted(available)}.  Add them via "
                "Species.from_dict(..., gamete_labels=[...]); without them the "
                "preset would silently inherit nothing while still applying its "
                "fitness patch."
            )
        return active

    def gamete_modifier(self, host: "RecipeHost") -> Optional[GameteModifier]:
        """Tag maternal gametes: default-glab → *glab_name* for matching slabs.

        Uses the declarative :class:`GameteConversionRuleSet`: one whole-gtype
        conversion per target glab, restricted to female producers whose
        ztype slab matches and to gametes currently carrying the default
        glab specified by ``default_glab``.

        Raises:
            ValueError: If a required source or target label is unknown.
        """
        if not self._maternal_map:
            return None

        glab_to_idx = host.index_registry.glab_to_index
        # A missing label is a configuration error, not a silent no-op.
        active_map = self._active_maternal_map(list(glab_to_idx))

        ruleset = GameteConversionRuleSet()
        if self.default_glab not in host.registry.glab_labels:
            raise ValueError(f"Preset '{self.name}': unknown default_glab {self.default_glab!r}")
        for slab_name, glab_name in active_map.items():
            ruleset.add_gtype_convert(
                to=f"*@{glab_name}",
                rate=1.0,
                filters={
                    "parent_sex": "female",
                    "parent": f"*@{slab_name}",
                    "current": f"*@{self.default_glab}",
                },
            )

        return ruleset.to_gamete_modifier(host) if ruleset.rules else None  # type: ignore[return-type]  # structurally satisfies the GameteModifier protocol

    def zygote_modifier(self, host: "RecipeHost") -> Optional[ZygoteModifier]:
        """Redirect zygotes: tagged maternal gamete + any paternal → target slab.

        For each (slab_name, glab_name) in ``_maternal_map``: when the
        maternal gamete (c1) carries *glab_name*, redirect default-slab
        zygote outcomes to *slab_name* using conversion rules. The source
        label is explicitly configured by ``default_slab``. Other slabs and the
        genotype are preserved, including earlier modifiers' changes.

        Raises:
            ValueError: If a required source or target label is unknown.
        """
        if not self._maternal_map:
            return None

        glab_to_idx = host.index_registry.glab_to_index
        # A missing label is a configuration error, not a silent no-op.
        active_map = self._active_maternal_map(list(glab_to_idx))

        ruleset = ZygoteConversionRuleSet()
        if self.default_slab not in host.registry.slab_labels:
            raise ValueError(f"Preset '{self.name}': unknown default_slab {self.default_slab!r}")
        for slab_name, glab_name in active_map.items():
            ruleset.add_ztype_convert(
                to=f"*@{slab_name}",
                rate=1.0,
                filters={
                    "maternal": f"*@{glab_name}",
                    "current": f"*@{self.default_slab}",
                },
            )

        return ruleset.to_zygote_modifier(host) if ruleset.rules else None  # type: ignore[return-type]  # structurally satisfies the ZygoteModifier protocol

    @staticmethod
    def apply_zygote_redirect(
        z2g_expanded: NDArray[np.float64],
        glab_name: str,
        slab_name: str,
        gamete_labels: List[str],
        somatic_labels: List[str],
        n_slabs: int,
        n_genotypes_raw: int,
        n_hg: int,
        n_glabs: int,
    ) -> None:
        """Redirect zygote columns: glab-tagged maternal gametes → target slab.

        Looks up *glab_name* and *slab_name* in the label lists (not
        the registry — this runs in build_population_config which has
        no registry access).  No-op if either label is missing.
        """
        if glab_name not in gamete_labels or slab_name not in somatic_labels:
            return
        glab_idx = gamete_labels.index(glab_name)
        slab_idx = somatic_labels.index(slab_name)
        for g_raw in range(n_genotypes_raw):
            z_dst = g_raw * n_slabs + slab_idx
            z_src = g_raw * n_slabs + 0
            for hg_f in range(n_hg):
                hl_f = hg_f * n_glabs + glab_idx
                for hg_m in range(n_hg):
                    for gm in range(n_glabs):
                        hl_m = hg_m * n_glabs + gm
                        val = z2g_expanded[hl_f, hl_m, z_src]
                        if val > 0:
                            z2g_expanded[hl_f, hl_m, z_dst] += val
                            z2g_expanded[hl_f, hl_m, z_src] = 0.0


class Wolbachia(CytoplasmicPreset):
    """Maternal infection inheritance with optional cross-specific incompatibility.

    Offspring of uninfected mothers paired with infected fathers carry the
    incompatibility cost. Infected mothers provide complete rescue. The
    incompatible offspring label records origin, not an inherited infection.
    """

    def __init__(
        self,
        name: str,
        infected_slab: str = "infected",
        normal_slab: str = "normal",
        viability_scaling: float = 1.0,
        fecundity_scaling: Optional[float] = None,
        species: Optional[Species] = None,
        priority: int = 0,
        *,
        default_glab: str = "default",
        incompatibility_cost: float = 0.0,
        incompatibility_effect: Literal["zygote_viability", "viability", "fecundity"] = "zygote_viability",
        incompatibility_slab: str = "incompatible",
        paternal_glab: str = "wolbachia_ci",
    ) -> None:
        """Configure infection costs and a separate incompatible-cross loss.

        Args:
            name: Preset name.
            infected_slab: Somatic label for infected individuals.
            normal_slab: Uninfected label and source of offspring relabeling.
            viability_scaling: Finite nonnegative infected-carrier multiplier.
            fecundity_scaling: Finite nonnegative infected-carrier multiplier;
                ``None`` leaves fecundity unchanged.
            species: Optional species binding.
            priority: Modifier and fitness application priority.
            default_glab: Source gamete label eligible for tagging.
            incompatibility_cost: Fraction lost, finite and in [0, 1]. Zero
                disables incompatibility without requiring additional labels.
            incompatibility_effect: Fitness of the incompatible offspring:
                embryo survival before competition, ordinary viability at the
                last juvenile age, or its own fecundity when reproducing.
                The scalar multiplier applies to both sexes.
            incompatibility_slab: Uninfected offspring-origin label required
                for positive incompatibility costs. Carriers do not pass
                this label maternally and remain susceptible to incompatibility.
            paternal_glab: Gamete tag required for positive incompatibility costs;
                signals paternal induction, never paternal infection inheritance.

        Raises:
            ValueError: If costs, effect, or label roles are invalid.
        """
        super().__init__(
            name=name, species=species, priority=priority,
            default_glab=default_glab, default_slab=normal_slab,
        )
        self._maternal_map = {infected_slab: "wolbachia"}
        self.infected_slab = infected_slab
        self.normal_slab = normal_slab
        self.viability_scaling = viability_scaling
        self.fecundity_scaling = fecundity_scaling
        self.incompatibility_cost = incompatibility_cost
        self.incompatibility_effect = incompatibility_effect
        self.incompatibility_slab = incompatibility_slab
        self.paternal_glab = paternal_glab
        self._validate_options()

    def _validate_options(self) -> None:
        # Revalidate on compilation: runtime preset reconfiguration replaces
        # attributes without invoking the constructor.
        for name, value in (("viability_scaling", self.viability_scaling),
                            ("fecundity_scaling", self.fecundity_scaling)):
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.incompatibility_cost) or not 0 <= self.incompatibility_cost <= 1:
            raise ValueError("incompatibility_cost must be finite and in [0, 1]")
        if self.incompatibility_effect not in ("zygote_viability", "viability", "fecundity"):
            raise ValueError("Unknown incompatibility_effect")
        if self.incompatibility_cost > 0:
            if self.infected_slab == self.normal_slab:
                raise ValueError("infected_slab and normal_slab must differ for incompatibility")
            if self.incompatibility_slab in (self.infected_slab, self.normal_slab):
                raise ValueError("incompatibility_slab must differ from infection labels")
            if self.default_glab == "wolbachia":
                raise ValueError("default_glab must differ from the maternal infection tag")
            if self.paternal_glab in (self.default_glab, "wolbachia"):
                raise ValueError("paternal_glab must differ from source and maternal tags")

    def _validate_host(self, host: "RecipeHost") -> None:
        self._validate_options()
        if self.incompatibility_cost > 0:
            if self.paternal_glab not in host.registry.glab_labels:
                raise ValueError(f"Unknown paternal_glab {self.paternal_glab!r}")
            if self.incompatibility_slab not in host.registry.slab_labels:
                raise ValueError(f"Unknown incompatibility_slab {self.incompatibility_slab!r}")

    def gamete_modifier(self, host: "RecipeHost") -> Optional[GameteModifier]:
        """Tag infection maternally and incompatible-cross induction paternally."""
        self._validate_host(host)
        maternal = super().gamete_modifier(host)
        if self.incompatibility_cost == 0:
            return maternal
        rules = GameteConversionRuleSet()
        rules.add_gtype_convert(
            to="*@wolbachia", rate=1.0,
            filters={"parent_sex": "female", "parent": f"*@{self.infected_slab}",
                     "current": f"*@{self.default_glab}"},
        )
        rules.add_gtype_convert(
            to=f"*@{self.paternal_glab}", rate=1.0,
            filters={"parent_sex": "male", "parent": f"*@{self.infected_slab}",
                     "current": f"*@{self.default_glab}"},
        )
        return rules.to_gamete_modifier(host)  # type: ignore[return-type]  # compiled rule structurally implements GameteModifier

    def zygote_modifier(self, host: "RecipeHost") -> Optional[ZygoteModifier]:
        """Retain maternal inheritance and mark incompatible uninfected offspring."""
        self._validate_host(host)
        maternal = super().zygote_modifier(host)
        if self.incompatibility_cost == 0:
            return maternal
        rules = ZygoteConversionRuleSet()
        rules.add_ztype_convert(
            to=f"*@{self.infected_slab}", rate=1.0,
            filters={"maternal": "*@wolbachia", "current": f"*@{self.normal_slab}"},
        )
        rules.add_ztype_convert(
            to=f"*@{self.incompatibility_slab}", rate=1.0,
            filters={"maternal": f"*@{self.default_glab}",
                     "paternal": f"*@{self.paternal_glab}", "current": f"*@{self.normal_slab}"},
        )
        return rules.to_zygote_modifier(host)  # type: ignore[return-type]  # compiled rule structurally implements ZygoteModifier

    def fitness_patch(self) -> PresetFitnessPatch:
        """Apply carrier fitness separately from the incompatible-cross effect."""
        self._validate_options()
        patch: PresetFitnessPatch = {
            "viability_per_slab": {self.infected_slab: self.viability_scaling},
        }
        if self.fecundity_scaling is not None:
            patch["fecundity_per_slab"] = {self.infected_slab: self.fecundity_scaling}
        if self.incompatibility_cost == 0:
            return patch
        factor = 1.0 - self.incompatibility_cost
        if self.incompatibility_effect == "zygote_viability":
            patch["zygote_per_slab"] = {self.incompatibility_slab: factor}
        elif self.incompatibility_effect == "viability":
            patch["viability_per_slab"][self.incompatibility_slab] = factor
        else:
            patch.setdefault("fecundity_per_slab", {})[self.incompatibility_slab] = factor
        return patch


class TransgenicBackground(GeneticPreset):
    """Fitness scaling for a transgenic background slab.

    Applies fecundity and/or viability scaling to individuals carrying
    the *tg_slab* somatic label.  Does NOT implement outcrossing
    clearance — that requires a separate inheritance mechanism.
    """

    def __init__(
        self,
        name: str,
        tg_slab: str,
        wt_slab: str = "WT_bg",
        fecundity_scaling: float = 1.0,
        viability_scaling: Optional[float] = None,
        species: Optional[Species] = None,
        priority: int = 0,
    ) -> None:
        """Initialize a TransgenicBackground preset.

        Args:
            name: Preset name.
            tg_slab: Somatic slab label for transgenic individuals.
            wt_slab: Somatic slab label for wild-type background.
            fecundity_scaling: Fecundity multiplier for transgenic carriers.
            viability_scaling: Optional viability multiplier for transgenic
                carriers. ``None`` means no viability effect.
            species: Optional species for validation.
            priority: Modifier and fitness application priority.
        """
        super().__init__(name=name, species=species, priority=priority)
        self.tg_slab = tg_slab
        self.wt_slab = wt_slab
        self.fecundity_scaling = fecundity_scaling
        self.viability_scaling = viability_scaling

    def gamete_modifier(self, host: "RecipeHost") -> Optional[GameteModifier]:
        """Return no gamete modifier — transgenic background is slab-only."""
        return None

    def zygote_modifier(self, host: "RecipeHost") -> Optional[ZygoteModifier]:
        """Return no zygote modifier — transgenic background is slab-only."""
        return None

    def fitness_patch(self) -> PresetFitnessPatch:
        """Build fitness patch applying fecundity and optional viability scaling.

        Returns:
            A fitness patch dict with ``fecundity_per_slab`` and optionally
            ``viability_per_slab`` entries for the transgenic slab.
        """
        patch: PresetFitnessPatch = {}
        patch['fecundity_per_slab'] = {self.tg_slab: self.fecundity_scaling}
        if self.viability_scaling is not None:
            patch['viability_per_slab'] = {self.tg_slab: self.viability_scaling}
        return patch
