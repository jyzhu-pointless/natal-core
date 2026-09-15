"""Cytoplasmic inheritance and slab-based presets.

Public module — provides CytoplasmicPreset, Wolbachia, and TransgenicBackground.
"""

# pyright: reportPrivateUsage=false

from typing import TYPE_CHECKING, List, Optional

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
    """Maternally-inherited endosymbiont with explicit source labels.

    Infected mothers tag offspring that still carry ``normal_slab``,
    regardless of the father. Other somatic labels are preserved.

    Requires Species with:
      - gamete_labels including ``default_glab`` and ``"wolbachia"``
      - somatic_labels including ``normal_slab`` and ``infected_slab``
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
    ) -> None:
        """Initialize a Wolbachia cytoplasmic preset.

        Args:
            name: Preset name.
            infected_slab: Somatic slab label for infected individuals.
            normal_slab: Somatic label for uninfected individuals and the
                source label eligible for offspring infection tagging.
            viability_scaling: Viability multiplier for infected carriers.
            fecundity_scaling: Fecundity multiplier for infected carriers.
                ``None`` means no fecundity effect.
            species: Optional species for validation.
            priority: Modifier and fitness application priority.
            default_glab: Source gamete label eligible for maternal tagging.
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

    def fitness_patch(self) -> PresetFitnessPatch:
        """Build fitness patch applying viability and fecundity scaling.

        Returns:
            A fitness patch dict with optional ``viability_per_slab`` and
            ``fecundity_per_slab`` entries for the infected slab.
        """
        patch: PresetFitnessPatch = {}
        patch['viability_per_slab'] = {self.infected_slab: self.viability_scaling}
        if self.fecundity_scaling is not None:
            patch['fecundity_per_slab'] = {self.infected_slab: self.fecundity_scaling}
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
