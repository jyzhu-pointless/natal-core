"""Cytoplasmic inheritance and slab-based presets.

Public module — provides CytoplasmicPreset, Wolbachia, and TransgenicBackground.
"""

# pyright: reportPrivateUsage=false

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from natal.frontend.genetics import (
    Genotype,
    Species,
)
from natal.frontend.modifiers.gamete_conversion import (
    GameteConversionRuleSet,
)
from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier

from ._base import GeneticPreset
from ._types import PresetFitnessPatch

if TYPE_CHECKING:
    from natal.frontend.genetics.compile import RecipeHost


# ---------------------------------------------------------------------------
# Slab-aware presets
# ---------------------------------------------------------------------------


class CytoplasmicPreset(GeneticPreset):
    """Base class for maternally-inherited cytoplasmic elements.

    Child slab = mother slab regardless of father.  The mechanism:
    1. *Gamete tagging* — ``gamete_modifier`` builds a declarative
       ``GameteConversionRuleSet`` with pre-compiled probability
       transition matrices.  Default-glab gametes from specific
       maternal genotypes are reassigned to the cytoplasmic glab.
    2. *Zygote redirect* — ``zygote_modifier`` uses pre-computed
       ``glab → target_slab`` index lookup to redirect zygote
       columns: maternal gametes tagged with the cytoplasmic glab
       produce offspring in the matching slab.

    Subclasses must provide ``_maternal_map`` — a dict mapping
    ``{maternal_slab_name: glab_name}``.  Each maternal slab that
    should be heritable gets a unique glab for tagging.

    Example (Wolbachia):
        _maternal_map = {"infected": "wolbachia"}
    """

    _maternal_map: dict[str, str] = {}  # {slab_name: glab_name}

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
        glab.
        """
        if not self._maternal_map:
            return None

        glab_to_idx = host.index_registry.glab_to_index
        # A missing label is a configuration error, not a silent no-op.
        active_map = self._active_maternal_map(list(glab_to_idx))

        ruleset = GameteConversionRuleSet()
        for slab_name, glab_name in active_map.items():
            ruleset.add_gtype_convert(
                to=f"*@{glab_name}",
                rate=1.0,
                filters={
                    "parent_sex": "female",
                    "parent": f"*@{slab_name}",
                    "current": "*@default",
                },
            )

        return ruleset.to_gamete_modifier(host) if ruleset.rules else None  # type: ignore[return-type]  # structurally satisfies the GameteModifier protocol

    def zygote_modifier(self, host: "RecipeHost") -> Optional[ZygoteModifier]:
        """Redirect zygotes: tagged maternal gamete + any paternal → target slab.

        For each (slab_name, glab_name) in ``_maternal_map``: when the
        maternal gamete (c1) carries *glab_name*, redirect default-slab
        zygote outcomes to *slab_name*.  Pre-compiled c1-indexing avoids
        per-call registry lookups.
        """
        if not self._maternal_map:
            return None

        glab_to_idx = host.index_registry.glab_to_index
        # A missing label is a configuration error, not a silent no-op.
        active_map = self._active_maternal_map(list(glab_to_idx))

        registry = host.registry
        g2z = host.config.gametes_to_zygotes_map
        default_slab = registry.slab_labels[0]
        n_gtypes = g2z.shape[0]

        # Pre-compute: glab_name → [c1 indices where glab matches]
        glab_c1: dict[str, list[int]] = {}
        for c1 in range(n_gtypes):
            _, glab = registry.index_to_gtype[c1]
            glab_c1.setdefault(glab, []).append(c1)

        def modifier_func(*_args: object, **_kwargs: object) -> Dict[
            Tuple[int, int], Dict[int, float]
        ]:
            result: Dict[Tuple[int, int], Dict[int, float]] = {}

            for slab_name, glab_name in active_map.items():
                for c1 in glab_c1.get(glab_name, []):
                    for c2 in range(n_gtypes):
                        row = g2z[c1, c2]
                        total = float(row.sum())
                        if total == 0:
                            continue
                        dist: Dict[int, float] = {}
                        for ztype_idx in range(len(row)):
                            val = float(row[ztype_idx])
                            if val <= 0:
                                continue
                            gt, slab = cast(
                                tuple[object, str],
                                registry.index_to_ztype[ztype_idx],
                            )
                            if slab == default_slab:
                                dst_z = registry.ztype_index(
                                    cast(Genotype, gt), slab_name,
                                )
                                dist[dst_z] = dist.get(dst_z, 0.0) + val
                            else:
                                dist[ztype_idx] = dist.get(ztype_idx, 0.0) + val
                        if dist:
                            result[(c1, c2)] = dist

            return result

        return modifier_func  # type: ignore[return-type]  # inner func matches ZygoteModifier protocol

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
    """Maternally-inherited endosymbiont.  Infected mothers pass the
    infection to all offspring regardless of the father.

    Requires Species with:
      - gamete_labels including ``"wolbachia"``
      - somatic_labels including ``"normal"``, ``"infected"``
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
    ) -> None:
        """Initialize a Wolbachia cytoplasmic preset.

        Args:
            name: Preset name.
            infected_slab: Somatic slab label for infected individuals.
            normal_slab: Somatic slab label for uninfected individuals.
            viability_scaling: Viability multiplier for infected carriers.
            fecundity_scaling: Fecundity multiplier for infected carriers.
                ``None`` means no fecundity effect.
            species: Optional species for validation.
            priority: Modifier and fitness application priority.
        """
        super().__init__(name=name, species=species, priority=priority)
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
