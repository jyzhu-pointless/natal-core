"""GeneticPreset abstract base class and apply_preset_to_population.

Public module — provides the core preset infrastructure.
"""

# pyright: reportPrivateUsage=false

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Optional, Self, Tuple

from natal.genetics import Gene, Species
from natal.modifiers.module import GameteModifier, ZygoteModifier
from natal.utils.types import Sex

from ._types import (
    PresetFitnessPatch,
    _AlleleSpecifier,
    _SexSpecificRates,
)

if TYPE_CHECKING:
    from natal.population.base import BasePopulation


def apply_preset_to_population(population: 'BasePopulation[Any]', preset: 'GeneticPreset') -> None:
    """Apply a genetic preset to a population by registering its modifiers and fitness effects.

    This function handles the mechanical application of a preset to a population,
    including:
    1. Species binding and validation
    2. Registration of gamete modifiers
    3. Registration of zygote modifiers
    4. Application of fitness patches

    Args:
        population: The BasePopulation instance to modify.
        preset: The GeneticPreset instance to apply.

    Note:
        This is typically called through the modern API:
        ``population.apply_preset(preset)``

        The legacy API ``preset.apply(population)`` is deprecated but still supported.

    Raises:
        ValueError: If preset is bound to a different species than the population
        RuntimeError: If preset has no bound species
    """
    from natal.fitness._patch import apply_preset_fitness_patch

    preset.bind_species(population.species)

    gamete_mod = preset.gamete_modifier(population)
    zygote_mod = preset.zygote_modifier(population)

    if gamete_mod is not None:
        population.add_gamete_modifier(
            gamete_mod,
            name=f"{preset.name}/gamete",
            refresh=False,
        )

    if zygote_mod is not None:
        population.add_zygote_modifier(
            zygote_mod,
            name=f"{preset.name}/zygote",
            refresh=False,
        )

    if gamete_mod is not None or zygote_mod is not None:
        population.refresh_modifier_maps()

    # Preferred path: declarative fitness patch
    patch = preset.fitness_patch()
    if patch:
        apply_preset_fitness_patch(population, patch)


class GeneticPreset(ABC):
    """Abstract base for genetic modification presets including gene drives, mutations, and allele conversions.

    A preset bundles gamete modifiers, zygote modifiers, and fitness effects
    that form a cohesive genetic system. This can include:
    - Gene drives (e.g., CRISPR/Cas9 homing drives)
    - General mutations (point mutations, insertions, deletions)
    - Complex allele conversion systems

    Presets should implement:
      - gamete_modifier(): returns GameteModifier callable or None
      - zygote_modifier(): returns ZygoteModifier callable or None
      - fitness_patch(): returns declarative fitness configuration dict or None

    All methods are optional (can return None). At least one method should be implemented
    for the preset to have any effect.

    Examples:
        >>> population.apply_preset(preset)

    Attributes:
        name (str): Human-readable preset name.
        hook_id (Optional[int]): Optional identifier used when registering modifiers.
    """

    def __init__(
        self,
        name: str = "",
        species: Optional[Species] = None,
        priority: int = 0,
    ):
        """Initialize the preset.

        Args:
            name: Optional human-readable name for the preset.
            species: Optional species bound at construction time.
            priority: Execution order — lower values apply first.
                Same priority uses registration order (stable sort).
        """
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.hook_id: Optional[int] = None
        self._bound_species: Optional[Species] = species
        self._custom_fitness_patch: Optional[Callable[[], Optional[PresetFitnessPatch]]] = None

    def bind_species(self, species: Species) -> None:
        """Bind this preset instance to a concrete species.

        This enables delayed species injection: users can construct presets
        without passing species, and binding happens automatically when the
        preset is applied to a population.
        """
        if self._bound_species is None:
            self._bound_species = species
            return

        if self._bound_species is species:
            return

        raise ValueError(
            f"Preset '{self.name}' is already bound to species "
            f"'{self._bound_species.name}' and cannot be applied to population species '{species.name}'."
        )

    def _require_bound_species(self) -> Species:
        """Return the bound species or raise if preset has not been injected yet."""
        if self._bound_species is None:
            raise RuntimeError(
                f"Preset '{self.name}' is not bound to a species. "
                "Apply it through population.apply_preset(...) or builder.presets(...).build()."
            )
        return self._bound_species

    def _resolve_bound_gene(self, allele_name: str) -> Gene:
        """Resolve an allele name into a Gene using the currently bound species."""
        species = self._require_bound_species()
        gene = species.gene_index.get(allele_name)
        if gene is None:
            raise ValueError(
                f"Allele '{allele_name}' not found in species '{species.name}' "
                f"for preset '{self.name}'."
            )
        return gene

    @abstractmethod
    def gamete_modifier(self, population: 'BasePopulation[Any]') -> Optional[GameteModifier]:
        """Return a gamete modifier or None.

        The modifier should return:

            Dict[(sex_idx, ztype_idx) -> Dict[compressed_hg_glab_idx -> freq]]

        where compressed_hg_glab_idx is an integer index into the compressed
        haploid genotype space.
        """
        return None

    @abstractmethod
    def zygote_modifier(self, population: 'BasePopulation[Any]') -> Optional[ZygoteModifier]:
        """Return a zygote modifier or None.

        The modifier should return:

            Dict[(c1, c2) -> (idx_modified | Genotype | Dict[idx -> prob])]

        where c1, c2 are compressed coordinate pairs representing the parental
        diploid genotypes.
        """
        return None

    def fitness_patch(self) -> Optional[PresetFitnessPatch]:
        """Return declarative fitness patch.

        Returns:
            Fitness patch from custom function if set, otherwise None.
            Subclasses should override this method for built-in behavior.
        """
        if self._custom_fitness_patch is not None:
            return self._custom_fitness_patch()
        return None

    def with_fitness_patch(
        self,
        patch_func: Callable[[], Optional[PresetFitnessPatch]]
    ) -> Self:
        """Set a custom fitness patch function and return self for chaining.

        This allows dynamic modification of fitness effects at runtime
        without subclassing, using a fluent interface.

        Args:
            patch_func: Callable that returns a PresetFitnessPatch or None.

        Returns:
            Self for method chaining.

        Example:
            >>> preset = (HomingDrive(...)
            ...     .with_fitness_patch(lambda: {
            ...         'viability_allele': {'Drive': (0.8, 'dominant')}
            ...     }))
            >>> population.apply_preset(preset)

            >>> # Also works with complex custom logic
            >>> def conditional_patch():
            ...     if some_condition:
            ...         return {'fecundity_allele': {'Mut': (0.5, 'recessive')}}
            ...     return None
            >>>
            >>> preset = HomingDrive(...).with_fitness_patch(conditional_patch)

        Note:
            This overrides any fitness patch defined in subclasses.
            To preserve subclass behavior while adding modifications,
            subclass and call super().fitness_patch() instead.
        """
        if not callable(patch_func):
            raise TypeError(f"patch_func must be callable, got {type(patch_func)}")
        self._custom_fitness_patch = patch_func
        return self

    def clear_fitness_patch(self) -> 'GeneticPreset':
        """Remove any custom fitness patch, restoring default behavior.

        Returns:
            Self for method chaining.
        """
        self._custom_fitness_patch = None
        return self

    def _resolve_allele_name(self, allele: _AlleleSpecifier) -> str:
        """Helper to resolve allele inputs to their string names."""
        if isinstance(allele, Gene):
            return allele.name
        return allele

    def _resolve_rates(
        self, rate: _SexSpecificRates
    ) -> Tuple[float, float]:
        """Helper to resolve rate inputs into a tuple of (female_rate, male_rate)."""
        if isinstance(rate, (int, float)):
            return (rate, rate)
        if isinstance(rate, tuple):
            return rate
        female_rate = rate.get(Sex.FEMALE) or rate.get("female") or rate.get("f") or rate.get("F") or 0.0
        male_rate = rate.get(Sex.MALE) or rate.get("male") or rate.get("m") or rate.get("M") or 0.0
        return (female_rate, male_rate)

    def apply(self, population: 'BasePopulation[Any]') -> None:
        """Register this preset onto a population (DEPRECATED).

        .. deprecated::
            Use population.apply_preset(preset) instead.
            This method is kept for backwards compatibility and may be removed in future versions.

        Args:
            population: The BasePopulation instance to modify.

        See Also:
            :meth:`natal.population.base.BasePopulation.apply_preset` - Preferred modern API
        """
        apply_preset_to_population(population, self)
