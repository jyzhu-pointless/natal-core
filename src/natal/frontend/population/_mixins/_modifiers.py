"""Modifier and preset management mixin for BasePopulation.

Extracted from :mod:`natal.frontend.population.base` to reduce the
BasePopulation ABC to its core lifecycle contract.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Sequence,
    cast,
)

from natal.frontend.population._mixins._hooks import HookManagerMixin

if TYPE_CHECKING:
    from natal.frontend.population.base import BasePopulation

if TYPE_CHECKING:
    from natal.frontend.genetics import Species
    from natal.frontend.modifiers.module import GameteModifier, ZygoteModifier
    from natal.frontend.presets import GeneticPreset


class ModifierPresetMixin(HookManagerMixin):
    """Mixin providing modifier and preset management.

    Builds on HookManagerMixin since modifier compilation triggers
    hook rebuilds.

    Expects the host class (BasePopulation) to define these attributes:
    ``_config``, ``_presets``, ``_species``,
    ``_gamete_modifiers``, ``_zygote_modifiers``, ``_manual_gamete``,
    ``_manual_zygote``, ``_index_registry``.
    """

    # Declared here so pyright knows these come from the host class.
    _config: Any  # type: ignore[assignment]
    _presets: list[Any]  # type: ignore[assignment]
    _species: Any  # type: ignore[assignment]
    _gamete_modifiers: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _zygote_modifiers: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _manual_gamete: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _manual_zygote: list[tuple[int, Optional[str], Any]]  # type: ignore[assignment]
    _index_registry: Any  # type: ignore[assignment]

    def _session_target(self) -> Any:
        """Resolve the idle-session commit target for this population.

        Returns:
            The resolved ``_UpdateTarget`` (typed ``Any`` here: the
            runtime target classes are internal to the configurator
            package and the host mixin only forwards them).

        Raises:
            RuntimeError: If a run holds the session borrow.
        """
        from natal.frontend.configurator._runtime import idle_session_target

        return idle_session_target(cast("BasePopulation[Any]", self))

    # ========================================================================
    # Modifier management
    # ========================================================================
    def reapply_preset_fitness(self) -> None:
        """Reset fitness tensors to 1.0 and re-apply all preset fitness patches.

        Called after structural changes to presets (addition, removal, or
        reconfiguration).  Only preset-derived fitness is restored — any
        fitness values set directly via ``pop.update().fitness()`` will be
        overwritten. This explicit reset clears stored manual-fitness patches.
        """
        from natal.frontend.configurator._runtime import reset_preset_fitness

        if self._config is None:
            return
        reset_preset_fitness(self._session_target())

    def refresh_modifiers(self, rebuild_maps: bool = True) -> None:
        """Rebuild derived modifier lists and maps from _presets + _manual_*.

        Presets are applied in priority order, then manual modifiers are
        appended.  Modifier maps (zygotes_to_gametes_map,
        gametes_to_zygotes_map, offspring_tensor) are rebuilt from the
        combined list.

        Args:
            rebuild_maps: If ``True`` (default), also commit the rebuilt
                maps.  Set to ``False`` when the caller plans to batch
                multiple modifier registrations and will commit once
                afterward.
        """
        from natal.frontend.configurator._runtime import (
            commit_genetic_update,
            compile_runtime_candidate,
            read_declaration,
        )

        target = self._session_target()
        declaration = read_declaration(target)
        old = target.live_draft()
        # map-only updates preserve current native fitness.
        products = compile_runtime_candidate(
            target.species, old, target.registry, declaration, preserve_fitness=True,
        )
        if rebuild_maps:
            commit_genetic_update(
                target, old, products.config, declaration,
                products.gamete_modifiers, products.zygote_modifiers,
            )
        else:
            self._gamete_modifiers = list(products.gamete_modifiers)
            self._zygote_modifiers = list(products.zygote_modifiers)

    def refresh_modifier_maps(self) -> None:
        """Rebuild the three modifier maps from current modifier lists.

        Recomputes ``zygotes_to_gametes_map``,
        ``gametes_to_zygotes_map``, and the derived ``offspring_tensor``
        through the unified compiler
        (:func:`natal.frontend.genetics.compile.compile_modifier_maps`)
        — the same spelling the build path uses, so the two entry
        points cannot drift (pinned bit-for-bit by the parity safety
        net).

        .. note::

            This method is called automatically by :meth:`refresh_modifiers`
            and by individual ``add_gamete_modifier`` /
            ``add_zygote_modifier`` when ``refresh=True``.
        """
        from natal.frontend.configurator._runtime import recompile_modifier_maps

        if self._config is None or self._index_registry is None:
            return
        if not self._index_registry.index_to_haplo or not self._index_registry.index_to_genotype:
            return
        recompile_modifier_maps(self._session_target())

    def add_gamete_modifier(
        self,
        modifier: GameteModifier,
        name: Optional[str] = None,
        modifier_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a gamete-level modifier.

        Args:
            modifier: A ``GameteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            modifier_id: Optional numeric priority used for ordering.
            refresh: If True (default), immediately rebuild modifier maps.
                Set to False when adding multiple modifiers in a batch;
                call :meth:`refresh_modifiers` or
                :meth:`refresh_modifier_maps` afterward to apply all at once.
        """
        from natal.frontend.configurator._runtime import add_manual_modifier

        add_manual_modifier(
            self._session_target(), "gamete", modifier, name, modifier_id,
            refresh=refresh,
        )

    def add_zygote_modifier(
        self,
        modifier: ZygoteModifier,
        name: Optional[str] = None,
        modifier_id: Optional[int] = None,
        refresh: bool = True,
    ) -> None:
        """Register a zygote-level modifier.

        Args:
            modifier: A ``ZygoteModifier`` callable or object.
            name: Optional human-readable name for debugging.
            modifier_id: Optional numeric priority used for ordering.
            refresh: If True (default), immediately rebuild modifier maps.
                Set to False when adding multiple modifiers in a batch;
                call :meth:`refresh_modifiers` or
                :meth:`refresh_modifier_maps` afterward to apply all at once.
        """
        from natal.frontend.configurator._runtime import add_manual_modifier

        add_manual_modifier(
            self._session_target(), "zygote", modifier, name, modifier_id,
            refresh=refresh,
        )

    def add_preset(self, preset: GeneticPreset) -> None:
        """Add a preset to this population.

        Registration is idempotent by object identity: if the exact same
        preset instance is already in ``_presets``, this is a no-op.
        This prevents double-registration when ``presets(drive)`` is
        called twice — the alternative (appending twice) would cause
        ``refresh_modifiers()`` to build two copies of the preset's
        gamete/zygote modifier, double-applying its effect in the
        offspring tensor.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).
        """
        if not any(p is preset for p in self._presets):
            self._presets.append(preset)

    def apply_preset(self, preset: GeneticPreset) -> None:
        """Apply a genetic preset to this population.

        This is the preferred API for registering presets. The preset's
        gamete modifiers, zygote modifiers, and fitness effects are
        registered in the correct order.

        Args:
            preset: A GeneticPreset instance (e.g., HomingDrive or custom preset).

        Examples:
            >>> from natal.frontend.presets import HomingDrive
            >>> drive = HomingDrive(
            ...     name="MyDrive",
            ...     drive_allele="Drive",
            ...     target_allele="WT",
            ...     drive_conversion_rate=0.95
            ... )
            >>> population.apply_preset(drive)

        See Also:
:class:`natal.frontend.presets.GeneticPreset` - Base class for creating custom presets
:class:`natal.frontend.presets.HomingDrive` - Built-in gene drive preset
        """
        from natal.frontend.configurator._runtime import (
            apply_runtime_presets,
        )

        apply_runtime_presets(self._session_target(), (preset,))

    @classmethod
    def builder(cls, species: Species) -> Any:
        """Create a builder for this population type.

        This is the recommended way to construct populations with presets.

        Args:
            species: Genetic architecture for the population.

        Returns:
            A builder instance for this population type.

        Examples:
            >>> pop = (AgeStructuredPopulation.builder(species)
            ...     .set_age_structure(n_ages=10)
            ...     .add_preset(HomingModificationDrive(...))
            ...     .build())
        """
        raise NotImplementedError(f"{cls.__name__} must implement builder()")

    def register_gamete_labels(self, labels: Optional[Sequence[str]]) -> None:
        """
        Register gamete labels in the IndexRegistry.

        Args:
            labels: Sequence of string labels to register. Labels must be
                unique in the provided sequence. Existing labels are ignored.
        """
        if not hasattr(self, "_index_registry") or self._index_registry is None:
            raise RuntimeError("IndexRegistry not initialized; cannot register gamete labels")

        if labels is None:
            return

        # Normalize and validate input
        try:
            seq = list(labels)
        except Exception as e:
            raise TypeError("labels must be a sequence of strings") from e

        # Ensure provided labels are unique
        if len(set(seq)) != len(seq):
            raise ValueError("labels must be unique")

        # Register each string label if not already present
        for lab in seq:
            if lab not in self._index_registry.glab_labels:
                self._index_registry.glab_labels.append(lab)
