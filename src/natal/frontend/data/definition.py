"""The frozen user declaration of one model (plan 5.1, slice 3).

:class:`ModelDefinition` is the replayable record of what the user
declared: the species, the granularity, and the ordered declaration
journal (every public chaining call with its explicitly-passed
arguments, live object references preserved).  ``build()`` freezes it
onto the population; the definition never changes afterwards — runtime
updates go through the writers, not through this snapshot.

The definition is deliberately reference-light: it holds no arrays of
its own, so mutating a draft array in place cannot leak into it, and
the only mutable things it references are the user's own objects
(presets, hooks), which the plan explicitly keeps outside the
no-side-effects guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from natal.frontend.configurator import Configurator
    from natal.frontend.genetics import Species


@dataclass(frozen=True)
class ModelDefinition:
    """The frozen declaration snapshot attached to a built population.

    Attributes:
        species: The genetic architecture the model declares.
        discrete_generation: Whether the declaration targets the
            discrete-generation granularity.
        journal: The ordered declaration entries
            ``(method_name, explicitly_passed_kwargs)`` captured by the
            ``@_declared`` journal — the replayable source of truth.
        build_name: The population name the chain declared.
    """

    species: Species
    discrete_generation: bool
    # Any: arbitrary explicitly-passed declaration values (floats,
    # presets, selectors, BatchSettings, ...) kept by reference.
    journal: tuple[tuple[str, dict[str, Any]], ...] = field(default=())
    build_name: str | None = None

    def replay(
        self, factory: Callable[[], Configurator]
    ) -> Configurator:
        """Rebuild a configurator from this definition's journal.

        Args:
            factory: Zero-argument constructor producing a fresh,
                empty configurator of the right granularity.

        Returns:
            The configurator after replaying every journal entry.
        """
        from natal.frontend.configurator._base import replay_declarations

        return replay_declarations(factory, list(self.journal))

    def entry_names(self) -> tuple[str, ...]:
        """Return the declaration method names in order.

        Returns:
            The ordered ``(method_name, ...)`` sequence of the journal.
        """
        return tuple(name for name, _ in self.journal)
