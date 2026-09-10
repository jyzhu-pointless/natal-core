"""PopulationBuilder subpackage — build-side chains and the runtime updater.

Provides the unified PopulationBuilder API for constructing ``ModelDraft``
(the former age-structured/discrete subclass split into one class driven
by the route table), plus the runtime update handle:

- :class:`PopulationBuilder` — the build-side chain: domain methods
  (``.competition()``, ``.reproduction()``) that write through the
  declarative route table, created via ``PopulationBuilder.from_species()``
  and finalized with ``build()``.  Batch writers live in
  :mod:`._writers`, routing logic in :mod:`._routes`.
- :class:`RuntimeUpdater` — the single runtime-update handle returned by
  ``pop.update()`` and ``ctx.update()``: eight domain methods, a commit
  target (idle session or event transaction), and no build capability.

Utility symbols:
  - ``set_param`` — write a scalar parameter by name, usable from pure
    Python or the route-table writers.
"""

from natal.frontend.builder._base import (
    PopulationBuilder,
    set_param,
)
from natal.frontend.builder._routes import (
    ROUTES,
    ROUTES_BY_METHOD,
    dispatch,
)
from natal.frontend.builder._runtime import (
    RuntimeUpdater,
)

__all__ = [
    "PopulationBuilder",
    "ROUTES",
    "ROUTES_BY_METHOD",
    "RuntimeUpdater",
    "dispatch",
    "set_param",
]
