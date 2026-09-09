"""Configurator subpackage — chainable ModelDraft builders.

Provides the unified Configurator API for constructing and modifying
``ModelDraft`` (the former age-structured/discrete
subclass split into one class driven by the route table):

- :class:`Configurator` — chainable domain methods
  (``.competition()``, ``.reproduction()``) that write through the
  declarative route table.  Created via ``Configurator.from_species()``
  or bound to a running simulation via ``for_population()`` for runtime
  changes.  Batch writers live in :mod:`._writers`, routing logic in
  :mod:`._routes`.

Utility symbols:
  - ``set_param`` — write a scalar parameter by name, usable from pure
    Python or the route-table writers.
"""

from natal.frontend.configurator._base import (
    Configurator,
    set_param,
)
from natal.frontend.configurator._routes import (
    ROUTES,
    ROUTES_BY_METHOD,
    dispatch,
)
from natal.frontend.configurator._writers import (
    ConfigWriter,
    CoreConfigWriter,
    DraftWriter,
)

__all__ = [
    "Configurator",
    "CoreConfigWriter",
    "ConfigWriter",
    "DraftWriter",
    "ROUTES",
    "ROUTES_BY_METHOD",
    "dispatch",
    "set_param",
]
