"""Boundary-layer contracts between the frontend and execution backends.

This package defines the data contracts (``Blueprint``, ``Params``,
``SimState``, and — from slice ④ — the ``Program`` bytecode container),
the materialization bridge from the build-time draft, the backend
capability table and selector, and the cross-backend conformance
suite.

The frontend produces these objects; every backend (reference / rust)
consumes them.  Neither side may reach into the other's internals.

Contract discipline:
    - The frozen/mutable split is absolute: ``Blueprint`` holds only
      rebuild-to-change data; every runtime-mutable value lives in
      ``Params``.
    - Bump ``CONTRACTS_VERSION`` on any contract change; when the Rust
      mirror lands (slice ②), update it in the same change.

Participation in the top-level lazy export follows the package rule:
this ``__init__`` declares a non-empty literal ``__all__``.
"""

from natal.contracts.blueprint import Blueprint, format_type_name
from natal.contracts.materialize import (
    Materialized,
    gtype_names_from_registry,
    materialize,
    ztype_names_from_registry,
)
from natal.contracts.params import CustomValue, Params
from natal.contracts.state import SimState

__all__ = [
    "CONTRACTS_VERSION",
    "Blueprint",
    "CustomValue",
    "Materialized",
    "Params",
    "SimState",
    "format_type_name",
    "gtype_names_from_registry",
    "materialize",
    "ztype_names_from_registry",
]

CONTRACTS_VERSION: int = 2
