"""Boundary-layer contracts between the frontend and execution backends.

This package defines the four data contracts (``Blueprint``, ``ParamsBlock``,
``SimState``, ``Program``), the backend capability table, the backend selector,
and the cross-backend conformance suite.  The frontend produces these objects;
every backend (reference / numba / rust) consumes them.  Neither side may
reach into the other's internals.

Contract discipline:
    - Fields are append-only: never remove or repurpose a field; opcode
      values are never reused.
    - ``CONTRACTS_VERSION`` must be bumped on any contract change and the
      Rust mirror in ``rust/src/contracts.rs`` updated in the same commit.

The package is intentionally empty during Phase 0 (directory reorganisation);
types land in Phase 1.
"""

__all__: list[str] = []
