"""Typed, lifetime-checked interfaces for native Python-hook transactions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray


class EventTransaction(Protocol):
    """Owned Rust candidate bound to one callback invocation."""

    def state_arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Materialize isolated state arrays on first access."""
        ...

    def validate_state(self) -> None:
        """Validate any materialized state before committing metadata."""
        ...

    def ecology(self) -> dict[str, object]:
        """Return heterogeneous scalar/vector/custom ecology candidates."""
        ...

    def get_scalar(self, name: str) -> float:
        """Read one scalar candidate."""
        ...

    def get_tensor(self, name: str) -> NDArray[np.float64]:
        """Read an isolated tensor candidate."""
        ...

    def apply(self, writes: dict[str, float]) -> None:
        """Apply a validated batch of scalar candidates."""
        ...

    def tensor_write(self, name: str, values: NDArray[np.float64]) -> None:
        """Replace one validated tensor candidate."""
        ...

    def refresh_params(self, fields: list[str], source: object) -> None:
        """Import a heterogeneous compiled Params object into the candidate."""
        ...

    def get_custom_slots(self) -> dict[str, bool | int | float | NDArray[np.float64]]:
        """Read isolated custom values from this callback candidate."""
        ...

    def set_custom_slots(self, source: object) -> None:
        """Validate and replace the callback's custom values."""
        ...

    def sample(self, kind: str, a: float, b: float) -> float:
        """Sample from the candidate of the owning Rust RNG stream."""
        ...


class HookRng:
    """Lifetime-checked sampling from the owning deme's Rust random stream."""

    def __init__(self, transaction: EventTransaction, validate: Callable[[], None]) -> None:
        """Bind the native transaction and the callback lifetime check."""
        self._transaction = transaction
        self._validate = validate

    def _draw(self, kind: str, a: float, b: float, size: int | tuple[int, ...] | None) -> float | NDArray[np.float64]:
        """Sample requested dimensions after checking the callback lifetime."""
        self._validate()
        if size is None:
            return self._transaction.sample(kind, a, b)
        shape = (size,) if isinstance(size, int) else size
        if any(d < 0 for d in shape):
            raise ValueError("negative sampling dimensions are not allowed")
        return np.array([self._transaction.sample(kind, a, b) for _ in range(int(np.prod(shape)))], dtype=np.float64).reshape(shape)

    def random(self, size: int | tuple[int, ...] | None = None) -> float | NDArray[np.float64]:
        """Sample independent uniform values in [0, 1)."""
        return self._draw("uniform", 0.0, 1.0, size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size: int | tuple[int, ...] | None = None) -> float | NDArray[np.float64]:
        """Sample uniform values between the supplied bounds."""
        return self._draw("uniform", low, high, size)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size: int | tuple[int, ...] | None = None) -> float | NDArray[np.float64]:
        """Sample normal values with a supplied location and standard deviation."""
        return self._draw("normal", loc, scale, size)

    def integers(self, low: int, high: int | None = None, size: int | tuple[int, ...] | None = None, endpoint: bool = False) -> int | NDArray[np.int64]:
        """Sample integers with exclusive high unless endpoint is requested."""
        if high is None:
            low, high = 0, low
        upper = high + int(endpoint)
        result = self._draw("integers", float(low), float(upper), size)
        return result.astype(np.int64) if isinstance(result, np.ndarray) else int(result)

    def binomial(self, n: int | NDArray[np.int64], p: float | NDArray[np.float64], size: int | tuple[int, ...] | None = None) -> int | NDArray[np.int64]:
        """Sample binomial counts using the Rust numerical sampler."""
        self._validate()
        if isinstance(n, np.ndarray) or isinstance(p, np.ndarray):
            n_values, p_values = np.broadcast_arrays(n, p)
            if size is not None:
                shape = (size,) if isinstance(size, int) else size
                n_values = np.broadcast_to(n_values, shape)
                p_values = np.broadcast_to(p_values, shape)
            sampled = [self._transaction.sample("binomial", float(count), float(probability)) for count, probability in zip(n_values.flat, p_values.flat)]
            return np.asarray(sampled, dtype=np.int64).reshape(n_values.shape)
        result = self._draw("binomial", float(n), float(p), size)
        return result.astype(np.int64) if isinstance(result, np.ndarray) else int(result)


class GuardedConfigurator:
    """Keep every retained chain method tied to its original callback."""

    def __init__(self, configurator: object, validate: Callable[[], None]) -> None:
        """Bind a heterogeneous Configurator facade and its lifetime guard."""
        self._configurator = configurator
        self._validate = validate

    def __getattr__(self, name: str) -> Callable[..., GuardedConfigurator]:
        """Wrap an arbitrary builder method without changing its call syntax."""
        self._validate()
        # Any is required here because Configurator methods have heterogeneous
        # keyword signatures; the facade forwards them without interpreting them.
        def invoke(*args: Any, **kwargs: Any) -> GuardedConfigurator:
            """Check lifetime at invocation, including a previously saved method."""
            self._validate()
            getattr(self._configurator, name)(*args, **kwargs)
            return self
        return invoke
