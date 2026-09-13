"""Contract tests for publication stage guards and directory projections."""

import pytest

from natal.frontend.model.publication import IndexProjection
from natal.frontend.registry.index import IndexRegistry


def _registry() -> IndexRegistry:
    registry = IndexRegistry()
    registry.register_somatic_label("default")
    registry.register_gamete_label("default")
    registry.register_ztype("z0", "default")
    registry.register_gtype("g0", "default")
    return registry


def test_published_registry_rejects_registration_and_compression() -> None:
    registry = _registry()
    registry.mark_published()
    with pytest.raises(RuntimeError):
        registry.register_ztype("z1", "default")
    with pytest.raises(RuntimeError):
        registry.compress([], [])  # type: ignore[arg-type]


def test_projection_binds_source_keys_and_rejects_mismatch() -> None:
    full = _registry()
    runtime = _registry()
    projection = IndexProjection.from_registry(full, runtime)
    projection.validate_layout(full)
    other = _registry()
    other.register_ztype("z1", "default")
    with pytest.raises(ValueError):
        projection.validate_layout(other)
