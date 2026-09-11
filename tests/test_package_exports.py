from __future__ import annotations

import natal as nt


def test_observation_is_exported_from_natal() -> None:
    assert hasattr(nt, "Observation")
    assert hasattr(nt.Observation, "apply")
