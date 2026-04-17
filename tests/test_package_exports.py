from __future__ import annotations

import natal as nt


def test_observation_is_exported_from_natal() -> None:
    assert hasattr(nt, "Observation")
    assert hasattr(nt, "ObservationFilter")
    assert hasattr(nt.Observation, "apply")


def test_ui_package_is_exported_from_natal() -> None:
    assert hasattr(nt, "ui")
    assert callable(nt.ui.launch)
