"""Value-based assertions for independently owned configuration snapshots."""

import numpy as np

from natal.frontend.data import ModelDraft


def assert_config_equal(actual: ModelDraft, expected: ModelDraft) -> None:
    """Compare every declared field without assuming shared snapshot identity."""
    for name in ModelDraft._fields:
        np.testing.assert_equal(
            getattr(actual, name), getattr(expected, name),
            err_msg=f"Configuration field {name} differs",
        )
