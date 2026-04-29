import math

import natal.hooks.executor as executor


def _call_python_impl(fn, *args):  # type: ignore[no-untyped-def]
    """Call the pure-Python implementation of *fn*, bypassing any Numba wrapper.

    When Numba is enabled, ``@njit_switch`` returns a njit dispatcher whose
    ``.py_func`` attribute holds the original Python function.  When Numba is
    disabled the function is already plain Python.

    This helper keeps tests agnostic to the current Numba state so they pass
    in both modes.
    """
    return fn.py_func(*args) if hasattr(fn, "py_func") else fn(*args)


def test_apply_target_without_sperm_rounds_current_count_in_stochastic_mode(monkeypatch) -> None:
    captured: dict[str, float] = {}

    def fake_sample_survivors(n_base: float, survival_prob: float, stochastic_flag: bool, dirichlet_flag: bool) -> float:
        captured["n_base"] = n_base
        captured["survival_prob"] = survival_prob
        captured["stochastic_flag"] = 1.0 if stochastic_flag else 0.0
        captured["dirichlet_flag"] = 1.0 if dirichlet_flag else 0.0
        return 123.0

    monkeypatch.setattr(executor, "_sample_survivors", fake_sample_survivors)

    result = _call_python_impl(executor._apply_target_without_sperm, 9.999999999, 8.0, True, False)

    assert result == 123.0
    assert captured["n_base"] == 10.0
    assert math.isclose(captured["survival_prob"], 0.8, rel_tol=0.0, abs_tol=1e-12)
    assert captured["stochastic_flag"] == 1.0
    assert captured["dirichlet_flag"] == 0.0


def test_apply_target_without_sperm_keeps_continuous_mode_unrounded(monkeypatch) -> None:
    captured: dict[str, float] = {}

    def fake_sample_survivors(n_base: float, survival_prob: float, stochastic_flag: bool, dirichlet_flag: bool) -> float:
        captured["n_base"] = n_base
        captured["survival_prob"] = survival_prob
        captured["stochastic_flag"] = 1.0 if stochastic_flag else 0.0
        captured["dirichlet_flag"] = 1.0 if dirichlet_flag else 0.0
        return 321.0

    monkeypatch.setattr(executor, "_sample_survivors", fake_sample_survivors)

    result = _call_python_impl(executor._apply_target_without_sperm, 9.999999999, 8.0, True, True)

    assert result == 321.0
    assert not math.isclose(captured["n_base"], 10.0, rel_tol=0.0, abs_tol=1e-12)
    assert captured["stochastic_flag"] == 1.0
    assert captured["dirichlet_flag"] == 1.0
