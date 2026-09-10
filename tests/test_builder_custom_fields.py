"""Test builder .custom() method for registering custom named slots.

The draft stores custom slots as a plain ``{name: value}`` dict — the same
shape as the runtime ``Params.custom_slots`` contract.  Values are
normalized at the ``build_custom_slots`` boundary: NumPy scalars collapse
to native Python values and 3-D arrays become owned float64 copies.
"""

import numpy as np
import pytest

import natal as nt

sp = nt.Species.from_dict(name="__custom_test__", structure={"auto": {"A": ["WT", "Var"]}})


def _build(custom_kwargs):
    return (
        nt.DiscreteGenerationPopulation
        .setup(species=sp, name="test", stochastic=False)
        .initial_state({"female": {"WT|WT": 5000}, "male": {"WT|WT": 5000}})
        .reproduction(eggs_per_female=50, sex_ratio=0.5)
        .competition(carrying_capacity=10000)
        .custom(**custom_kwargs)
        .build()
    )


class TestBuilderCustomFields:
    def test_scalar_custom(self):
        """Custom scalar slots are stored in config.custom."""
        pop = _build({"temperature": 25.0, "threshold": 100.0})
        cfg = pop.config
        assert cfg.custom["temperature"] == 25.0
        assert cfg.custom["threshold"] == 100.0

    def test_array_custom(self):
        """Custom 3-D ndarray slots are stored as float64 copies."""
        n_ages, n_gen = 2, 2
        habitat = np.zeros((2, n_ages, n_gen), dtype=np.float64)
        habitat[1, 0, 0] = 0.42
        pop = _build({"habitat": habitat})
        assert pop.config.custom["habitat"][1, 0, 0] == 0.42
        assert pop.config.custom["habitat"].shape == (2, 2, 2)
        assert pop.config.custom["habitat"].dtype == np.float64
        assert pop.config.custom["habitat"] is not habitat

    def test_mixed_scalar_and_array(self):
        """Scalars and arrays can be mixed in one call."""
        habitat = np.ones((2, 2, 2), dtype=np.float64)
        pop = _build({"rainfall": 0.8, "terrain": habitat})
        assert pop.config.custom["rainfall"] == 0.8
        assert pop.config.custom["terrain"].shape == (2, 2, 2)

    def test_custom_snapshot_isolated_and_runtime_update_commits(self):
        """Custom query edits stay local; explicit updates reach the session."""
        pop = _build({"temperature": 25.0})
        snapshot = pop.config
        snapshot.custom["temperature"] = 30.0
        assert pop.config.custom["temperature"] == 25.0
        pop.update().custom(temperature=30.0)
        assert pop.config.custom["temperature"] == 30.0

    def test_bool_custom(self):
        """bool values normalize to native Python bool."""
        pop = _build({"debug": True, "verbose": False})
        assert pop.config.custom["debug"] is True
        assert pop.config.custom["verbose"] is False

    def test_numpy_bool_custom(self):
        """np.bool_ values normalize to native Python bool."""
        pop = _build({"debug": np.bool_(True), "verbose": np.bool_(False)})
        assert pop.config.custom["debug"] is True
        assert type(pop.config.custom["verbose"]) is bool

    def test_int_custom(self):
        """int values normalize to native Python int."""
        pop = _build({"mode": 3, "iterations": 100})
        assert pop.config.custom["mode"] == 3
        assert type(pop.config.custom["iterations"]) is int

    def test_numpy_integer_custom(self):
        """np.integer values normalize to native Python int."""
        pop = _build({"mode": np.int64(3), "iterations": np.int32(100)})
        assert pop.config.custom["mode"] == 3
        assert pop.config.custom["iterations"] == 100
        assert type(pop.config.custom["iterations"]) is int

    def test_numpy_floating_custom(self):
        """np.floating values normalize to native Python float."""
        pop = _build({"temperature": np.float64(25.5)})
        assert pop.config.custom["temperature"] == 25.5
        assert type(pop.config.custom["temperature"]) is float

    def test_unsupported_custom_type_raises(self):
        """Unsupported custom scalar type raises TypeError early."""
        with pytest.raises(TypeError):
            _build({"label": "hot"})

    @pytest.mark.parametrize("shape", [(), (4,), (2, 3), (1, 2, 3, 4)])
    def test_custom_array_rank_is_preserved(self, shape):
        """Custom arrays keep their declared rank in isolated native storage."""
        values = np.zeros(shape, dtype=np.float64)
        pop = _build({"payload": values})
        assert pop.config.custom["payload"].shape == shape
        assert not np.shares_memory(pop.config.custom["payload"], values)

    def test_custom_snapshot_edits_do_not_change_runtime_values(self):
        """A helper may edit its owned snapshot without changing the session."""
        pop = _build({"temperature": 25.0, "threshold": 100.0})

        def read_custom(config):
            return config.custom["temperature"] + config.custom["threshold"]

        def write_custom(config):
            config.custom["temperature"] = 99.0
            return 0

        assert read_custom(pop.config) == 125.0
        snapshot = pop.config
        write_custom(snapshot)
        assert snapshot.custom["temperature"] == 99.0
        assert pop.config.custom["temperature"] == 25.0
        pop.update().custom(temperature=99.0)
        assert pop.config.custom["temperature"] == 99.0


class TestBuildCustomArrayRemoved:
    """Negative contract: the structured-array storage path is gone."""

    def test_build_custom_array_not_importable(self):
        import natal as nt
        from natal.frontend import data

        with pytest.raises(ImportError):
            from natal.frontend.data import (
                build_custom_array,  # type: ignore[attr-defined]  # noqa: F401  # negative contract: must not import
            )
        with pytest.raises(ImportError):
            from natal.frontend.data._engine import (
                build_custom_array,  # type: ignore[attr-defined]  # noqa: F401  # negative contract: must not import
            )
        assert not hasattr(nt, "build_custom_array")
        assert not hasattr(data, "build_custom_array")

    def test_draft_custom_is_a_dict(self):
        pop = _build({"temperature": 25.0})
        cfg = pop.config.custom
        assert isinstance(cfg, dict)
        assert cfg["temperature"] == 25.0
        # The runtime contract mirrors it: materialized params carry the
        # same value, owned by Params.
        p = __import__("natal.contracts.materialize", fromlist=["materialize"]).materialize(pop.config).params
        assert p.custom_slots["temperature"] == 25.0
