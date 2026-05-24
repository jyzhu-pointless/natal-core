"""Test builder .custom() method for registering custom named fields."""

import numpy as np
import natal as nt
import pytest

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
        """Custom scalar fields are stored in config.custom."""
        pop = _build({"temperature": 25.0, "threshold": 100.0})
        cfg = pop.config
        assert cfg.custom["temperature"][()] == 25.0
        assert cfg.custom["threshold"][()] == 100.0

    def test_array_custom(self):
        """Custom 3-D ndarray fields are stored as sub-arrays."""
        n_ages, n_gen = 2, 2
        habitat = np.zeros((2, n_ages, n_gen), dtype=np.float64)
        habitat[1, 0, 0] = 0.42
        pop = _build({"habitat": habitat})
        assert pop.config.custom["habitat"][1, 0, 0] == 0.42
        assert pop.config.custom["habitat"].shape == (2, 2, 2)

    def test_mixed_scalar_and_array(self):
        """Scalars and arrays can be mixed in one call."""
        habitat = np.ones((2, 2, 2), dtype=np.float64)
        pop = _build({"rainfall": 0.8, "terrain": habitat})
        assert pop.config.custom["rainfall"][()] == 0.8
        assert pop.config.custom["terrain"].shape == (2, 2, 2)

    def test_custom_mutable(self):
        """Custom fields are mutable in-place."""
        pop = _build({"temperature": 25.0})
        pop.config.custom["temperature"][()] = 30.0
        assert pop.config.custom["temperature"][()] == 30.0

    def test_bool_custom(self):
        """bool values preserve type as np.bool_."""
        pop = _build({"debug": True, "verbose": False})
        assert bool(pop.config.custom["debug"][()]) is True
        assert bool(pop.config.custom["verbose"][()]) is False
        assert pop.config.custom["debug"][()].dtype == np.bool_

    def test_int_custom(self):
        """int values preserve type as np.int64."""
        pop = _build({"mode": 3, "iterations": 100})
        assert pop.config.custom["mode"][()] == 3
        assert pop.config.custom["iterations"][()] == 100
        assert pop.config.custom["mode"][()].dtype == np.int64

    def test_numpy_floating_custom(self):
        """np.floating values are accepted as float custom fields."""
        pop = _build({"temperature": np.float64(25.5)})
        assert pop.config.custom["temperature"][()] == 25.5
        assert pop.config.custom["temperature"][()].dtype == np.float64

    def test_unsupported_custom_type_raises(self):
        """Unsupported custom scalar type raises TypeError early."""
        with pytest.raises(TypeError):
            _build({"label": "hot"})

    def test_custom_accessible_in_numba(self):
        """Custom fields can be read and mutated from @njit functions."""
        from natal.numba_utils import njit_switch

        pop = _build({"temperature": 25.0, "threshold": 100.0})

        @njit_switch(cache=True)
        def read_custom(config):
            return config.custom["temperature"][()] + config.custom["threshold"][()]

        @njit_switch(cache=True)
        def write_custom(config):
            config.custom["temperature"][()] = 99.0
            return 0

        assert read_custom(pop.config) == 125.0
        write_custom(pop.config)
        assert pop.config.custom["temperature"][()] == 99.0
