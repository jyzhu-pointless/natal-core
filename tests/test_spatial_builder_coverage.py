"""Comprehensive tests for ``natal.spatial_builder``.

Covers:
- Homogeneous builds (discrete_generation and age_structured pop_types)
- Heterogeneous builds with batch_setting
- Observation chaining
- HexGrid topology
- Error paths (n_demes=0, migration kernel shape)
- Migration with kernel_bank + deme_kernel_ids
- Migration with batch_setting kernel
- BatchSetting class: scalar, array, spatial, expand, first_value
"""

from __future__ import annotations

import numpy as np
import pytest

import natal as nt
from natal.numba_utils import numba_disabled
from natal.spatial_builder import BatchSetting, SpatialBuilder, batch_setting
from natal.spatial_topology import HexGrid, SquareGrid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_species(name: str = "SpatialBuilderTestSpecies") -> nt.Species:
    """Return a minimal species with one biallelic locus."""
    return nt.Species.from_dict(
        name,
        {
            "Chr1": {
                "L1": ["WT", "Dr"],
            }
        },
        gamete_labels=["default"],
    )


# ===========================================================================
# BatchSetting unit tests
# ===========================================================================

class TestBatchSetting:
    """Unit tests for BatchSetting class and batch_setting() helper."""

    def test_batch_setting_scalar_kind(self) -> None:
        bs = batch_setting([100, 200, 300])
        assert bs.kind == "scalar"
        assert bs.first_value() == 100
        expanded = bs.expand(3)
        assert expanded == [100, 200, 300]

    def test_batch_setting_scalar_length_mismatch_raises(self) -> None:
        bs = batch_setting([100, 200])
        with pytest.raises(ValueError, match="BatchSetting has 2 values but 4 demes"):
            bs.expand(4)

    def test_batch_setting_1d_array_kind(self) -> None:
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        bs = batch_setting(arr)
        assert bs.kind == "array"
        assert bs.first_value() == 10.0
        expanded = bs.expand(4)
        assert expanded == [10.0, 20.0, 30.0, 40.0]

    def test_batch_setting_1d_array_wrong_size_raises(self) -> None:
        arr = np.array([1.0, 2.0])
        bs = batch_setting(arr)
        with pytest.raises(ValueError, match="BatchSetting array has 2 values but 4"):
            bs.expand(4)

    def test_batch_setting_2d_array_expand_flat(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        bs = batch_setting(arr)
        assert bs.kind == "array"
        assert bs.first_value() == 1.0
        expanded = bs.expand(4)
        assert expanded == [1.0, 2.0, 3.0, 4.0]

    def test_batch_setting_2d_array_shape_mismatch_raises(self) -> None:
        """2D array with correct total size but wrong grid shape raises."""
        arr = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [7.0, 8.0, 9.0]])  # shape (3,3) = 9 values
        bs = batch_setting(arr)
        topo = SquareGrid(rows=1, cols=9)  # 9 demes, but topology expects 1x9
        with pytest.raises(ValueError, match="shape"):
            bs.expand(9, topology=topo)

    def test_batch_setting_spatial_fn_flat(self) -> None:
        bs = batch_setting(lambda i: float(i * 10))
        assert bs.kind == "spatial"
        assert bs.first_value() is None
        topo = SquareGrid(rows=1, cols=3)
        expanded = bs.expand(3, topology=topo)
        assert expanded == [0.0, 10.0, 20.0]

    def test_batch_setting_spatial_fn_rowcol(self) -> None:
        bs = batch_setting(lambda r, c: float(r * 3 + c))
        assert bs.kind == "spatial"
        topo = SquareGrid(rows=2, cols=3)
        expanded = bs.expand(6, topology=topo)
        assert expanded == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_batch_setting_spatial_without_topology_raises(self) -> None:
        bs = batch_setting(lambda i: float(i))
        with pytest.raises(ValueError, match="requires topology"):
            bs.expand(3)

    def test_batch_setting_passthrough(self) -> None:
        bs1 = batch_setting([1, 2, 3])
        bs2 = batch_setting(bs1)
        assert bs2 is bs1

    def test_batch_setting_repr(self) -> None:
        bs_scalar = batch_setting([1, 2])
        assert "scalar" in repr(bs_scalar)
        bs_spatial = batch_setting(lambda i: 1.0)
        assert "spatial" in repr(bs_spatial)

    def test_batch_setting_first_value_edge_cases(self) -> None:
        # Empty list
        bs = batch_setting([])
        assert bs.first_value() is None
        # Scalar None entry
        bs = batch_setting([None, 1])
        assert bs.first_value() is None
        # numpy array with zeros
        arr = np.zeros((1,))
        bs = batch_setting(arr)
        assert bs.first_value() == 0.0

    def test_batch_setting_ndim_error(self) -> None:
        arr = np.ones((2, 2, 2))
        with pytest.raises(ValueError, match="must be 1D or 2D"):
            BatchSetting(arr)

    def test_batch_setting_scalar_none_raises(self) -> None:
        """BatchSetting expand with None scalar values (line 155)."""
        bs = batch_setting([1, 2, 3])
        object.__setattr__(bs, '_values', None)
        with pytest.raises(ValueError, match="scalar values are None"):
            bs.expand(3)

    def test_batch_setting_array_none_raises(self) -> None:
        """BatchSetting expand with None array values (line 165)."""
        bs = batch_setting(np.array([1.0, 2.0]))
        object.__setattr__(bs, '_values_array', None)
        with pytest.raises(ValueError, match="array values are None"):
            bs.expand(2)

    def test_batch_setting_unknown_kind_raises(self) -> None:
        """BatchSetting expand with unknown kind (line 202)."""
        bs = batch_setting([1, 2])
        object.__setattr__(bs, '_kind', 'bogus')
        with pytest.raises(ValueError, match="Unknown kind"):
            bs.expand(2)

    def test_batch_setting_spatial_fn_1param(self) -> None:
        """Spatial fn expand with 1 parameter (line 200)."""
        fn = lambda i: float(i * 2)
        bs = batch_setting(fn)
        result = bs.expand(4, SquareGrid(1, 4))
        assert result == [0.0, 2.0, 4.0, 6.0]

    def test_batch_setting_spatial_fn_2param(self) -> None:
        """Spatial fn expand with 2 parameters (lines 196-199)."""
        fn = lambda r, c: float(r * 2 + c)
        bs = batch_setting(fn)
        result = bs.expand(6, SquareGrid(2, 3))
        assert len(result) == 6

    def test_batch_setting_spatial_fn_no_signature(self) -> None:
        """Spatial fn expand when inspect.signature raises ValueError (lines 193-194)."""
        def _no_sig_fn(x: float) -> float:
            return float(x * 3)
        # Setting __signature__ to a non-Signature causes inspect.signature to
        # raise ValueError, which triggers the fallback to _fn_param_count = 1.
        _no_sig_fn.__signature__ = "not-a-signature"  # type: ignore[assignment]
        bs = batch_setting(_no_sig_fn)
        result = bs.expand(4, SquareGrid(1, 4))
        assert result == [0.0, 3.0, 6.0, 9.0]


# ===========================================================================
# Homogeneous build — discrete_generation
# ===========================================================================

class TestHomogeneousBuildDiscrete:
    """Homogeneous spatial build with discrete_generation pop_type."""

    def test_build_and_run_minimal(self) -> None:
        species = _simple_species("HomoDiscreteMin")
        topo = SquareGrid(2, 2)
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=topo,
                                             pop_type="discrete_generation")
                .setup(name="test_min", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(kernel=None, migration_rate=0.0)
                .build()
            )
            assert isinstance(spatial, nt.SpatialPopulation)
            assert spatial.n_demes == 4
            assert spatial.tick == 0

            spatial.run(5)

        assert spatial.tick == 5
        assert spatial.get_total_count() > 0
        for i in range(4):
            assert spatial.deme(i).get_total_count() >= 0

    def test_build_homogeneous_no_topology(self) -> None:
        species = _simple_species("HomoDiscreteNoTopo")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=3, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="test_no_topo", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 50},
                        "male": {"WT|WT": 50},
                    }
                )
                .reproduction(eggs_per_female=5)
                .competition(carrying_capacity=500)
                .build()
            )
            spatial.run(3)

        assert spatial.tick == 3
        assert spatial.get_total_count() > 0
        assert spatial.topology is None

    def test_all_deme_indices_present(self) -> None:
        species = _simple_species("HomoDiscreteIdx")
        topo = SquareGrid(2, 2)
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=topo,
                                             pop_type="discrete_generation")
                .setup(name="test_idx", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .build()
            )

        # The first deme keeps the original name (no _deme_0 suffix).
        assert [spatial.deme(i).name for i in range(4)] == [
            "test_idx",
            "test_idx_deme_1",
            "test_idx_deme_2",
            "test_idx_deme_3",
        ]


# ===========================================================================
# Homogeneous build — age_structured
# ===========================================================================

class TestHomogeneousBuildAgeStructured:
    """Homogeneous spatial build with age_structured pop_type."""

    def test_build_and_run(self) -> None:
        species = _simple_species("HomoAgeStruct")
        topo = SquareGrid(2, 2)
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=topo,
                                             pop_type="age_structured")
                .setup(name="test_age", stochastic=False)
                .age_structure(n_ages=4, new_adult_age=2)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": [0, 0, 100, 0]},
                        "male": {"WT|WT": [0, 0, 100, 0]},
                    }
                )
                .survival(
                    female_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
                    male_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
                )
                .reproduction(eggs_per_female=10)
                .competition(age_1_carrying_capacity=500)
                .build()
            )
            spatial.run(3)

        assert spatial.tick == 3
        assert spatial.n_demes == 4
        assert spatial.get_total_count() > 0


# ===========================================================================
# Heterogeneous build — batch_setting
# ===========================================================================

class TestHeterogeneousBuild:
    """Heterogeneous spatial build with batch_setting parameters."""

    def test_batch_carrying_capacity_discrete(self) -> None:
        species = _simple_species("HetCCDisc")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="het_cc", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=batch_setting([1000, 500, 500, 1000]))
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.n_demes == 4

    def test_batch_eggs_per_female(self) -> None:
        species = _simple_species("HetEggs")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=3, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="het_eggs", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=batch_setting([10, 20, 30]))
                .competition(carrying_capacity=2000)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_batch_low_density_growth_rate(self) -> None:
        species = _simple_species("HetGrowth")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=3, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="het_growth", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(
                    carrying_capacity=1000,
                    low_density_growth_rate=batch_setting([2.0, 4.0, 6.0]),
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_batch_juvenile_growth_mode(self) -> None:
        species = _simple_species("HetMode")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=3, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="het_mode", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(
                    carrying_capacity=1000,
                    juvenile_growth_mode=batch_setting([2, 2, 3]),
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2


# ===========================================================================
# with_observation chaining
# ===========================================================================

class TestObservation:
    """Tests for with_observation chaining."""

    def test_with_observation_dict(self) -> None:
        species = _simple_species("ObsDict")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="obs_dict", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .with_observation(
                    groups={"wt": {"genotype": ["WT|WT"]}},
                    collapse_age=True,
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.record_observation is not None

    def test_with_observation_list(self) -> None:
        species = _simple_species("ObsList")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="obs_list", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .with_observation(
                    groups=[{"genotype": ["WT|WT"], "sex": ["female"]}],
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.record_observation is not None


# ===========================================================================
# HexGrid topology
# ===========================================================================

class TestHexGridTopology:
    """Tests with HexGrid topology."""

    def test_hexgrid_build_and_run(self) -> None:
        species = _simple_species("HexGridSpec")
        topo = HexGrid(rows=3, cols=4)
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=12, topology=topo,
                                             pop_type="discrete_generation")
                .setup(name="hex", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(kernel=None, migration_rate=0.0)
                .build()
            )
            spatial.run(3)

        assert spatial.tick == 3
        assert spatial.n_demes == 12
        assert spatial.topology is not None
        assert isinstance(spatial.topology, HexGrid)
        assert spatial.topology.rows == 3
        assert spatial.topology.cols == 4

    def test_hexgrid_wrapping(self) -> None:
        species = _simple_species("HexWrapSpec")
        topo = HexGrid(rows=2, cols=2, wrap=True)
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=topo,
                                             pop_type="discrete_generation")
                .setup(name="hex_wrap", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2


# ===========================================================================
# Error paths
# ===========================================================================

class TestErrorPaths:
    """Error paths in SpatialBuilder and SpatialPopulation."""

    def test_n_demes_zero_raises(self) -> None:
        species = _simple_species("ZeroDemes")
        with pytest.raises(ValueError, match="n_demes must be >= 1"):
            SpatialBuilder(species, n_demes=0)

    def test_n_demes_negative_raises(self) -> None:
        species = _simple_species("NegDemes")
        with pytest.raises(ValueError, match="n_demes must be >= 1"):
            SpatialBuilder(species, n_demes=-1)

    def test_adjacency_mode_requires_kernel_raises(self) -> None:
        """Kernel mode with no kernel and no kernel_bank raises."""
        species = _simple_species("KernelModeNoKernel")
        with numba_disabled():
            builder = (
                nt.SpatialPopulation.builder(species, n_demes=1, topology=SquareGrid(1, 1),
                                             pop_type="discrete_generation")
                .setup(stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
            )
        with pytest.raises(ValueError, match="migration_kernel is required"):
            builder.migration(strategy="kernel").build()

    def test_migration_kernel_even_dimension_raises(self) -> None:
        species = _simple_species("EvenKernel")
        with numba_disabled():
            builder = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="even_kern", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
            )
        # Validation of odd dimensions happens at build() time.
        builder.migration(kernel=np.array([[0.1, 0.1], [0.1, 0.1]]), migration_rate=0.1)
        with pytest.raises(ValueError, match="odd dimensions"):
            builder.build()

    def test_age_structure_on_discrete_raises(self) -> None:
        species = _simple_species("AgeStructOnDisc")
        builder = SpatialBuilder(species, n_demes=2, pop_type="discrete_generation")
        with pytest.raises(TypeError, match="age_structure.*only valid.*age_structured"):
            builder.age_structure(n_ages=4, new_adult_age=2)

    def test_batch_kernel_and_kernel_bank_conflict(self) -> None:
        species = _simple_species("KernelConflict")
        builder = SpatialBuilder(species, n_demes=2, pop_type="discrete_generation")
        with pytest.raises(ValueError, match="Cannot use batch_setting for kernel when kernel_bank"):
            builder.migration(
                kernel=batch_setting([np.ones((3, 3)), np.ones((3, 3))]),
                kernel_bank=[np.ones((3, 3))],
            )

    def test_batch_kernel_and_deme_kernel_ids_conflict(self) -> None:
        species = _simple_species("KernelIdsConflict")
        builder = SpatialBuilder(species, n_demes=2, pop_type="discrete_generation")
        with pytest.raises(ValueError, match="Cannot use batch_setting for kernel when deme_kernel_ids"):
            builder.migration(
                kernel=batch_setting([np.ones((3, 3)), np.ones((3, 3))]),
                deme_kernel_ids=np.array([0, 0]),
            )

    def test_wrong_invalid_migration_strategy_raises(self) -> None:
        species = _simple_species("InvalidStrategy")
        with numba_disabled():
            builder = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="inv_strat", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(strategy="invalid")
            )
        # Validation of migration_strategy happens at build() time.
        with pytest.raises(ValueError, match="migration_strategy must be one of"):
            builder.build()


# ===========================================================================
# Migration with kernel_bank + deme_kernel_ids
# ===========================================================================

class TestMigrationKernelBank:
    """Heterogeneous migration via kernel_bank and deme_kernel_ids."""

    def test_kernel_bank_build_and_run(self) -> None:
        species = _simple_species("KernBankSpec")
        kernel_a = np.ones((3, 3), dtype=np.float64)
        kernel_a[1, 1] = 0.0
        kernel_b = np.ones((3, 3), dtype=np.float64)
        kernel_b[1, 1] = 0.0

        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="kern_bank", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(
                    kernel_bank=[kernel_a, kernel_b],
                    deme_kernel_ids=np.array([0, 0, 1, 1], dtype=np.int64),
                    migration_rate=0.1,
                    kernel_include_center=False,
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.kernel_bank is not None
        assert spatial.deme_kernel_ids is not None
        assert spatial.migration_mode == "kernel"


# ===========================================================================
# Migration with batch_setting kernel
# ===========================================================================

class TestMigrationBatchKernel:
    """Heterogeneous migration using batch_setting for kernel."""

    def test_batch_kernel_build_and_run(self) -> None:
        species = _simple_species("BatchKernSpec")
        kernel = np.ones((3, 3), dtype=np.float64)
        kernel[1, 1] = 0.0

        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="batch_kern", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(
                    kernel=batch_setting([kernel, kernel, kernel, kernel]),
                    migration_rate=0.0,
                    kernel_include_center=False,
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.get_total_count() > 0

    def test_adjacency_migration_with_rate_zero(self) -> None:
        """Migration with rate=0 should not affect population."""
        species = _simple_species("AdjZeroRate")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="adj_zero", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(strategy="adjacency", migration_rate=0.0)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.migration_mode == "adjacency"

    def test_kernel_strategy_migration(self) -> None:
        """Explicit kernel migration strategy."""
        species = _simple_species("KernelStratSpec")
        kernel = np.ones((3, 3), dtype=np.float64)
        kernel[1, 1] = 0.0

        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="kern_strat", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(kernel=kernel, migration_rate=0.1, strategy="kernel",
                           kernel_include_center=False)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.migration_mode == "kernel"

    def test_adjust_migration_on_edge(self) -> None:
        """adjust_migration_on_edge=True with kernel migration."""
        species = _simple_species("AdjustEdgeSpec")
        kernel = np.array([[0.0, 0.1, 0.0],
                           [0.1, 0.0, 0.1],
                           [0.0, 0.1, 0.0]], dtype=np.float64)

        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="adjust_edge", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(kernel=kernel, migration_rate=0.1, strategy="kernel",
                           adjust_migration_on_edge=True)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.adjust_migration_on_edge is True

    def test_hybrid_strategy(self) -> None:
        """Hybrid migration strategy (falls back to auto)."""
        species = _simple_species("HybridStrat")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="hybrid", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(strategy="hybrid", migration_rate=0.0)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2


# ===========================================================================
# Builder method chain coverage
# ===========================================================================

class TestBuilderMethodChaining:
    """Additional coverage for builder method chaining paths."""

    def test_survival_discrete_with_three_params(self) -> None:
        species = _simple_species("SurvDisc")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="surv_disc", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .survival(
                    female_age0_survival=0.8,
                    male_age0_survival=0.7,
                    adult_survival=0.9,
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_fitness_method(self) -> None:
        species = _simple_species("FitSpec")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="fitness", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100, "WT|Dr": 0, "Dr|Dr": 0},
                        "male": {"WT|WT": 100, "WT|Dr": 0, "Dr|Dr": 0},
                    }
                )
                .fitness(viability={"WT|WT": 1.0, "WT|Dr": 0.9, "Dr|Dr": 0.8})
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_modifiers_method(self) -> None:
        species = _simple_species("ModSpec")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="modifiers", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .modifiers(gamete_modifiers=None, zygote_modifiers=None)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_survival_age_structured(self) -> None:
        species = _simple_species("SurvAgeSpec")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="age_structured")
                .setup(name="surv_age", stochastic=False)
                .age_structure(n_ages=4, new_adult_age=2)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": [0, 0, 100, 0]},
                        "male": {"WT|WT": [0, 0, 100, 0]},
                    }
                )
                .survival(
                    female_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
                    male_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
                )
                .reproduction(eggs_per_female=10)
                .competition(age_1_carrying_capacity=500)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_old_juvenile_carrying_capacity_alias(self) -> None:
        species = _simple_species("OldJuvCC")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="age_structured")
                .setup(name="old_juv", stochastic=False)
                .age_structure(n_ages=4, new_adult_age=2)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": [0, 0, 100, 0]},
                        "male": {"WT|WT": [0, 0, 100, 0]},
                    }
                )
                .survival(
                    female_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
                    male_age_based_survival_rates=[1.0, 1.0, 1.0, 0.0],
                )
                .reproduction(eggs_per_female=10)
                .competition(old_juvenile_carrying_capacity=800)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_hooks_method(self) -> None:
        species = _simple_species("HookSpec")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="hooks", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .hooks()
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_presets_method(self) -> None:
        species = _simple_species("PresetSpec")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="presets", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .presets()
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_presets_with_batch_setting(self) -> None:
        """Cover presets with BatchSetting positional arg (lines 797-803)."""
        species = _simple_species("PresetBatch")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="preset_batch", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .presets(batch_setting([None, None]))
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_presets_with_batch_setting_non_none(self) -> None:
        """Cover presets BatchSetting with non-None first_value (lines 801, 1392, 1396)."""
        species = _simple_species("PresetBatchNN")
        drive = nt.HomingDrive(
            name="test_drive",
            drive_allele="Dr",
            target_allele="WT",
            species=species,
        )
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="preset_batch_nn", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .presets(batch_setting([drive, None]))
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_presets_with_mixed_args(self) -> None:
        """Cover non-BatchSetting preset alongside BatchSetting (lines 802-803, 1397-1398)."""
        species = _simple_species("PresetMixed")
        drive = nt.HomingDrive(
            name="test_drive2",
            drive_allele="Dr",
            target_allele="WT",
            species=species,
        )
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="preset_mixed", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .presets(batch_setting([None, None]), drive)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2

    def test_fitness_and_hooks_triggers_fallback(self) -> None:
        """Cover _build_heterogeneous fallback with unrecognized kwarg & hooks replay.

        ``viability`` is not a PopulationConfig field, so ``_can_use_replace``
        returns False for the second group, triggering the full-builder-replay
        fallback.  Also covers hooks replay in ``_build_template_for_group``.
        (Lines 1147, 1209, 1402-1404)
        """
        species = _simple_species("FitnessFallback")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="fitness_fallback", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .fitness(viability=batch_setting([{"WT|WT": 1.0}, None]))
                .hooks()
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2


# ===========================================================================
# _make_hashable indirect coverage
# ===========================================================================

class TestMakeHashable:
    """Indirect testing of _make_hashable via batch_setting._resolve_migration_kernels."""

    def test_batch_setting_with_mixed_kernels(self) -> None:
        """Different kernel arrays in batch_setting lead to deduplication via _make_hashable."""
        species = _simple_species("HashSpec")

        kernel_a = np.ones((3, 3), dtype=np.float64)
        kernel_a[1, 1] = 0.0
        kernel_b = np.ones((3, 3), dtype=np.float64)
        kernel_b[1, 1] = 0.0
        # kernel_a and kernel_b are equal -> they should be deduplicated

        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=SquareGrid(2, 2),
                                             pop_type="discrete_generation")
                .setup(name="hash_test", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(
                    kernel=batch_setting([kernel_a, kernel_b, kernel_a, kernel_b]),
                    migration_rate=0.0,
                )
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2
        assert spatial.kernel_bank is not None
        assert len(spatial.kernel_bank) == 1


# ===========================================================================
# _make_hashable with dict/tuple/list inputs
# ===========================================================================

class TestMakeHashableBranches:
    """Direct tests for _make_hashable covering dict/tuple/list branches."""

    def test_make_hashable_dict(self) -> None:
        from natal.spatial_builder import _make_hashable
        d = {"b": 2, "a": 1}
        h = _make_hashable(d)
        assert isinstance(h, tuple)
        assert h[0] == "__dict__"
        assert len(h[1]) == 2

    def test_make_hashable_list(self) -> None:
        from natal.spatial_builder import _make_hashable
        lst = [3, 1, 2]
        h = _make_hashable(lst)
        assert isinstance(h, tuple)
        assert h == (3, 1, 2)

    def test_make_hashable_tuple(self) -> None:
        from natal.spatial_builder import _make_hashable
        tup = (10, 20)
        h = _make_hashable(tup)
        assert isinstance(h, tuple)
        assert h == (10, 20)

    def test_make_hashable_ndarray(self) -> None:
        from natal.spatial_builder import _make_hashable
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        h = _make_hashable(arr)
        assert isinstance(h, tuple)
        assert h[0] == "__ndarray__"

    def test_make_hashable_scalar(self) -> None:
        from natal.spatial_builder import _make_hashable
        assert _make_hashable(42) == 42
        assert _make_hashable("hello") == "hello"


# ===========================================================================
# _detect_and_delegate_with_positional_args coverage
# ===========================================================================

class TestDetectAndDelegateWithArgs:
    """Cover _detect_and_delegate_with_positional_args via presets."""

    def test_presets_with_positional_args(self) -> None:
        species = _simple_species("PosPresetOk")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=1,
                                             pop_type="discrete_generation")
                .setup(name="pp_ok", stochastic=False)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100},
                        "male": {"WT|WT": 100},
                    }
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .presets()
                .build()
            )
            spatial.run(2)
        assert spatial.tick == 2


# ===========================================================================
# initial_state with sperm_storage (age_structured)
# ===========================================================================

class TestInitialStateWithSpermStorage:
    """Cover the 'sperm_storage is not None' branch in initial_state."""

    def test_initial_state_with_sperm_storage(self) -> None:
        species = _simple_species("InitSpermStore")
        topo = SquareGrid(1, 2)
        n_genotypes = 2
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=topo,
                                             pop_type="age_structured")
                .setup(name="sperm_init", stochastic=False)
                .age_structure(n_ages=4, new_adult_age=2)
                .initial_state(
                    individual_count={
                        "female": {"WT|WT": 100, "WT|Dr": 0},
                        "male": {"WT|WT": 100, "WT|Dr": 0},
                    },
                    sperm_storage={"WT|WT": {"WT|WT": {2: 5}}},
                )
                .survival(
                    female_age_based_survival_rates=[1.0, 0.9, 0.8, 0.0],
                    male_age_based_survival_rates=[1.0, 0.8, 0.7, 0.0],
                )
                .reproduction(
                    eggs_per_female=10,
                    female_age_based_mating_rates=[0.0, 0.0, 0.3, 0.5],
                    male_age_based_mating_rates=[0.0, 0.0, 0.3, 0.5],
                )
                .competition(carrying_capacity=1000, expected_num_adult_females=100)
                .build()
            )
            spatial.run(2)

        assert spatial.tick == 2


# ===========================================================================
# Additional coverage: BatchSetting first_value with empty array
# ===========================================================================

class TestBatchSettingFirstValueEdge:
    """Additional first_value edge cases."""

    def test_empty_array_first_value(self) -> None:
        """BatchSetting with empty array returns None (line 226)."""
        arr = np.array([1.0])
        bs = batch_setting(arr)
        object.__setattr__(bs, '_values_array', np.array([]))
        assert bs.first_value() is None


# ===========================================================================
# Migration with explicit adjacency
# ===========================================================================

class TestMigrationAdjacency:
    """Tests for migration with explicit adjacency (line 951)."""

    def test_migration_with_adjacency(self) -> None:
        from natal.spatial_topology import build_adjacency_matrix
        species = _simple_species("MigAdj")
        topo = SquareGrid(2, 2)
        adj = build_adjacency_matrix(topo)
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=4, topology=topo,
                                             pop_type="discrete_generation")
                .setup(name="mig_adj", stochastic=False)
                .initial_state(
                    individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}},
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .migration(kernel=np.ones((3, 3)), migration_rate=0.1, strategy="kernel",
                           adjacency=adj)
                .build()
            )
            spatial.run(2)
        assert spatial.tick == 2


# ===========================================================================
# Heterogeneous build with varying initial state
# ===========================================================================

class TestHeterogeneousIndividualCount:
    """Heterogeneous build with per-deme individual_count (covers lines 1131, 1201, 1267-1281)."""

    def test_batch_individual_count(self) -> None:
        species = _simple_species("HetIndiv")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="het_indiv", stochastic=False)
                .initial_state(
                    individual_count=batch_setting([
                        {"female": {"WT|WT": 100}, "male": {"WT|WT": 100}},
                        {"female": {"WT|WT": 50}, "male": {"WT|WT": 50}},
                    ]),
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=1000)
                .build()
            )
            spatial.run(2)
        assert spatial.tick == 2


# ===========================================================================
# Observation on heterogeneous build
# ===========================================================================

class TestObservationOnHeterogeneous:
    """Observation chaining on a heterogeneous build (line 1178)."""

    def test_observation_on_hetero(self) -> None:
        species = _simple_species("HetObs")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=None,
                                             pop_type="discrete_generation")
                .setup(name="het_obs", stochastic=False)
                .initial_state(
                    individual_count={"female": {"WT|WT": 100}, "male": {"WT|WT": 100}},
                )
                .reproduction(eggs_per_female=10)
                .competition(carrying_capacity=batch_setting([500, 1000]))
                .with_observation(groups={"wt": {"genotype": ["WT|WT"]}}, collapse_age=True)
                .build()
            )
            spatial.run(2)
        assert spatial.tick == 2
        assert spatial.record_observation is not None


# ===========================================================================
# Heterogeneous build with age_structured individual_count and sperm_storage
# ===========================================================================

class TestHeterogeneousAgeStructuredState:
    """Heterogeneous age_structured build with batch individual_count and
    sperm_storage (covers lines 1133-1135, 1268, 1284-1293)."""

    def test_batch_age_structured_state(self) -> None:
        species = _simple_species("HetAgeState")
        with numba_disabled():
            spatial = (
                nt.SpatialPopulation.builder(species, n_demes=2, topology=SquareGrid(1, 2),
                                             pop_type="age_structured")
                .setup(name="het_age_state", stochastic=False)
                .age_structure(n_ages=4, new_adult_age=2)
                .initial_state(
                    individual_count=batch_setting([
                        {"female": {"WT|WT": 100}, "male": {"WT|WT": 100}},
                        {"female": {"WT|WT": 50}, "male": {"WT|WT": 50}},
                    ]),
                    sperm_storage=batch_setting([
                        {"WT|WT": {"WT|WT": {2: 10}}},
                        {"WT|WT": {"WT|WT": {2: 5}}},
                    ]),
                )
                .survival(
                    female_age_based_survival_rates=[1.0, 0.9, 0.8, 0.0],
                    male_age_based_survival_rates=[1.0, 0.8, 0.7, 0.0],
                )
                .reproduction(
                    eggs_per_female=10,
                    female_age_based_mating_rates=[0.0, 0.0, 0.3, 0.5],
                    male_age_based_mating_rates=[0.0, 0.0, 0.3, 0.5],
                )
                .competition(carrying_capacity=1000, expected_num_adult_females=100)
                .build()
            )
            spatial.run(2)
        assert spatial.tick == 2
