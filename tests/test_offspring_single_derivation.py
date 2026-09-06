"""Batch-6 strict tests: the offspring derivation has a single spelling.

The derived tensor ``P[i,j,k] = Σ meiosis_f[i,a] · meiosis_m[j,b] ·
fusion[a,b,k]`` used to be spelled independently at five call sites
(writer channel, modifier refresh, registry compression, build-time
maps, species blueprint).  Batch 6 collapses them onto one wrapper,
``natal.frontend.data._engine.recompute_offspring_tensor``, whose
ztype/gtype counts derive from the live table shapes.

Every test here attacks one way that collapse could be wrong:

1. **Axis-combination equivalence**: for {compress, no-compress} ×
   {1, 2 gamete labels} × {slabs}, the produced tensor equals an
   independent hand einsum over the live tables bit-for-bit, and the
   written-back config counts equal the live table widths (the
   registry's ``n_gtypes`` formula must match the shape-derived width).
2. **Channel agreement**: the modifier-refresh derivation and the
   tensor-write derivation produce identical floats on a multi-label
   species (the drift the batch exists to prevent).
3. **Ownership**: the wrapper returns freshly owned float64 storage
   that shares no memory with its inputs.
4. **Negative contract**: no frontend module other than the wrapper
   spells the numeric kernel, and the writer re-exports *are* the
   data-layer definitions (identity, not copies).
5. **Module lifecycle**: importing ``data._config`` first (whose
   kernel call is a lazy function-level import of ``data._engine``)
   resolves cleanly even after ``_engine`` is evicted mid-session.
"""

from __future__ import annotations

import re
import struct
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

import natal as nt

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_population(
    species_name: str,
    *,
    compress: bool,
    gamete_labels: list[str],
    initial: dict[str, dict[str, float]],
) -> nt.DiscreteGenerationPopulation:
    """Build a deterministic discrete population on a one-locus species.

    Args:
        species_name: Unique singleton-scoped species name.
        compress: Whether the builder compresses unreachable axes.
        gamete_labels: Gamete label vocabulary for the species.
        initial: ``{sex: {genotype: count}}`` initial state.

    Returns:
        The built population (deterministic, discrete-generation).
    """
    species = nt.Species.from_dict(
        name=species_name,
        structure={"c1": {"l1": ["A", "B", "C"]}},
        gamete_labels=gamete_labels,
    )
    return (
        nt.DiscreteGenerationPopulation.setup(
            species, stochastic=False, compress=compress
        )
        .initial_state(individual_count=initial)
        .survival(female_age0_survival=1.0, male_age0_survival=1.0)
        .reproduction(eggs_per_female=2, sex_ratio=0.5)
        .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
        .build()
    )


def _einsum_reference(meiosis: np.ndarray, fusion: np.ndarray) -> np.ndarray:
    """Independent spelling of the offspring derivation contract.

    Args:
        meiosis: Live ``(2, n_ztypes, n_gtypes)`` meiosis table.
        fusion: Live ``(n_gtypes, n_gtypes, n_ztypes)`` fusion table.

    Returns:
        ``einsum('ia,jb,abk->ijk', meiosis[0], meiosis[1], fusion)``.
    """
    return np.einsum("ia,jb,abk->ijk", meiosis[0], meiosis[1], fusion)


def _load_rust_kernel() -> (
    Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]] | None
):
    """Return the raw Rust offspring pyfunction, or None without the build.

    Returns:
        ``natal._engine_rs.compute_offspring_tensor`` when the compiled
        extension is importable, else ``None`` (extension-less CI).
    """
    try:
        from natal._engine_rs import compute_offspring_tensor
    except ImportError:
        return None
    return compute_offspring_tensor


def _load_equilibrium_kernel() -> (
    Callable[
        [
            float,
            float,
            float,
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            int,
            int,
            NDArray[np.float64] | None,
            float | None,
        ],
        tuple[float, float],
    ]
    | None
):
    """Return the flat Rust equilibrium pyfunction, or None without the build.

    Returns:
        ``natal._engine_rs.equilibrium_metrics_flat`` when the compiled
        extension is importable, else ``None`` (extension-less CI).
    """
    try:
        from natal._engine_rs import equilibrium_metrics_flat
    except ImportError:
        return None
    return equilibrium_metrics_flat


# ── Axis-combination equivalence: compression × gamete labels ────────────────


class TestCompressedRegistryMultiLabel:
    """compress=True with 2 gamete labels: flat HL axis, pruned both axes."""

    def test_offspring_matches_einsum_exactly(self) -> None:
        """Compressed multi-label build equals the hand einsum bit-for-bit.

        Attack: compression prunes the flat ``n_hg * n_glabs`` gamete
        axis down to a non-Cartesian width (2 here, which is not
        ``n_hg * 2`` for any integral n_hg), so a derivation that reads
        n_gtypes from the blueprint's Cartesian dimensions reshapes the
        fusion table into the wrong buckets.  The einsum over the live
        tables is implementation-blind and must match exactly.
        """
        pop = _build_population(
            "__b6_compress_mlabel__",
            compress=True,
            gamete_labels=["default", "cas9"],
            initial={"female": {"B|C": 10}, "male": {"B|C": 10}},
        )
        meiosis = np.asarray(pop.config.zygotes_to_gametes_map)
        fusion = np.asarray(pop.config.gametes_to_zygotes_map)
        derived = np.asarray(pop.config.offspring_tensor)

        # From B|C only {B, C} gametes and {B|B, B|C, C|C} zygotes stay.
        assert meiosis.shape == (2, 3, 2)
        assert fusion.shape == (2, 2, 3)
        assert derived.shape == (3, 3, 3)
        np.testing.assert_array_equal(derived, _einsum_reference(meiosis, fusion))

        # Hand-computed Mendelian cells (atol=0: exact reals 0/0.5/1):
        # B|B × B|B → B|B only.
        np.testing.assert_allclose(derived[0, 0, :], [1.0, 0.0, 0.0], rtol=0, atol=0)
        # B|B × B|C → 1/2 B|B + 1/2 B|C.
        np.testing.assert_allclose(derived[0, 1, :], [0.5, 0.5, 0.0], rtol=0, atol=0)
        # B|C × C|C → 1/2 B|C + 1/2 C|C (asymmetric-axis pin).
        np.testing.assert_allclose(derived[1, 2, :], [0.0, 0.5, 0.5], rtol=0, atol=0)
        # Every (gf, gm) row of the derived tensor stays a distribution.
        np.testing.assert_allclose(derived.sum(axis=-1), 1.0, rtol=0, atol=1e-12)

    def test_config_counts_equal_live_table_widths(self) -> None:
        """The kept ``n_gtypes`` write-back matches the shape-derived width.

        Attack: the registry overrides dict still spells
        ``n_hg_effective if gtype_compressed else n_hg_effective *
        n_glabs_effective``; if its ``n_hg_effective`` source were stale
        (pre-slice) the written count would disagree with the live
        meiosis width and every downstream axis consumer mis-slices.
        """
        pop = _build_population(
            "__b6_compress_mlabel_wb__",
            compress=True,
            gamete_labels=["default", "cas9"],
            initial={"female": {"B|C": 10}, "male": {"B|C": 10}},
        )
        meiosis = np.asarray(pop.config.zygotes_to_gametes_map)
        # Compressed gamete axis: 2 is not any n_hg × 2 product, proving
        # the flat-axis layout the write-back formula must track.
        assert int(pop.config.n_gtypes) == 2
        assert int(pop.config.n_gtypes) == meiosis.shape[2]
        assert int(pop.config.n_ztypes) == meiosis.shape[1]
        assert pop.config.gtype_names == ("B:default", "C:default")


class TestUncompressedBuildMultiLabel:
    """compress=False with 2 gamete labels: full Cartesian glab product."""

    def test_offspring_matches_einsum_exactly(self) -> None:
        """Uncompressed multi-label build equals the hand einsum exactly.

        Attack: the old build-time spelling passed
        ``n_gtypes = n_hg_effective * n_glabs_effective``; the new one
        reads ``meiosis.shape[2]``.  On the uncompressed axis these must
        coincide (3 haplotypes × 2 labels = 6), and the derivation over
        the live 6-wide tables must be exact — including the all-zero
        non-baseline glab columns contributing exact zeros.
        """
        pop = _build_population(
            "__b6_nocompress_mlabel__",
            compress=False,
            gamete_labels=["default", "cas9"],
            initial={"female": {"B|C": 10}, "male": {"B|C": 10}},
        )
        meiosis = np.asarray(pop.config.zygotes_to_gametes_map)
        fusion = np.asarray(pop.config.gametes_to_zygotes_map)
        derived = np.asarray(pop.config.offspring_tensor)

        assert meiosis.shape == (2, 6, 6)
        assert fusion.shape == (6, 6, 6)
        assert derived.shape == (6, 6, 6)
        assert int(pop.config.n_gtypes) == 6
        np.testing.assert_array_equal(derived, _einsum_reference(meiosis, fusion))
        np.testing.assert_allclose(derived.sum(axis=-1), 1.0, rtol=0, atol=1e-12)


class TestSpeciesBlueprintMultiLabelSlab:
    """Species blueprint: 2 gamete labels × 2 somatic slabs, cached."""

    def test_blueprint_offspring_matches_einsum_exactly(self) -> None:
        """Blueprint tensor equals the hand einsum; metadata matches shapes.

        Attack: the blueprint derivation previously passed
        ``n_ztypes = n_g * n_slabs`` / ``n_gtypes = n_hg * n_glabs``
        explicitly; the shape-derived spelling must reproduce the same
        (z=6, g=4) layout on a labels × slabs species, and the cached
        blueprint metadata must stay consistent with the array shapes
        (downstream builders read both).
        """
        species = nt.Species.from_dict(
            name="__b6_blueprint_mlabel_slab__",
            structure={"c1": {"l1": ["A", "B"]}},
            gamete_labels=["default", "cas9"],
            somatic_labels=["S0", "S1"],
        )
        blueprint = species.get_config_blueprint()
        z2g = np.asarray(blueprint["zygotes_to_gametes_map"])
        g2z = np.asarray(blueprint["gametes_to_zygotes_map"])
        offspring = np.asarray(blueprint["offspring_tensor"])

        # 3 unordered genotypes × 2 slabs = 6 ztypes;
        # 2 haplotypes × 2 glabs = 4 gtypes.
        assert z2g.shape == (2, 6, 4)
        assert g2z.shape == (4, 4, 6)
        assert offspring.shape == (6, 6, 6)
        np.testing.assert_array_equal(offspring, _einsum_reference(z2g, g2z))
        # Metadata written next to the arrays matches the shapes the
        # derivation actually used.
        assert blueprint["n_ztypes"] == z2g.shape[1]
        assert blueprint["n_gtypes"] == z2g.shape[2]

    def test_blueprint_cache_returns_identical_tensor(self) -> None:
        """Second blueprint call is the cached object (no re-derivation).

        Attack: if the cache key missed, a second derivation could
        return a different tensor object with (barring bugs) equal
        values — the identity check pins the caching transition.
        """
        species = nt.Species.from_dict(
            name="__b6_blueprint_cache__",
            structure={"c1": {"l1": ["A", "B"]}},
            gamete_labels=["default", "cas9"],
            somatic_labels=["S0", "S1"],
        )
        first = species.get_config_blueprint()
        second = species.get_config_blueprint()
        assert first is second
        assert first["offspring_tensor"] is second["offspring_tensor"]


# ── Channel agreement: refresh vs tensor_write on a multi-label species ───────


class TestChannelAgreementMultiLabel:
    """The refresh and write channels derive identical floats (no drift)."""

    def test_refresh_matches_write_channel_bit_for_bit(self) -> None:
        """Modifier-refresh and tensor-write recomputes are bit-identical.

        Attack: the historical P6 drift — the writer channel and the
        modifier refresh were separate spellings of the derivation and
        could disagree in the last ulp.  With a HomingDrive on a
        2-label species (drive mass lives in the cas9 columns), a
        no-op round trip of the live meiosis table through
        ``tensor_write`` must reproduce the refresh output exactly.
        """
        species = nt.Species.from_dict(
            name="__b6_channel_mlabel__",
            structure={"chr1": {"loc": ["WT", "Dr"]}},
            gamete_labels=["default", "cas9"],
        )
        drive = nt.HomingDrive(
            name="hd_b6",
            drive_allele="Dr",
            target_allele="WT",
            resistance_allele="WT",
            drive_conversion_rate=0.95,
        )
        pop = (
            nt.DiscreteGenerationPopulation.setup(species, stochastic=False)
            .presets(drive)
            .initial_state(
                individual_count={"female": {"WT|Dr": 10}, "male": {"WT|Dr": 10}}
            )
            .survival(female_age0_survival=1.0, male_age0_survival=1.0)
            .reproduction(eggs_per_female=2, sex_ratio=0.5)
            .competition(carrying_capacity=100000.0, low_density_growth_rate=2.0)
            .build()
        )
        pop.refresh_modifier_maps()
        meiosis = np.asarray(pop.config.zygotes_to_gametes_map)
        fusion = np.asarray(pop.config.gametes_to_zygotes_map)
        refreshed = np.asarray(pop.config.offspring_tensor)

        # Multi-label layout with live drive mass (3 ztypes × 4 gtypes).
        assert meiosis.shape == (2, 3, 4)
        np.testing.assert_array_equal(refreshed, _einsum_reference(meiosis, fusion))
        # The drive really biases the cas9-labelled columns (non-trivial
        # input to the derivation, not the Mendelian baseline).
        assert meiosis[:, 1, 2:].sum() > 0.0

        # Drop the refresh sentinel so only the write's own marking is
        # asserted, then round-trip the same table through the write
        # channel: identical floats in, identical tensor out.
        pop._rust_dirty.clear()
        pop.params.tensor_write("meiosis_map", pop.params.meiosis_map.array)
        np.testing.assert_array_equal(
            np.asarray(pop.config.offspring_tensor), refreshed
        )
        assert pop._rust_dirty == {"meiosis_map", "offspring_tensor"}


# ── Ownership: the wrapper owns its output ────────────────────────────────────


class TestWrapperOwnership:
    """``recompute_offspring_tensor`` returns freshly owned storage."""

    def _tables(self) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic normalized meiosis/fusion table pair.

        Returns:
            A ``(meiosis, fusion)`` pair where meiosis rows sum to 1 and
            fusion maps gamete pair (a, b) to zygote ``(a + b) % n_z``.
        """
        rng = np.random.default_rng(7)
        n_z, n_g = 5, 4
        meiosis = rng.random((2, n_z, n_g))
        meiosis /= meiosis.sum(axis=2, keepdims=True)
        fusion = np.zeros((n_g, n_g, n_z))
        for a in range(n_g):
            for b in range(n_g):
                fusion[a, b, (a + b) % n_z] = 1.0
        return meiosis, fusion

    def test_result_shares_no_memory_and_inputs_survive_mutation(
        self,
    ) -> None:
        """Mutating the result leaves the inputs bit-identical.

        Attack: a wrapper that returns a view over the meiosis or fusion
        buffer would let callers corrupt the live tables through the
        derived tensor.
        """
        from natal.frontend.data._engine import recompute_offspring_tensor

        meiosis, fusion = self._tables()
        meiosis_before, fusion_before = meiosis.copy(), fusion.copy()
        result = recompute_offspring_tensor(meiosis, fusion)

        assert not np.shares_memory(result, meiosis)
        assert not np.shares_memory(result, fusion)
        result.fill(-1.0)
        np.testing.assert_array_equal(meiosis, meiosis_before)
        np.testing.assert_array_equal(fusion, fusion_before)

    def test_result_dtype_layout_and_values(self) -> None:
        """Output is fresh float64 C-contiguous and equals the einsum.

        Attack: a wrapper that skipped the dtype coercion or the
        contiguity guarantee would leak layout surprises into configs
        that are later written into the Rust session as flat buffers.
        """
        from natal.frontend.data._engine import recompute_offspring_tensor

        meiosis, fusion = self._tables()
        result = recompute_offspring_tensor(meiosis, fusion)
        assert result.dtype == np.float64
        assert result.flags["C_CONTIGUOUS"]
        np.testing.assert_array_equal(result, _einsum_reference(meiosis, fusion))

    def test_float32_input_is_coerced_without_aliasing(self) -> None:
        """A float32 caller buffer is never aliased by the float64 result.

        Attack: ``np.asarray(..., dtype=float64)`` on a float32 input
        must copy; if it somehow kept a reference, mutating the result
        would scribble on the caller's table.
        """
        from natal.frontend.data._engine import recompute_offspring_tensor

        meiosis, fusion = self._tables()
        meiosis32 = meiosis.astype(np.float32)
        result = recompute_offspring_tensor(meiosis32, fusion)
        assert result.dtype == np.float64
        assert not np.shares_memory(result, meiosis32)
        poisoned = result.copy()
        result.fill(-1.0)
        assert not np.shares_memory(meiosis32, poisoned)
        np.testing.assert_allclose(
            meiosis32, meiosis.astype(np.float32), rtol=0, atol=0
        )


# ── Negative contract: single spelling + re-export identity ──────────────────


class TestSingleSpellingContract:
    """The kernel is spelled only in its definition file and the wrapper."""

    _FRONTEND_MODULES_THAT_MUST_NOT_SPELL_KERNEL = (
        "natal/frontend/configurator/_writers.py",
        "natal/frontend/configurator/_registry_builder.py",
        "natal/frontend/data/_config.py",
        "natal/frontend/genetics/structures/_mapping.py",
        "natal/frontend/population/_mixins/_modifiers.py",
        "natal/frontend/spatial/population.py",
    )

    def test_replaced_call_sites_do_not_spell_the_kernel(self) -> None:
        """None of the four replaced call sites names the kernel directly.

        Attack: a partial revert that reintroduces
        ``compute_offspring_probability_tensor`` at any former call site
        recreates the drift surface batch 6 collapsed.  This is the
        route-scoped complement to the repo-wide ledger test.
        """
        src_root = _REPO_ROOT / "src"
        for rel in self._FRONTEND_MODULES_THAT_MUST_NOT_SPELL_KERNEL:
            text = (src_root / rel).read_text(encoding="utf-8")
            assert "compute_offspring_probability_tensor" not in text, rel

    def test_no_frontend_module_outside_the_wrapper_spells_the_kernel(
        self,
    ) -> None:
        """Whole-frontend scan: the kernel name exists only in ``_engine``.

        Attack: new frontend code (not among the four replaced sites)
        importing the kernel directly would add a sixth spelling.
        """
        src_root = _REPO_ROOT / "src" / "natal" / "frontend"
        offenders = [
            str(path.relative_to(_REPO_ROOT / "src"))
            for path in sorted(src_root.rglob("*.py"))
            if path.name != "_engine.py"
            and "compute_offspring_probability_tensor"
            in path.read_text(encoding="utf-8")
        ]
        assert offenders == []

    def test_writer_reexports_are_the_data_layer_definitions(self) -> None:
        """``_writers`` re-exports ARE ``data._engine`` definitions.

        Attack: a re-export that accidentally redefined (or later
        shadowed) the functions locally would fork the spelling again
        at the import level while looking identical to readers.
        """
        from natal.frontend.configurator import _writers

        # The exact import spelling spatial/population.py depends on.
        from natal.frontend.configurator._writers import (
            recompute_offspring_tensor,
            validate_meiosis_table,
        )
        from natal.frontend.data import _engine

        assert _writers.recompute_offspring_tensor is _engine.recompute_offspring_tensor
        assert _writers.validate_meiosis_table is _engine.validate_meiosis_table
        assert recompute_offspring_tensor is _engine.recompute_offspring_tensor
        assert validate_meiosis_table is _engine.validate_meiosis_table
        # Definition point really moved to the data layer.
        assert _engine.recompute_offspring_tensor.__module__ == (
            "natal.frontend.data._engine"
        )
        assert _engine.validate_meiosis_table.__module__ == (
            "natal.frontend.data._engine"
        )
        for name in ("recompute_offspring_tensor", "validate_meiosis_table"):
            assert name in _writers.__all__
            assert name in _engine.__all__

    def test_helpers_stay_internal_to_the_package(self) -> None:
        """The moved helpers are not re-exported on the public ``natal``.

        Attack: an ``__init__`` (or stub) re-export would widen the
        public API surface the batch explicitly kept private.
        """
        assert not hasattr(nt, "recompute_offspring_tensor")
        assert not hasattr(nt, "validate_meiosis_table")


# ── Module lifecycle: the lazy import inside data/_config.py ─────────────────


class TestModuleLifecycle:
    """``data/_config.py`` imports the wrapper lazily; prove it resolves."""

    def test_config_first_import_and_engine_eviction_resolve(self) -> None:
        """A fresh interpreter importing ``_config`` first stays healthy.

        Attack: ``_config`` imports ``_engine`` only inside
        ``build_config_maps`` while ``_engine`` imports ``_config`` at
        module level — a real cycle if the lazy edge ever moved to
        module scope.  Importing ``_config`` first, evicting ``_engine``
        from ``sys.modules`` (simulating a partially initialized or
        reloaded state), and re-resolving must produce a working
        derivation whose module is the data layer's.
        """
        program = (
            "import sys\n"
            "import natal.frontend.data._config as cfg\n"
            "assert 'natal.frontend.data._engine' in sys.modules\n"
            "del sys.modules['natal.frontend.data._engine']\n"
            "del sys.modules['natal.frontend.data._builders']\n"
            "import natal.frontend.data._engine as eng\n"
            "import numpy as np\n"
            "m = np.full((2, 3, 3), 1.0 / 3.0)\n"
            "f = np.zeros((3, 3, 3))\n"
            "for a in range(3):\n"
            "    f[a, a, a] = 1.0\n"
            "out = eng.recompute_offspring_tensor(m, f)\n"
            "assert eng.recompute_offspring_tensor.__module__ == "
            "'natal.frontend.data._engine'\n"
            "assert out.shape == (3, 3, 3)\n"
            "ref = np.einsum('ia,jb,abk->ijk', m[0], m[1], f)\n"
            "assert np.array_equal(out, ref)\n"
            "assert cfg.build_config_maps is not None\n"
            "print('lifecycle-ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "lifecycle-ok" in result.stdout

    def test_config_first_import_builds_exact_tensor(self) -> None:
        """Building via a ``_config``-first interpreter matches einsum.

        Attack: if the lazy import resolved to a different module object
        than the one the population build uses, the build-time
        derivation could silently route through a stale definition.
        Importing ``_config`` before anything else and then building the
        multi-label population must still produce the exact tensor.
        """
        program = (
            "import natal.frontend.data._config\n"
            "import numpy as np\n"
            "import natal as nt\n"
            "sp = nt.Species.from_dict(\n"
            "    name='__b6_subprocess_config_first__',\n"
            "    structure={'c1': {'l1': ['A', 'B', 'C']}},\n"
            "    gamete_labels=['default', 'cas9'],\n"
            ")\n"
            "pop = (\n"
            "    nt.DiscreteGenerationPopulation.setup(\n"
            "        sp, stochastic=False, compress=False\n"
            "    )\n"
            "    .initial_state(\n"
            "        individual_count={'female': {'B|C': 10}, "
            "'male': {'B|C': 10}}\n"
            "    )\n"
            "    .survival(female_age0_survival=1.0, male_age0_survival=1.0)\n"
            "    .reproduction(eggs_per_female=2, sex_ratio=0.5)\n"
            "    .competition(carrying_capacity=100000.0, "
            "low_density_growth_rate=2.0)\n"
            "    .build()\n"
            ")\n"
            "m = np.asarray(pop.config.zygotes_to_gametes_map)\n"
            "f = np.asarray(pop.config.gametes_to_zygotes_map)\n"
            "ref = np.einsum('ia,jb,abk->ijk', m[0], m[1], f)\n"
            "assert np.array_equal(np.asarray(pop.config.offspring_tensor), "
            "ref)\n"
            "print('build-ok')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(_REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "build-ok" in result.stdout


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


class TestRustKernelParity:
    """The Rust kernel and the pure-Python fallback are bit-identical."""

    def test_rust_kernel_bitwise_matches_python_kernel(self) -> None:
        """Random tables (with zero entries) agree bitwise across kernels.

        The Rust kernel and the reference Python kernel use the same
        statement order and zero-skips, so any divergence is a kernel
        bug, not float noise.
        """
        rust_available = True
        try:
            from natal._engine_rs import compute_offspring_tensor as rust_kernel
        except ImportError:
            rust_available = False
        if not rust_available:
            from natal.frontend.data._engine import _rust_offspring_kernel

            assert _rust_offspring_kernel(
                np.zeros((2, 1, 1)), np.zeros((1, 1, 1))
            ) is None
            return

        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

        rng = np.random.default_rng(2026)
        for z, g in ((2, 2), (5, 4), (7, 6), (13, 3)):
            meiosis = rng.random((2, z, g))
            meiosis[meiosis < 0.3] = 0.0  # exercise the skip-zero path
            # Renormalize rows so entries stay probability-like (values
            # themselves do not need to be probabilities for the parity
            # check, but keep them finite and in [0, 1]).
            fusion = (rng.random((g, g, z)) < 0.5).astype(np.float64)

            rust = np.asarray(
                rust_kernel(
                    np.ascontiguousarray(meiosis), np.ascontiguousarray(fusion)
                )
            ).reshape(z, z, z)
            py = compute_offspring_probability_tensor(
                meiosis_f=meiosis[0],
                meiosis_m=meiosis[1],
                haplo_to_genotype_map=fusion,
                n_ztypes=z,
                n_gtypes=g,
            )
            np.testing.assert_array_equal(
                rust, py, err_msg=f"kernel parity broke at z={z}, g={g}"
            )

    def test_wrapper_prefers_rust_kernel_when_available(self) -> None:
        """The single wrapper dispatches to the Rust kernel path.

        With the extension importable, the wrapper's result must equal
        the direct kernel call exactly (identity of code path, not just
        of values).
        """
        from natal.frontend.data._engine import _rust_offspring_kernel

        rng = np.random.default_rng(99)
        meiosis = np.ascontiguousarray(rng.random((2, 4, 3)))
        fusion = np.ascontiguousarray(
            (rng.random((3, 3, 4)) < 0.5).astype(np.float64)
        )
        wrapped = _rust_offspring_kernel(meiosis, fusion)
        if wrapped is None:
            pytest.skip("rust extension not built")
        import natal._engine_rs as rs

        direct = np.asarray(rs.compute_offspring_tensor(meiosis, fusion)).reshape(
            4, 4, 4
        )
        np.testing.assert_array_equal(wrapped, direct)


class TestRustKernelAdversarialParity:
    """Batch-10 attacks: the Rust kernel is the *same* arithmetic.

    Category map (adversarial-review): shape/sparsity grid = axis
    combination; order probe and FMA probe = invariant (statement-order
    and rounding identity); NaN/inf = invariant (propagation identity);
    forced-ImportError = state transition (dispatch branch); shape
    violations = error path; per-call storage = ownership.  Every value
    assertion is a byte-level comparison (``tobytes``), never a
    tolerance — the contract is bit identity, not closeness.
    """

    _GRID = (
        # (name, z, g, meiosis zero-fraction): single element, wide/tall
        # degenerate axes, empty tables, squares, non-squares, dense.
        ("single", 1, 1, 0.0),
        ("wide_z1", 1, 5, 0.3),
        ("tall_g1", 6, 1, 0.5),
        ("empty", 0, 0, 0.0),
        ("square2", 2, 2, 0.3),
        ("square9", 9, 9, 0.2),
        ("z5_g4", 5, 4, 0.4),
        ("z7_g6", 7, 6, 0.1),
        ("z13_g3", 13, 3, 0.6),
        ("z4_g10", 4, 10, 0.25),
        ("dense", 10, 7, 0.0),
    )

    def test_shape_sparsity_grid_is_bit_identical(self) -> None:
        """Every shape corner and sparsity level matches the Python kernel.

        Attack: an indexing bug that only shows on non-square ``z != g``
        tables (fusion strides use ``z``, meiosis strides use ``g``), on
        degenerate 1-wide axes, or on the skip-zero path (an all-zero
        female ztype row is injected whenever ``z >= 2``) would flip
        cells that square dense tables never visit.
        """
        rust = _load_rust_kernel()
        if rust is None:
            from natal.frontend.data._engine import _rust_offspring_kernel

            assert _rust_offspring_kernel(
                np.zeros((2, 1, 1)), np.zeros((1, 1, 1))
            ) is None
            return

        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

        rng = np.random.default_rng(20260906)
        for name, z, g, sparsity in self._GRID:
            meiosis = rng.random((2, z, g))
            meiosis[rng.random((2, z, g)) < sparsity] = 0.0
            if z >= 2:
                meiosis[0, 1, :] = 0.0  # all-zero female ztype row
            fusion = (rng.random((g, g, z)) < 0.55).astype(np.float64)

            rust_out = np.asarray(
                rust(np.ascontiguousarray(meiosis), np.ascontiguousarray(fusion))
            ).reshape(z, z, z)
            py_out = compute_offspring_probability_tensor(
                meiosis_f=meiosis[0],
                meiosis_m=meiosis[1],
                haplo_to_genotype_map=fusion,
                n_ztypes=z,
                n_gtypes=g,
            )
            assert rust_out.shape == (z, z, z), name
            assert rust_out.dtype == np.float64, name
            assert rust_out.tobytes() == np.ascontiguousarray(py_out).tobytes(), name

        # All-zero tables: exact zeros everywhere, both kernels.
        zero_out = np.asarray(
            rust(np.zeros((2, 3, 4)), np.zeros((4, 4, 3)))
        ).reshape(3, 3, 3)
        assert zero_out.tobytes() == np.zeros((3, 3, 3)).tobytes()

    def test_order_probe_discriminates_and_rust_uses_reference_order(
        self,
    ) -> None:
        """Rust accumulates in the reference (hf, hm) order — proven, not
        assumed.

        Attack: two kernels could agree on random dense tables by luck
        while summing in different orders.  Tables built from
        0.1/0.2/0.3-family values make summation order visible in the
        low bits; this test first *proves the probe has power* (a
        reversed (hf, hm) accumulation differs somewhere) and only then
        asserts the Rust bytes equal the forward-order Python bytes.
        """
        rust = _load_rust_kernel()
        if rust is None:
            pytest.skip("rust extension not built")

        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

        vals = np.array(
            [0.1, 0.2, 0.3, 0.7, 0.9, 1.0 / 3.0, 0.123456789, 0.987654321]
        )
        z, g = 3, 4
        rng = np.random.default_rng(424242)
        meiosis = vals[rng.integers(0, len(vals), size=(2, z, g))]
        fusion = vals[rng.integers(0, len(vals), size=(g, g, z))]

        forward = compute_offspring_probability_tensor(
            meiosis_f=meiosis[0],
            meiosis_m=meiosis[1],
            haplo_to_genotype_map=fusion,
            n_ztypes=z,
            n_gtypes=g,
        )
        # Reversed (hf, hm) accumulation — a legal but *different* order.
        flat = np.ascontiguousarray(fusion).reshape(g * g, z)
        reversed_out = np.zeros((z, z, z))
        for gf in range(z):
            for gm in range(z):
                for go in range(z):
                    s = 0.0
                    for hf in range(g - 1, -1, -1):
                        mf = meiosis[0, gf, hf]
                        if mf == 0.0:
                            continue
                        for hm in range(g - 1, -1, -1):
                            mm = meiosis[1, gm, hm]
                            if mm == 0.0:
                                continue
                            s += mf * mm * flat[hf * g + hm, go]
                    reversed_out[gf, gm, go] = s

        n_diff = int(
            np.count_nonzero(
                forward.view(np.uint64) != reversed_out.view(np.uint64)
            )
        )
        # Probe power: if this fails the probe went vacuous — redesign it,
        # do not delete it.
        assert n_diff >= 1

        rust_out = np.asarray(
            rust(np.ascontiguousarray(meiosis), np.ascontiguousarray(fusion))
        ).reshape(z, z, z)
        assert rust_out.tobytes() == np.ascontiguousarray(forward).tobytes()

    def test_release_build_performs_unfused_multiply_add(self) -> None:
        """The release build rounds every multiply and add separately.

        Attack: an optimizer that contracts ``s += mf * mm * fusion``
        into a fused multiply-add (one rounding instead of two) would
        shift results by 1 ulp on tables where the two spellings differ.
        The constants below were found by search so that the separate
        and fused spellings differ; the kernel must land on the
        separate value.  ``float(Fraction ... )`` reproduces the exact
        single-rounding fma semantics portably (no ``math.fma`` — the
        project supports Python 3.9).
        """
        rust = _load_rust_kernel()
        if rust is None:
            pytest.skip("rust extension not built")

        a, b, c, d, e, f = (
            0.09266663117715868,
            0.009808112625737497,
            187.1507120918915,
            0.2709929808816529,
            420.6642290190864,
            5.228309038042963,
        )
        # (2, 1, 2) meiosis / (2, 2, 1) fusion: the cell accumulates
        # exactly two nonzero terms, (hf=0,hm=0) -> a*b*c and
        # (hf=1,hm=1) -> d*e*f.
        meiosis = np.array([[[a, d]], [[b, e]]])
        fusion = np.array([[[c], [0.0]], [[0.0], [f]]])

        kernel_value = float(np.asarray(rust(meiosis, fusion))[0])
        separate = a * b * c + d * e * f  # Python: no contraction
        # Contraction of the LAST multiply-add only: the earlier products
        # stay rounded, the final p2*f + s collapses to one rounding.
        fused = float(Fraction(d * e) * Fraction(f) + Fraction(a * b * c))

        assert separate != fused  # the constants discriminate (1 ulp)
        assert kernel_value.hex() == separate.hex()
        assert kernel_value.hex() != fused.hex()

    def test_nan_inf_propagation_is_bit_identical(self) -> None:
        """NaN and inf inputs propagate to identical bit patterns.

        Attack: a kernel that reordered terms (e.g. an FMA or a different
        skip rule for non-finite values) produces different NaN cells —
        ``0 * inf = nan`` and ``finite + nan = nan`` are order-sensitive
        poison.  ``tobytes`` compares NaN payloads; ``array_equal``
        cannot.
        """
        rust = _load_rust_kernel()
        if rust is None:
            pytest.skip("rust extension not built")

        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

        rng = np.random.default_rng(8080)
        meiosis = rng.random((2, 3, 3))
        # Exact zero placed so it pairs with the inf fusion entry below:
        # with the skip-zero rule the 0 * inf = nan poison is never
        # computed and the cell stays inf; a kernel that dropped the
        # skip materializes NaN instead (order-sensitive).
        meiosis[0, 0, 1] = 0.0
        meiosis[0, 1, 2] = np.nan
        fusion = rng.random((3, 3, 3))
        fusion[1, 1, 0] = np.inf

        rust_out = np.asarray(
            rust(np.ascontiguousarray(meiosis), np.ascontiguousarray(fusion))
        ).reshape(3, 3, 3)
        py_out = compute_offspring_probability_tensor(
            meiosis_f=meiosis[0],
            meiosis_m=meiosis[1],
            haplo_to_genotype_map=fusion,
            n_ztypes=3,
            n_gtypes=3,
        )
        # Probe power: the poison really reaches the outputs — the NaN
        # row poisons every (gf=1) cell, the inf fusion entry surfaces
        # in the (gf=2, go=0) cells, and the zeroed female entry at
        # (gf=0, hf=1) keeps the whole (gf=0) row finite (a kernel
        # without the skip turns that row into NaN via 0 * inf).
        assert np.isnan(py_out[1, :, :]).all()
        assert np.isinf(py_out[2, :, 0]).all()
        assert np.isfinite(py_out[0, :, :]).all()
        assert rust_out.tobytes() == np.ascontiguousarray(py_out).tobytes()

    def test_forced_import_failure_routes_to_python_and_stays_bit_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Blocking the extension forces the fallback — same bytes out.

        Attack: the dispatch could take the Python branch but diverge
        (different dtype coercion, einsum-style reorder, cached stale
        tensor).  A ``None`` entry in ``sys.modules`` makes the
        function-level ``from natal._engine_rs import ...`` raise
        ImportError, which is exactly the branch the wrapper must
        survive.  After the block lifts, the wrapper must retry the
        import (no negative caching of the failed lookup).
        """
        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )
        from natal.frontend.data._engine import (
            _rust_offspring_kernel,
            recompute_offspring_tensor,
        )

        rng = np.random.default_rng(31337)
        meiosis = rng.random((2, 5, 4))
        meiosis[meiosis < 0.25] = 0.0
        fusion = (rng.random((4, 4, 5)) < 0.5).astype(np.float64)

        extension_result = recompute_offspring_tensor(meiosis, fusion)

        monkeypatch.setitem(sys.modules, "natal._engine_rs", None)
        assert _rust_offspring_kernel(meiosis, fusion) is None
        fallback_result = recompute_offspring_tensor(meiosis, fusion)

        # Byte identity across the two dispatch branches.
        assert fallback_result.tobytes() == extension_result.tobytes()
        # And the fallback really spells the reference kernel.
        direct = compute_offspring_probability_tensor(
            meiosis_f=meiosis[0],
            meiosis_m=meiosis[1],
            haplo_to_genotype_map=fusion,
            n_ztypes=5,
            n_gtypes=4,
        )
        assert fallback_result.tobytes() == np.ascontiguousarray(direct).tobytes()

        # Lift the block: the next call must find the extension again.
        monkeypatch.undo()
        if _load_rust_kernel() is not None:
            assert (
                recompute_offspring_tensor(meiosis, fusion).tobytes()
                == extension_result.tobytes()
            )

    def test_shape_violations_raise_pyvalueerror_from_rust(self) -> None:
        """Invalid shapes/dtype raise the documented errors at the boundary.

        Attack: a kernel that indexes out of bounds instead of validating
        would segfault or read garbage rather than raise; the messages
        pin *which* validation fired so a swapped check cannot pass.
        """
        rust = _load_rust_kernel()
        if rust is None:
            pytest.skip("rust extension not built")

        good_fusion = np.zeros((2, 2, 2))
        with pytest.raises(ValueError, match="leading sex axis of 2"):
            rust(np.ones((3, 2, 2)), good_fusion)
        with pytest.raises(ValueError, match="leading sex axis of 2"):
            rust(np.ones((1, 2, 2)), good_fusion)
        with pytest.raises(
            ValueError, match=re.escape("fusion shape must be (4, 4, 3)")
        ):
            rust(np.ones((2, 3, 4)), np.zeros((4, 4, 4)))
        with pytest.raises(ValueError, match="must be C-contiguous"):
            rust(
                np.asfortranarray(np.ones((2, 3, 4))), np.zeros((4, 4, 3))
            )
        with pytest.raises(TypeError):
            rust(np.ones((2, 3, 4), dtype=np.float32), np.zeros((4, 4, 3)))

    def test_wrapper_rejects_bad_fusion_and_leaves_inputs_untouched(self) -> None:
        """The wrapper surfaces the ValueError and mutates nothing.

        Attack: an error path that had already scribbled on the caller's
        tables before raising would corrupt live config state; the
        wrapper is documented pure, so inputs must survive the raise
        bit-identically on *both* dispatch branches.
        """
        from natal.frontend.data._engine import recompute_offspring_tensor

        meiosis = np.ones((2, 3, 4))
        fusion = np.zeros((4, 4, 4))
        m_before, f_before = meiosis.copy(), fusion.copy()
        with pytest.raises(ValueError):
            recompute_offspring_tensor(meiosis, fusion)
        np.testing.assert_array_equal(meiosis, m_before)
        np.testing.assert_array_equal(fusion, f_before)

    def test_kernel_result_freshly_owned_each_call(self) -> None:
        """Each kernel call returns fresh storage; inputs stay read-only.

        Attack: a wrapper that cached the flat result, or a Rust side
        that reused a buffer, would hand two callers the same memory —
        poisoning one result would poison the other, and mutating the
        output could scribble on the input tables.
        """
        rust = _load_rust_kernel()
        if rust is None:
            pytest.skip("rust extension not built")

        rng = np.random.default_rng(5150)
        meiosis = np.ascontiguousarray(rng.random((2, 4, 4)))
        fusion = np.ascontiguousarray(rng.random((4, 4, 4)))
        m_before, f_before = meiosis.copy(), fusion.copy()

        r1 = np.asarray(rust(meiosis, fusion))
        expected = r1.copy()
        r1.fill(-1.0)  # poison the caller's view of the first result
        r2 = np.asarray(rust(meiosis, fusion))

        assert r1 is not r2
        assert r2.tobytes() == expected.tobytes()  # not cached, not aliased
        assert not np.shares_memory(r2, meiosis)
        assert not np.shares_memory(r2, fusion)
        np.testing.assert_array_equal(meiosis, m_before)
        np.testing.assert_array_equal(fusion, f_before)

    def test_wrapper_matches_rust_kernel_and_python_reference(self) -> None:
        """Wrapper, raw kernel, and reference kernel agree byte-for-byte.

        Attack: the wrapper reshapes with ``meiosis.shape[1]`` while the
        kernel derives ``z`` internally — a mismatch (or an
        F-order/dtype coercion bug in the wrapper's ``ascontiguousarray``
        funnel) would show as a reshape into the wrong cube or shifted
        values.  Comparing against the *independent* Python spelling
        also catches a wrapper that merely round-trips its own output.
        """
        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )
        from natal.frontend.data._engine import recompute_offspring_tensor

        rng = np.random.default_rng(99)
        meiosis = rng.random((2, 4, 3))
        fusion = (rng.random((3, 3, 4)) < 0.5).astype(np.float64)

        wrapped = recompute_offspring_tensor(meiosis, fusion)
        assert wrapped.shape == (4, 4, 4)

        py = compute_offspring_probability_tensor(
            meiosis_f=meiosis[0],
            meiosis_m=meiosis[1],
            haplo_to_genotype_map=fusion,
            n_ztypes=4,
            n_gtypes=3,
        )
        assert wrapped.tobytes() == np.ascontiguousarray(py).tobytes()

        # F-order inputs are value-preserved through the funnel.
        wrapped_f = recompute_offspring_tensor(
            np.asfortranarray(meiosis), np.asfortranarray(fusion)
        )
        assert wrapped_f.tobytes() == wrapped.tobytes()

        rust = _load_rust_kernel()
        if rust is not None:
            direct = np.asarray(
                rust(np.ascontiguousarray(meiosis), np.ascontiguousarray(fusion))
            ).reshape(4, 4, 4)
            assert wrapped.tobytes() == direct.tobytes()


class TestEquilibriumKernelParity:
    """The Rust equilibrium kernel matches the Python reference bitwise."""

    def _python_reference(self, **kw: object) -> tuple[float, float]:  # object: probe mirrors the production kwargs dict (heterogeneous value types)
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )

        return compute_equilibrium_metrics(**kw)  # type: ignore[arg-type]  # probe mirrors the production kwargs

    def test_equilibrium_rust_kernel_bitwise_matches_python(self) -> None:
        """Random demographic inputs agree bitwise across both kernels."""
        try:
            from natal._engine_rs import equilibrium_metrics_flat as rust_metrics
        except ImportError:
            pytest.skip("rust extension not built")

        rng = np.random.default_rng(4242)
        for trial in range(8):
            n_ages = int(rng.integers(2, 9))
            new_adult = int(rng.integers(1, n_ages))
            survival = rng.random((2, n_ages))
            mating = rng.random((2, n_ages))
            reproduction = rng.random(n_ages) if trial % 2 else None
            fertility = rng.random(n_ages)
            competition = rng.random(n_ages)
            sex_ratio = float(rng.random())
            k = float(rng.random() * 5000)
            eggs = float(rng.random() * 40)
            declared = rng.random((2, n_ages)) * 100 if trial % 3 == 0 else None
            external = float(rng.random() * 3000) if trial % 4 == 0 else None

            py = self._python_reference(
                carrying_capacity=k,
                eggs_per_female=eggs,
                age_based_survival_rates=survival,
                age_based_mating_rates=mating,
                age_based_reproduction_rates=reproduction,
                female_age_based_fertility=fertility,
                relative_competition_strength=competition,
                sex_ratio=sex_ratio,
                new_adult_age=new_adult,
                n_ages=n_ages,
                equilibrium_individual_count=declared,
                external_expected_eggs=external,
            )
            resolved = reproduction if reproduction is not None else mating[0]
            rust = rust_metrics(
                k, eggs, sex_ratio,
                np.ascontiguousarray(survival),
                np.ascontiguousarray(resolved),
                np.ascontiguousarray(fertility),
                np.ascontiguousarray(competition),
                new_adult, n_ages,
                (
                    np.ascontiguousarray(declared)
                    if declared is not None
                    else None
                ),
                external,
            )
            assert rust == py, f"equilibrium parity broke at trial {trial}"

    def test_sync_path_uses_rust_and_matches_fallback_bitwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The production sync entry returns bitwise-identical metrics
        under the Rust dispatch and the forced Python fallback.

        Attack: the two dispatch branches could drift (dtype coercion,
        sentinel normalization, a stale cached import).  A ``None`` entry
        in ``sys.modules`` makes the function-level ``from
        natal._engine_rs import ...`` raise ImportError — exactly the
        branch the sync must survive — without replacing the
        process-global ``builtins.__import__`` (the same blocking method
        the batch-10 kernel tests use).  Both results are also anchored
        against an independent Python computation on the draft's own
        fields, so a shared bug in both branches cannot pass.
        """
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )
        from natal.frontend.configurator._routes import sync_equilibrium_for_draft

        sp = nt.Species.from_dict(
            name="__eq_parity_species__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        draft = (
            nt.AgeStructuredPopulation.setup(sp, stochastic=False)
            .age_structure(n_ages=5, new_adult_age=2)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 0, 40, 0, 0]},
                    "male": {"A|A": [0, 0, 40, 0, 0]},
                }
            )
            .competition(carrying_capacity=1234.0, juvenile_growth_mode=3)
            .reproduction(eggs_per_female=17.0, sex_ratio=0.5)
            .survival(female_age_based_survival=0.85, male_age_based_survival=0.8)
            .build()
        ).config
        # Distinct fertility/competition vectors: the builder defaults
        # are all-ones, where a kernel wiring swap (fertility read as
        # competition) is numerically invisible.  Nonzero survival keeps
        # the derive branch's produced_age_0 nonzero and value-sensitive.
        rng = np.random.default_rng(424242)
        draft = draft._replace(
            female_age_based_fertility=rng.random(5),
            age_based_relative_competition_strength=rng.random(5),
        )
        survival_before = draft.age_based_survival_rates.copy()
        mating_before = draft.age_based_mating_rates.copy()
        fertility_before = draft.female_age_based_fertility.copy()
        competition_before = (
            draft.age_based_relative_competition_strength.copy()
        )

        rust_synced = sync_equilibrium_for_draft(draft)

        monkeypatch.setitem(sys.modules, "natal._engine_rs", None)
        py_synced = sync_equilibrium_for_draft(draft)

        assert py_synced.expected_competition_strength == (
            rust_synced.expected_competition_strength
        )
        assert py_synced.expected_survival_rate == (
            rust_synced.expected_survival_rate
        )
        # Anchor both branches against the reference kernel on the
        # draft's own demographic fields (a wrong-but-agreed pair fails).
        expected = compute_equilibrium_metrics(
            carrying_capacity=float(draft.carrying_capacity),
            eggs_per_female=float(draft.eggs_per_female),
            sex_ratio=float(draft.sex_ratio),
            age_based_survival_rates=draft.age_based_survival_rates,
            age_based_mating_rates=draft.age_based_mating_rates,
            age_based_reproduction_rates=draft.age_based_reproduction_rates,
            female_age_based_fertility=draft.female_age_based_fertility,
            relative_competition_strength=(
                draft.age_based_relative_competition_strength
            ),
            new_adult_age=int(draft.new_adult_age),
            n_ages=int(draft.n_ages),
            equilibrium_individual_count=None,
            external_expected_eggs=None,
        )
        assert rust_synced.expected_competition_strength == expected[0]
        assert rust_synced.expected_survival_rate == expected[1]
        # Ownership: sync must read the draft, never scribble on it —
        # every demographic array survives both dispatch branches
        # bit-identically, and the caches keep their build-time values.
        draft_cache = (
            draft.expected_competition_strength,
            draft.expected_survival_rate,
        )
        np.testing.assert_array_equal(
            draft.age_based_survival_rates, survival_before
        )
        np.testing.assert_array_equal(
            draft.age_based_mating_rates, mating_before
        )
        np.testing.assert_array_equal(
            draft.female_age_based_fertility, fertility_before
        )
        np.testing.assert_array_equal(
            draft.age_based_relative_competition_strength, competition_before
        )
        assert (
            draft.expected_competition_strength,
            draft.expected_survival_rate,
        ) == draft_cache

        # Lift the block: the next call must find the extension again
        # (no negative caching of the failed import).
        monkeypatch.undo()
        again = sync_equilibrium_for_draft(draft)
        assert again.expected_competition_strength == (
            rust_synced.expected_competition_strength
        )
        assert again.expected_survival_rate == rust_synced.expected_survival_rate

    def test_sync_resolves_none_reproduction_to_mating_row(self) -> None:
        """A draft carrying ``age_based_reproduction_rates=None`` syncs to
        the mating-row fallback on both dispatch branches.

        Attack: the Optional annotation (batch 11) exists because the
        runtime intermediate state can hold None — the 1102 incident.
        The Rust pre-resolution could pick the male row, clamp
        differently, or crash on the None; the Python fallback must keep
        its own ``mating[0]`` semantics.  Bitwise equality of the two
        branches plus the None-vs-resolved draft comparison pins the row
        choice.
        """
        from natal.frontend.configurator._routes import sync_equilibrium_for_draft

        sp = nt.Species.from_dict(
            name="__eq_none_repro_species__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        draft = (
            nt.AgeStructuredPopulation.setup(sp, stochastic=False)
            .age_structure(n_ages=4, new_adult_age=1)
            .initial_state(
                individual_count={
                    "female": {"A|A": [0, 40, 0, 0]},
                    "male": {"A|A": [0, 40, 0, 0]},
                }
            )
            .competition(carrying_capacity=977.0, juvenile_growth_mode=3)
            .reproduction(eggs_per_female=23.0, sex_ratio=0.5)
            .survival(female_age_based_survival=0.9, male_age_based_survival=0.7)
            .build()
        ).config
        # Overwrite the two mating rows with clearly different adult
        # values so a wrong row choice changes the output.
        draft.age_based_mating_rates[0][1:] = 0.9
        draft.age_based_mating_rates[1][1:] = 0.4
        female_row = draft.age_based_mating_rates[0].copy()
        male_row = draft.age_based_mating_rates[1].copy()
        assert not np.array_equal(female_row, male_row)

        none_draft = draft._replace(age_based_reproduction_rates=None)
        synced_none = sync_equilibrium_for_draft(none_draft)
        synced_resolved = sync_equilibrium_for_draft(draft._replace(
            age_based_reproduction_rates=female_row
        ))
        # None must behave exactly like the female mating row.
        assert synced_none.expected_competition_strength == (
            synced_resolved.expected_competition_strength
        )
        assert synced_none.expected_survival_rate == (
            synced_resolved.expected_survival_rate
        )
        # And unlike the male row (the wrong-row detector).
        synced_male = sync_equilibrium_for_draft(draft._replace(
            age_based_reproduction_rates=male_row
        ))
        assert synced_male.expected_competition_strength != (
            synced_none.expected_competition_strength
        )

    def test_equilibrium_parity_edge_axis_matrix(self) -> None:
        """Bit-pattern parity across the edge/axis product of inputs.

        Attack: the random trial in the sibling test draws every value
        from ``[0, 1)`` with positive K and eggs — the degenerate
        branches (sex_ratio 0/1, all-zero survival/fertility, K=0,
        eggs=0, external=0.0) are never entered, so a kernel that
        mishandles only a degenerate guard would pass it.  This test
        enumerates the sentinel corners and compares full 64-bit
        patterns (``struct.pack``), so even a ``-0.0``/``0.0`` or
        last-ulp difference fails.
        """
        rust = _load_equilibrium_kernel()
        if rust is None:
            pytest.skip("rust extension not built")
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )

        rng = np.random.default_rng(20260906)
        base_vectors = {
            n: (
                rng.random((2, n)),
                rng.random((2, n)),
                rng.random(n),
                rng.random(n),
            )
            for n in (2, 3, 5, 8)
        }

        def both_ways(k, eggs, sr, surv, mat, repro, fert, comp, na, n,
                      declared, external):
            py = compute_equilibrium_metrics(
                carrying_capacity=k,
                eggs_per_female=eggs,
                age_based_survival_rates=surv,
                age_based_mating_rates=mat,
                age_based_reproduction_rates=repro,
                female_age_based_fertility=fert,
                relative_competition_strength=comp,
                sex_ratio=sr,
                new_adult_age=na,
                n_ages=n,
                equilibrium_individual_count=declared,
                external_expected_eggs=external,
            )
            resolved = repro if repro is not None else mat[0]
            ru = rust(
                float(k), float(eggs), float(sr),
                np.ascontiguousarray(surv, dtype=np.float64),
                np.ascontiguousarray(resolved, dtype=np.float64),
                np.ascontiguousarray(fert, dtype=np.float64),
                np.ascontiguousarray(comp, dtype=np.float64),
                int(na), int(n),
                (np.ascontiguousarray(declared, dtype=np.float64)
                 if declared is not None else None),
                external,
            )
            return (
                struct.pack(">d", float(py[0])) == struct.pack(">d", ru[0])
                and struct.pack(">d", float(py[1])) == struct.pack(">d", ru[1])
            )

        for n_ages, (survival, mating, repro, comp) in base_vectors.items():
            fert = rng.random(n_ages)
            na = max(1, n_ages // 2)
            assert both_ways(
                1234.0, 17.0, 0.5, survival, mating, repro, fert, comp,
                na, n_ages, None, None,
            ), f"derive parity broke at n_ages={n_ages}"
            # Python's own fallback branch: reproduction=None -> mating[0].
            assert both_ways(
                1234.0, 17.0, 0.5, survival, mating, None, fert, comp,
                na, n_ages, None, None,
            ), f"mating-fallback parity broke at n_ages={n_ages}"
            declared = rng.random((2, n_ages)) * 400.0
            assert both_ways(
                1234.0, 17.0, 0.5, survival, mating, repro, fert, comp,
                na, n_ages, declared, None,
            ), f"declared parity broke at n_ages={n_ages}"
            assert both_ways(
                1234.0, 17.0, 0.5, survival, mating, repro, fert, comp,
                na, n_ages, declared, 3131.0,
            ), f"external parity broke at n_ages={n_ages}"

        # Degenerate corners (each targets one guard in the reference).
        surv, mat, repro, comp = base_vectors[3]
        fert = np.array([0.0, 1.0, 0.9])
        corners = [
            ("sex_ratio=0", 0.0, 800.0, 25.0, surv, repro, fert),
            ("sex_ratio=1", 1.0, 800.0, 25.0, surv, repro, fert),
            ("survival all zero", 0.5, 800.0, 25.0, np.zeros((2, 3)), repro, fert),
            ("fertility all zero", 0.5, 800.0, 25.0, surv, repro, np.zeros(3)),
            ("K=0", 0.5, 0.0, 25.0, surv, repro, fert),
            ("eggs=0", 0.5, 800.0, 0.0, surv, repro, fert),
        ]
        for label, sr, k, eggs, s, r, f in corners:
            assert both_ways(
                k, eggs, sr, s, mat, r, f, comp, 1, 3, None, None
            ), f"{label} parity broke"
        # external=0.0 (not None) must keep the "rate degenerates to 1.0"
        # branch identical on both sides.
        assert both_ways(
            800.0, 25.0, 0.5, surv, mat, repro, fert, comp, 1, 3, None, 0.0
        ), "external=0.0 parity broke"

    def test_equilibrium_clamp_fires_identically_on_both_kernels(self) -> None:
        """clamp01 must actually run — and identically — on both kernels.

        Attack: the sibling random trial only feeds ``rng.random()``
        values in [0, 1), where clamp01 is the identity, so deleting the
        clamp entirely would still pass it.  The invariant here is
        mathematical: clamp01(2.0) == clamp01(1.0) and clamp01(-3.0) ==
        clamp01(0.0), so a kernel with a working clamp returns *bitwise
        equal* results for those input pairs, while an identity "clamp"
        (or a wrong-direction one) does not.  Parity with the Python
        reference is asserted on the same out-of-range inputs.
        """
        rust = _load_equilibrium_kernel()
        if rust is None:
            pytest.skip("rust extension not built")
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )

        rng = np.random.default_rng(616)
        n = 4
        survival = rng.random((2, n))
        mating = rng.random((2, n))
        fert = rng.random(n) + 0.5
        comp = rng.random(n) + 0.5
        k, eggs = 1500.0, 12.0

        def call_rust(repro):
            return rust(
                k, eggs, 0.5,
                np.ascontiguousarray(survival), np.ascontiguousarray(repro),
                np.ascontiguousarray(fert), np.ascontiguousarray(comp),
                1, n, None, None,
            )

        high = np.array([0.0, 2.0, 1.7, 50.0])
        unit = np.array([0.0, 1.0, 1.0, 1.0])
        low = np.array([0.0, -3.0, -0.1, -100.0])
        zero = np.zeros(n)

        # Sensitivity floor: the reproduction vector must move the output
        # (otherwise the equalities below are vacuous).
        mid = np.array([0.0, 1.0, 0.5, 1.0])
        assert call_rust(high) != call_rust(mid)
        assert call_rust(high) == call_rust(unit), "clamp(>1) must equal clamp(=1)"
        assert call_rust(low) == call_rust(zero), "clamp(<0) must equal clamp(=0)"

        # Out-of-range values through the Python fallback row too.
        mating_out = mating.copy()
        mating_out[0] = np.array([-2.0, 0.4, 1.9, 3.0])
        py = compute_equilibrium_metrics(
            carrying_capacity=k, eggs_per_female=eggs,
            age_based_survival_rates=survival,
            age_based_mating_rates=mating_out,
            age_based_reproduction_rates=None,
            female_age_based_fertility=fert,
            relative_competition_strength=comp,
            sex_ratio=0.5, new_adult_age=1, n_ages=n,
            equilibrium_individual_count=None, external_expected_eggs=None,
        )
        ru = rust(
            k, eggs, 0.5,
            np.ascontiguousarray(survival),
            np.ascontiguousarray(mating_out[0]),
            np.ascontiguousarray(fert), np.ascontiguousarray(comp),
            1, n, None, None,
        )
        assert ru == py, "out-of-range fallback-row parity broke"

    def test_equilibrium_sentinel_and_shape_contracts(self) -> None:
        """None/empty declared derive; every wrong shape names its field.

        Attack: a kernel that treated the empty derive-mode sentinel as
        a declared zero distribution (or validated shapes in the wrong
        order, or validated none at all) would silently mis-derive or
        read out of bounds.  Error paths must also leave the caller's
        arrays bit-identical (a validation that scribbles before raising
        corrupts live config).
        """
        rust = _load_equilibrium_kernel()
        if rust is None:
            pytest.skip("rust extension not built")

        n = 4
        survival = np.array([[0.8, 0.9, 0.7, 0.6], [0.85, 0.75, 0.65, 0.55]])
        repro = np.array([0.0, 0.8, 0.7, 0.6])
        fert = np.array([0.0, 1.0, 0.9, 0.8])
        comp = np.array([1.0, 0.8, 0.7, 0.6])

        def call(declared, external=None, surv=survival, r=repro, f=fert,
                 c=comp, na=1, ages=n):
            return rust(
                400.0, 30.0, 0.5,
                np.ascontiguousarray(surv), np.ascontiguousarray(r),
                np.ascontiguousarray(f), np.ascontiguousarray(c),
                na, ages, declared, external,
            )

        from_none = call(None)
        # All three empty spellings derive, bit-identically to None.
        assert call(np.zeros((0, 0))) == from_none
        assert call(np.zeros((2, 0))) == from_none

        # Wrong declared shapes: ValueError naming the field and shape.
        for bad in (np.zeros((2, n + 1)), np.zeros((2, n - 1)), np.zeros((3, n))):
            with pytest.raises(ValueError, match="declared_distribution shape"):
                call(bad)
        # Wrong survival shapes and vector lengths: each names its field.
        with pytest.raises(ValueError, match="survival_rates shape"):
            call(None, surv=np.zeros((2, n + 1)))
        with pytest.raises(ValueError, match="survival_rates shape"):
            call(None, surv=np.zeros((1, n)))
        with pytest.raises(ValueError, match="reproduction_rates must have length"):
            call(None, r=np.zeros(n - 1))
        with pytest.raises(ValueError, match="fertility must have length"):
            call(None, f=np.zeros(n + 1))
        with pytest.raises(ValueError, match="competition_weights must have length"):
            call(None, c=np.zeros(n + 2))
        # Non-contiguous inputs are rejected, not silently copied
        # (direct kernel calls — the sync path always pre-normalizes).
        with pytest.raises(ValueError, match="must be C-contiguous"):
            rust(
                400.0, 30.0, 0.5,
                np.asfortranarray(survival), np.ascontiguousarray(repro),
                np.ascontiguousarray(fert), np.ascontiguousarray(comp),
                1, n, None, None,
            )
        wide = np.zeros((2, 2 * n))
        with pytest.raises(
            ValueError, match="declared_distribution must be C-contiguous"
        ):
            rust(
                400.0, 30.0, 0.5,
                np.ascontiguousarray(survival), np.ascontiguousarray(repro),
                np.ascontiguousarray(fert), np.ascontiguousarray(comp),
                1, n, np.asfortranarray(wide[:, :n]), None,
            )

        # Ownership/error-path: the caller's arrays survive every raise
        # above bit-identically (no partial writes before the error).
        np.testing.assert_array_equal(
            survival, [[0.8, 0.9, 0.7, 0.6], [0.85, 0.75, 0.65, 0.55]]
        )
        np.testing.assert_array_equal(repro, [0.0, 0.8, 0.7, 0.6])
        np.testing.assert_array_equal(fert, [0.0, 1.0, 0.9, 0.8])
        np.testing.assert_array_equal(comp, [1.0, 0.8, 0.7, 0.6])

        # A declared input mutated after the call must not leak into any
        # retained kernel state (the kernel copies into owned storage).
        declared = np.array([[0.0, 200.0, 150.0, 100.0],
                             [0.0, 200.0, 150.0, 100.0]])
        r1 = call(declared)
        declared.fill(0.0)
        r2 = call(declared)
        assert r1 != r2, "kernel retained the first call's declared storage"

    def test_contract_wrapper_and_flat_entry_agree_bitwise(self) -> None:
        """The core extraction refactor changed no contract-path bits.

        Attack: ``equilibrium_metrics`` (bp/params, consumed by SimConfig
        assembly) now funnels through ``equilibrium_metrics_core``; a
        slice off-by-one, a swapped field, or a re-ordered argument in
        the wrapper would change the contract results while the flat
        entry (independently wired) stayed right.  Three-way bit
        equality — contract vs flat vs Python reference — across
        derive/declared/external modes pins the wrapper.
        """
        from natal import _engine_rs
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )
        from natal.contracts.materialize import materialize

        sp = nt.Species.from_dict(
            name="__eq_refactor_zero_change__",
            structure={"chr1": {"loc": ["A", "B"]}},
            gamete_labels=["default"],
        )
        rng = np.random.default_rng(70707)
        for trial in range(3):
            n = int(rng.integers(2, 9))
            na = int(rng.integers(1, n))
            draft = (
                nt.AgeStructuredPopulation.setup(sp, stochastic=False)
                .age_structure(n_ages=n, new_adult_age=na)
                .initial_state(
                    individual_count={
                        "female": {"A|A": [40, 0, 0, 0, 0, 0, 0, 0][:n]},
                        "male": {"A|A": [40, 0, 0, 0, 0, 0, 0, 0][:n]},
                    }
                )
                .competition(
                    carrying_capacity=float(rng.uniform(100, 4000)),
                    juvenile_growth_mode=3,
                )
                .reproduction(
                    eggs_per_female=float(rng.uniform(5, 60)),
                    sex_ratio=float(rng.uniform(0.2, 0.8)),
                )
                .survival(
                    female_age_based_survival=float(rng.uniform(0.4, 0.95)),
                    male_age_based_survival=float(rng.uniform(0.4, 0.95)),
                )
                .build()
            ).config
            # Distinct fertility/competition vectors (builder defaults are
            # all-ones, hiding a wiring swap in the core extraction).
            draft = draft._replace(
                female_age_based_fertility=rng.random(n) + 0.25,
                age_based_relative_competition_strength=rng.random(n) + 0.25,
            )
            contracts = materialize(draft)
            bp, params = contracts.blueprint, contracts.params

            def py_call(d, declared, external):
                return compute_equilibrium_metrics(
                    carrying_capacity=float(d.carrying_capacity),
                    eggs_per_female=float(d.eggs_per_female),
                    sex_ratio=float(d.sex_ratio),
                    age_based_survival_rates=d.age_based_survival_rates,
                    age_based_mating_rates=d.age_based_mating_rates,
                    age_based_reproduction_rates=(
                        d.age_based_reproduction_rates
                    ),
                    female_age_based_fertility=d.female_age_based_fertility,
                    relative_competition_strength=(
                        d.age_based_relative_competition_strength
                    ),
                    new_adult_age=int(d.new_adult_age),
                    n_ages=int(d.n_ages),
                    equilibrium_individual_count=declared,
                    external_expected_eggs=external,
                )

            def flat_call(d, declared, external):
                return _engine_rs.equilibrium_metrics_flat(
                    float(d.carrying_capacity),
                    float(d.eggs_per_female),
                    float(d.sex_ratio),
                    np.ascontiguousarray(
                        d.age_based_survival_rates, dtype=np.float64
                    ),
                    np.ascontiguousarray(
                        d.age_based_reproduction_rates, dtype=np.float64
                    ),
                    np.ascontiguousarray(
                        d.female_age_based_fertility, dtype=np.float64
                    ),
                    np.ascontiguousarray(
                        d.age_based_relative_competition_strength,
                        dtype=np.float64,
                    ),
                    int(d.new_adult_age),
                    int(d.n_ages),
                    (np.ascontiguousarray(declared, dtype=np.float64)
                     if declared is not None else None),
                    external,
                )

            # Derive mode: contract carries the empty sentinel column.
            c, s = _engine_rs.equilibrium_metrics(bp, params)
            fc, fs = flat_call(draft, None, None)
            pc, ps = py_call(draft, None, None)
            assert struct.pack(">d", c) == struct.pack(">d", fc) == (
                struct.pack(">d", float(pc))
            ), f"derive C* three-way mismatch at trial {trial}"
            assert struct.pack(">d", s) == struct.pack(">d", fs) == (
                struct.pack(">d", float(ps))
            ), f"derive s* three-way mismatch at trial {trial}"

            # Declared mode: widen the contract column and mirror it in
            # the flat/python calls.
            declared = rng.random((2, n)) * 300.0
            params.equilibrium_distribution = declared.ravel().copy()
            c, s = _engine_rs.equilibrium_metrics(bp, params)
            fc, fs = flat_call(draft, declared, None)
            pc, ps = py_call(draft, declared, None)
            assert struct.pack(">d", c) == struct.pack(">d", fc) == (
                struct.pack(">d", float(pc))
            ), f"declared C* three-way mismatch at trial {trial}"
            assert struct.pack(">d", s) == struct.pack(">d", fs) == (
                struct.pack(">d", float(ps))
            ), f"declared s* three-way mismatch at trial {trial}"

            # External override: only the survival-rate path may move.
            external = float(rng.uniform(100, 9000))
            params.external_expected_eggs = external
            c2, s2 = _engine_rs.equilibrium_metrics(bp, params)
            fc2, fs2 = flat_call(draft, declared, external)
            pc2, ps2 = py_call(draft, declared, external)
            assert struct.pack(">d", c2) == struct.pack(">d", c), (
                "external override must not move C*"
            )
            assert struct.pack(">d", c2) == struct.pack(">d", fc2) == (
                struct.pack(">d", float(pc2))
            )
            assert struct.pack(">d", s2) == struct.pack(">d", fs2) == (
                struct.pack(">d", float(ps2))
            ), f"external s* three-way mismatch at trial {trial}"

    def test_build_path_uses_rust_and_matches_fallback_bitwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build-time metrics are bitwise-identical across both kernels.

        The build path keeps its own None semantics (reproduction is
        pre-normalized to a possibly all-zero array — never the sync
        path's mating-row fallback), and the Rust dispatch and the forced
        Python fallback must agree bit for bit on those inputs.
        """
        if _load_equilibrium_kernel() is None:
            pytest.skip("rust extension not built")
        from natal.frontend.data import _engine as engine_module
        from natal.frontend.data._config import build_population_config

        def build_once() -> nt.ModelDraft:
            return build_population_config(
                n_genotypes=3,
                n_gtypes=2,
                n_glabs=1,
                n_ages=4,
                new_adult_age=2,
                age_based_survival_rates=np.array(
                    [[1.0, 0.9, 0.7, 0.4], [1.0, 0.85, 0.6, 0.3]]
                ),
                age_based_mating_rates=np.array(
                    [[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.5, 0.0]]
                ),
                female_age_based_fertility=np.array([0.0, 0.0, 0.9, 0.5]),
                age_based_relative_competition_strength=np.array(
                    [1.0, 0.6, 0.2, 0.0]
                ),
                carrying_capacity=750.0,
                eggs_per_female=23.0,
                sex_ratio=0.5,
            )

        rust_built = build_once()
        rust_metrics = _draft_metrics(rust_built)
        assert rust_metrics[0] > 0.0  # non-degenerate
        assert 0.0 < rust_metrics[1] <= 1.0

        # The build function imports the dispatch from the engine module
        # inside its body, so the swap must target that source module.
        def force_fallback(  # object: probe shim mirrors any dispatch call shape
            *args: object, **kwargs: object
        ) -> None:
            return None

        monkeypatch.setattr(
            engine_module, "equilibrium_metrics_dispatch", force_fallback
        )
        py_built = build_once()

        assert _pack_metrics(rust_metrics) == _pack_metrics(
            _draft_metrics(py_built)
        )


class _AgeBuildCase(NamedTuple):
    """One typed build-path parameter set for the parity matrix."""

    tag: str
    n_ages: int
    new_adult_age: int
    survival: NDArray[np.float64]
    mating: NDArray[np.float64]
    reproduction: NDArray[np.float64] | None
    fertility: NDArray[np.float64]
    competition: NDArray[np.float64]
    k: float
    eggs: float
    sr: float
    declared: NDArray[np.float64] | None
    external: float | None


def _age_build_cases() -> list[_AgeBuildCase]:
    """The six build-path axes: undeclared/declared/all-zero reproduction
    x derive/declared distribution x external egg override, plus the
    minimal age ladder and an asymmetric sex ratio."""
    declared = np.array(
        [[0.0, 240.0, 180.0, 90.0], [0.0, 230.0, 150.0, 60.0]]
    )
    return [
        # 1. Undeclared reproduction: _validate_or_default_array must
        #    normalize to ones-with-juveniles-zeroed ([0, 0, 1, 1]) — the
        #    mating rows are deliberately unlike that default so the
        #    sync-path's mating-row fallback would give a different answer.
        _AgeBuildCase(
            tag="undeclared-repro",
            n_ages=4,
            new_adult_age=2,
            survival=np.array([[1.0, 0.9, 0.7, 0.4], [1.0, 0.85, 0.6, 0.3]]),
            mating=np.array([[0.0, 0.25, 0.5, 0.75], [0.0, 0.9, 0.8, 0.7]]),
            reproduction=None,
            fertility=np.array([0.0, 0.0, 0.9, 0.5]),
            competition=np.array([1.0, 0.6, 0.2, 0.0]),
            k=750.0,
            eggs=23.0,
            sr=0.5,
            declared=None,
            external=None,
        ),
        # 2. Declared reproduction + declared (2, n_ages) distribution.
        _AgeBuildCase(
            tag="declared-dist",
            n_ages=4,
            new_adult_age=2,
            survival=np.array([[0.95, 0.88, 0.7, 0.5], [0.92, 0.8, 0.6, 0.4]]),
            mating=np.array([[0.0, 1.0, 0.9, 0.0], [0.0, 0.8, 0.7, 0.0]]),
            reproduction=np.array([0.0, 0.6, 0.9, 0.3]),
            fertility=np.array([0.0, 0.0, 0.85, 0.45]),
            competition=np.array([1.0, 0.7, 0.1, 0.0]),
            k=1200.0,
            eggs=41.0,
            sr=0.5,
            declared=declared,
            external=None,
        ),
        # 3. External Champer egg override (moves the survival rate only).
        _AgeBuildCase(
            tag="external-eggs",
            n_ages=4,
            new_adult_age=2,
            survival=np.array([[0.95, 0.88, 0.7, 0.5], [0.92, 0.8, 0.6, 0.4]]),
            mating=np.array([[0.0, 1.0, 0.9, 0.0], [0.0, 0.8, 0.7, 0.0]]),
            reproduction=np.array([0.0, 0.6, 0.9, 0.3]),
            fertility=np.array([0.0, 0.0, 0.85, 0.45]),
            competition=np.array([1.0, 0.7, 0.1, 0.0]),
            k=1200.0,
            eggs=41.0,
            sr=0.5,
            declared=None,
            external=3131.0,
        ),
        # 4. Minimal ladder: new_adult_age=1, n_ages=2 — the juvenile
        #    competition loop over ages 1..new_adult_age is empty, so the
        #    egg mass alone feeds C*.
        _AgeBuildCase(
            tag="minimal-ladder",
            n_ages=2,
            new_adult_age=1,
            survival=np.array([[0.9, 0.5], [0.8, 0.4]]),
            mating=np.array([[0.0, 0.7], [0.0, 0.6]]),
            reproduction=None,
            fertility=np.array([0.0, 0.95]),
            competition=np.array([1.0, 0.3]),
            k=600.0,
            eggs=19.0,
            sr=0.5,
            declared=None,
            external=None,
        ),
        # 5. Declared all-zero reproduction: produced_age_0 == 0, so the
        #    survival-rate guard must fire (s* == 1.0 exactly) while the
        #    juvenile age-1 mass still drives C* > 0.
        _AgeBuildCase(
            tag="zero-repro",
            n_ages=4,
            new_adult_age=2,
            survival=np.array([[1.0, 0.9, 0.7, 0.4], [1.0, 0.85, 0.6, 0.3]]),
            mating=np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.5, 0.0]]),
            reproduction=np.zeros(4),
            fertility=np.array([0.0, 0.0, 0.9, 0.5]),
            competition=np.array([1.0, 0.6, 0.2, 0.0]),
            k=750.0,
            eggs=23.0,
            sr=0.5,
            declared=None,
            external=None,
        ),
        # 6. Asymmetric sex ratio on a deeper ladder.
        _AgeBuildCase(
            tag="deep-ladder",
            n_ages=6,
            new_adult_age=3,
            survival=np.array(
                [
                    [0.97, 0.93, 0.88, 0.8, 0.7, 0.5],
                    [0.96, 0.9, 0.84, 0.75, 0.6, 0.4],
                ]
            ),
            mating=np.array(
                [
                    [0.0, 0.0, 0.8, 0.9, 0.7, 0.2],
                    [0.0, 0.0, 0.6, 0.8, 0.5, 0.1],
                ]
            ),
            reproduction=np.array([0.0, 0.0, 0.0, 0.85, 0.9, 0.4]),
            fertility=np.array([0.0, 0.0, 0.0, 0.9, 0.75, 0.3]),
            competition=np.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.0]),
            k=2100.0,
            eggs=37.0,
            sr=0.3,
            declared=np.array(
                [
                    [0.0, 260.0, 200.0, 320.0, 250.0, 140.0],
                    [0.0, 240.0, 180.0, 300.0, 220.0, 110.0],
                ]
            ),
            external=None,
        ),
    ]


def _build_age_draft(case: _AgeBuildCase) -> nt.ModelDraft:
    """Run one matrix case through the public age-structured builder."""
    from natal.frontend.data._config import build_population_config

    return build_population_config(
        n_genotypes=3,
        n_gtypes=2,
        n_glabs=1,
        n_ages=case.n_ages,
        new_adult_age=case.new_adult_age,
        age_based_survival_rates=case.survival,
        age_based_mating_rates=case.mating,
        age_based_reproduction_rates=case.reproduction,
        female_age_based_fertility=case.fertility,
        age_based_relative_competition_strength=case.competition,
        carrying_capacity=case.k,
        eggs_per_female=case.eggs,
        sex_ratio=case.sr,
        equilibrium_individual_distribution=case.declared,
        external_expected_eggs=case.external,
    )


def _pack_metrics(metrics: tuple[float, float]) -> tuple[bytes, bytes]:
    """Full 64-bit bit patterns of a metric pair (catches -0.0/ulp drift)."""
    return struct.pack(">d", float(metrics[0])), struct.pack(
        ">d", float(metrics[1])
    )


def _draft_metrics(draft: nt.ModelDraft) -> tuple[float, float]:
    """The two equilibrium caches of a built draft."""
    return (
        float(draft.expected_competition_strength),
        float(draft.expected_survival_rate),
    )


def _assert_draft_fields_equal(
    left: nt.ModelDraft, right: nt.ModelDraft, context: str
) -> None:
    """Every NamedTuple field of two builds on identical input is equal."""
    right_fields = right._asdict()
    for name, left_value in left._asdict().items():
        right_value = right_fields[name]
        if isinstance(left_value, np.ndarray):
            np.testing.assert_array_equal(
                left_value, right_value, err_msg=f"{context}: field {name}"
            )
        else:
            assert left_value == right_value, (
                f"{context}: field {name}: {left_value!r} != {right_value!r}"
            )


class TestBuildPathEquilibriumDispatch:
    """Adversarial tests for the batch-12 build-path dispatch collapse."""

    def test_build_matrix_rust_matches_forced_fallback_bitwise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Six build axes: rust dispatch vs forced fallback are bit-equal.

        Attack: the two build branches could drift — the Rust branch
        normalizes dtypes/contiguity through ``ascontiguousarray`` while
        the Python fallback consumes the raw arrays, a sentinel could be
        treated as declared on one side only, or a vector could be wired
        into the wrong kernel slot.  Each case pins the full 64-bit
        pattern of both metrics AND every other draft field (only the
        kernel may differ, nothing else), plus per-case sensitivity
        floors so equality is never vacuous.
        """
        if _load_equilibrium_kernel() is None:
            pytest.skip("rust extension not built")
        from natal.frontend.data import _engine as engine_module

        def force_fallback(  # object: probe shim mirrors any dispatch call shape
            *args: object, **kwargs: object
        ) -> None:
            return None

        cases = _age_build_cases()
        for case in cases:
            rust_draft = _build_age_draft(case)
            rust_metrics = _draft_metrics(rust_draft)
            # Sensitivity floors per case family.
            assert rust_metrics[0] > 0.0, f"{case.tag}: degenerate C*"
            assert 0.0 < rust_metrics[1] <= 1.0 or case.tag == "zero-repro", (
                f"{case.tag}: implausible s*"
            )
            if case.tag == "zero-repro":
                # produced_age_0 == 0 fires the survival-rate guard.
                assert rust_metrics[1] == 1.0, (
                    f"{case.tag}: zero reproduction must yield s* == 1.0"
                )
            if case.tag == "external-eggs":
                # The override moves only the survival rate — compare
                # against the same case without the override.
                no_ext = _age_build_cases()[1]
                assert no_ext.tag == "declared-dist"
                base = _draft_metrics(_build_age_draft(no_ext))
                assert rust_metrics[0] != base[0], (
                    "external case lost its distinct declared/derive setup"
                )

            monkeypatch.setattr(
                engine_module, "equilibrium_metrics_dispatch", force_fallback
            )
            try:
                py_draft = _build_age_draft(case)
            finally:
                monkeypatch.undo()
            assert _pack_metrics(rust_metrics) == _pack_metrics(
                _draft_metrics(py_draft)
            ), f"{case.tag}: rust/forced-fallback metrics differ in bits"
            _assert_draft_fields_equal(
                rust_draft, py_draft, context=case.tag
            )

    def test_build_and_sync_none_semantics_differ_deliberately(self) -> None:
        """Undeclared reproduction: build uses the normalized default,
        sync uses the female mating row — and both match their own
        reference exactly.

        Attack: the None handling could be silently unified in either
        direction (build inheriting the mating fallback, or sync losing
        it).  The mating row here is unlike the normalized default, so
        each wrong unification moves the metrics.  As a bonus invariant,
        the built draft (reproduction stored non-None) is a fixed point
        of sync.
        """
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )
        from natal.frontend.configurator._routes import (
            sync_equilibrium_for_draft,
        )

        case = _age_build_cases()[0]
        assert case.tag == "undeclared-repro"
        draft = _build_age_draft(case)
        # The builder normalized the undeclared reproduction in place.
        np.testing.assert_array_equal(
            draft.age_based_reproduction_rates, [0.0, 0.0, 1.0, 1.0]
        )

        def reference(reproduction: NDArray[np.float64]) -> tuple[float, float]:
            return compute_equilibrium_metrics(
                carrying_capacity=case.k,
                eggs_per_female=case.eggs,
                sex_ratio=case.sr,
                age_based_survival_rates=case.survival,
                age_based_mating_rates=case.mating,
                age_based_reproduction_rates=reproduction,
                female_age_based_fertility=case.fertility,
                relative_competition_strength=case.competition,
                new_adult_age=case.new_adult_age,
                n_ages=case.n_ages,
                equilibrium_individual_count=None,
                external_expected_eggs=None,
            )

        ref_default = reference(np.array([0.0, 0.0, 1.0, 1.0]))
        ref_mating = reference(case.mating[0].copy())
        # Non-vacuous: the two candidate semantics genuinely differ.
        assert ref_default != ref_mating

        build_metrics = _draft_metrics(draft)
        assert _pack_metrics(build_metrics) == _pack_metrics(ref_default), (
            "build must consume the normalized default, not mating[0]"
        )
        assert _pack_metrics(build_metrics) != _pack_metrics(ref_mating)

        # Sync on a draft carrying None resolves to the female mating row.
        synced_none = sync_equilibrium_for_draft(
            draft._replace(age_based_reproduction_rates=None)
        )
        assert _pack_metrics(_draft_metrics(synced_none)) == _pack_metrics(
            ref_mating
        ), "sync(None) must consume the female mating row"
        assert _pack_metrics(_draft_metrics(synced_none)) != _pack_metrics(
            ref_default
        )

        # Fixed point: sync on the stored (non-None) reproduction is the
        # build result — build->sync cannot move the caches.
        synced_stored = sync_equilibrium_for_draft(draft)
        assert _pack_metrics(_draft_metrics(synced_stored)) == (
            _pack_metrics(build_metrics)
        )

    def test_dispatch_normalizes_empty_sentinels_to_derive(self) -> None:
        """None, (0,0), (2,0), and 1-D empty declared all derive; a real
        (2, n) declaration switches branch and matches the reference.

        Attack: a dispatch that forwarded the empty sentinel as a
        declared zero distribution would read out of bounds or derive a
        degenerate all-zero distribution instead of K.
        """
        if _load_equilibrium_kernel() is None:
            pytest.skip("rust extension not built")
        from natal.backends.reference.simulation.age_structured import (
            compute_equilibrium_metrics,
        )
        from natal.frontend.data._engine import equilibrium_metrics_dispatch

        survival = np.array([[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]])
        reproduction = np.array([0.0, 0.8, 0.6])
        fertility = np.array([0.0, 1.0, 0.9])
        competition = np.array([1.0, 0.5, 0.2])

        def call(
            declared: NDArray[np.float64] | None,
            external: float | None = None,
        ) -> tuple[float, float]:
            result = equilibrium_metrics_dispatch(
                400.0,
                30.0,
                0.5,
                survival,
                reproduction,
                fertility,
                competition,
                1,
                3,
                declared,
                external,
            )
            assert result is not None  # extension guarded above
            return result

        from_none = call(None)
        for empty in (
            np.zeros((0, 0)),
            np.zeros((2, 0)),
            np.zeros(0),
        ):
            assert _pack_metrics(call(empty)) == _pack_metrics(from_none), (
                f"sentinel {empty.shape} must derive like None"
            )

        declared = np.array(
            [[0.0, 200.0, 150.0], [0.0, 180.0, 120.0]]
        )
        declared_metrics = call(declared)
        assert _pack_metrics(declared_metrics) != _pack_metrics(from_none), (
            "declared distribution must switch the kernel branch"
        )
        # Declared branch matches the Python reference bitwise.
        ref = compute_equilibrium_metrics(
            carrying_capacity=400.0,
            eggs_per_female=30.0,
            sex_ratio=0.5,
            age_based_survival_rates=survival,
            age_based_mating_rates=np.zeros((2, 3)),  # unused: reproduction given
            age_based_reproduction_rates=reproduction,
            female_age_based_fertility=fertility,
            relative_competition_strength=competition,
            new_adult_age=1,
            n_ages=3,
            equilibrium_individual_count=declared,
            external_expected_eggs=None,
        )
        assert _pack_metrics(declared_metrics) == _pack_metrics(ref)

        # External override moves only the survival rate.
        external_metrics = call(None, external=3131.0)
        assert struct.pack(">d", external_metrics[0]) == struct.pack(
            ">d", from_none[0]
        ), "external override must not move C*"
        assert struct.pack(">d", external_metrics[1]) != struct.pack(
            ">d", from_none[1]
        ), "external override must move s*"

    def test_dispatch_block_recover_state_transition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """rust build -> blocked dispatch (None) -> fallback build ->
        unblocked rust build: every state produces the same bits.

        Attack: a negative import cache would keep the fallback active
        after the block lifts; a fallback that drifts from the kernel
        would flip the metrics between the blocked and unblocked builds.
        """
        if _load_equilibrium_kernel() is None:
            pytest.skip("rust extension not built")
        import natal._engine_rs  # noqa: F401  # pin the module in sys.modules
        from natal.frontend.data._engine import equilibrium_metrics_dispatch

        case = _age_build_cases()[0]
        pre_block = _draft_metrics(_build_age_draft(case))

        monkeypatch.setitem(sys.modules, "natal._engine_rs", None)
        assert equilibrium_metrics_dispatch(
            400.0, 30.0, 0.5,
            np.zeros((2, 3)), np.zeros(3), np.zeros(3), np.zeros(3),
            1, 3, None, None,
        ) is None, "blocked import must return None"
        blocked_build = _draft_metrics(_build_age_draft(case))
        assert _pack_metrics(blocked_build) == _pack_metrics(pre_block), (
            "fallback build drifted from the rust build"
        )

        monkeypatch.undo()
        after = _draft_metrics(_build_age_draft(case))
        assert _pack_metrics(after) == _pack_metrics(pre_block), (
            "dispatch must rediscover the extension after the block lifts"
        )

    def test_single_dispatch_point_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Build and sync both resolve the kernel through the engine
        module attribute — and neither spells the rust import inline.

        Attack: an inline ``from natal._engine_rs import ...`` (the
        batch-11 spelling) inside either caller would bypass the module
        attribute, so the recorder below would not fire.  The recorder
        also pins the caller-policy contract: sync resolves None
        reproduction to the female mating row *before* dispatching,
        build passes the already-normalized array.
        """
        import inspect

        from natal.frontend.configurator import _routes as routes_module
        from natal.frontend.configurator._routes import (
            sync_equilibrium_for_draft,
        )
        from natal.frontend.data import _engine as engine_module
        from natal.frontend.data._config import build_population_config

        # object: recorded positional/keyword payloads of arbitrary dispatch calls
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        def recorder(  # object: recorder mirrors any dispatch call shape
            *args: object, **kwargs: object
        ) -> None:
            calls.append((args, kwargs))
            return None  # forces each caller's fallback translation

        case = _age_build_cases()[0]
        draft = _build_age_draft(case)
        expected_metrics = _draft_metrics(draft)

        monkeypatch.setattr(
            engine_module, "equilibrium_metrics_dispatch", recorder
        )
        try:
            rebuilt = build_population_config(
                n_genotypes=3,
                n_gtypes=2,
                n_glabs=1,
                n_ages=case.n_ages,
                new_adult_age=case.new_adult_age,
                age_based_survival_rates=case.survival,
                age_based_mating_rates=case.mating,
                age_based_reproduction_rates=case.reproduction,
                female_age_based_fertility=case.fertility,
                age_based_relative_competition_strength=case.competition,
                carrying_capacity=case.k,
                eggs_per_female=case.eggs,
                sex_ratio=case.sr,
            )
            synced = sync_equilibrium_for_draft(
                draft._replace(age_based_reproduction_rates=None)
            )
        finally:
            monkeypatch.undo()

        assert len(calls) == 2, "build and sync must each dispatch exactly once"
        (build_args, build_kwargs), (sync_args, _sync_kwargs) = calls
        # The build caller spells keywords; the sync caller spells
        # positionals (slot 5 / index 4 is the resolved reproduction).
        np.testing.assert_array_equal(
            np.asarray(build_kwargs["reproduction_rates"]),
            [0.0, 0.0, 1.0, 1.0],
            err_msg="build must dispatch the normalized default vector",
        )
        np.testing.assert_array_equal(
            np.asarray(sync_args[4]), case.mating[0],
            err_msg="sync must dispatch the female mating row for None",
        )
        # The None return was translated into the fallback on both sides.
        assert _pack_metrics(_draft_metrics(rebuilt)) == _pack_metrics(
            expected_metrics
        )
        assert _pack_metrics(_draft_metrics(synced)) == _pack_metrics(
            _draft_metrics(
                sync_equilibrium_for_draft(
                    draft._replace(
                        age_based_reproduction_rates=case.mating[0].copy()
                    )
                )
            )
        )

        # Negative contract: no inline rust import survives in _routes,
        # and the dispatch is plumbing, not public API.
        routes_source = inspect.getsource(routes_module)
        assert "equilibrium_metrics_flat" not in routes_source
        assert "_engine_rs" not in routes_source
        import natal.frontend.data as data_package

        assert not hasattr(data_package, "equilibrium_metrics_dispatch")
        assert "equilibrium_metrics_dispatch" not in data_package.__all__

    def test_dispatch_error_paths_and_ownership(self) -> None:
        """Wrong shapes raise ValueError naming the field, leave the
        caller's arrays bit-identical, and no state is retained between
        calls.

        Attack: a validation that scribbles before raising corrupts live
        config; a kernel that retains the declared storage would ignore
        later mutations of the same array object.
        """
        if _load_equilibrium_kernel() is None:
            pytest.skip("rust extension not built")
        from natal.frontend.data._engine import equilibrium_metrics_dispatch

        survival = np.array([[0.9, 0.8, 0.7], [0.85, 0.75, 0.65]])
        reproduction = np.array([0.0, 0.8, 0.6])
        fertility = np.array([0.0, 1.0, 0.9])
        competition = np.array([1.0, 0.5, 0.2])
        snapshots = [a.copy() for a in (survival, reproduction, fertility, competition)]

        def call(
            survival_rates: NDArray[np.float64] = survival,
            reproduction_rates: NDArray[np.float64] = reproduction,
            declared: NDArray[np.float64] | None = None,
        ) -> tuple[float, float]:
            result = equilibrium_metrics_dispatch(
                400.0, 30.0, 0.5,
                survival_rates, reproduction_rates, fertility, competition,
                1, 3, declared, None,
            )
            assert result is not None  # extension guarded above
            return result

        with pytest.raises(ValueError, match="reproduction_rates must have length"):
            call(reproduction_rates=np.zeros(2))
        with pytest.raises(ValueError, match="declared_distribution shape"):
            call(declared=np.zeros((2, 4)))
        with pytest.raises(ValueError, match="survival_rates shape"):
            call(survival_rates=np.zeros((1, 3)))
        # State clean after every raise.
        for original, current in zip(
            snapshots, (survival, reproduction, fertility, competition)
        ):
            np.testing.assert_array_equal(current, original)

        # Ownership: an F-order caller is copied, not reordered in place.
        survival_f = np.asfortranarray(survival)
        survival_f_snapshot = survival_f.copy()
        call(survival_rates=survival_f)
        assert not survival_f.flags.c_contiguous
        np.testing.assert_array_equal(survival_f, survival_f_snapshot)

        # No retained kernel state: mutating the declared array between
        # calls must change the next result.
        declared = np.array([[0.0, 200.0, 150.0], [0.0, 180.0, 120.0]])
        first = call(declared=declared)
        declared[0, 1] = 999.0
        second = call(declared=declared)
        assert _pack_metrics(first) != _pack_metrics(second)

    def test_discrete_build_path_parity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The discrete factory funnels through the same changed
        build_config_maps, so its caches are kernel-independent too.

        Attack: build_discrete_engine_config normalizes its demographic
        vectors (survival [1,0], reproduction [0,1], new_adult_age=1)
        before the shared computation; a dispatch wiring that mishandled
        those discrete defaults would flip the discrete caches while the
        age-structured matrix stayed green.
        """
        if _load_equilibrium_kernel() is None:
            pytest.skip("rust extension not built")
        from natal.frontend.data import _engine as engine_module
        from natal.frontend.data._engine import build_discrete_engine_config

        # Minimal Mendelian maps: 1 locus, 2 alleles, 3 genotypes.
        meiosis = np.array(
            [
                [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
                [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
            ]
        )
        fusion = np.zeros((2, 2, 3))
        for gf, female_allele in enumerate(("A", "B")):
            for gm, male_allele in enumerate(("A", "B")):
                zygote = {"AA": 0, "AB": 1, "BA": 1, "BB": 2}[
                    female_allele + male_allele
                ]
                fusion[gf, gm, zygote] = 1.0

        def build_once() -> nt.ModelDraft:
            return build_discrete_engine_config(
                n_genotypes=3,
                n_gtypes=2,
                n_glabs=1,
                zygotes_to_gametes_map=meiosis,
                gametes_to_zygotes_map=fusion,
                carrying_capacity=850.0,
                eggs_per_female=31.0,
                sex_ratio=0.55,
                stochastic=False,
            )

        rust_draft = build_once()
        rust_metrics = _draft_metrics(rust_draft)
        assert rust_metrics[0] > 0.0
        assert 0.0 < rust_metrics[1] <= 1.0

        def force_fallback(  # object: probe shim mirrors any dispatch call shape
            *args: object, **kwargs: object
        ) -> None:
            return None

        monkeypatch.setattr(
            engine_module, "equilibrium_metrics_dispatch", force_fallback
        )
        py_draft = build_once()
        assert _pack_metrics(rust_metrics) == _pack_metrics(
            _draft_metrics(py_draft)
        )
        _assert_draft_fields_equal(
            rust_draft,
            py_draft,
            context="discrete",
        )
