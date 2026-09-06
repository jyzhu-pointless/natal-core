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
import subprocess
import sys
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

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
        from natal.frontend.data._engine import (
            _rust_offspring_kernel,
            recompute_offspring_tensor,
        )
        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
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
        from natal.frontend.data._engine import recompute_offspring_tensor
        from natal.backends.reference.simulation.age_structured import (
            compute_offspring_probability_tensor,
        )

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
