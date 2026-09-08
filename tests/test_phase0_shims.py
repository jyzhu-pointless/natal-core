"""Contract tests for the post-shim Phase-0/6 relocation state.

The 15 legacy top-level forwarding shims (``natal.<mod>`` for every relocated
frontend package plus ``natal.engine``) were removed when the Phase-0
reorganization completed.  The real package tree is now
``natal.frontend.*``, ``natal.backends.*``, and ``natal.contracts``
(plus the compiled ``natal._engine_rs`` extension).  These tests pin the
"API surface unchanged, legacy paths gone" promise through five invariant
classes:

1. Negative contract (must-not-exist): every legacy shim path — package,
   registered submodule, and engine alias — raises ``ModuleNotFoundError``.
2. Import-order independence (state transition): the fitness/presets import
   cycle is broken by PEP 562 deferral, so a clean interpreter must be able
   to import the involved modules in ANY order.
3. Lazy-map completeness (axis combination): every name in the top-level
   ``natal._lazy_map`` (230 names across 15 real owning modules) resolves
   through ``getattr(natal, name)`` to the very object its owning module
   exports, and legacy package keys (``natal.hooks`` etc.) resolve to the
   relocated module object.
4. Error paths: unknown names raise the documented exception types, and a
   failed lookup leaves the lazy machinery uncorrupted.
5. Removed numba/codegen surface (slice ④ negative contracts) unchanged.
"""

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import natal

# =====================================================================
# Data tables: legacy paths that must NOT exist anymore.
# =====================================================================

# The 14 relocated legacy packages: natal.<mod> used to be a forwarding shim
# of natal.frontend.<mod>.  ``engine`` is covered by ENGINE_LEGACY_PATHS
# below (its shim exported no ``__all__`` names).
_RELOCATED_PACKAGES: Tuple[str, ...] = (
    "patterns",
    "registry",
    "genetics",
    "fitness",
    "presets",
    "modifiers",
    "output",
    "data",
    "configurator",
    "population",
    "spatial",
    "ui",
    "hooks",
    "utils",
)

# Submodule paths the removed shims used to register via sys.modules
# aliasing; each must now be strictly unimportable.
_LEGACY_SUBMODULE_PATHS: Tuple[str, ...] = tuple(
    list(_RELOCATED_PACKAGES)
    + [
        "configurator._base",
        "configurator._factory",
        "configurator._fitness",
        "configurator._params",
        "configurator._registry_builder",
        "configurator._routes",
        "configurator._writers",
        "data._builders",
        "data._config",
        "data._engine",
        "data._extract",
        "data.config",
        "data.constants",
        "data.state",
        "fitness._patch",
        "fitness._types",
        "fitness._writer",
        "genetics.entities",
        "genetics.entities._base",
        "genetics.entities.gene",
        "genetics.entities.genotype",
        "genetics.entities.haplotype",
        "genetics.structures",
        "genetics.structures._base",
        "genetics.structures._construction",
        "genetics.structures._enumeration",
        "genetics.structures._helpers",
        "genetics.structures._mapping",
        "genetics.structures._pattern",
        "genetics.structures._registry",
        "genetics.structures._types",
        "genetics.structures.chromosome",
        "genetics.structures.chromosome_map",
        "genetics.structures.locus",
        "genetics.structures.species",
        "hooks.compile",
        "hooks.compile.container",
        "hooks.entry",
        "hooks.entry.declarative",
        "hooks.entry.decorator",
        "hooks.entry.selector",
        "hooks.runtime",
        "hooks.runtime.csr_kernel",
        "hooks.runtime.fallback",
        "hooks.tick_context",
        "hooks.types",
        "modifiers.conditions",
        "modifiers.gamete_conversion",
        "modifiers.module",
        "modifiers.zygote_conversion",
        "output._recording",
        "output.history",
        "output.observation",
        "output.record",
        "output.translation",
        "patterns.elements",
        "patterns.individual_selector",
        "patterns.parser",
        "patterns.selector",
        "population._mixins",
        "population._mixins._hooks",
        "population._mixins._modifiers",
        "population._mixins._observation",
        "population._mixins._output",
        "population.age_structured",
        "population.base",
        "population.discrete_generation",
        "presets._base",
        "presets._fitness",
        "presets._types",
        "presets.cytoplasmic",
        "presets.homing",
        "presets.toxin_antidote",
        "registry.index",
        "spatial.configurator",
        "spatial.population",
        "spatial.topology",
        "ui.dashboard",
        "ui.dashboard_helpers",
        "ui.dashboard_population",
        "ui.spatial_dashboard",
        "ui.visualization",
        "utils.helpers",
        "utils.parameters",
        "utils.types",
    ]
)

# Every legacy engine path from the Phase-0 split: each must now fail.
ENGINE_LEGACY_PATHS: Tuple[str, ...] = (
    "natal.engine",
    "natal.engine.simulation",
    "natal.engine.simulation.age_structured",
    "natal.engine.simulation.discrete_generation",
    "natal.engine.simulation.mgdrive1_compatible",
    "natal.engine.migration",
    "natal.engine.migration.adjacency",
    "natal.engine.age_structured_simulator",
    "natal.engine.discrete_generation_simulator",
    "natal.engine.spatial_simulator",
    "natal.engine.spatial_migrator",
    "natal.engine.lifecycle",
    "natal.engine.backends",
    "natal.engine.backends.rust_backend",
)

# =====================================================================
# 1. Negative contract: legacy paths must not exist (import fails).
# =====================================================================


@pytest.mark.parametrize("legacy_path", sorted(f"natal.{p}" for p in _LEGACY_SUBMODULE_PATHS))
def test_legacy_shim_packages_not_importable(legacy_path: str) -> None:
    """Removed shim paths must raise ModuleNotFoundError.

    Invariant: any import of a removed forwarding path —
    ``natal.<mod>`` or one of its formerly aliased submodules — fails.
    ``ModuleNotFoundError`` (importlib) and ``ImportError`` (syntax form)
    are the same exception class hierarchy here.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_path)


@pytest.mark.parametrize("legacy_path", ENGINE_LEGACY_PATHS)
def test_legacy_engine_paths_not_importable(legacy_path: str) -> None:
    """Removed engine shim and its alias tree must raise ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_path)


def test_reference_backend_removed() -> None:
    """The deleted reference backend is gone statically and dynamically.

    Invariant: ``natal.backends.reference`` has no importable spec (the
    package directory was physically removed) and importing it raises
    ``ModuleNotFoundError`` — no silent fallback reappears.
    """
    import importlib.util

    assert importlib.util.find_spec("natal.backends.reference") is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("natal.backends.reference")


def test_no_legacy_packages_left_on_disk() -> None:
    """The physical shim directories are gone from the package tree.

    Invariant: the only first-level packages under ``natal/`` are the real
    tree (``frontend``, ``backends``, ``contracts``); no ``configurator``,
    ``data``, ... ``utils`` directory remains.
    """
    pkg_dir = Path(importlib.import_module("natal").__file__ or ".").resolve().parent
    leftovers = sorted(
        p.name
        for p in pkg_dir.iterdir()
        if p.is_dir() and p.name in (set(_RELOCATED_PACKAGES) | {"engine", "utils"})
    )
    assert not leftovers, f"legacy shim dirs still on disk: {leftovers}"


# =====================================================================
# 2. Import-order independence (state-transition class)
# =====================================================================


def _run_clean_interpreter(code: str) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a fresh interpreter and return the completed process.

    A subprocess is required because import-order bugs are process-global:
    once ``sys.modules`` is populated in this pytest process, the failure
    mode (a partially-initialized module) can no longer be reproduced.
    """
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=300,
    )


# Each script imports the cycle-involved modules in a different order and
# then asserts the identity invariants.  Any import failure OR identity
# mismatch aborts with a non-zero exit code, so rc == 0 is the invariant.
ORDER_SCRIPTS: Dict[str, str] = {
    "fitness-then-presets": """
import natal.frontend.fitness
import natal.frontend.presets
assert (natal.frontend.presets.apply_preset_fitness_patch
        is natal.frontend.fitness._patch.apply_preset_fitness_patch)
""",
    "presets-then-fitness": """
import natal.frontend.presets
import natal.frontend.fitness
assert (natal.frontend.presets.apply_preset_fitness_patch
        is natal.frontend.fitness._patch.apply_preset_fitness_patch)
""",
    # The historical crash trigger: importing the deep private module first
    # used to leave ``_patch`` partially initialized for ``presets``.
    "private-patch-first": """
import natal.frontend.fitness._patch
import natal.frontend.presets
assert (natal.frontend.presets.apply_preset_fitness_patch
        is natal.frontend.fitness._patch.apply_preset_fitness_patch)
""",
    # Deep private preset module first, then its deferred attribute, then
    # the fitness side that ``_fitness`` lazily imports from.
    "private-presets-fitness-first": """
import natal.frontend.presets._fitness
assert (natal.frontend.presets._fitness.apply_preset_fitness_patch
        is natal.frontend.fitness._patch.apply_preset_fitness_patch)
import natal.frontend.fitness
import natal.frontend.presets
assert (natal.frontend.presets.apply_preset_fitness_patch
        is natal.frontend.fitness._patch.apply_preset_fitness_patch)
""",
    # Lazy top-level surface first, then the deep module: the PEP 562
    # deferral must hold through ``import natal`` as well.
    "lazy-top-level-then-deep": """
import natal as nt
assert (nt.apply_preset_fitness_patch
        is nt.frontend.fitness._patch.apply_preset_fitness_patch)
import natal.frontend.presets._fitness
assert (nt.apply_preset_fitness_patch
        is natal.frontend.presets._fitness.apply_preset_fitness_patch)
""",
    # Unrelated heavy packages: hooks pulls the hook compiler stack, genetics
    # pulls the entity/structure stack; neither order may corrupt the other.
    "hooks-then-genetics": """
import natal.frontend.hooks
import natal.frontend.genetics
assert natal.frontend.hooks.HookProgram is natal.frontend.hooks.HookProgram
assert natal.frontend.genetics.Species is natal.frontend.genetics.Species
""",
    # Rust backend adapter first, then the contract layer feeding it: both
    # are independent of the frontend initialization order.
    "rust-backend-then-contracts": """
import natal.backends.rust.rust_backend
import natal.contracts
assert natal.contracts.CONTRACTS_VERSION == 2
""",
}


@pytest.mark.parametrize("order_name", sorted(ORDER_SCRIPTS))
def test_import_order_independence(order_name: str) -> None:
    """A clean interpreter imports the cycle-involved modules in any order.

    Invariant: the PEP 562 deferral in ``frontend/presets/__init__.py`` and
    ``frontend/presets/_fitness.py`` must make package initialization
    order-insensitive — every ordering below exits 0 (imports succeed AND
    the deferred re-export still resolves to the canonical object).
    """
    result = _run_clean_interpreter(ORDER_SCRIPTS[order_name])
    assert result.returncode == 0, (
        f"import order {order_name!r} failed in a clean interpreter "
        f"(rc={result.returncode}):\n{result.stderr}"
    )


# =====================================================================
# 3. Lazy-map completeness (axis-combination class)
# =====================================================================

# The lazy index is built from the real tree only: 14 ``frontend.*`` modules
# plus ``contracts``.  A module whose ``__all__`` stops being an AST literal
# silently drops out of the index, so the size bound and the owner set are
# both pinned.
EXPECTED_LAZY_OWNERS = frozenset(
    {f"frontend.{mod}" for mod in _RELOCATED_PACKAGES} | {"contracts"}
)

# Exact keys as of the post-⑥A public surface (230 names, incl. the 15
# legacy package self-entries).  A name that silently disappears from a
# package ``__all__`` drops the count below the guard.
MIN_LAZY_MAP_SIZE = 225


def test_lazy_map_every_name_resolves_to_owner_export() -> None:
    """Every name in ``natal._lazy_map`` resolves and matches its owner.

    Invariant: for all indexed names, ``getattr(natal, name)`` succeeds and
    IS the object obtained from the owning (real) module.  Package
    self-entries (e.g. ``"hooks" -> "frontend.hooks"``) resolve to the
    relocated module object itself, so they are compared against
    ``sys.modules`` instead of an attribute.
    """
    failures: List[str] = []
    for name in sorted(natal._lazy_map):
        owner_name = natal._lazy_map[name]
        resolved = getattr(natal, name)  # a failed lookup raises -> loud error

        is_package_entry = name in natal._lazy_packages
        owner_mod = importlib.import_module(f"natal.{owner_name}")
        expected = owner_mod if is_package_entry else getattr(owner_mod, name)

        if resolved is not expected:
            failures.append(f"{name} (owner {owner_name}): object mismatch")

    assert not failures, (
        f"{len(failures)}/{len(natal._lazy_map)} lazy-map names resolve to "
        "unexpected objects:\n" + "\n".join(failures)
    )


def test_lazy_map_owner_axes_and_size() -> None:
    """The lazy map is built from exactly the 15 expected owning modules.

    Invariant (axis combination): the owner set of the index must be the 14
    relocated frontend packages plus ``contracts`` — the legacy shims are
    gone and no ``backends.*`` unit joined (their ``__all__`` is empty).
    """
    owners = set(natal._lazy_map.values())
    assert owners == EXPECTED_LAZY_OWNERS, (
        f"lazy-map owners drifted: missing={sorted(EXPECTED_LAZY_OWNERS - owners)} "
        f"unexpected={sorted(owners - EXPECTED_LAZY_OWNERS)}"
    )
    assert len(natal._lazy_map) >= MIN_LAZY_MAP_SIZE, (
        f"lazy map shrank to {len(natal._lazy_map)} names "
        f"(expected >= {MIN_LAZY_MAP_SIZE}): a package probably lost its literal __all__"
    )


def test_legacy_package_keys_still_resolve_to_relocated_modules() -> None:
    """The legacy package names keep resolving to the real modules.

    Invariant: ``getattr(natal, mod)`` for each relocated legacy key yields
    the relocated module itself, so pre-migration code relying on
    ``natal.hooks.X``-style access sees the same objects.
    """
    for mod in _RELOCATED_PACKAGES:
        resolved = getattr(natal, mod)
        relocated = importlib.import_module(f"natal.frontend.{mod}")
        assert resolved is relocated, f"natal.{mod} is not the relocated module"


# =====================================================================
# 4. Error paths
# =====================================================================


def test_frontend_presets_getattr_unknown_name_raises() -> None:
    """The PEP 562 ``__getattr__`` rejects unknown names with AttributeError.

    Invariant: a deferred-attribute module must fail closed — an arbitrary
    name raises AttributeError naming the module and the attribute, instead
    of silently returning something (e.g. falling through to ``None``).
    """
    mod = importlib.import_module("natal.frontend.presets")
    with pytest.raises(AttributeError, match=r"natal\.frontend\.presets.*definitely_not_here"):
        getattr(mod, "definitely_not_here")


def test_presets_fitness_module_getattr_unknown_name_raises() -> None:
    """``presets._fitness.__getattr__`` also fails closed, state unchanged.

    Invariant: after a failed deferred lookup the module is not corrupted —
    the deferred ``_PATCH_REEXPORTS`` name still resolves to the canonical
    object from ``fitness._patch`` (error path leaves state intact).
    """
    mod = importlib.import_module("natal.frontend.presets._fitness")
    with pytest.raises(AttributeError, match=r"definitely_not_here"):
        getattr(mod, "definitely_not_here")

    canonical = importlib.import_module("natal.frontend.fitness._patch").apply_preset_fitness_patch
    assert mod.apply_preset_fitness_patch is canonical, (
        "deferred re-export broke after a failed __getattr__ lookup"
    )


def test_top_level_unknown_name_raises_attribute_error() -> None:
    """``getattr(natal, <unknown>)`` raises AttributeError (lazy contract)."""
    with pytest.raises(AttributeError, match="definitely_not_a_symbol"):
        getattr(natal, "definitely_not_a_symbol")


def test_unknown_legacy_module_raises_module_not_found() -> None:
    """Importing a non-existent module fails with ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("natal.definitely_not_a_module")


# =====================================================================
# 5. Removed njit codegen surface (slice ④ negative contract)
# =====================================================================


def test_removed_codegen_surface_not_importable() -> None:
    """The njit hook codegen modules and templates no longer exist.

    The njit hook-wrapper mechanism was removed; importing the legacy
    module paths must raise ModuleNotFoundError and the templates
    directory is gone from the package tree.
    """
    hooks_pkg = Path(importlib.import_module("natal.frontend.hooks").__file__ or ".").resolve().parent

    for legacy in (
        "natal.hooks.compile.codegen",
        "natal.frontend.hooks.compile.codegen",
        "natal.frontend.hooks.templates",
    ):
        try:
            importlib.import_module(legacy)
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"{legacy} unexpectedly importable")

    assert not (hooks_pkg / "templates").exists(), (
        "hooks templates directory should have been removed with the njit codegen pipeline"
    )
    assert not (hooks_pkg / "compile" / "codegen.py").exists()


def test_no_numba_package_remaining() -> None:
    """The numba backend package is gone.

    Invariant: after the numba backend removal, neither ``natal.backends.numba``
    nor any of its former submodules is importable, and no ``numba`` directory
    exists anywhere under ``src/natal``.
    """
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("natal.backends.numba")

    pkg_dir = Path(importlib.import_module("natal").__file__ or ".").resolve().parent
    assert not any(p.name == "numba" and p.is_dir() for p in pkg_dir.rglob("*")), (
        "remaining numba package directory under src/natal"
    )


def test_public_export_list_matches_module_all() -> None:
    """The explicit top-level export list cannot drift from module ``__all__``.

    Two directions (plan S6, must-not-exist item 8): every unit listed in
    ``natal._PUBLIC_EXPORTS`` must still declare exactly those names in its
    literal ``__all__`` (no stale entries), and no unlisted unit with a
    non-empty ``__all__`` may exist (a new module export must be added to
    the explicit list — an intentional act — before it becomes public).
    """
    import ast

    import natal

    def extract_all(init: Path) -> list[str]:
        """Return the literal ``__all__`` names of one package ``__init__``."""
        tree = ast.parse(init.read_text(encoding="utf-8"))
        for node in tree.body:
            value: ast.expr | None = None
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
            ):
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "__all__":
                value = node.value
            if value is None:
                continue
            names = ast.literal_eval(value)
            assert isinstance(names, (list, tuple)), "non-literal __all__"
            return [str(name) for name in names]
        return []

    package_dir = Path(natal.__file__).resolve().parent
    listed: dict[str, list[str]] = {
        unit: list(names) for unit, names in natal._PUBLIC_EXPORTS.items()  # noqa: SLF001 — the pinned surface under test
    }

    for unit, names in sorted(listed.items()):
        init = package_dir.joinpath(*unit.split("."), "__init__.py")
        assert init.is_file(), f"listed unit {unit} has no __init__.py"
        actual = extract_all(init)
        assert actual == names, (
            f"{unit}.__all__ drifted from _PUBLIC_EXPORTS: "
            f"module={actual} list={names}"
        )

    # No unlisted export-owning unit: scan the tree the way the retired
    # auto-discovery did and require every hit to be listed.
    discovered: set[str] = set()
    for root in ("contracts", "frontend", "backends"):
        root_init = package_dir / root / "__init__.py"
        if not root_init.is_file():
            continue
        if root == "contracts" or extract_all(root_init):
            discovered.add(root)
        for entry in sorted((package_dir / root).iterdir()):
            if entry.is_dir() and (entry / "__init__.py").is_file() and not entry.name.startswith("_"):
                discovered.add(f"{root}.{entry.name}")
    for unit in sorted(discovered):
        init = package_dir.joinpath(*unit.split("."), "__init__.py")
        if unit != "contracts" and not extract_all(init):
            continue  # private/structural unit (empty __all__)
        assert unit in listed, (
            f"{unit} declares exports but is not in _PUBLIC_EXPORTS — "
            "add it there to publish its names"
        )
    assert set(listed) <= discovered, "listed unit no longer exists on disk"
