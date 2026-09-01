"""Contract tests for the Phase-0 directory reorganization forwarding shims.

The reorganization moved 13 frontend packages to ``natal/frontend/<mod>``,
split the engine across ``natal/backends/{reference,numba,rust}``, and moved
the top-level ``natal.numba`` to ``natal.backends.numba``.  Every legacy path
survives as a forwarding shim (literal ``__all__`` re-export plus
``sys.modules`` submodule aliasing).  These tests pin the "zero behavior
change" promise through five invariant classes:

1. Forwarding identity (ownership): a name resolved via the legacy path IS
   the same object as the one resolved via the new path — ``is`` equality,
   never a copy or a re-executed module.
2. Import-order independence (state transition): the fitness/presets import
   cycle is broken by PEP 562 deferral, so a clean interpreter must be able
   to import the involved modules in ANY order.
3. Lazy-map completeness (axis combination): every name in the top-level
   ``natal._lazy_map`` (242 names across 15 owning modules) resolves through
   ``getattr(natal, name)`` to the very object its shim exports.
4. Error paths: unknown names raise the documented exception types, and a
   failed lookup leaves the lazy machinery uncorrupted.
5. Template exclusion: the ``*.tmpl.py`` codegen templates are data files,
   not importable modules, while codegen can still read them from disk.
"""

import importlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

import natal

# =====================================================================
# Data tables: the full set of relocated packages and engine aliases.
# =====================================================================

# The 13 frontend packages: natal.<mod> (shim) -> natal.frontend.<mod>.
FRONTEND_MODULE_NAMES: Tuple[str, ...] = (
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
)

# The relocated top-level numba package: natal.numba (shim) -> natal.backends.numba.
NUMBA_PACKAGE_NAME = "numba"

# 13 frontend modules + numba: every package whose legacy path is a literal
# ``__all__``-forwarding shim (engine has no ``__all__`` shim; it is covered
# by the alias table below).
RELOCATED_PACKAGES: Tuple[str, ...] = FRONTEND_MODULE_NAMES + (NUMBA_PACKAGE_NAME,)


def _new_root(mod: str) -> str:
    """Return the new canonical import path for a relocated package."""
    if mod == NUMBA_PACKAGE_NAME:
        return "natal.backends.numba"
    return f"natal.frontend.{mod}"


# Every legacy engine path from the Phase-0 split (see the engine shim
# docstring): each must alias the exact relocated module object.
ENGINE_ALIAS_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("natal.engine.simulation", "natal.backends.reference.simulation"),
    ("natal.engine.migration", "natal.backends.reference.migration"),
    ("natal.engine.age_structured_simulator", "natal.backends.reference.age_structured_simulator"),
    ("natal.engine.discrete_generation_simulator", "natal.backends.reference.discrete_generation_simulator"),
    ("natal.engine.spatial_simulator", "natal.backends.reference.spatial_simulator"),
    ("natal.engine.spatial_migrator", "natal.backends.reference.spatial_migrator"),
    ("natal.engine.lifecycle", "natal.backends.numba.lifecycle"),
    ("natal.engine.lifecycle_wrappers", "natal.backends.numba.lifecycle_wrappers"),
    ("natal.engine.backends.rust_backend", "natal.backends.rust.rust_backend"),
    # Nested submodules aliased by the shim's sys.modules copy loop.
    ("natal.engine.simulation.age_structured", "natal.backends.reference.simulation.age_structured"),
    ("natal.engine.simulation.discrete_generation", "natal.backends.reference.simulation.discrete_generation"),
    ("natal.engine.migration.kernel", "natal.backends.reference.migration.kernel"),
    ("natal.engine.migration.adjacency", "natal.backends.reference.migration.adjacency"),
)

# Lazy-map axis check: the owning modules the top-level index is built from.
# Participation is rule-based (non-empty literal ``__all__`` in the package
# ``__init__``), so every relocated package plus ``utils`` owns names —
# including ``fitness``, whose absence from the former hand-maintained
# allowlist was a pre-existing omission, fixed when the allowlist was
# replaced by the rule.  The canonical owner of the doubly-exported
# ``apply_preset_fitness_patch`` is ``fitness`` (alphabetical first-wins).
EXPECTED_LAZY_OWNERS = frozenset(RELOCATED_PACKAGES) | {"utils"}

# Lower bound for the lazy map size.  If a shim loses its literal ``__all__``
# (e.g. someone rebuilds it dynamically), the AST scan in natal/__init__.py
# silently contributes ZERO names and the top-level API collapses.  The exact
# count is 244 (240 symbols + 15 public package names); the bound guards
# against silent shrinkage.
MIN_LAZY_MAP_SIZE = 244


# =====================================================================
# 1. Forwarding identity (ownership class)
# =====================================================================


@pytest.mark.parametrize("mod", RELOCATED_PACKAGES)
def test_forward_identity_of_every_public_name(mod: str) -> None:
    """Every ``__all__`` name of a shim IS the object at the new path.

    Invariant: for all 14 relocated packages, ``getattr(old_path, n) is
    getattr(new_path, n)`` for every public name — the shim forwards the
    original object, never a copy — and the public surfaces (``__all__``)
    are identical, so nothing was dropped or added by the move.
    """
    old_mod = importlib.import_module(f"natal.{mod}")
    new_mod = importlib.import_module(_new_root(mod))

    old_all = set(old_mod.__all__)
    new_all = set(new_mod.__all__)
    assert old_all == new_all, (
        f"shim natal.{mod} and {_new_root(mod)} expose different public "
        f"surfaces: only-old={sorted(old_all - new_all)} "
        f"only-new={sorted(new_all - old_all)}"
    )

    mismatches = [
        name
        for name in sorted(old_all)
        if getattr(old_mod, name) is not getattr(new_mod, name)
    ]
    assert not mismatches, (
        f"natal.{mod} forwards different objects than {_new_root(mod)} "
        f"for: {mismatches}"
    )


@pytest.mark.parametrize("old_path,new_path", ENGINE_ALIAS_PAIRS)
def test_engine_legacy_paths_are_module_aliases(old_path: str, new_path: str) -> None:
    """Each legacy engine path resolves to the relocated module object.

    Invariant: ``sys.modules[old_path] is sys.modules[new_path]`` and the
    module's ``__name__`` is the NEW path.  A distinct ``__name__`` would
    mean the file was re-executed under the legacy name (a second module
    object with duplicate compiled functions), which the shim must prevent.
    """
    old_mod = importlib.import_module(old_path)
    new_mod = importlib.import_module(new_path)

    assert old_mod is new_mod, (
        f"{old_path} resolved to a different module object than {new_path}"
    )
    assert sys.modules[old_path].__name__ == new_path, (
        f"{old_path} is not an alias: its __name__ is "
        f"{sys.modules[old_path].__name__!r} instead of {new_path!r}"
    )


def test_all_legacy_frontend_submodule_paths_are_module_aliases() -> None:
    """Every registered legacy submodule path IS its frontend counterpart.

    Invariant: after importing each of the 13 shims, every ``sys.modules``
    key of the form ``natal.<mod>.<sub>`` must be the very same module
    object as ``natal.frontend.<mod>.<sub>``.  The pairs are derived from
    ``sys.modules`` (not hardcoded) so the test automatically covers every
    alias the shims register, including nested ones like
    ``natal.population._mixins._hooks``.
    """
    for mod in FRONTEND_MODULE_NAMES:
        # Force the shim to run so it registers its submodule aliases.
        importlib.import_module(f"natal.{mod}")

    failures: List[str] = []
    for key in sorted(sys.modules):
        parts = key.split(".")
        # Skip non-legacy keys and the shim roots themselves: the root
        # ``natal.<mod>`` is a separate shim module by design (its exported
        # names are covered by the identity test above); only the submodule
        # aliases must be the identical module objects.
        if len(parts) < 3 or parts[0] != "natal" or parts[1] not in FRONTEND_MODULE_NAMES:
            continue
        new_key = "natal.frontend." + ".".join(parts[1:])
        new_mod = sys.modules.get(new_key)
        if new_mod is None:
            failures.append(f"{key}: no counterpart {new_key} in sys.modules")
        elif sys.modules[key] is not new_mod:
            failures.append(f"{key}: module object differs from {new_key}")

    assert not failures, "legacy submodule aliasing is broken:\n" + "\n".join(failures)


def test_legacy_numba_submodules_are_aliases() -> None:
    """``natal.numba.compat``/``.utils`` alias ``natal.backends.numba`` ones."""
    importlib.import_module("natal.numba")
    for sub in ("compat", "utils"):
        assert (
            sys.modules[f"natal.numba.{sub}"] is sys.modules[f"natal.backends.numba.{sub}"]
        ), f"natal.numba.{sub} is not an alias of the relocated module"


def test_deferred_fitness_reexport_returns_canonical_object() -> None:
    """All three lazy chains converge on the canonical ``fitness._patch`` object.

    Invariant: ``apply_preset_fitness_patch`` reached through
    (a) the frontend presets package ``__getattr__``,
    (b) the legacy ``natal.presets`` shim, and
    (c) the ``presets._fitness`` shim module's own ``__getattr__``
    must all be the single object defined in
    ``natal.frontend.fitness._patch`` — no re-export may produce a
    second copy of the function.
    """
    canonical = importlib.import_module("natal.frontend.fitness._patch").apply_preset_fitness_patch

    assert (
        getattr(importlib.import_module("natal.frontend.presets"), "apply_preset_fitness_patch")
        is canonical
    )
    assert (
        getattr(importlib.import_module("natal.presets"), "apply_preset_fitness_patch")
        is canonical
    )
    assert (
        getattr(importlib.import_module("natal.frontend.presets._fitness"), "apply_preset_fitness_patch")
        is canonical
    )


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
import natal.presets
assert (natal.presets.apply_preset_fitness_patch
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
    # Legacy shim paths only — the fitness/presets cycle must also stay
    # broken when entered through the old import locations.
    "legacy-fitness-presets": """
import natal.fitness
from natal.presets import apply_preset_fitness_patch
import natal.frontend.fitness._patch as patch
assert apply_preset_fitness_patch is patch.apply_preset_fitness_patch
""",
    # Unrelated heavy shims: hooks pulls the hook compiler stack, genetics
    # pulls the entity/structure stack; neither order may corrupt the other.
    "hooks-then-genetics": """
import natal.hooks
import natal.genetics
assert natal.hooks.HookProgram is natal.frontend.hooks.HookProgram
assert natal.genetics.Species is natal.frontend.genetics.Species
""",
    # Engine shim first, then the relocated reference modules it forwards to.
    "engine-then-reference": """
import natal.engine
import natal.backends.reference.simulation.age_structured
import natal.engine.simulation.age_structured as legacy
assert legacy is natal.backends.reference.simulation.age_structured
import natal.engine.backends.rust_backend as rb
assert rb is natal.backends.rust.rust_backend
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


# FIXED (was a real bug found by this test): the engine shim originally
# aliased only nested ``simulation``/``migration`` submodules already loaded
# at shim-execution time, so a legacy-first import of
# ``simulation.mgdrive1_compatible`` re-executed the file under the legacy
# name.  The shim now eagerly imports every leaf via pkgutil before
# registering the legacy aliases; identity holds in both orders.
_MGDRIVE1_LEGACY = "natal.engine.simulation.mgdrive1_compatible"
_MGDRIVE1_NEW = "natal.backends.reference.simulation.mgdrive1_compatible"


@pytest.mark.parametrize(
    "order",
    [
        pytest.param("new-first", id="new-first"),
        pytest.param("legacy-first", id="legacy-first"),
    ],
)
def test_engine_deep_submodule_identity_is_import_order_independent(order: str) -> None:
    """Deep engine submodule identity must not depend on import order.

    Invariant: whichever of the two paths is imported first, both names
    must end up referring to the SAME module object.  The new-first order
    works today because the engine shim's copy loop sees the loaded module;
    the legacy-first order currently creates a duplicate module.
    """
    if order == "new-first":
        code = (
            f"import {_MGDRIVE1_NEW} as new_mod\n"
            f"import natal.engine\n"
            f"import {_MGDRIVE1_LEGACY} as legacy_mod\n"
            "assert legacy_mod is new_mod\n"
        )
    else:
        code = (
            f"import {_MGDRIVE1_LEGACY} as legacy_mod\n"
            f"import {_MGDRIVE1_NEW} as new_mod\n"
            "assert legacy_mod is new_mod\n"
        )
    result = _run_clean_interpreter(code)
    assert result.returncode == 0, (
        f"{order}: mgdrive1_compatible identity broken "
        f"(rc={result.returncode}):\n{result.stderr}"
    )


# =====================================================================
# 3. Lazy-map completeness (axis-combination class)
# =====================================================================


def test_lazy_map_every_name_resolves_to_shim_export() -> None:
    """Every name in ``natal._lazy_map`` resolves and matches its shim.

    Invariant: for all 242 indexed names, ``getattr(natal, name)`` succeeds
    and IS the object obtained from the owning (shim) module — the top-level
    lazy table and the forwarding shims must agree name-for-name.  Package
    self-entries (e.g. ``"data" -> "data"``) resolve to the shim module
    object itself, so they are compared against ``sys.modules`` instead of
    an attribute.
    """
    failures: List[str] = []
    for name in sorted(natal._lazy_map):
        owner_name = natal._lazy_map[name]
        resolved = getattr(natal, name)  # a failed lookup raises -> loud error

        is_package_entry = name == owner_name and name in natal._lazy_packages
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

    Invariant (axis combination): the owner set of the index must be the 15
    relocated packages (13 frontend + numba + fitness) plus the unmoved
    ``utils``.
    A shim whose ``__all__`` stops being an AST literal silently drops out
    of the index, which the size bound (>= 240, actual 242) additionally
    guards against.
    """
    owners = set(natal._lazy_map.values())
    assert owners == EXPECTED_LAZY_OWNERS, (
        f"lazy-map owners drifted: missing={sorted(EXPECTED_LAZY_OWNERS - owners)} "
        f"unexpected={sorted(owners - EXPECTED_LAZY_OWNERS)}"
    )
    assert len(natal._lazy_map) >= MIN_LAZY_MAP_SIZE, (
        f"lazy map shrank to {len(natal._lazy_map)} names "
        f"(expected >= {MIN_LAZY_MAP_SIZE}): a shim probably lost its literal __all__"
    )


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


def test_from_legacy_shim_unknown_name_raises_import_error() -> None:
    """``from natal.patterns import <unknown>`` raises ImportError.

    Invariant: the legacy ``from``-import path must surface unknown names
    as ImportError (``from X import Y`` semantics), consistent with a
    normal package rather than silently succeeding via the shim.
    """
    with pytest.raises(ImportError, match="definitely_not_here"):
        from natal.patterns import definitely_not_here  # noqa: F401


def test_top_level_unknown_name_raises_attribute_error() -> None:
    """``getattr(natal, <unknown>)`` raises AttributeError (lazy contract)."""
    with pytest.raises(AttributeError, match="definitely_not_a_symbol"):
        getattr(natal, "definitely_not_a_symbol")


def test_unknown_legacy_module_raises_module_not_found() -> None:
    """Importing a non-existent legacy module fails with ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("natal.definitely_not_a_module")


# =====================================================================
# 5. Template exclusion
# =====================================================================

# The codegen template directory (data files, not an importable package).
_TEMPLATES_DIR = (
    Path(importlib.import_module("natal.frontend.hooks").__file__ or ".").resolve().parent
    / "templates"
)


def test_templates_directory_exposes_no_importable_modules() -> None:
    """``*.tmpl.py`` files are data, not importable modules.

    Invariant: for every template file, neither its intended module name
    (``combined_hook``) nor the dotted file stem (``combined_hook.tmpl``)
    is importable — ``find_spec`` returns None and ``import_module`` raises
    ModuleNotFoundError.  (The templates directory itself resolves as a
    PEP 420 namespace package because it lacks ``__init__.py``; that is
    unavoidable and harmless — the contract is that NO module inside it
    is importable.)
    """
    # Import the namespace parent first so child find_spec lookups are
    # well-defined instead of failing on a missing parent package.
    importlib.import_module("natal.frontend.hooks.templates")

    template_files = sorted(_TEMPLATES_DIR.glob("*.tmpl.py"))
    assert len(template_files) >= 4, (
        f"expected the known codegen templates under {_TEMPLATES_DIR}, "
        f"found only {[f.name for f in template_files]}"
    )

    failures: List[str] = []
    for file in template_files:
        # "combined_hook.tmpl.py" -> both "combined_hook" and "combined_hook.tmpl".
        dotted_stem = file.name[: -len(".py")]
        plain_stem = dotted_stem[: -len(".tmpl")] if dotted_stem.endswith(".tmpl") else dotted_stem
        for candidate in (plain_stem, dotted_stem):
            full = f"natal.frontend.hooks.templates.{candidate}"
            try:
                spec = importlib.util.find_spec(full)
            except ModuleNotFoundError:
                spec = None  # missing parent path also means "not importable"
            if spec is not None:
                failures.append(f"{full}: find_spec unexpectedly returned {spec}")
            try:
                importlib.import_module(full)
            except ModuleNotFoundError:
                pass  # the invariant holds for this name
            else:
                failures.append(f"{full}: import unexpectedly succeeded")

    assert not failures, "template files are importable as modules:\n" + "\n".join(failures)


def test_codegen_reads_combined_hook_template() -> None:
    """Codegen still reads the template as raw text from disk.

    Invariant: ``_read_hook_template("combined_hook.tmpl.py")`` returns a
    non-empty string containing the codegen placeholder markers — proof
    that excluding templates from the import path did not break the
    file-based template pipeline.
    """
    from natal.frontend.hooks.compile.codegen import _read_hook_template

    text = _read_hook_template("combined_hook.tmpl.py")
    assert isinstance(text, str) and text.strip(), "template read returned empty content"
    # The raw template must carry the markers compile_combined_hook replaces.
    assert "_combined_hook_TEMPLATE" in text
    assert "# PLACEHOLDER_SCHEDULE_BODY" in text
