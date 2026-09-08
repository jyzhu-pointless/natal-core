"""
Genetic Simulation Utilities
============================

Core components for genetic simulation: structures, entities, and population models.
"""

import ast
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, List, cast

__version__ = "0.2.0b"

# Maps exported symbol names to the module that defines them.
#
# The package intentionally does not import any child modules during initialization.
# It only builds a name index up front, for Examples:
# {"Sex": "frontend.utils", "AgeStructuredPopulation": "frontend.population"}
# When code first accesses natal.Sex, the matching module is imported on demand.
#
# This index is built from the real (non-shim) package tree only:
# ``frontend.*`` subpackages, ``contracts``, and ``backends.*``.  The legacy
# top-level forwarding shims (``natal.frontend.data``, ``natal.frontend.hooks`` ...) were removed
# when the Phase-0 reorganization completed; each legacy package name below
# still resolves to its relocated module so ``natal.frontend.hooks``-style access keeps
# working, but the physical packages no longer exist.
_lazy_map: Dict[str, str] = {}
_lazy_packages: set[str] = set()


def _extract_module_exports(module_file: Path) -> list[str]:
    """Return literal ``__all__`` entries from a module source file.

    This uses static source parsing instead of importing the module so the package
    can support true lazy loading. Importing the package only reads source text and
    builds the export table; it does not execute child-module top-level code.

    This requires each child module's ``__all__`` to be a literal value that
    ``ast.literal_eval`` can resolve, for Examples:

        __all__ = ["Sex", "Age"]

    If ``__all__`` is built dynamically at runtime, this function returns an empty
    list and that module will not participate in package-level lazy exports.
    """
    try:
        # Read source text and parse an AST only; this never executes module code.
        source = module_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_file))
    except (OSError, SyntaxError):
        # Ignore unreadable files or modules with syntax errors so one broken file
        # does not prevent the package itself from importing.
        return []

    # Only inspect top-level statements. The top-level __all__ assignment defines
    # the public symbols this package can expose lazily.
    for node in tree.body:
        value_node = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                value_node = node.value

        if value_node is None:
            continue

        try:
            # ast.literal_eval only resolves safe literal structures and will not
            # execute expressions.
            exports = ast.literal_eval(value_node)
        except Exception:
            return []

        if isinstance(exports, list):
            list_exports = cast(List[object], exports)
            if all(isinstance(item, str) for item in list_exports):
                return [cast(str, item) for item in list_exports]
        if isinstance(exports, tuple):
            tuple_exports = cast(tuple[object, ...], exports)
            if all(isinstance(item, str) for item in tuple_exports):
                return [cast(str, item) for item in tuple_exports]
        return []

    return []


def _scan_unit(module_name: str, allow_legacy_key: bool) -> list[str]:
    """Extract the literal ``__all__`` of one unit and register its exports.

    Args:
        module_name: Dotted module name of the unit (e.g. ``frontend.hooks``).
        allow_legacy_key: Whether the legacy short package key (e.g. ``hooks``)
            is registered alongside the exported names.  ``contracts`` keeps its
            own key because the package path did not change; ``frontend`` and
            ``backends`` subpackages keep their pre-Phase-0 keys so that
            ``natal.<legacy-key>`` attribute access keeps resolving.

    Returns:
        The unit's own exported names (may be empty).
    """
    module_file = package_dir.joinpath(*module_name.split("."), "__init__.py")
    if not module_file.is_file():
        return []
    exports = _extract_module_exports(module_file)
    if allow_legacy_key:
        short = module_name.rsplit(".", 1)[-1]
        _lazy_map.setdefault(short, module_name)
        if exports:
            _lazy_packages.add(short)
    for name in exports:
        _lazy_map.setdefault(name, module_name)
    return exports


# Scan the package tree and build the export-name -> module-name index.
#
# This only scans and parses files. It does not import modules, so importing natal
# remains lightweight.  Only real entity packages participate: the direct
# children of ``frontend``, the direct children of ``backends``, and
# ``contracts`` itself.  A unit joins the public lazy-export index when its
# ``__init__.py`` declares a non-empty literal ``__all__``; an empty (or
# missing) ``__all__`` marks the package as private/structural (e.g.
# ``frontend`` itself and ``backends``).
package_dir = Path(__file__).resolve().parent

# Every first-level package of the real tree, sorted for deterministic
# first-wins semantics on repeated names (keeps e.g. ``apply_preset_fitness_patch``
# owned by ``frontend.fitness``, alphabetically before ``frontend.presets``).
scan_units: list[str] = []
for root in ("contracts", "frontend", "backends"):
    root_init = package_dir / root / "__init__.py"
    if not root_init.is_file():
        continue
    if root == "contracts" or not _extract_module_exports(root_init):
        # ``contracts`` is itself an export owner; ``frontend``/``backends`` are
        # structural, so their only role is hosting the subpackages below.
        scan_units.append(root)
    for _, submodule_name, is_package in sorted(
        pkgutil.iter_modules([str(package_dir / root)]),
        key=lambda item: item[1],
    ):
        if is_package:
            scan_units.append(f"{root}.{submodule_name}")

for unit in sorted(set(scan_units)):
    # Skip empty-``__all__`` units: their submodule aliases would be registered
    # anyway for package-key access, but exporting nothing keeps their names out
    # of the attribute namespace.
    if unit != "contracts" and not _extract_module_exports(package_dir.joinpath(*unit.split("."), "__init__.py")):
        continue
    _scan_unit(unit, allow_legacy_key=True)

# Public export list.
#
# This keeps from natal import * aligned with the package's public API and also
# helps dir(natal) and some tooling discover these names.
__all__ = list(_lazy_map)  # type: ignore  # TODO


def __getattr__(name: str) -> Any:
    # When code accesses natal.<name> and that attribute is not present yet,
    # Python calls the module-level __getattr__. Import the owning child module
    # here so the symbol is loaded only on first access.
    if name in _lazy_map:
        module = importlib.import_module(f".{_lazy_map[name]}", __name__)
        if name in _lazy_packages:
            globals()[name] = module
            return module
        value = getattr(module, name)
        # Cache the resolved object in this module's globals so future accesses do
        # not have to go through __getattr__ again.
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    # Expose lazily exported names to dir() and completion tools.
    return sorted(set(globals().keys()) | set(__all__))
