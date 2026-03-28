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

# Maps exported symbol names to the module that defines them.
#
# The package intentionally does not import any child modules during initialization.
# It only builds a name index up front, for example:
# {"Sex": "type_def", "AgeStructuredPopulation": "age_structured_population"}
# When code first accesses natal.Sex, the matching module is imported on demand.
_lazy_map: Dict[str, str] = {}


def _extract_module_exports(module_file: Path) -> list[str]:
    """Return literal ``__all__`` entries from a module source file.

    This uses static source parsing instead of importing the module so the package
    can support true lazy loading. Importing the package only reads source text and
    builds the export table; it does not execute child-module top-level code.

    This requires each child module's ``__all__`` to be a literal value that
    ``ast.literal_eval`` can resolve, for example:

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

# Scan the package directory and build the export-name -> module-name index.
#
# This only scans and parses files. It does not import modules, so importing natal
# remains lightweight.
package_dir = Path(__file__).resolve().parent
for _, module_name, is_package in sorted(pkgutil.iter_modules(__path__), key=lambda item: item[1]):

    # Skip private modules. Most subpackages are ignored, except known lazy-export
    # providers such as `hooks`.
    if module_name.startswith("_"):
        continue

    if is_package:
        if module_name != "hooks":
            continue
        module_file = package_dir / module_name / "__init__.py"
    else:
        module_file = package_dir / f"{module_name}.py"

    if not module_file.is_file():
        continue

    # If multiple modules export the same name, keep the first mapping instead of
    # silently letting a later one overwrite it. Sorting by module name makes the
    # result stable and predictable.
    for name in _extract_module_exports(module_file):
        _lazy_map.setdefault(name, module_name)

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
        value = getattr(module, name)
        # Cache the resolved object in this module's globals so future accesses do
        # not have to go through __getattr__ again.
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    # Expose lazily exported names to dir() and completion tools.
    return sorted(set(globals().keys()) | set(__all__))
