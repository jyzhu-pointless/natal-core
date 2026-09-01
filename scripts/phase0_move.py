"""Phase-0 module migration helper: relocate a subpackage and leave a shim.

Automates the validated pilot workflow:

1. ``git mv src/natal/<mod> src/natal/<dest>/<mod>``
2. Generate a forwarding shim at the legacy path that re-exports the
   package's literal ``__all__`` (required by the top-level lazy-export
   table), plus any extra names imported from the legacy path elsewhere in
   the codebase, and alias-registers every subpackage/submodule under the
   legacy path via ``sys.modules``.
3. Run ``ruff check --fix`` on the shim.

Usage:
    python scripts/phase0_move.py <module> <dest>
    # e.g. python scripts/phase0_move.py registry frontend
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "natal"


def _literal_all(init_path: Path) -> list[str]:
    """Extract the literal ``__all__`` list from a package ``__init__.py``."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        if value is None:
            continue
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
    return []


def _extra_imported_names(module: str) -> list[str]:
    """Names imported from the legacy path elsewhere in src/demos (not in __all__)."""
    pattern = re.compile(rf"from natal\.{re.escape(module)} import ([\w, ()\n]+?)(?:$|#)", re.M)
    names: set[str] = set()
    files = list(SRC.rglob("*.py")) + list((REPO / "demos").rglob("*.py"))
    for py in files:
        rel = py.relative_to(SRC).as_posix() if py.is_relative_to(SRC) else ""
        if rel.startswith(module + "/"):
            continue  # files inside the module itself
        text = py.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            for name in match.group(1).replace("(", " ").replace(")", " ").replace("\n", " ").split(","):
                name = name.strip()
                if name and name.isidentifier():
                    names.add(name)
    return sorted(names)


def _walk_submodules(pkg_dir: Path, prefix: str) -> list[str]:
    """All importable dotted subpaths under a package directory."""
    subs: list[str] = []
    for child in sorted(pkg_dir.rglob("*.py")):
        # ``*.tmpl.py`` files are codegen text templates (dotted filenames,
        # not importable modules) — skip them.
        if child.name.endswith(".tmpl.py"):
            continue
        if child.name == "__init__.py":
            rel = child.parent.relative_to(pkg_dir).as_posix()
        else:
            rel = child.with_suffix("").relative_to(pkg_dir).as_posix()
        if rel == ".":
            continue
        subs.append(f"{prefix}.{rel.replace('/', '.')}")
    return subs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="subpackage name under src/natal")
    parser.add_argument("dest", help="destination package under src/natal (e.g. frontend)")
    args = parser.parse_args()

    module, dest = args.module, args.dest
    old_dir = SRC / module
    new_dir = SRC / dest / module
    if not old_dir.is_dir():
        print(f"error: {old_dir} does not exist")
        return 1
    if new_dir.exists():
        print(f"error: {new_dir} already exists")
        return 1

    all_names = _literal_all(old_dir / "__init__.py")
    if not all_names:
        print(f"error: {module}/__init__.py has no literal __all__; handle manually")
        return 1

    extras = [n for n in _extra_imported_names(module) if n not in all_names]

    # 1. git mv (history-preserving)
    subprocess.run(["git", "mv", str(old_dir), str(new_dir)], cwd=REPO, check=True)

    # 1b. Rewrite legacy absolute self-references inside the relocated tree to
    # direct paths.  Eager self-references through the legacy shim create
    # circular imports (shim -> new package -> shim), so the migrated tree
    # must import itself directly.
    self_ref = re.compile(rf"natal\.{re.escape(module)}\b")
    rewritten = 0
    for py in new_dir.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        new_text = self_ref.sub(f"natal.{dest}.{module}", text)
        if new_text != text:
            py.write_text(new_text, encoding="utf-8")
            rewritten += 1

    # 2. enumerate submodules from the NEW location
    submods = _walk_submodules(new_dir, f"natal.{dest}.{module}")

    # 3. generate shim
    all_block = "\n".join(
        line for line in [
            "from natal.%s.%s import (" % (dest, module),
            *[f"    {name}," for name in sorted(set(all_names + extras))],
            ")",
        ]
    )
    alias_block = ""
    if submods:
        alias_block = (
            "\n\n# Alias-register submodules under the legacy path so both\n"
            "# ``import natal.{m}.<sub>`` and ``from natal.{m}.<sub> import X`` work.\n"
            "import sys as _sys\n".format(m=module)
        )
        for i, sm in enumerate(submods):
            legacy = sm.replace(f"natal.{dest}.{module}", f"natal.{module}")
            alias_block += f"import {sm} as _m{i}\n_sys.modules[\"{legacy}\"] = _m{i}\n"

    shim = (
        '"""Forwarding shim: the ``{m}`` package now lives at\n'
        '``natal.{d}.{m}``.\n\n'
        "This module preserves the legacy import path during the Phase-0\n"
        "directory reorganisation; it will be removed once the migration\n"
        'completes.\n"""\n\n'
        "{imports}\n"
        "{aliases}\n"
        "__all__ = [\n{alllist}\n]\n"
    ).format(
        m=module, d=dest,
        imports=all_block,
        aliases=alias_block,
        alllist="\n".join(f'    "{n}",' for n in sorted(all_names)),
    )
    old_dir.mkdir(parents=True)
    (old_dir / "__init__.py").write_text(shim, encoding="utf-8")

    # 4. ruff fix
    result = subprocess.run(
        ["ruff", "check", str(old_dir), "--fix"],
        cwd=REPO, capture_output=True, text=True)
    if result.returncode not in (0, 1):
        print(result.stdout, result.stderr)

    print(f"moved natal.{module} -> natal.{dest}.{module}")
    print(f"  __all__ names: {len(all_names)}, extra re-exports: {extras or 'none'}")
    print(f"  submodules aliased: {len(submods)}, self-refs rewritten in {rewritten} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
