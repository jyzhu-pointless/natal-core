"""The internal compilation dependency graph (FRONTEND_REFACTOR_PLAN.md §4.3).

``dependencies.jsonc`` declares every derived product and what it depends
on; this module loads that file, validates it (unknown node names, missing
compute implementations, cycles) and executes the nodes a phase asks for in
topological order.  The graph arranges computation only: the compute
functions live in the owning modules, and user-facing semantics — preset
priority, fitness replace/multiply order, manual rule combination — are
unchanged by it.

The graph is internal tooling for the compiler, not user API.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

_GRAPH_FILE = Path(__file__).parent / "dependencies.jsonc"

_Compute = Callable[[], object]


def _strip_jsonc_comments(text: str) -> str:
    """Drop ``//`` line comments that live outside string literals."""
    out: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < len(text):
                out.append(text[i + 1])
                i += 2
                continue
            if char == '"':
                in_string = False
            i += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            i += 1
            continue
        if char == "/" and i + 1 < len(text) and text[i + 1] == "/":
            while i < len(text) and text[i] != "\n":
                i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class DependencyGraph:
    """Validated node/dependency declarations from the JSONC config."""

    inputs: frozenset[str]
    dependencies: Mapping[str, tuple[str, ...]]

    def order(self) -> tuple[str, ...]:
        """Return every computed node in a dependency-respecting order.

        Raises:
            ValueError: If a node names a dependency the graph does not
                declare, or if the dependencies form a cycle.
        """
        for node, deps in self.dependencies.items():
            known = set(self.dependencies) | set(self.inputs)
            unknown = [dep for dep in deps if dep not in known]
            if unknown:
                raise ValueError(
                    f"Dependency graph node {node!r} names unknown "
                    f"dependencies {unknown!r}"
                )
        resolved: list[str] = []
        done: set[str] = set()
        remaining = dict(self.dependencies)
        while remaining:
            ready = sorted(
                node for node, deps in remaining.items()
                if all(dep in done or dep in self.inputs for dep in deps)
            )
            if not ready:
                raise ValueError(
                    "Dependency graph contains a cycle among "
                    f"{sorted(remaining)!r}"
                )
            for node in ready:
                resolved.append(node)
                done.add(node)
                del remaining[node]
        return tuple(resolved)


@dataclass
class DerivationPipeline:
    """Execute one phase's nodes of a :class:`DependencyGraph`.

    A phase registers a compute for each node it owns.  Running the
    pipeline validates that every node it should execute has a compute —
    a missing implementation is a configuration error, not a silent skip —
    and then executes the nodes in the graph's topological order.

    The publish-side nodes (compression, hooks, observation) are declared
    in the graph for order-of-record; the publish phase runs them in the
    order :meth:`DependencyGraph.order` yields.
    """

    graph: DependencyGraph
    _computes: dict[str, _Compute] = field(default_factory=dict[str, _Compute])

    def register(self, node: str, compute: _Compute) -> None:
        """Attach one node's compute function to this pipeline."""
        if node not in self.graph.dependencies:
            raise ValueError(
                f"Dependency graph declares no computed node {node!r}"
            )
        self._computes[node] = compute

    def run(self, nodes: tuple[str, ...]) -> dict[str, object]:
        """Execute the given nodes, and only those, in graph order.

        Args:
            nodes: The nodes this phase owns; they run in the graph's
                topological order restricted to this set.

        Returns:
            Each executed node's compute result keyed by node name.

        Raises:
            ValueError: If a requested node has no compute registered.
        """
        missing = [node for node in nodes if node not in self._computes]
        if missing:
            raise ValueError(
                f"Dependency graph nodes missing compute implementations: "
                f"{sorted(missing)!r}"
            )
        results: dict[str, object] = {}
        for node in self.graph.order():
            if node in nodes:
                results[node] = self._computes[node]()
        return results


def load_dependency_graph(text: str | None = None) -> DependencyGraph:
    """Load and validate the compilation dependency graph.

    Args:
        text: JSONC source; defaults to the packaged config file.

    Returns:
        The validated graph.

    Raises:
        ValueError: If the config is malformed, declares an input that is
            also a computed node, or fails validation in
            :meth:`DependencyGraph.order`.
    """
    source = text if text is not None else _GRAPH_FILE.read_text(encoding="utf-8")
    data = json.loads(_strip_jsonc_comments(source))
    try:
        inputs_raw = data["inputs"]
        nodes_raw = data["nodes"]
    except KeyError as exc:
        raise ValueError(f"Dependency graph config is missing {exc.args[0]!r}") from exc
    inputs = frozenset(inputs_raw)
    dependencies = {
        node: tuple(spec.get("depends", ()))
        for node, spec in nodes_raw.items()
    }
    overlap = sorted(set(inputs) & set(dependencies))
    if overlap:
        raise ValueError(
            f"Dependency graph nodes are both inputs and computed: {overlap!r}"
        )
    return DependencyGraph(inputs=inputs, dependencies=dependencies)
