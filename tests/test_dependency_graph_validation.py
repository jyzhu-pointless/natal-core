"""Load-time dependency validation and isolation of the shared graph."""

import json

import pytest

from natal.frontend.model.dependency_graph import (
    DependencyGraph,
    DerivationPipeline,
    load_dependency_graph,
)


@pytest.mark.parametrize('source, message', [
    ('[]', 'must be an object'),
    ('{"inputs": 1, "nodes": {}}', 'list of names'),
    ('{"inputs": [1], "nodes": {}}', 'list of names'),
    ('{"inputs": [], "nodes": []}', 'must be an object'),
    ('{"inputs": [], "nodes": {"a": 1}}', 'must be an object'),
    ('{"inputs": [], "nodes": {"a": {"depends": "a"}}}', 'list of names'),
    ('{"inputs": [], "nodes": {"a": {"depends": [1]}}}', 'list of names'),
    ('{"inputs": [], "nodes": {"a": {"depends": ["missing"]}}}', 'unknown'),
    ('{"inputs": [], "nodes": {"a": {"depends": ["b"]}, "b": {"depends": ["a"]}}}', 'cycle'),
])
def test_invalid_graph_fails_at_load(source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_dependency_graph(text=source)


def test_graph_owns_dependencies_and_cached_graph_is_immutable() -> None:
    dependencies = {'a': ('input',)}
    graph = DependencyGraph(frozenset({'input'}), dependencies)
    dependencies['a'] = ('a',)
    assert graph.order() == ('a',)
    shared = load_dependency_graph()
    with pytest.raises(TypeError):
        shared.dependencies['hooks'] = ('hooks',)
    assert load_dependency_graph() is shared
    assert shared.order().index('compression') < shared.order().index('hooks')


def test_jsonc_strings_are_not_treated_as_comments() -> None:
    name = 'https://host/"quoted"'
    graph = load_dependency_graph('// comment\n' + json.dumps({
        'inputs': [name], 'nodes': {'output': {'depends': [name]}},
    }))
    assert graph.order() == ('output',)


def test_missing_phase_dependency_fails_before_any_compute() -> None:
    graph = DependencyGraph(frozenset(), {'first': (), 'second': ('first',)})
    pipeline = DerivationPipeline(graph)
    seen = []
    pipeline.register('second', lambda: seen.append('second'))
    with pytest.raises(ValueError, match='unfinished products'):
        pipeline.run(('second',))
    assert seen == []
    pipeline.run(('second',), completed=('first',))
    assert seen == ['second']


def test_unimplemented_graph_node_is_rejected() -> None:
    graph = DependencyGraph(frozenset(), {'extra': ()})
    with pytest.raises(ValueError, match='missing compute implementations'):
        graph.require_implementations(())
    graph.require_implementations(('extra',))
