"""Independent negative checks for installed artifact verification."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def load_verifier():
    spec = importlib.util.spec_from_file_location('evaluated_verifier', ROOT / 'scripts/verify_wheel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frontend_cli_reports_missing_build_prerequisite(monkeypatch):
    """The executable entry point must fail clearly instead of packaging stale assets."""
    import runpy
    import shutil

    monkeypatch.setattr(shutil, 'which', lambda command: None)
    with pytest.raises(SystemExit, match='Corepack'):
        runpy.run_path(str(ROOT / 'scripts/build_frontend.py'), run_name='__main__')


@pytest.mark.parametrize('failure', ['package-version', 'metadata-version', 'package-path', 'engine-path', 'python-engine', None])
def test_installed_import_guard_detects_contamination(tmp_path, failure):
    verifier = load_verifier()
    # Execute the real guard in a separate interpreter with controlled imports;
    # every corrupted case must fail before any simulation can be trusted.
    setup = f'''
import sys, types, importlib.metadata, importlib.machinery
from pathlib import Path
root = Path({str(tmp_path)!r})
sys.prefix = str(root)
natal = types.ModuleType('natal')
natal.__path__ = []
natal.__version__ = '0.3.0b1'
natal.__file__ = str(root / 'natal/__init__.py')
engine = types.ModuleType('natal._engine_rs')
engine.__file__ = str(root / ('natal/_engine_rs' + importlib.machinery.EXTENSION_SUFFIXES[0]))
natal._engine_rs = engine
sys.modules['natal'] = natal
sys.modules['natal._engine_rs'] = engine
importlib.metadata.version = lambda name: '0.3.0b1'
sys.argv = ['guard', '0.3.0b1']
failure = {failure!r}
if failure == 'package-version': natal.__version__ = '0.2.0b'
if failure == 'metadata-version': importlib.metadata.version = lambda name: '0.2.0b'
if failure == 'package-path': natal.__file__ = str(root.parent / 'checkout/natal/__init__.py')
if failure == 'engine-path': engine.__file__ = str(root.parent / 'checkout/_engine_rs.so')
if failure == 'python-engine': engine.__file__ = str(root / 'natal/_engine_rs.py')
'''
    result = subprocess.run([sys.executable, '-I', '-c', setup + verifier.IMPORT_CHECK], capture_output=True, text=True)
    assert (result.returncode == 0) == (failure is None), result.stderr
    if failure is not None:
        assert 'RuntimeError' in result.stderr


@pytest.mark.parametrize('failed_index', range(5))
def test_install_failures_stop_remaining_checks_and_remove_sandbox(tmp_path, monkeypatch, failed_index):
    verifier = load_verifier()
    sandboxes = []
    monkeypatch.setattr(verifier.venv.EnvBuilder, 'create', lambda self, path: None)

    def run(command, *, cwd, env, check):
        sandboxes.append(cwd)
        if len(sandboxes) - 1 == failed_index:
            raise subprocess.CalledProcessError(23, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(verifier.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        verifier.verify_install(tmp_path / 'candidate.whl', Version('0.3.0b1'))
    assert failure.value.returncode == 23
    assert len(sandboxes) == failed_index + 1
    assert not sandboxes[0].exists()


@pytest.mark.parametrize('damage', [None, 'missing-js', 'html-instead-of-js', 'fallback-page'])
def test_installed_dashboard_guard_rejects_broken_assets(tmp_path, monkeypatch, damage):
    """The actual wheel guard must reject a 200 fallback or broken asset links."""
    import natal.frontend.webui.app as app

    assets = tmp_path / 'assets'
    assets.mkdir()
    (tmp_path / 'index.html').write_text(
        '<div id="app"></div><script src="/assets/app.js"></script>'
        '<link rel="stylesheet" href="/assets/app.css">'
    )
    (assets / 'app.css').write_text('body { color: black; }')
    if damage != 'missing-js':
        (assets / 'app.js').write_text('console.log("ready")')
    if damage == 'html-instead-of-js':
        from starlette.responses import HTMLResponse
        from starlette.routing import Route

        create_app = app.create_app

        async def fallback(request):
            return HTMLResponse('<h1>Wrong fallback</h1>', status_code=200)

        def wrong_content_app(*args, **kwargs):
            result = create_app(*args, **kwargs)
            result.router.routes.insert(0, Route('/assets/app.js', endpoint=fallback))
            return result

        monkeypatch.setattr(app, 'create_app', wrong_content_app)
    if damage == 'fallback-page':
        (tmp_path / 'index.html').write_text('<h1>Frontend build not found</h1>')
    monkeypatch.setattr(app, '_DIST_DIR', tmp_path)
    verifier = load_verifier()
    if damage is None:
        exec(verifier.WEBUI_CHECK, {})
    else:
        with pytest.raises(AssertionError):
            exec(verifier.WEBUI_CHECK, {})


def test_locked_baseline_uses_same_explicit_python_when_running():
    """A repository .python-version must not silently switch the locked venv."""
    import shlex
    import yaml

    workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
    commands = [shlex.split(step['run']) for step in workflow['jobs']['baseline']['steps'] if 'run' in step]
    sync = next(command for command in commands if command[:2] == ['uv', 'sync'])
    run = next(command for command in commands if command[:2] == ['uv', 'run'])
    assert '--locked' in sync
    assert '--no-sync' in run
    assert sync[sync.index('--python') + 1] == run[run.index('--python') + 1]


@pytest.mark.parametrize('inherited', [None, '/foreign/python'])
def test_rust_linking_uses_running_interpreter(monkeypatch, inherited):
    """Linking and the runtime Python library lookup must select one Python."""
    spec = importlib.util.spec_from_file_location('evaluated_rust', ROOT / 'scripts/check_rust.py')
    rust = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rust)
    if inherited is None:
        monkeypatch.delenv('PYO3_PYTHON', raising=False)
    else:
        monkeypatch.setenv('PYO3_PYTHON', inherited)
    assert rust._cargo_env().get('PYO3_PYTHON') == sys.executable


def test_required_rust_gate_rejects_missing_crate(tmp_path, monkeypatch):
    """A missing prerequisite cannot count as a successful required check."""
    spec = importlib.util.spec_from_file_location('evaluated_missing_rust', ROOT / 'scripts/check_rust.py')
    rust = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rust)
    monkeypatch.setattr(rust, 'RUST_DIR', tmp_path / 'absent')
    assert rust.main() != 0


def load_rust():
    spec = importlib.util.spec_from_file_location('evaluated_protocol', ROOT / 'scripts/check_rust.py')
    rust = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rust)
    return rust


def test_lsp_read_reassembles_partial_header_and_body(monkeypatch):
    from types import SimpleNamespace
    rust = load_rust()
    client = rust._LspClient.__new__(rust._LspClient)
    stdout = SimpleNamespace(fileno=lambda: 123)
    client.process = SimpleNamespace(stdout=stdout)
    client.buffer = b''
    chunks = iter([b'Content-Len', b'gth: 8\r\n\r\n{"id', b'":1}'])
    monkeypatch.setattr(rust.select, 'select', lambda readers, *args: (readers, [], []))
    monkeypatch.setattr(rust.os, 'read', lambda descriptor, count: next(chunks))
    assert client.read(2) == {'id': 1}
    assert client.buffer == b''


@pytest.mark.parametrize('value', [None, 1, {'key': 'value'}])
def test_vscode_expansion_preserves_nonstring_json(tmp_path, value):
    rust = load_rust()
    assert rust._expand_vscode_variable(value, tmp_path) is value
    assert rust._expand_vscode_variable('${workspaceFolder}/src', tmp_path) == f'{tmp_path}/src'


def test_analyzer_environment_expands_strings_and_stringifies_json(tmp_path, monkeypatch):
    from types import SimpleNamespace
    rust = load_rust()
    environments = []
    monkeypatch.setattr(rust, 'ROOT_DIR', tmp_path)
    monkeypatch.setattr(rust, '_load_rust_analyzer_config', lambda: {'server.extraEnv': {'ROOT': '${workspaceFolder}', 'COUNT': 2}})

    class ExitedClient:
        process = SimpleNamespace(poll=lambda: 1)

        def __init__(self, analyzer, env):
            environments.append(env)

        def send(self, message):
            pass

        def close(self):
            pass

    monkeypatch.setattr(rust, '_LspClient', ExitedClient)
    assert rust._rust_analyzer_gate(tmp_path / 'analyzer') == 0
    assert environments[0]['ROOT'] == str(tmp_path)
    assert environments[0]['COUNT'] == '2'
