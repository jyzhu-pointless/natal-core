"""Contracts for CI failure propagation, artifact identity, and release gating."""

from __future__ import annotations

import ast
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

from packaging.tags import sys_tags
from packaging.version import Version
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ci():
    return _load('ci_full')


@pytest.fixture
def verifier():
    return _load('verify_wheel')


def test_all_stages_and_interpreter_are_shared(ci, monkeypatch):
    commands = []
    monkeypatch.setattr(ci.subprocess, 'run', lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=0))
    assert ci.main([]) == 0
    assert len(commands) == 7
    assert all(command[0] == sys.executable for command in commands)
    assert any('scripts/generate_init_pyi.py' in command and '--check' in command for command in commands)
    assert any('scripts/phase0_baseline.py' in command and '--check' in command for command in commands)
    assert any('scripts/build_rust_wheel.py' in command for command in commands)


def test_selected_stages_stop_on_failure(ci, monkeypatch):
    commands = []
    monkeypatch.setattr(ci.subprocess, 'run', lambda command, **kwargs: commands.append(command) or SimpleNamespace(returncode=7))
    assert ci.main(['--only', 'tests', 'baseline']) == 7
    assert len(commands) == 1
    assert 'pytest' in commands[0]


@pytest.mark.parametrize('args', [['--only'], ['--only', 'typo'], ['--only', 'tests', '--skip-tests']])
def test_stage_selection_errors_cannot_succeed(ci, args):
    with pytest.raises(SystemExit) as error:
        ci.main(args)
    assert error.value.code == 2


def test_legacy_skip_flags_do_not_skip_new_required_checks(ci, monkeypatch):
    seen = []
    monkeypatch.setattr(ci, 'run_stage', lambda name, command: seen.append(name) or 0)
    assert ci.main(['--skip-rust', '--skip-tests', '--skip-pyright', '--skip-ruff', '--skip-wheel']) == 0
    assert seen == ['stubs', 'baseline']


def _project(tmp_path, metadata='0.3.0b1', source='0.3.0b1'):
    (tmp_path / 'pyproject.toml').write_text(f'[project]\nversion="{metadata}"\n')
    package = tmp_path / 'src/natal'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text(f'__version__ = "{source}"\n')
    return tmp_path


def test_project_version_is_read_without_importing_package(verifier, tmp_path):
    root = _project(tmp_path, '0.3.0b', '0.3.0b0')
    with (root / 'src/natal/__init__.py').open('a') as stream:
        stream.write("raise RuntimeError('must not import the source package')\n")
    assert verifier.project_version(root) == Version('0.3.0b0')


@pytest.mark.parametrize('source', ['0.2.0b', '0.3.0b2'])
def test_mismatched_source_version_fails(verifier, tmp_path, source):
    with pytest.raises(ValueError, match='must agree'):
        verifier.project_version(_project(tmp_path, source=source))


@pytest.mark.parametrize('tag,valid', [('v0.3.0b', True), ('v0.3.0b0', True), ('0.3.0b0', False), ('v0.3.0b1', False), ('vbad', False)])
def test_release_tag_contract(verifier, tag, valid):
    if valid:
        verifier.check_tag(tag, Version('0.3.0b0'))
    else:
        with pytest.raises(ValueError):
            verifier.check_tag(tag, Version('0.3.0b0'))


def _wheel(tmp_path, *, version='0.3.0b1', metadata_version=None, extension=True, tag=None, name='natal_core', frontend=True):
    tag = tag or str(next(sys_tags()))
    path = tmp_path / f'{name}-{version}-{tag}.whl'
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr(f'{name}-{version}.dist-info/METADATA', f'Name: {name}\nVersion: {metadata_version or version}\n')
        suffix = importlib.machinery.EXTENSION_SUFFIXES[0] if extension else '.py'
        archive.writestr(f'natal/_engine_rs{suffix}', b'not executed in identity tests')
        if frontend:
            archive.writestr('natal/frontend/webui/dist/index.html', '<div id="app"></div>')
            archive.writestr('natal/frontend/webui/dist/assets/index.js', 'console.log("built")')
    return path


def test_wheel_identity_returns_exact_artifact(verifier, tmp_path):
    wheel = _wheel(tmp_path)
    assert verifier.wheel_identity(tmp_path, Version('0.3.0b1')) == wheel.resolve()


@pytest.mark.parametrize('options', [
    {'version': '0.3.0b2'}, {'metadata_version': '0.3.0b2'},
    {'extension': False}, {'tag': 'cp27-cp27m-win32'}, {'name': 'other'}, {'frontend': False},
])
def test_wrong_wheels_fail_before_install(verifier, tmp_path, options):
    _wheel(tmp_path, **options)
    with pytest.raises(ValueError):
        verifier.wheel_identity(tmp_path, Version('0.3.0b1'))


@pytest.mark.parametrize('count', [0, 2])
def test_empty_or_stale_mixed_output_fails(verifier, tmp_path, count):
    for index in range(count):
        _wheel(tmp_path, version=f'0.3.0b{index}')
    with pytest.raises(ValueError, match='exactly one wheel'):
        verifier.wheel_identity(tmp_path, Version('0.3.0b1'))


def test_bad_metadata_layout_fails(verifier, tmp_path):
    path = _wheel(tmp_path)
    with zipfile.ZipFile(path, 'a') as archive:
        archive.writestr('other.dist-info/METADATA', 'Name: other\nVersion: 1\n')
    with pytest.raises(ValueError, match='metadata record'):
        verifier.wheel_identity(tmp_path, Version('0.3.0b1'))


def test_isolated_install_scrubs_paths_and_uses_copied_tests(verifier, tmp_path, monkeypatch):
    wheel = _wheel(tmp_path)
    commands = []
    monkeypatch.setenv('PYTHONPATH', str(ROOT / 'src'))
    monkeypatch.setenv('PYTEST_ADDOPTS', '--ignore=everything')
    monkeypatch.setattr(verifier.venv.EnvBuilder, 'create', lambda self, path: None)

    def run(command, *, cwd, env, check):
        assert check is True
        assert '-I' in command
        assert not cwd.is_relative_to(ROOT)
        assert 'PYTHONPATH' not in env and 'PYTEST_ADDOPTS' not in env
        assert env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] == '1'
        commands.append(command)
        if 'pytest' in command:
            assert (cwd / 'pytest.ini').read_text() == '[pytest]\n'
            for name in verifier.WHEEL_TESTS:
                assert (cwd / name).read_bytes() == (ROOT / 'tests' / name).read_bytes()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(verifier.subprocess, 'run', run)
    verifier.verify_install(wheel, Version('0.3.0b1'))
    assert str(wheel) in commands[0]
    assert any('check' in command and 'pip' in command for command in commands)
    assert any(verifier.IMPORT_CHECK in command for command in commands)
    assert any(verifier.WEBUI_CHECK in command for command in commands)
    assert 'pytest' in commands[-1]


def test_verification_cli_propagates_install_failure(verifier, tmp_path, monkeypatch):
    _wheel(tmp_path)
    monkeypatch.setattr(verifier, 'project_version', lambda: Version('0.3.0b1'))
    def fail(*args):
        raise subprocess.CalledProcessError(17, ['pip'])
    monkeypatch.setattr(verifier, 'verify_install', fail)
    assert verifier.main(['--wheel-dir', str(tmp_path)]) == 1


def test_verification_cli_requires_work_and_accepts_tag_only(verifier, monkeypatch):
    with pytest.raises(SystemExit) as error:
        verifier.main([])
    assert error.value.code == 2
    monkeypatch.setattr(verifier, 'project_version', lambda: Version('0.3.0b1'))
    assert verifier.main(['--check-tag', 'v0.3.0b1']) == 0


@pytest.mark.parametrize('failed_stage', [0, 1, 2, 3, None])
def test_build_never_verifies_or_installs_after_a_failure(tmp_path, monkeypatch, failed_stage):
    builder = _load('build_rust_wheel')
    commands = []
    output = tmp_path / 'new-build'
    def run(command, **kwargs):
        index = len(commands)
        commands.append(command)
        if index == 1:
            _wheel(output)
        return SimpleNamespace(returncode=19 if index == failed_stage else 0)
    monkeypatch.setattr(builder.subprocess, 'run', run)
    assert builder.main(['--out', str(output), '--install']) == (19 if failed_stage is not None else 0)
    assert len(commands) == (failed_stage + 1 if failed_stage is not None else 4)
    assert commands[0] == [sys.executable, 'scripts/build_frontend.py']
    if failed_stage != 0:
        assert commands[1][commands[1].index('--interpreter') + 1] == sys.executable
        assert '--locked' in commands[1]


@pytest.mark.parametrize('failed_stage', [0, 1, 2, 3, None])
def test_frontend_build_stops_on_failed_check(monkeypatch, failed_stage):
    builder = _load('build_frontend')
    commands = []
    monkeypatch.setattr(builder.shutil, 'which', lambda name: '/tools/corepack')
    def run(command, **kwargs):
        index = len(commands)
        commands.append(command)
        assert kwargs['cwd'] == ROOT / 'ui'
        return SimpleNamespace(returncode=9 if index == failed_stage else 0)
    monkeypatch.setattr(builder.subprocess, 'run', run)
    assert builder.main() == (9 if failed_stage is not None else 0)
    assert len(commands) == (failed_stage + 1 if failed_stage is not None else 4)
    assert commands[0][-2:] == ['install', '--frozen-lockfile']
    if failed_stage is None:
        assert '--emptyOutDir' in commands[-1]


def test_frontend_build_requires_corepack(monkeypatch):
    builder = _load('build_frontend')
    monkeypatch.setattr(builder.shutil, 'which', lambda name: None)
    with pytest.raises(SystemExit, match='Corepack'):
        builder.main()


def test_build_rejects_old_output_before_running_commands(tmp_path):
    builder = _load('build_rust_wheel')
    with pytest.raises(SystemExit) as error:
        builder.main(['--out', str(tmp_path)])
    assert error.value.code == 2


def test_default_build_outputs_are_unique(tmp_path, monkeypatch):
    builder = _load('build_rust_wheel')
    monkeypatch.setattr(builder, 'ROOT_DIR', tmp_path)
    monkeypatch.setenv('CARGO_TARGET_DIR', str(tmp_path / 'shared-cargo-cache'))
    outputs = []
    def run(command, **kwargs):
        if '--out' in command:
            outputs.append(command[command.index('--out') + 1])
        assert kwargs['env']['CARGO_TARGET_DIR'] == str(tmp_path / 'shared-cargo-cache')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(builder.subprocess, 'run', run)
    assert builder.main([]) == builder.main([]) == 0
    assert len(set(outputs)) == 2


def test_stub_check_detects_drift_without_writing(tmp_path, monkeypatch):
    generator = _load('generate_init_pyi')
    monkeypatch.setattr(generator, 'ROOT_DIR', tmp_path)
    output = tmp_path / '__init__.pyi'
    output.write_text('user changes\n')
    monkeypatch.setattr(generator, 'OUTPUT_FILE', output)
    monkeypatch.setattr(generator, 'iter_public_modules', lambda path: [('unit', ['Export'])])
    monkeypatch.setattr(generator.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='formatted\n'))
    assert generator.main(['--check']) == 1
    assert output.read_text() == 'user changes\n'
    assert generator.main([]) == 0
    assert generator.main(['--check']) == 0


def test_stub_formatter_failure_never_overwrites(tmp_path, monkeypatch):
    generator = _load('generate_init_pyi')
    output = tmp_path / '__init__.pyi'
    output.write_text('existing\n')
    monkeypatch.setattr(generator, 'OUTPUT_FILE', output)
    monkeypatch.setattr(generator.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=2, stderr='formatter failed'))
    assert generator.main([]) == 2
    assert output.read_text() == 'existing\n'


def test_stub_imports_follow_runtime_canonical_owners():
    import natal
    generator = _load('generate_init_pyi')
    modules = generator.iter_public_modules(generator.PACKAGE_DIR)
    names = [name for _, names in modules for name in names]
    assert len(names) == len(set(names))
    for module, exports in modules:
        assert all(natal._lazy_map[name] == module for name in exports)
    assert set(names) == {name for exports in natal._PUBLIC_EXPORTS.values() for name in exports}
    assert ast.parse(generator.render_stub(modules))


def _workflow(name):
    return yaml.load((ROOT / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader)


@pytest.mark.parametrize('event', ['push', 'workflow_dispatch', 'pull_request'])
@pytest.mark.parametrize('ref', ['refs/tags/v0.3.0b1', 'refs/heads/main'])
@pytest.mark.parametrize('dry_run', [True, False])
def test_publish_guard_blocks_dry_runs_and_non_release_events(event, ref, dry_run):
    workflow = _workflow('wheels.yml')
    expression = workflow['jobs']['publish']['if'].replace('&&', ' and ').replace('||', ' or ').replace('!', 'not ')
    github = SimpleNamespace(ref=ref, event_name=event)
    actual = eval(expression, {'__builtins__': {}}, {
        'github': github, 'inputs': SimpleNamespace(dry_run=dry_run),
        'startsWith': str.startswith,
    })
    expected = ref.startswith('refs/tags/v') and (event == 'push' or (event == 'workflow_dispatch' and not dry_run))
    assert actual == expected
    assert workflow['on']['workflow_dispatch']['inputs']['dry_run']['type'] == 'boolean'
    assert workflow['on']['workflow_dispatch']['inputs']['dry_run']['default'] == 'true'


@pytest.mark.parametrize('status', ['success', 'failure', 'skipped', 'cancelled'])
def test_summary_gate_rejects_any_unsuccessful_required_job(status):
    workflow = _workflow('ci.yml')
    gate = workflow['jobs']['ci-success']
    script = gate['steps'][0]['run'].split("<<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
    results = {name: {'result': 'success'} for name in gate['needs']}
    results['wheel-build']['result'] = status
    result = subprocess.run([sys.executable, '-c', script], env={**os.environ, 'RESULTS': json.dumps(results)}, capture_output=True)
    assert (result.returncode == 0) == (status == 'success')


def test_workflow_release_reuses_verified_artifacts_and_full_matrix():
    ci = _workflow('ci.yml')
    wheels = _workflow('wheel-build.yml')
    release = _workflow('wheels.yml')
    assert 'workflow_call' in ci['on']
    assert release['jobs']['checks']['uses'] == './.github/workflows/ci.yml'
    assert set(release['jobs']['publish']['needs']) == {'checks', 'release-identity'}
    assert ci['jobs']['wheel-build']['uses'] == './.github/workflows/wheel-build.yml'
    matrix = wheels['jobs']['build']['strategy']['matrix']
    assert matrix['python-version'] == ci['jobs']['test']['strategy']['matrix']['python-version']
    assert matrix['python-version'] == ['3.10', '3.11', '3.12', '3.13']
    assert len(matrix['os']) == 5 and 'macos-13' not in matrix['os']
    steps = wheels['jobs']['build']['steps']
    build = next(step for step in steps if step.get('uses', '').startswith('PyO3/maturin-action'))
    assert "runner.os == 'Linux'" in build['with']['args']
    assert "format('python{0}', matrix.python-version)" in build['with']['args']
    assert "|| 'python'" in build['with']['args']
    verify_index = next(i for i, step in enumerate(steps) if 'verify_wheel.py' in step.get('run', ''))
    upload_index = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('actions/upload-artifact'))
    assert verify_index < upload_index
    assert release['jobs']['publish']['steps'][0]['with']['pattern'] == 'verified-wheel-*'
    assert not any('maturin' in str(step) for step in release['jobs']['publish']['steps'])
