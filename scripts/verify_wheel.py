#!/usr/bin/env python3
"""Validate release identity and test an exact wheel outside the source tree."""

from __future__ import annotations

import argparse
import ast
import email
import importlib.machinery
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

from packaging.tags import sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version

ROOT_DIR = Path(__file__).resolve().parents[1]
WHEEL_TESTS = (
    "test_complex_genetics_e2e.py",
    "test_complex_population_e2e.py",
    "test_runtime_updater_contracts.py",
)

IMPORT_CHECK = """
import importlib.metadata
import importlib.machinery
from pathlib import Path
import sys
from packaging.version import Version
import natal
import natal._engine_rs as engine
expected = Version(sys.argv[1])
if Version(natal.__version__) != expected or Version(importlib.metadata.version('natal-core')) != expected:
    raise RuntimeError('Installed package version does not match the wheel')
for module in (natal, engine):
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f'Import escaped the isolated environment: {path}')
    print(f'Imported {module.__name__} from {path}')
if not any(str(engine.__file__).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
    raise RuntimeError('Rust backend is not a native extension')
"""

WEBUI_CHECK = """
import asyncio
from html.parser import HTMLParser
import httpx
import natal as nt
from natal.frontend.webui.app import create_app

class Assets(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if (tag, name) in {('script', 'src'), ('link', 'href')} and value:
                self.paths.append(value)

async def check():
    species = nt.Species.from_dict(name='wheel-ui', structure={'chr1': {'loc1': ['WT']}})
    population = (nt.DiscreteGenerationPopulation.setup(species=species, name='wheel-ui')
        .initial_state(individual_count={'male': {'WT|WT': 10}, 'female': {'WT|WT': 10}})
        .reproduction(eggs_per_female=2)
        .competition(carrying_capacity=100, juvenile_growth_mode='fixed').build())
    app = create_app(population)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://wheel') as client:
        page = await client.get('/')
        assert page.status_code == 200 and '<div id="app">' in page.text, page.text
        assets = Assets()
        assets.feed(page.text)
        assert any(path.endswith('.js') for path in assets.paths), 'Missing compiled JavaScript'
        assert any(path.endswith('.css') for path in assets.paths), 'Missing compiled stylesheet'
        for path in assets.paths:
            response = await client.get(path)
            assert response.status_code == 200 and response.content, path
            assert 'text/html' not in response.headers['content-type'], path
        meta = await client.get('/api/meta')
        assert meta.status_code == 200 and meta.json()['population_name'] == 'wheel-ui'
        assert 'backend' not in meta.json()
    print('Installed dashboard HTML, linked assets and API passed')

asyncio.run(check())
"""


def project_version(root: Path = ROOT_DIR) -> Version:
    """Require the distribution and import version declarations to agree."""
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    expected = Version(metadata["project"]["version"])
    tree = ast.parse((root / "src/natal/__init__.py").read_text(encoding="utf-8"))
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign)
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
    ]
    if len(versions) != 1 or Version(versions[0]) != expected:
        raise ValueError("pyproject.toml and natal.__version__ must agree")
    return expected


def check_tag(tag: str, expected: Version) -> None:
    """Require a version tag matching the package, including beta normalization."""
    if not tag.startswith("v") or Version(tag[1:]) != expected:
        raise ValueError(f"Tag {tag!r} does not match package version {expected}")


def wheel_identity(directory: Path, expected: Version) -> Path:
    """Require exactly one compatible wheel with matching metadata and native code."""
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError(f"Expected exactly one wheel in {directory}; found {len(wheels)}")
    wheel = wheels[0].resolve()
    name, version, _, tags = parse_wheel_filename(wheel.name)
    if name != "natal-core" or version != expected or not (tags & set(sys_tags())):
        raise ValueError(f"Wrong package, version, or interpreter/platform tags: {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("Wheel must have exactly one distribution metadata record")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if (canonicalize_name(metadata.get("Name", "")) != "natal-core"
                or Version(metadata.get("Version", "0")) != expected):
            raise ValueError("Wheel metadata does not match the project version/name")
        if not any(f"natal/_engine_rs{suffix}" in names
                   for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            raise ValueError("Wheel has no native natal._engine_rs extension")
        prefix = "natal/frontend/webui/dist/"
        if (prefix + "index.html" not in names
                or not any(name.startswith(prefix + "assets/") and name.endswith(".js")
                           for name in names)):
            raise ValueError("Wheel has no compiled dashboard assets")
    return wheel


def verify_install(wheel: Path, expected: Version) -> None:
    """Install and test the wheel in a fresh venv, without checkout pytest config.

    Reuse independent numerical E2E assertions instead of maintaining a second
    set of simulation expectations specifically for packaging.
    """
    with tempfile.TemporaryDirectory(prefix="natal-wheel-check-") as temporary:
        sandbox = Path(temporary)
        environment = sandbox / "venv"
        venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONHOME", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(key, None)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        subprocess.run(
            [str(python), "-I", "-m", "pip", "install", str(wheel), "pytest>=7",
             "packaging>=24", "httpx>=0.27,<0.29"],
            cwd=sandbox, env=env, check=True,
        )
        subprocess.run(
            [str(python), "-I", "-m", "pip", "check"], cwd=sandbox, env=env, check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", IMPORT_CHECK, str(expected)],
            cwd=sandbox, env=env, check=True,
        )
        subprocess.run(
            [str(python), "-I", "-c", WEBUI_CHECK], cwd=sandbox, env=env, check=True,
        )
        for name in WHEEL_TESTS:
            shutil.copyfile(ROOT_DIR / "tests" / name, sandbox / name)
        config = sandbox / "pytest.ini"
        config.write_text("[pytest]\n", encoding="utf-8")
        subprocess.run(
            [str(python), "-I", "-m", "pytest", "-q", "-c", str(config),
             "--confcutdir", str(sandbox), *WHEEL_TESTS],
            cwd=sandbox, env=env, check=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Validate a tag, a wheel, or both; any mismatch prevents publication."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--check-tag")
    args = parser.parse_args(argv)
    if args.wheel_dir is None and args.check_tag is None:
        parser.error("provide --wheel-dir and/or --check-tag")
    try:
        expected = project_version()
        if args.check_tag is not None:
            check_tag(args.check_tag, expected)
        if args.wheel_dir is not None:
            wheel = wheel_identity(args.wheel_dir, expected)
            verify_install(wheel, expected)
            print(f"Wheel installation and E2E checks passed: {wheel}")
        return 0
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"Release verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
