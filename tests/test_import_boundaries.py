"""Import-boundary tests: which packages may reach which dependencies.

These are static import-graph checks, not runtime ones, because the failures they prevent are
not catchable at runtime.

PSYCOPG KILLS THE SERVERLESS NOTEBOOK KERNEL (environment fact, C-1). ``psycopg[binary]`` 3.3.4
aborts at IMPORT time on serverless compute: the libpq extension abort fires inside
``psycopg/pq/__init__.py`` ``import_from_libpq`` and the kernel exits 134 before any of our code
runs. There is no try/except that survives a SIGABRT, so the only defence is never importing it
there — which makes ``src/database/lakebase.py`` app-container-only and puts every pipeline and
modeling module off limits.

The check is TRANSITIVE. A direct ``import psycopg`` in a pipeline module is the easy case and
nobody writes it; the realistic regression is ``src/pipelines/x.py`` importing a helper that
imports ``src.database.lakebase`` three hops away — for instance wiring the watchlist read into
the ingestion universe, which ``src.ingestion.resolve_universe`` has a TODO inviting. So the walk
follows first-party ``src.*`` edges to exhaustion and reports the path it found.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

#: Packages that run on a cluster or in a notebook kernel, and therefore must never reach psycopg.
CLUSTER_PACKAGES = ("src.ingestion", "src.pipelines", "src.models")

FORBIDDEN_ON_CLUSTER = "psycopg"


def _module_name(path: Path) -> str:
    relative = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _imports_of(path: Path, *, module_level_only: bool = False) -> set[str]:
    """Every module name imported by ``path``, absolute and relative alike.

    ``module_level_only`` restricts the scan to imports that run at import time, which is the
    distinction the lazy imports in ``src/agent/tools.py`` rely on: importing that module must
    not drag psycopg in, even though two of its functions use it.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _module_name(path)
    if path.name != "__init__.py":
        package = package.rsplit(".", 1)[0]

    nodes = tree.body if module_level_only else ast.walk(tree)
    found: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # a relative import, resolved against the containing package
                base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                found.add(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                found.add(node.module)
                # `from src.database import lakebase` imports a submodule, not just the package.
                found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def _source_for(module: str) -> Path | None:
    """The file backing a first-party module name, if there is one."""
    relative = Path(*module.split("."))
    for candidate in (REPO_ROOT / relative.with_suffix(".py"), REPO_ROOT / relative / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _import_path_to(
    start_package: str,
    target: str,
    *,
    module_level_only: bool = False,
) -> list[str] | None:
    """Breadth-first walk of first-party imports from ``start_package``, returning the first path
    that reaches a module whose name starts with ``target``, or ``None``."""
    roots = [
        _module_name(path)
        for path in sorted((REPO_ROOT / Path(*start_package.split("."))).rglob("*.py"))
    ]
    queue: list[list[str]] = [[root] for root in roots]
    seen: set[str] = set(roots)

    while queue:
        chain = queue.pop(0)
        source = _source_for(chain[-1])
        if source is None:
            continue
        for imported in sorted(_imports_of(source, module_level_only=module_level_only)):
            root = imported.split(".")[0]
            if imported == target or root == target:
                return [*chain, imported]
            if root != "src" or imported in seen:
                continue
            seen.add(imported)
            queue.append([*chain, imported])
    return None


@pytest.mark.parametrize("package", CLUSTER_PACKAGES)
def test_cluster_packages_never_reach_psycopg(package):
    """No chain of first-party imports may take a cluster module to psycopg (C-1).

    The failure this prevents is not an exception: the serverless kernel SIGABRTs at import and
    the task dies with exit 134 and no traceback of ours.
    """
    chain = _import_path_to(package, FORBIDDEN_ON_CLUSTER)

    assert chain is None, (
        f"{package} can reach {FORBIDDEN_ON_CLUSTER} via {' -> '.join(chain)}. psycopg aborts the "
        "serverless notebook kernel at import time (exit 134), so Lakebase access is "
        "app-container-only."
    )


def test_importing_the_agent_does_not_import_psycopg():
    """``src/agent/tools.py`` may USE Lakebase; it may not import it at module level (C-1).

    The two write tools need Postgres and the two read tools do not, so the import sits inside
    the functions that need it. That keeps ``import src.agent.tools`` — for the tool schemas, or
    from a notebook — safe on serverless compute, where the import itself would kill the kernel.
    """
    chain = _import_path_to("src.agent", FORBIDDEN_ON_CLUSTER, module_level_only=True)

    assert chain is None, (
        f"src.agent reaches {FORBIDDEN_ON_CLUSTER} at import time via {' -> '.join(chain)}. "
        "Import lakebase inside the function that needs it."
    )


def test_lakebase_is_the_only_module_importing_psycopg():
    """One module owns the dependency, so the boundary has exactly one place to check."""
    importers = {
        _module_name(path)
        for path in sorted(SRC_ROOT.rglob("*.py"))
        if re.search(rf"^\s*(import|from)\s+{FORBIDDEN_ON_CLUSTER}", path.read_text(encoding="utf-8"), re.M)
    }

    assert importers == {"src.database.lakebase"}


def test_psycopg_is_absent_from_the_cluster_requirements():
    """The install list has to agree with the boundary, or the next reader re-adds it (C-1)."""
    text = (REPO_ROOT / "requirements-databricks.txt").read_text(encoding="utf-8")
    installs = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert not any(line.lower().startswith("psycopg") for line in installs), installs
    # It stays in the local venv and in the app, which are the two places it works.
    for path in ("requirements.txt", "app/requirements.txt"):
        other = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert re.search(r"^psycopg", other, re.M), f"{path} should still install psycopg"
