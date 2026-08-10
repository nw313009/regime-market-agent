"""Model access and telemetry (spec C-3).

There is no routing subsystem. Every model call goes through one lightweight abstraction,
``call_model(task, ...)``, and configuration decides which Databricks endpoint that means.

Explicitly not here: model tiers, semantic routing, AI Gateway, escalation graphs, routing
benchmarks, middleware.

THE CONFIG READ LIVES HERE, shared by both modules in the package: ``call_model`` resolves an
endpoint name from ``model.*`` and ``telemetry`` resolves its write mode from ``telemetry.mode``,
and neither takes a config argument in its spec signature. The repository has no central config
loader — every other entry point does its own ``yaml.safe_load`` in a ``main`` — so this is the
same pattern scoped to the one package whose public functions cannot be handed a config.

:func:`set_config` is the injection seam: tests and the Streamlit app can install a mapping
without a file on disk.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "CONFIG_PATH",
    "ConfigError",
    "config_section",
    "load_config",
    "set_config",
]

#: Repository root, derived from this file's location (``src/llm/__init__.py``).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: The project's single config file (spec B0).
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


class ConfigError(RuntimeError):
    """The config file is missing, unreadable, or not a mapping."""


_lock = threading.Lock()
_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None, *, refresh: bool = False) -> Mapping[str, Any]:
    """Read ``config/config.yaml`` once and cache it.

    Cached because a model call is on the interactive path and re-reading a file per call buys
    nothing: the config is static for the life of a process. Pass ``refresh=True`` or call
    :func:`set_config` to replace it.
    """
    global _cache

    with _lock:
        if _cache is not None and not refresh and path is None:
            return _cache

        target = Path(path) if path is not None else CONFIG_PATH
        try:
            with open(target, encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle)
        except OSError as exc:
            raise ConfigError(f"cannot read config file {target}: {type(exc).__name__}") from exc

        if not isinstance(loaded, Mapping):
            raise ConfigError(f"config file {target} did not parse to a mapping")

        _cache = dict(loaded)
        return _cache


def set_config(config: Mapping[str, Any] | None) -> None:
    """Install ``config`` as the cached configuration, or clear the cache with ``None``."""
    global _cache

    with _lock:
        _cache = None if config is None else dict(config)


def config_section(name: str, config: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    """Return a top-level section of the config, or an empty mapping when it is absent.

    Absent-means-empty rather than an error: a caller that needs a specific key reports a better
    message than "section missing" ever could.
    """
    source = config if config is not None else load_config()
    section = source.get(name)
    return section if isinstance(section, Mapping) else {}
