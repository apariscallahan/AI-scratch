"""Where NeuroBlocks keeps projects, data files, saved models, runs and caches.

Everything lives under one "home" folder (default ``~/NeuroBlocks``; override
with the ``NEUROBLOCKS_HOME`` environment variable). Cloud bundles set the home to
the bundle folder so that ``data/`` and ``models/`` travel with the job.
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
EXAMPLES_DIR = PACKAGE_DIR / "examples"
ASSETS_DIR = PACKAGE_DIR / "runtime" / "assets"


def home() -> Path:
    env = os.environ.get("NEUROBLOCKS_HOME")
    path = Path(env).expanduser() if env else Path.home() / "NeuroBlocks"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def sub(name: str) -> Path:
    path = home() / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def projects_dir() -> Path:
    return sub("projects")


def data_dir() -> Path:
    return sub("data")


def models_dir() -> Path:
    return sub("models")


def runs_dir() -> Path:
    return sub("runs")


def cache_dir() -> Path:
    return sub("cache")


def resolve_data_file(path: str) -> Path:
    """Find a data file the user referred to by name or path."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    candidates = [home() / p, data_dir() / p, Path.cwd() / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    return (home() / p) if len(p.parts) > 1 else (data_dir() / p)


def resolve_model_file(path: str, for_writing: bool = False) -> Path:
    """Model files without a folder go to ``models/``; other relative paths are relative to home."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    if len(p.parts) == 1:
        target = models_dir() / p
        if for_writing or target.exists():
            return target
        for c in (home() / p, Path.cwd() / p):
            if c.exists():
                return c.resolve()
        return target
    target = home() / p
    if not for_writing and not target.exists() and (Path.cwd() / p).exists():
        return (Path.cwd() / p).resolve()
    return target
