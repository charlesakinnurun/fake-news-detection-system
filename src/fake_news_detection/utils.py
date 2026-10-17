"""Small, dependency-light helpers shared across the package.

Everything here is deterministic and free of global state so that it can be
unit tested in isolation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def set_global_seeds(seed: int) -> None:
    """Seed every RNG that can influence a training run.

    ``PYTHONHASHSEED`` cannot be changed once the interpreter has started, so
    it is documented in the README instead of being set here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Time / files
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a :class:`Path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def display_path(path: str | Path, base: str | Path | None = None) -> str:
    """Return ``path`` relative to ``base`` when possible, else the full path.

    CLI flags allow output directories outside the repository; logging must not
    fail just because a path cannot be made relative.
    """
    candidate = Path(path)
    if base is None:
        return str(candidate)
    try:
        return str(candidate.resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(candidate)


def human_bytes(num_bytes: float) -> str:
    """Format a byte count using binary units (``B``, ``KiB``, ``MiB`` ...)."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.2f} TiB"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file without loading it in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    """Compute the SHA-256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Compute the SHA-256 hex digest of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def stable_int_hash(text: str, bits: int = 64) -> int:
    """Return a process-independent integer hash of ``text``.

    Python's built-in :func:`hash` is randomised per process for strings, which
    would make duplicate detection irreproducible. ``blake2b`` is stable across
    runs and platforms.
    """
    size = bits // 8
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=size).digest()
    return int.from_bytes(digest, "big", signed=False)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    """Write ``payload`` as UTF-8 JSON, creating parent directories as needed.

    ``allow_nan=False`` makes a non-finite metric (``NaN``/``Infinity``) fail
    loudly here instead of producing a file that is not valid JSON and that a
    downstream consumer may silently mis-parse.
    """
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False, default=str, allow_nan=False),
        "utf-8",
    )
    return path


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""
    return json.loads(Path(path).read_text("utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def to_jsonable(payload: Any) -> Any:
    """Recursively convert numpy/pandas scalars to plain Python objects."""
    return json.loads(json.dumps(payload, default=_json_default))


# ---------------------------------------------------------------------------
# Environment introspection
# ---------------------------------------------------------------------------

_CORE_PACKAGES = (
    "python",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "pydantic",
    "fastapi",
    "uvicorn",
    "streamlit",
    "matplotlib",
)


def package_versions(names: tuple[str, ...] = _CORE_PACKAGES) -> dict[str, str]:
    """Return a ``{distribution: version}`` map for the installed packages."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for name in names:
        if name == "python":
            versions["python"] = platform.python_version()
            continue
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def environment_summary() -> dict[str, Any]:
    """Describe the host in a way that is safe to publish in metadata."""
    import platform

    return {
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


@contextmanager
def timer() -> Iterator[_TimerResult]:
    """Context manager measuring wall-clock duration.

    ``with timer() as t: ...`` then read ``t.seconds``.
    """
    result = _TimerResult()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


class _TimerResult:
    """Mutable holder written by :func:`timer` on exit."""

    __slots__ = ("seconds",)

    def __init__(self) -> None:
        self.seconds = 0.0


def percentile(values: list[float], q: float) -> float:
    """Return the ``q``-th percentile (0-100) of ``values`` as a float."""
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype="float64"), q))


def truncate(text: str, limit: int = 160) -> str:
    """Collapse whitespace and truncate ``text`` for logging/error messages."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "…"
