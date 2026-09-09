"""Resilient access helpers for transient nflverse network failures.

These wrappers do not change data or modeling semantics. They only retry transient
transport/server failures so a short nflverse/GitHub outage does not knock out a
near-kick production workflow.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar
from urllib.error import URLError

import nflreadpy as nfl

T = TypeVar("T")

# HTTPError (including 5xx responses such as the observed 502) subclasses
# URLError. OSError also covers lower-level socket/transport failures surfaced by
# urllib on some runners. Data/schema errors such as ValueError are intentionally
# not retried.
TRANSIENT_ERRORS = (URLError, TimeoutError, ConnectionError, OSError)


def with_network_retry(
    func: Callable[[], T],
    *,
    label: str,
    attempts: int = 4,
    initial_delay_seconds: float = 2.0,
) -> T:
    """Run a network-backed operation with bounded exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(1, attempts + 1):
        try:
            return func()
        except TRANSIENT_ERRORS as exc:
            if attempt >= attempts:
                print(
                    f"{label} failed after {attempts} attempt(s): "
                    f"{type(exc).__name__}: {exc}"
                )
                raise
            delay = initial_delay_seconds * (2 ** (attempt - 1))
            print(
                f"Transient failure loading {label} on attempt {attempt}/{attempts}: "
                f"{type(exc).__name__}: {exc}. Retrying in {delay:.1f}s."
            )
            time.sleep(delay)

    raise RuntimeError("unreachable retry state")


def load_schedules(seasons, *, attempts: int = 4):
    """Load nflverse schedules with retry protection for transient failures."""
    return with_network_retry(
        lambda: nfl.load_schedules(seasons),
        label=f"nflverse schedules {seasons}",
        attempts=attempts,
    )


def load_pbp(seasons, *, attempts: int = 4):
    """Load nflverse play-by-play with retry protection for transient failures."""
    return with_network_retry(
        lambda: nfl.load_pbp(seasons),
        label=f"nflverse PBP {seasons}",
        attempts=attempts,
    )
