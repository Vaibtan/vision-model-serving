"""Independent physical-TTL janitor for private request and result files."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import signal
from threading import Event

from .storage import EphemeralJobStore


def run_janitor(
    store: EphemeralJobStore,
    *,
    interval_seconds: float,
    stop: Event,
) -> None:
    if interval_seconds <= 0:
        raise ValueError("janitor interval must be positive")
    while not stop.is_set():
        store.cleanup_expired()
        stop.wait(interval_seconds)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep expired private prediction files.")
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    stop = Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    run_janitor(
        EphemeralJobStore(args.job_root),
        interval_seconds=args.interval_seconds,
        stop=stop,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
