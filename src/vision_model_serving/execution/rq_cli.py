"""Command-line adapter for the standard forked RQ prediction worker."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .rq_worker import create_prediction_rq_worker


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the standard forked RQ prediction worker.",
    )
    parser.add_argument("--redis-url", default="redis://127.0.0.1:6379/0")
    parser.add_argument("--queue-name", default="gpu-inference")
    parser.add_argument("--executor-socket", type=Path, required=True)
    parser.add_argument("--executor-timeout-seconds", type=float, required=True)
    parser.add_argument("--burst", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--logging-level", default="INFO")
    args = parser.parse_args(argv)

    from redis import Redis

    from vision_model_serving.observability import configure_structured_logging

    configure_structured_logging(args.logging_level)
    redis_client = Redis.from_url(args.redis_url)
    redis_client.ping()
    worker = create_prediction_rq_worker(
        redis_client=redis_client,
        queue_name=args.queue_name,
        executor_socket_path=args.executor_socket,
        executor_timeout_seconds=args.executor_timeout_seconds,
    )
    worker.work(
        burst=args.burst,
        max_jobs=args.max_jobs,
        logging_level=args.logging_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
