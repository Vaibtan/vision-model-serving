"""Gunicorn configuration for the packaged web image."""

import os

bind = "0.0.0.0:8000"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = 60
graceful_timeout = 30
# Workers stay persistent; recycling would leak prometheus multiprocess shards.
max_requests = 0
# The application emits sanitized telemetry; the default access log would leak
# capability prediction identifiers in request lines.
accesslog = None
errorlog = "-"


def child_exit(server, worker):
    """Reap the prometheus multiprocess shard of a dead gunicorn worker."""
    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker.pid)
    except Exception:
        # Metrics may be disabled or prometheus_client absent; never block reaping.
        pass
