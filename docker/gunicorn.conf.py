"""Gunicorn configuration for the packaged web image."""

import os

bind = "0.0.0.0:8000"
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = 60
graceful_timeout = 30
# Keep worker lifetime stable; child_exit still removes every exact-PID metrics
# shard after an unexpected exit or operator-driven restart.
max_requests = 0
# The application emits sanitized telemetry; the default access log would leak
# capability prediction identifiers in request lines.
accesslog = None
errorlog = "-"


def child_exit(server, worker):
    """Reap all prometheus multiprocess shards of a dead gunicorn worker."""
    try:
        from vision_model_serving.observability import cleanup_multiprocess_pid

        cleanup_multiprocess_pid(worker.pid)
    except Exception:
        # Metrics may be disabled or prometheus_client absent; never block reaping.
        pass
