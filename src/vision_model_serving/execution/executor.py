"""Private Unix-socket adapter for the long-lived GPU executor."""

from __future__ import annotations

import argparse
import json
import re
import signal
import socket
import socketserver
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .contracts import PredictionId

_TOKEN = re.compile(r"[A-Za-z0-9_-]{32}")
_MAX_MESSAGE_BYTES = 4_096


class GpuExecutorError(RuntimeError):
    pass


class GpuExecutorUnavailable(GpuExecutorError):
    pass


class GpuExecutorConfigurationError(GpuExecutorError):
    pass


class _Executor(Protocol):
    def execute(self, prediction_id: PredictionId, locator: str) -> None: ...


class GpuExecutorClient:
    """Execute one opaque prediction through the local GPU-owner process."""

    def __init__(self, socket_path: Path, *, timeout_seconds: float):
        if not isinstance(socket_path, Path):
            raise TypeError("executor socket path must be a pathlib.Path")
        if timeout_seconds <= 0:
            raise ValueError("executor timeout must be positive")
        self._socket_path = socket_path.expanduser().resolve()
        self._timeout_seconds = float(timeout_seconds)

    def execute(self, prediction_id: PredictionId, locator: str) -> None:
        prediction = str(prediction_id)
        if _TOKEN.fullmatch(prediction) is None or _TOKEN.fullmatch(locator) is None:
            raise GpuExecutorError("prediction executor identifiers are invalid")
        request = _encode(
            {
                "schema_version": 1,
                "prediction_id": prediction,
                "locator": locator,
            }
        )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout_seconds)
                connection.connect(str(self._socket_path))
                connection.sendall(request)
                response = _read_message(connection)
        except (OSError, TimeoutError):
            raise GpuExecutorUnavailable("prediction executor is unavailable") from None
        if response != {"schema_version": 1, "ok": True}:
            raise GpuExecutorError("prediction execution failed")


class GpuExecutorServer:
    """Serve serialized prediction commands from RQ work-horses."""

    def __init__(self, socket_path: Path, executor: _Executor):
        if not isinstance(socket_path, Path):
            raise TypeError("executor socket path must be a pathlib.Path")
        if not callable(getattr(executor, "execute", None)):
            raise TypeError("executor must implement execute(prediction_id, locator)")
        self._socket_path = socket_path.expanduser().resolve()
        self._executor = executor
        self._prepare_socket_path()
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                outer._handle(self.rfile, self.wfile)

        try:
            self._server = socketserver.UnixStreamServer(
                str(self._socket_path),
                Handler,
            )
            self._socket_path.chmod(0o600)
        except Exception:
            self._unlink_socket()
            raise

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.5)

    def shutdown(self) -> None:
        self._server.shutdown()

    def close(self) -> None:
        self._server.server_close()
        self._unlink_socket()

    def _handle(self, reader: object, writer: object) -> None:
        response = {"schema_version": 1, "ok": False}
        try:
            raw = reader.readline(_MAX_MESSAGE_BYTES + 1)
            if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                raise ValueError
            request = json.loads(raw)
            if (
                not isinstance(request, dict)
                or set(request) != {"schema_version", "prediction_id", "locator"}
                or request.get("schema_version") != 1
            ):
                raise ValueError
            prediction = request.get("prediction_id")
            locator = request.get("locator")
            if (
                not isinstance(prediction, str)
                or not isinstance(locator, str)
                or _TOKEN.fullmatch(prediction) is None
                or _TOKEN.fullmatch(locator) is None
            ):
                raise ValueError
            self._executor.execute(PredictionId(prediction), locator)
            response["ok"] = True
        except Exception:  # noqa: BLE001 - sanitize the process seam
            response = {"schema_version": 1, "ok": False}
        try:
            writer.write(_encode(response))
            writer.flush()
        except OSError:
            pass

    def _prepare_socket_path(self) -> None:
        self._socket_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not self._socket_path.exists():
            return
        if not self._socket_path.is_socket():
            raise GpuExecutorConfigurationError(
                "executor socket path exists and is not a socket"
            )
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.settimeout(0.2)
                probe.connect(str(self._socket_path))
        except OSError:
            self._socket_path.unlink()
            return
        raise GpuExecutorConfigurationError("prediction executor is already running")

    def _unlink_socket(self) -> None:
        try:
            if self._socket_path.is_socket():
                self._socket_path.unlink()
        except OSError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from vision_model_serving.pipeline import (
        LocalCudaPipelineConfig,
        build_local_cuda_pipeline,
    )

    from .rq_worker import PredictionJobWorker

    pipeline = build_local_cuda_pipeline(
        LocalCudaPipelineConfig(
            project_root=args.project_root,
            artifact_root=args.artifact_root,
            tokenizer_root=args.tokenizer_root,
            focalnet_root=args.focalnet_root,
            mmbcd_root=args.mmbcd_root,
            dino_root=args.dino_root,
            device=args.device,
            require_history_for_full=not args.allow_empty_history,
            retain_models=True,
        )
    )
    worker = PredictionJobWorker(
        job_root=args.job_root,
        pipeline=pipeline,
        result_ttl_seconds=args.result_ttl_seconds,
    )
    server = GpuExecutorServer(args.socket_path, worker)

    def stop(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        pipeline.close()
    return 0


def _parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Run the persistent local GPU prediction executor.",
    )
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--job-root", type=Path, required=True)
    parser.add_argument("--result-ttl-seconds", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--focalnet-root", type=Path, required=True)
    parser.add_argument("--mmbcd-root", type=Path, required=True)
    parser.add_argument("--dino-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-empty-history", action="store_true")
    return parser


def _encode(value: object) -> bytes:
    encoded = (
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if len(encoded) > _MAX_MESSAGE_BYTES:
        raise GpuExecutorError("prediction executor message is too large")
    return encoded


def _read_message(connection: socket.socket) -> object:
    chunks = bytearray()
    while len(chunks) <= _MAX_MESSAGE_BYTES:
        chunk = connection.recv(min(1024, _MAX_MESSAGE_BYTES + 1 - len(chunks)))
        if not chunk:
            break
        chunks.extend(chunk)
        if chunks.endswith(b"\n"):
            break
    if len(chunks) > _MAX_MESSAGE_BYTES or not chunks.endswith(b"\n"):
        raise GpuExecutorUnavailable("prediction executor response is unavailable")
    try:
        return json.loads(chunks)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise GpuExecutorUnavailable(
            "prediction executor response is unavailable"
        ) from None


if __name__ == "__main__":
    raise SystemExit(main())
