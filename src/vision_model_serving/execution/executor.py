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

from .contracts import GpuExecutorStatus, PredictionId

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

    def status(self) -> GpuExecutorStatus: ...


class PersistentGpuExecutor:
    """Own prediction execution and sanitized runtime state in one process."""

    def __init__(self, worker: object, *, device_name: str):
        if not callable(getattr(worker, "execute", None)) or not callable(
            getattr(worker, "status", None)
        ):
            raise TypeError("worker must implement execute() and status()")
        if not isinstance(device_name, str) or not device_name:
            raise ValueError("device name must not be empty")
        self._worker = worker
        self._device_name = device_name

    def execute(self, prediction_id: PredictionId, locator: str) -> None:
        self._worker.execute(prediction_id, locator)

    def status(self) -> GpuExecutorStatus:
        runtime = self._worker.status()
        state_value = getattr(getattr(runtime, "state", None), "value", None)
        if not isinstance(state_value, str) or not state_value:
            raise GpuExecutorError("prediction runtime status is invalid")
        active_model = getattr(runtime, "active_model", None)
        residents = getattr(runtime, "resident_models", ())
        last_error = getattr(runtime, "last_error", None)
        error_code = None if last_error is None else getattr(last_error, "code", None)
        if active_model is not None and not isinstance(active_model, str):
            raise GpuExecutorError("prediction runtime status is invalid")
        if not isinstance(residents, tuple) or not all(
            isinstance(model, str) for model in residents
        ):
            raise GpuExecutorError("prediction runtime status is invalid")
        if error_code is not None and not isinstance(error_code, str):
            raise GpuExecutorError("prediction runtime status is invalid")
        return GpuExecutorStatus(
            ready=state_value != "failed",
            verified_artifacts=True,
            device=True,
            native_operator=True,
            runtime_state=state_value,
            active_model=active_model,
            resident_models=residents,
            device_name=self._device_name,
            last_error=error_code,
        )


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
                "operation": "execute",
                "prediction_id": prediction,
                "locator": locator,
            }
        )
        response = self._exchange(request)
        if response == {"schema_version": 1, "ok": True}:
            return
        if response == {
            "schema_version": 1,
            "ok": False,
            "error": "runtime_unavailable",
        }:
            raise GpuExecutorUnavailable("prediction executor is unavailable")
        raise GpuExecutorError("prediction execution failed")

    def status(self) -> GpuExecutorStatus:
        response = self._exchange(_encode({"schema_version": 1, "operation": "status"}))
        if (
            not isinstance(response, dict)
            or set(response) != {"schema_version", "ok", "status"}
            or response.get("schema_version") != 1
            or response.get("ok") is not True
        ):
            raise GpuExecutorUnavailable("prediction executor status is unavailable")
        return _status_from_dict(response.get("status"))

    def _exchange(self, request: bytes) -> object:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self._timeout_seconds)
                connection.connect(str(self._socket_path))
                connection.sendall(request)
                response = _read_message(connection)
        except (OSError, TimeoutError):
            raise GpuExecutorUnavailable("prediction executor is unavailable") from None
        return response


class GpuExecutorServer:
    """Serve serialized prediction commands from RQ work-horses."""

    def __init__(self, socket_path: Path, executor: _Executor):
        if not isinstance(socket_path, Path):
            raise TypeError("executor socket path must be a pathlib.Path")
        if not callable(getattr(executor, "execute", None)) or not callable(
            getattr(executor, "status", None)
        ):
            raise TypeError("executor must implement execute() and status()")
        self._socket_path = socket_path.expanduser().resolve()
        self._executor = executor
        self._prepare_socket_path()
        outer = self

        class Handler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                outer._handle(self.rfile, self.wfile)

        try:
            self._server = _threading_unix_server_type()(
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
        response = {
            "schema_version": 1,
            "ok": False,
            "error": "execution_failed",
        }
        try:
            raw = reader.readline(_MAX_MESSAGE_BYTES + 1)
            if len(raw) > _MAX_MESSAGE_BYTES or not raw.endswith(b"\n"):
                raise ValueError
            request = json.loads(raw)
            if not isinstance(request, dict) or request.get("schema_version") != 1:
                raise ValueError
            operation = request.get("operation")
            if operation == "status":
                if set(request) != {"schema_version", "operation"}:
                    raise ValueError
                response = {
                    "schema_version": 1,
                    "ok": True,
                    "status": _status_to_dict(self._executor.status()),
                }
                writer.write(_encode(response))
                writer.flush()
                return
            if operation != "execute" or set(request) != {
                "schema_version",
                "operation",
                "prediction_id",
                "locator",
            }:
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
            response = {"schema_version": 1, "ok": True}
        except Exception:  # noqa: BLE001 - sanitize the process seam
            try:
                runtime_unavailable = not self._executor.status().ready
            except Exception:  # noqa: BLE001 - preserve the sanitized seam
                runtime_unavailable = True
            response = {
                "schema_version": 1,
                "ok": False,
                "error": (
                    "runtime_unavailable" if runtime_unavailable else "execution_failed"
                ),
            }
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
    executor = PersistentGpuExecutor(worker, device_name=args.device)
    server = GpuExecutorServer(args.socket_path, executor)

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


def _status_to_dict(status: GpuExecutorStatus) -> dict[str, object]:
    return {
        "ready": status.ready,
        "verified_artifacts": status.verified_artifacts,
        "device": status.device,
        "native_operator": status.native_operator,
        "runtime_state": status.runtime_state,
        "active_model": status.active_model,
        "resident_models": list(status.resident_models),
        "device_name": status.device_name,
        "last_error": status.last_error,
    }


def _status_from_dict(value: object) -> GpuExecutorStatus:
    if not isinstance(value, dict) or set(value) != {
        "ready",
        "verified_artifacts",
        "device",
        "native_operator",
        "runtime_state",
        "active_model",
        "resident_models",
        "device_name",
        "last_error",
    }:
        raise GpuExecutorUnavailable("prediction executor status is unavailable")
    boolean_fields = ("ready", "verified_artifacts", "device", "native_operator")
    if not all(isinstance(value.get(field), bool) for field in boolean_fields):
        raise GpuExecutorUnavailable("prediction executor status is unavailable")
    runtime_state = value.get("runtime_state")
    active_model = value.get("active_model")
    residents = value.get("resident_models")
    device_name = value.get("device_name")
    last_error = value.get("last_error")
    if (
        not isinstance(runtime_state, str)
        or not isinstance(device_name, str)
        or active_model is not None
        and not isinstance(active_model, str)
        or last_error is not None
        and not isinstance(last_error, str)
        or not isinstance(residents, list)
        or not all(isinstance(model, str) for model in residents)
    ):
        raise GpuExecutorUnavailable("prediction executor status is unavailable")
    return GpuExecutorStatus(
        ready=value["ready"],
        verified_artifacts=value["verified_artifacts"],
        device=value["device"],
        native_operator=value["native_operator"],
        runtime_state=runtime_state,
        active_model=active_model,
        resident_models=tuple(residents),
        device_name=device_name,
        last_error=last_error,
    )


def _threading_unix_server_type() -> type[socketserver.BaseServer]:
    unix_server = getattr(socketserver, "UnixStreamServer", None)
    if unix_server is None:
        raise GpuExecutorConfigurationError(
            "Unix-domain sockets are unavailable on this platform"
        )
    return type(
        "ThreadingUnixStreamServer",
        (socketserver.ThreadingMixIn, unix_server),
        {"daemon_threads": True},
    )


if __name__ == "__main__":
    raise SystemExit(main())
