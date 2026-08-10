from __future__ import annotations

from pathlib import Path
import socketserver
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.execution.contracts import (  # noqa: E402
    GpuExecutorStatus,
    PredictionId,
)
from vision_model_serving.execution.executor import (  # noqa: E402
    GpuExecutorClient,
    GpuExecutorError,
    GpuExecutorServer,
    GpuExecutorUnavailable,
)
from vision_model_serving.model_ids import DETECTOR_MODEL_ID  # noqa: E402


class _FailingHealthyExecutor:
    def execute(self, _prediction_id: PredictionId, _locator: str) -> None:
        raise RuntimeError("inference failed")

    def status(self) -> GpuExecutorStatus:
        return GpuExecutorStatus(
            verified_artifacts=True,
            runtime_initialized=True,
            device_available=True,
            native_operator_available=True,
            runtime_state="ready",
            active_model=DETECTOR_MODEL_ID,
            resident_models=(DETECTOR_MODEL_ID,),
            device_name="cuda:0",
            last_error=None,
        )


class GpuExecutorProtocolTests(unittest.TestCase):
    @unittest.skipUnless(
        hasattr(socketserver, "UnixStreamServer"),
        "Unix-domain socket server is unavailable on this platform",
    )
    def test_execution_failure_from_healthy_runtime_is_not_retryable(self) -> None:
        with TemporaryDirectory(prefix="vms-executor-") as directory:
            socket_path = Path(directory) / "executor.sock"
            server = GpuExecutorServer(socket_path, _FailingHealthyExecutor())
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                client = GpuExecutorClient(socket_path, timeout_seconds=2.0)

                with self.assertRaises(GpuExecutorError) as raised:
                    client.execute(PredictionId("a" * 32), "b" * 32)

                self.assertNotIsInstance(raised.exception, GpuExecutorUnavailable)
            finally:
                server.shutdown()
                server.close()
                thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
