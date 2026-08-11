from __future__ import annotations

import json
import os
import stat
import sys
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from vision_model_serving.execution.contracts import (  # noqa: E402
    PredictionId,
    PredictionRequest,
)
from vision_model_serving.execution.storage import (  # noqa: E402
    EphemeralJobStore,
    JobPayloadExpired,
    JobPayloadNotFound,
    JobResultNotFound,
)
from vision_model_serving.pipeline import CaseInput, PredictionMode  # noqa: E402


class FakeClock:
    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def request(dicom_bytes: bytes = b"private-dicom") -> PredictionRequest:
    return PredictionRequest(
        case=CaseInput(BytesIO(dicom_bytes), "private history"),
        mode=PredictionMode.FULL,
    )


class EphemeralJobStoreRetentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.clock = FakeClock()
        self.store = EphemeralJobStore(self.root, clock=self.clock)
        self.prediction_id = PredictionId("a" * 32)

    def test_expired_request_fails_closed_and_is_deleted(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(),
            ttl_seconds=10,
        )
        self.clock.advance(11)
        with self.assertRaises(JobPayloadExpired):
            self.store.load_request(self.prediction_id, stored.locator)
        self.assertFalse((self.root / stored.locator).exists())

    def test_request_at_exact_expiry_boundary_fails_closed(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(),
            ttl_seconds=10,
        )
        self.clock.advance(10)
        with self.assertRaises(JobPayloadExpired):
            self.store.load_request(self.prediction_id, stored.locator)

    def test_tampered_payload_is_rejected(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(),
            ttl_seconds=100,
        )
        (self.root / stored.locator / "input.dcm").write_bytes(b"ATTACKER-DICOM")
        with self.assertRaises(JobPayloadNotFound):
            self.store.load_request(self.prediction_id, stored.locator)

    def test_tampered_metadata_is_rejected(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(),
            ttl_seconds=100,
        )
        metadata_path = self.root / stored.locator / "request.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["clinical_history"] = "attacker history"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaises(JobPayloadNotFound):
            self.store.load_request(self.prediction_id, stored.locator)

    def test_valid_request_round_trips(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(b"payload-bytes"),
            ttl_seconds=100,
        )
        loaded = self.store.load_request(self.prediction_id, stored.locator)
        self.assertEqual(loaded.case.dicom_stream.read(), b"payload-bytes")
        self.assertEqual(loaded.case.clinical_history, "private history")

    @unittest.skipIf(os.name == "nt", "Windows does not expose POSIX mode bits")
    def test_job_payload_is_owner_writable_and_validation_group_readable(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(),
            ttl_seconds=100,
        )
        directory = self.root / stored.locator

        self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o750)
        self.assertEqual(stat.S_IMODE((directory / "input.dcm").stat().st_mode), 0o640)
        self.assertEqual(stat.S_IMODE((directory / "request.json").stat().st_mode), 0o640)

        self.store.store_result(stored.locator, _prediction_result(), ttl_seconds=100)
        self.assertEqual(stat.S_IMODE((directory / "result.json").stat().st_mode), 0o640)

    def test_expired_result_is_deleted_on_access(self) -> None:
        stored = self.store.store_request(
            self.prediction_id,
            request(),
            ttl_seconds=100,
        )
        result = _prediction_result()
        self.store.store_result(stored.locator, result, ttl_seconds=10)
        self.clock.advance(11)
        with self.assertRaises(JobResultNotFound):
            self.store.load_result(stored.locator)
        self.assertFalse((self.root / stored.locator).exists())


class EphemeralJobStoreJanitorTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.clock = FakeClock()
        self.store = EphemeralJobStore(self.root, clock=self.clock)

    def test_cleanup_removes_stale_staging_orphans(self) -> None:
        stale_dir = self.root / ".tmp-deadbeefdeadbeefdeadbeefdeadbeef"
        stale_dir.mkdir()
        (stale_dir / "input.dcm").write_bytes(b"orphaned-phi")
        stale_file = self.root / ".result-deadbeefdeadbeefdeadbeefdeadbeef.tmp"
        stale_file.write_text("{}", encoding="utf-8")
        old = 1.0
        os.utime(stale_dir, (old, old))
        os.utime(stale_file, (old, old))
        fresh_dir = self.root / ".tmp-feedfacefeedfacefeedfacefeedface"
        fresh_dir.mkdir()

        removed = self.store.cleanup_expired()

        self.assertEqual(removed, 2)
        self.assertFalse(stale_dir.exists())
        self.assertFalse(stale_file.exists())
        self.assertTrue(fresh_dir.exists())

    def test_maybe_cleanup_is_rate_limited(self) -> None:
        stale = self.root / ".tmp-deadbeefdeadbeefdeadbeefdeadbeef"
        stale.mkdir()
        os.utime(stale, (1.0, 1.0))

        self.assertEqual(self.store.maybe_cleanup(interval_seconds=60), 1)
        stale.mkdir()
        os.utime(stale, (1.0, 1.0))
        self.assertEqual(self.store.maybe_cleanup(interval_seconds=60), 0)
        self.clock.advance(61)
        self.assertEqual(self.store.maybe_cleanup(interval_seconds=60), 1)

    def test_cleanup_removes_expired_locator_directories(self) -> None:
        prediction_id = PredictionId("b" * 32)
        stored = self.store.store_request(prediction_id, request(), ttl_seconds=10)
        self.clock.advance(11)
        removed = self.store.cleanup_expired()
        self.assertEqual(removed, 1)
        self.assertFalse((self.root / stored.locator).exists())

    def test_active_execution_lease_protects_expired_request(self) -> None:
        prediction_id = PredictionId("c" * 32)
        stored = self.store.store_request(prediction_id, request(), ttl_seconds=10)
        self.store.acquire_lease(stored.locator, ttl_seconds=100)
        self.clock.advance(11)

        self.assertEqual(self.store.cleanup_expired(), 0)
        loaded = self.store.load_request(prediction_id, stored.locator)
        self.assertEqual(loaded.case.dicom_stream.read(), b"private-dicom")

        self.store.release_lease(stored.locator)
        self.assertEqual(self.store.cleanup_expired(), 1)

    def test_expired_execution_lease_does_not_block_cleanup(self) -> None:
        prediction_id = PredictionId("d" * 32)
        stored = self.store.store_request(prediction_id, request(), ttl_seconds=10)
        self.store.acquire_lease(stored.locator, ttl_seconds=20)
        self.clock.advance(21)

        self.assertEqual(self.store.cleanup_expired(), 1)
        self.assertFalse((self.root / stored.locator).exists())

    def test_direct_discard_refuses_to_delete_active_execution(self) -> None:
        prediction_id = PredictionId("e" * 32)
        stored = self.store.store_request(prediction_id, request(), ttl_seconds=10)
        self.store.acquire_lease(stored.locator, ttl_seconds=100)

        self.assertFalse(self.store.discard_job(stored.locator))
        self.assertTrue((self.root / stored.locator).is_dir())


def _prediction_result() -> object:
    from tests.test_prediction_pipeline import (
        DecoderFake,
        FullRuntimeFake,
        canonical_mammogram,
        classifier_result,
        detector_result,
    )
    from vision_model_serving.pipeline import PredictionPipeline

    mammogram = canonical_mammogram()
    pipeline = PredictionPipeline(
        decoder=DecoderFake(mammogram),
        runtime=FullRuntimeFake(
            detector_result(mammogram),
            classifier_result(),
            mammogram,
        ),
    )
    return pipeline.infer(
        CaseInput(BytesIO(b"source"), "  prior   surgery  "),
        PredictionMode.FULL,
    )


if __name__ == "__main__":
    unittest.main()
