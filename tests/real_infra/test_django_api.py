"""Real Redis and public-DICOM acceptance for the Django HTTP seam."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from io import BytesIO, StringIO
from pathlib import Path

EXPECTED_DICOM_SHA256 = (
    "9f70081672a460f29231bb471e8a9e26dd3ed26a2ebbd91c064e575e7842a19c"
)
OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9_-]{32}")


def main() -> int:
    redis_url = _required_environment("VMS_TEST_REDIS_URL")
    dicom_path = Path(_required_environment("VMS_TEST_DICOM_PATH")).resolve()
    dicom = dicom_path.read_bytes()
    observed_hash = hashlib.sha256(dicom).hexdigest()
    if observed_hash != EXPECTED_DICOM_SHA256:
        raise AssertionError(
            f"public DICOM hash mismatch: expected {EXPECTED_DICOM_SHA256}, "
            f"observed {observed_hash}"
        )
    from pydicom import dcmread

    dataset = dcmread(BytesIO(dicom), stop_before_pixels=True)
    dicom_identifiers = tuple(
        str(value)
        for value in (
            dataset.get("PatientID"),
            dataset.get("PatientName"),
            dataset.get("StudyInstanceUID"),
            dataset.get("SOPInstanceUID"),
        )
        if value
    )

    with tempfile.TemporaryDirectory(prefix="vms-django-real-") as job_root:
        os.environ["DJANGO_SETTINGS_MODULE"] = "vision_model_serving.web.settings"
        os.environ["VMS_REDIS_URL"] = redis_url
        os.environ["VMS_JOB_ROOT"] = job_root
        os.environ["VMS_QUEUE_CAPACITY"] = "2"
        os.environ["VMS_SYNC_WAIT_SECONDS"] = "0.1"
        os.environ["VMS_ALLOWED_HOSTS"] = "testserver,localhost"

        import django

        django.setup()

        telemetry_stream = StringIO()
        telemetry_handler = logging.StreamHandler(telemetry_stream)
        telemetry_handler.setFormatter(logging.Formatter("%(message)s"))
        telemetry_logger = logging.getLogger("vision_model_serving.telemetry")
        telemetry_logger.addHandler(telemetry_handler)

        from django.core.files.uploadedfile import SimpleUploadedFile
        from django.test import Client
        from redis import Redis
        from rq.job import Job
        from rq.serializers import JSONSerializer

        redis = Redis.from_url(redis_url)
        redis.ping()
        redis.flushdb()
        try:
            client = Client(enforce_csrf_checks=True)
            inspection = client.get("/")
            _assert_status(inspection.status_code, 200, inspection.content)
            if "csrftoken" not in inspection.cookies:
                raise AssertionError("inspection workbench did not issue a CSRF cookie")
            for required_surface in (
                b'id="inspection-form"',
                b'id="mammogram-viewport"',
                b'id="prediction-status"',
                b'aria-live="polite"',
                b"Research use only",
            ):
                if required_surface not in inspection.content:
                    raise AssertionError(
                        f"inspection workbench omitted {required_surface!r}"
                    )

            csrf_token = inspection.cookies["csrftoken"].value
            preview = client.post(
                "/api/v1/dicom-preview",
                {
                    "dicom": SimpleUploadedFile(
                        "public-mammogram.dcm",
                        dicom,
                        content_type="application/dicom",
                    )
                },
                HTTP_X_CSRFTOKEN=csrf_token,
            )
            _assert_status(preview.status_code, 200, preview.content)
            if preview["Content-Type"] != "image/png":
                raise AssertionError("DICOM preview did not return a PNG")
            if preview.get("Cache-Control") != "no-store":
                raise AssertionError("DICOM preview was not marked private and ephemeral")
            from PIL import Image

            with Image.open(BytesIO(preview.content)) as image:
                if image.size != (1024, 1024) or image.mode != "L":
                    raise AssertionError(
                        f"unexpected canonical preview: {image.mode} {image.size!r}"
                    )
            if any(value.encode() in preview.content for value in dicom_identifiers):
                raise AssertionError("DICOM preview retained a source identifier")

            monitoring = client.get("/monitoring")
            _assert_status(monitoring.status_code, 200, monitoring.content)
            if monitoring.get("Cache-Control") != "no-store":
                raise AssertionError("monitoring console was not marked no-store")
            for required_surface in (
                b'id="signal-rail"',
                b'id="operations-status"',
                b'id="queue-panel"',
                b'id="latency-panel"',
                b'aria-live="polite"',
                b"Operational telemetry only",
            ):
                if required_surface not in monitoring.content:
                    raise AssertionError(
                        f"monitoring console omitted {required_surface!r}"
                    )

            operations = client.get("/api/v1/operations")
            _assert_status(operations.status_code, 200, operations.content)
            if operations.get("Cache-Control") != "no-store":
                raise AssertionError("operations snapshot was not marked no-store")
            operations_body = operations.json()
            if operations_body.get("schema_version") != 1:
                raise AssertionError("operations snapshot schema is not versioned")
            if operations_body.get("status") != "not_ready":
                raise AssertionError(
                    f"unexpected degraded operations state: {operations_body!r}"
                )
            if operations_body.get("checks", {}).get("redis") is not True:
                raise AssertionError("operations snapshot did not verify real Redis")
            if operations_body.get("checks", {}).get("rq_worker") is not False:
                raise AssertionError("operations snapshot claimed an absent RQ worker")
            queue = operations_body.get("queue", {})
            if queue.get("available") is not True or queue.get("capacity") != 2:
                raise AssertionError(f"operations queue state is invalid: {queue!r}")
            if any(queue.get(name) != 0 for name in ("active", "queued", "running")):
                raise AssertionError(f"fresh operations queue is not empty: {queue!r}")
            model_ids = [model["id"] for model in operations_body.get("models", [])]
            if model_ids != ["focalnet-dino-detector", "mmbcd-classifier"]:
                raise AssertionError("operations snapshot omitted pinned model identity")
            serialized_operations = json.dumps(operations_body, sort_keys=True)
            for private_value in (
                "public-mammogram.dcm",
                str(job_root),
                *dicom_identifiers,
            ):
                if private_value in serialized_operations:
                    raise AssertionError("private data leaked into operations snapshot")

            liveness = client.get("/livez")
            _assert_status(liveness.status_code, 200, liveness.content)
            if liveness.json() != {"status": "alive"}:
                raise AssertionError(f"unexpected liveness body: {liveness.json()!r}")

            readiness = client.get("/readyz")
            _assert_status(readiness.status_code, 503, readiness.content)
            readiness_body = readiness.json()
            if readiness_body.get("status") != "not_ready":
                raise AssertionError(f"unexpected readiness body: {readiness_body!r}")
            if readiness_body.get("checks", {}).get("redis") is not True:
                raise AssertionError("readiness did not verify the real Redis server")
            if readiness_body.get("checks", {}).get("rq_worker") is not False:
                raise AssertionError("readiness claimed an absent RQ worker")
            if "rq_worker_unavailable" not in readiness_body.get("reasons", []):
                raise AssertionError("readiness omitted its bounded worker reason")

            models = client.get("/api/v1/models")
            _assert_status(models.status_code, 200, models.content)
            model_body = models.json()
            if [item["id"] for item in model_body["models"]] != [
                "focalnet-dino-detector",
                "mmbcd-classifier",
            ]:
                raise AssertionError(f"unexpected model inventory: {model_body!r}")
            if any(
                "filename" in item or "path" in item for item in model_body["models"]
            ):
                raise AssertionError("model endpoint exposed a filesystem detail")

            schema = client.get("/api/schema/")
            _assert_status(schema.status_code, 200, schema.content)
            if b"/api/v1/predictions" not in schema.content:
                raise AssertionError("OpenAPI schema omitted the prediction interface")
            if b"/api/v1/dicom-preview" not in schema.content:
                raise AssertionError("OpenAPI schema omitted the preview interface")
            if b"/api/v1/operations" not in schema.content:
                raise AssertionError("OpenAPI schema omitted the operations interface")
            schema_json_response = client.get("/api/schema/?format=json")
            _assert_status(
                schema_json_response.status_code,
                200,
                schema_json_response.content,
            )
            schema_json = json.loads(schema_json_response.content)
            prediction_operation = schema_json["paths"]["/api/v1/predictions"]["post"]
            multipart_schema = prediction_operation["requestBody"]["content"][
                "multipart/form-data"
            ]["schema"]
            if not multipart_schema:
                raise AssertionError("OpenAPI omitted the multipart request contract")
            if "504" not in prediction_operation["responses"]:
                raise AssertionError("OpenAPI omitted the prediction-timeout contract")
            docs = client.get("/api/docs/")
            _assert_status(docs.status_code, 200, docs.content)

            metrics = client.get("/metrics")
            _assert_status(metrics.status_code, 503, metrics.content)
            if metrics.json()["error"]["code"] != "metrics_not_configured":
                raise AssertionError(
                    "metrics integration point returned an unstable error"
                )

            unsupported_media = client.post(
                "/api/v1/predictions",
                data=json.dumps({"mode": "detection"}),
                content_type="application/json",
            )
            _assert_status(
                unsupported_media.status_code, 415, unsupported_media.content
            )
            if unsupported_media.json()["error"]["code"] != "unsupported_media_type":
                raise AssertionError("unsupported media error code is unstable")

            malformed_dicom = client.post(
                "/api/v1/predictions",
                {
                    "dicom": SimpleUploadedFile(
                        "malformed.dcm",
                        b"this is not a DICOM object",
                        content_type="application/dicom",
                    ),
                    "mode": "detection",
                },
            )
            _assert_status(
                malformed_dicom.status_code,
                400,
                malformed_dicom.content,
            )
            if malformed_dicom.json()["error"]["code"] != "dicom_invalid":
                raise AssertionError("malformed DICOM error code is unstable")

            oversized = client.post(
                "/api/v1/predictions",
                {
                    "dicom": SimpleUploadedFile(
                        "oversized.dcm",
                        b"0" * (64 * 1024 * 1024 + 1),
                        content_type="application/dicom",
                    ),
                    "mode": "detection",
                },
            )
            _assert_status(oversized.status_code, 413, oversized.content)
            if oversized.json()["error"]["code"] != "dicom_encoded_size_exceeded":
                raise AssertionError("encoded DICOM size error code is unstable")

            missing_history = client.post(
                "/api/v1/predictions",
                {
                    "dicom": SimpleUploadedFile(
                        "public-mammogram.dcm",
                        dicom,
                        content_type="application/dicom",
                    ),
                    "mode": "full",
                },
            )
            _assert_status(missing_history.status_code, 422, missing_history.content)
            if missing_history.json()["error"]["code"] != "clinical_history_required":
                raise AssertionError("missing clinical history error code is unstable")

            response = client.post(
                "/api/v1/predictions",
                {
                    "dicom": SimpleUploadedFile(
                        "public-mammogram.dcm",
                        dicom,
                        content_type="application/dicom",
                    ),
                    "mode": "detection",
                },
                HTTP_PREFER="respond-async",
                HTTP_IDEMPOTENCY_KEY="real-http-public-dicom",
            )
            _assert_status(response.status_code, 202, response.content)
            handle = response.json()
            prediction_id = handle["prediction_id"]
            if OPAQUE_TOKEN.fullmatch(prediction_id) is None:
                raise AssertionError("HTTP response exposed a non-opaque prediction ID")
            if handle["state"] != "queued":
                raise AssertionError(f"unexpected initial state: {handle['state']!r}")

            status_response = client.get(
                f"/api/v1/predictions/{prediction_id}",
            )
            _assert_status(
                status_response.status_code,
                200,
                status_response.content,
            )
            if status_response.json()["state"] != "queued":
                raise AssertionError("queued prediction was not pollable through HTTP")

            sync_timeout = client.post(
                "/api/v1/predictions",
                {
                    "dicom": SimpleUploadedFile(
                        "public-mammogram.dcm",
                        dicom,
                        content_type="application/dicom",
                    ),
                    "mode": "detection",
                },
                HTTP_IDEMPOTENCY_KEY="real-http-sync-timeout",
            )
            _assert_status(sync_timeout.status_code, 202, sync_timeout.content)
            if sync_timeout.json()["state"] != "queued":
                raise AssertionError(
                    "bounded sync wait did not return a pollable handle"
                )

            saturated = client.post(
                "/api/v1/predictions",
                {
                    "dicom": SimpleUploadedFile(
                        "public-mammogram.dcm",
                        dicom,
                        content_type="application/dicom",
                    ),
                    "mode": "detection",
                },
                HTTP_PREFER="respond-async",
                HTTP_IDEMPOTENCY_KEY="real-http-saturated-request",
            )
            _assert_status(saturated.status_code, 429, saturated.content)
            if saturated.json()["error"]["code"] != "prediction_queue_full":
                raise AssertionError("queue saturation error code is unstable")

            job = Job.fetch(
                prediction_id,
                connection=redis,
                serializer=JSONSerializer,
            )
            if len(job.args) != 2 or any(
                not isinstance(value, str) or OPAQUE_TOKEN.fullmatch(value) is None
                for value in job.args
            ):
                raise AssertionError(f"RQ arguments are not opaque: {job.args!r}")
            broker_bytes = b"".join(
                redis.dump(key) or b"" for key in redis.scan_iter("*")
            )
            if dicom[128:256] in broker_bytes:
                raise AssertionError("DICOM bytes leaked into Redis")
            if b"real-http-public-dicom" in broker_bytes:
                raise AssertionError("raw idempotency key leaked into Redis")

            events = [
                json.loads(line)
                for line in telemetry_stream.getvalue().splitlines()
                if line.startswith("{")
            ]
            observed_statuses = {
                event["status_code"]
                for event in events
                if event.get("event") == "http_response"
            }
            expected_statuses = {200, 202, 400, 413, 415, 422, 429, 503}
            if not expected_statuses.issubset(observed_statuses):
                raise AssertionError("structured logs omitted an HTTP outcome path")
            serialized_events = json.dumps(events, sort_keys=True)
            for private_value in (
                "malformed.dcm",
                "oversized.dcm",
                "public-mammogram.dcm",
                "real-http-public-dicom",
                job_root,
            ):
                if private_value in serialized_events:
                    raise AssertionError("private request data leaked into HTTP logs")

            print(
                json.dumps(
                    {
                        "dicom_sha256": observed_hash,
                        "http_status": response.status_code,
                        "prediction_state": status_response.json()["state"],
                        "redis": redis.info("server")["redis_version"],
                        "liveness": liveness.status_code,
                        "readiness": readiness.status_code,
                        "schema": schema.status_code,
                        "rq_arguments_are_opaque": True,
                        "private_payload_absent_from_redis": True,
                        "structured_http_logs_are_redacted": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        finally:
            telemetry_logger.removeHandler(telemetry_handler)
            redis.flushdb()
    return 0


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must point to real acceptance infrastructure")
    return value


def _assert_status(observed: int, expected: int, content: bytes) -> None:
    if observed != expected:
        raise AssertionError(
            f"expected HTTP {expected}, observed {observed}: "
            f"{content.decode('utf-8', errors='replace')}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
