from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SECRET_KEY_FOR_INTERPOLATION = "topology-test-secret"


def _compose_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["VMS_DICOM_PATH"] = str((REPOSITORY_ROOT / "ASSIGNMENT.md").resolve())
    environment["VMS_SECRET_KEY"] = SECRET_KEY_FOR_INTERPOLATION
    return environment


def _compose_config(profile: str) -> dict:
    completed = subprocess.run(
        [
            "docker",
            "compose",
            "--profile",
            profile,
            "config",
            "--format",
            "json",
        ],
        cwd=REPOSITORY_ROOT,
        env=_compose_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


class ComposeTopologyTests(unittest.TestCase):
    def test_browser_image_keeps_the_serving_runtime_out(self) -> None:
        dockerfile = (REPOSITORY_ROOT / "docker" / "browser.Dockerfile").read_text(encoding="utf-8")
        self.assertIn("uv sync --frozen --only-group browser", dockerfile)
        self.assertIn("validation/acceptance_constants.py", dockerfile)
        self.assertNotIn("COPY --chown=1000:1000 src ./src", dockerfile)

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_loopback_web_edge_does_not_expose_backend_services(self) -> None:
        configuration = _compose_config("gpu")
        services = configuration["services"]
        self.assertEqual(set(services["web"]["networks"]), {"backend", "edge"})
        self.assertEqual(
            [name for name, service in services.items() if "edge" in service["networks"]],
            ["web"],
        )
        self.assertEqual(
            services["web"]["ports"],
            [
                {
                    "mode": "ingress",
                    "target": 8000,
                    "published": "8000",
                    "protocol": "tcp",
                    "host_ip": "127.0.0.1",
                }
            ],
        )
        self.assertTrue(configuration["networks"]["backend"]["internal"])
        self.assertEqual(
            configuration["networks"]["edge"]["driver_opts"][
                "com.docker.network.bridge.enable_ip_masquerade"
            ],
            "false",
        )
        self.assertEqual(
            services["web"]["tmpfs"],
            ["/tmp:size=256m,mode=0700,uid=10001,gid=10001"],
        )

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_browser_uses_the_web_loopback_namespace(self) -> None:
        browser = _compose_config("browser")["services"]["browser-acceptance"]
        self.assertEqual(browser["network_mode"], "service:web")
        self.assertIn("http://127.0.0.1:8000", browser["command"])
        self.assertNotIn("networks", browser)

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_services_declare_resource_bounds(self) -> None:
        services = _compose_config("gpu")["services"]

        web = services["web"]
        self.assertEqual(web["mem_limit"], str(6 * 1024**3))
        self.assertEqual(web["cpus"], 2)
        self.assertEqual(web["pids_limit"], 256)

        worker = services["rq-worker"]
        self.assertEqual(worker["mem_limit"], str(1 * 1024**3))
        self.assertEqual(worker["cpus"], 1)
        self.assertEqual(worker["pids_limit"], 128)

        redis = services["redis"]
        self.assertEqual(redis["mem_limit"], str(256 * 1024**2))
        self.assertEqual(redis["pids_limit"], 64)

        executor = services["executor"]
        self.assertEqual(executor["mem_limit"], str(24 * 1024**3))
        self.assertEqual(executor["pids_limit"], 256)
        self.assertNotIn("cpus", executor)

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_redis_restarts_and_bounds_memory_without_eviction(self) -> None:
        redis = _compose_config("gpu")["services"]["redis"]
        self.assertEqual(redis["restart"], "unless-stopped")
        command = redis["command"]
        self.assertEqual(
            command[command.index("--maxmemory") :],
            ["--maxmemory", "192mb", "--maxmemory-policy", "noeviction"],
        )

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_timeout_hierarchy_socket_lt_job_lt_worker_lt_executor(self) -> None:
        services = _compose_config("gpu")["services"]

        worker_command = services["rq-worker"]["command"]
        socket_wait = int(worker_command[worker_command.index("--executor-timeout-seconds") + 1])
        self.assertEqual(socket_wait, 170)

        job_timeout = int(services["web"]["environment"]["VMS_JOB_TIMEOUT_SECONDS"])
        self.assertEqual(job_timeout, 180)

        self.assertEqual(services["rq-worker"]["stop_grace_period"], "3m10s")
        self.assertEqual(services["executor"]["stop_grace_period"], "3m30s")
        self.assertEqual(services["web"]["stop_grace_period"], "45s")
        self.assertLess(socket_wait, job_timeout)

        executor_health = services["executor"]["healthcheck"]
        self.assertEqual(executor_health["start_period"], "4m0s")

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_result_ttl_is_shared_between_web_and_executor(self) -> None:
        services = _compose_config("gpu")["services"]
        self.assertEqual(
            services["executor"]["environment"]["VMS_RESULT_TTL_SECONDS"],
            "900",
        )
        self.assertEqual(
            services["web"]["environment"]["VMS_RESULT_TTL_SECONDS"],
            "900",
        )
        executor_command = services["executor"]["command"]
        self.assertEqual(
            executor_command[executor_command.index("--result-ttl-seconds") + 1],
            "900",
        )

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_worker_healthcheck_requires_local_fresh_heartbeat(self) -> None:
        healthcheck = _compose_config("gpu")["services"]["rq-worker"]["healthcheck"]
        probe = healthcheck["test"][-1]
        self.assertIn("socket.gethostname()", probe)
        self.assertIn("last_heartbeat", probe)
        self.assertIn("total_seconds() < 60", probe)
        self.assertIn("socket_timeout=2", probe)
        self.assertIn("socket_connect_timeout=2", probe)

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_secret_key_interpolation_fails_closed(self) -> None:
        environment = _compose_environment()
        del environment["VMS_SECRET_KEY"]
        completed = subprocess.run(
            ["docker", "compose", "--profile", "gpu", "config", "--quiet"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("VMS_SECRET_KEY", completed.stderr)

        resolved = _compose_config("gpu")
        self.assertEqual(
            resolved["services"]["web"]["environment"]["VMS_SECRET_KEY"],
            SECRET_KEY_FOR_INTERPOLATION,
        )


if __name__ == "__main__":
    unittest.main()
