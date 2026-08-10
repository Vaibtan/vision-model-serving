from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ComposeTopologyTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_loopback_web_edge_does_not_expose_backend_services(self) -> None:
        environment = os.environ.copy()
        environment["VMS_DICOM_PATH"] = str(
            (REPOSITORY_ROOT / "ASSIGNMENT.md").resolve()
        )
        completed = subprocess.run(
            [
                "docker",
                "compose",
                "--profile",
                "gpu",
                "config",
                "--format",
                "json",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        configuration = json.loads(completed.stdout)
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
            configuration["networks"]["edge"]["driver_opts"]
            ["com.docker.network.bridge.enable_ip_masquerade"],
            "false",
        )

    @unittest.skipUnless(shutil.which("docker"), "Docker CLI is unavailable")
    def test_browser_uses_the_web_loopback_namespace(self) -> None:
        environment = os.environ.copy()
        environment["VMS_DICOM_PATH"] = str(
            (REPOSITORY_ROOT / "ASSIGNMENT.md").resolve()
        )
        completed = subprocess.run(
            [
                "docker",
                "compose",
                "--profile",
                "browser",
                "config",
                "--format",
                "json",
            ],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        browser = json.loads(completed.stdout)["services"]["browser-acceptance"]
        self.assertEqual(browser["network_mode"], "service:web")
        self.assertIn("http://127.0.0.1:8000", browser["command"])
        self.assertNotIn("networks", browser)


if __name__ == "__main__":
    unittest.main()
