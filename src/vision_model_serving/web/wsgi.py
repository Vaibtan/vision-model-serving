"""WSGI entry point for the CPU-only web process."""

from __future__ import annotations

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "vision_model_serving.web.settings",
)

application = get_wsgi_application()
