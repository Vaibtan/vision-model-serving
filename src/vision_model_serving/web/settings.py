"""Environment-owned Django settings for the CPU-only web process."""

from __future__ import annotations

import math
import os
from pathlib import Path


def _environment_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() == "true"


def _environment_positive_float(name: str, default: str) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


BASE_DIR = Path(__file__).resolve().parents[3]
SECRET_KEY = os.environ.get(
    "VMS_SECRET_KEY",
    "django-insecure-development-only-vision-model-serving-change-me",
)
DEBUG = _environment_bool("VMS_DEBUG")
ALLOWED_HOSTS = tuple(
    item.strip()
    for item in os.environ.get("VMS_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if item.strip()
)

ROOT_URLCONF = "vision_model_serving.web.urls"
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "vision_model_serving.web.errors.RequestIdMiddleware",
]
INSTALLED_APPS = [
    "rest_framework",
    "drf_spectacular",
]
DATABASES = {"default": {"ENGINE": "django.db.backends.dummy"}}
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "src" / "vision_model_serving" / "web" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {},
    }
]
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
SECURE_CONTENT_TYPE_NOSNIFF = True
SECURE_REFERRER_POLICY = "same-origin"
SECURE_CROSS_ORIGIN_OPENER_POLICY = "same-origin"
SECURE_SSL_REDIRECT = _environment_bool("VMS_SECURE_SSL_REDIRECT")
SECURE_HSTS_SECONDS = int(os.environ.get("VMS_SECURE_HSTS_SECONDS", "0"))
SECURE_HSTS_INCLUDE_SUBDOMAINS = _environment_bool("VMS_SECURE_HSTS_INCLUDE_SUBDOMAINS")
SECURE_HSTS_PRELOAD = _environment_bool("VMS_SECURE_HSTS_PRELOAD")
CSRF_COOKIE_SECURE = _environment_bool("VMS_CSRF_COOKIE_SECURE")
SESSION_COOKIE_SECURE = _environment_bool("VMS_SESSION_COOKIE_SECURE")
X_FRAME_OPTIONS = "DENY"

DATA_UPLOAD_MAX_MEMORY_SIZE = 65 * 1024 * 1024
DATA_UPLOAD_MAX_NUMBER_FILES = 1
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024 * 1024

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (),
    "DEFAULT_PERMISSION_CLASSES": (),
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "DEFAULT_THROTTLE_CLASSES": ("rest_framework.throttling.ScopedRateThrottle",),
    "DEFAULT_THROTTLE_RATES": {
        "predictions": os.environ.get("VMS_THROTTLE_PREDICTIONS", "30/minute"),
        "preview": os.environ.get("VMS_THROTTLE_PREVIEW", "20/minute"),
        "polling": os.environ.get("VMS_THROTTLE_POLLING", "300/minute"),
    },
    "EXCEPTION_HANDLER": "vision_model_serving.web.errors.exception_handler",
    "UNAUTHENTICATED_USER": None,
}
SPECTACULAR_SETTINGS = {
    "TITLE": "Vision Model Serving",
    "VERSION": "1.0.0",
}

# Empty means allow-any; the assessment fixture is Secondary Capture.
VMS_ALLOWED_MODALITIES = tuple(
    item.strip().upper()
    for item in os.environ.get("VMS_ALLOWED_MODALITIES", "").split(",")
    if item.strip()
)

VMS_REDIS_URL = os.environ.get("VMS_REDIS_URL", "redis://127.0.0.1:6379/0")
VMS_REDIS_CONNECT_TIMEOUT_SECONDS = _environment_positive_float(
    "VMS_REDIS_CONNECT_TIMEOUT_SECONDS",
    "1.0",
)
VMS_REDIS_SOCKET_TIMEOUT_SECONDS = _environment_positive_float(
    "VMS_REDIS_SOCKET_TIMEOUT_SECONDS",
    "5.0",
)
VMS_JOB_ROOT = Path(os.environ.get("VMS_JOB_ROOT", BASE_DIR / ".jobs")).resolve()
VMS_QUEUE_NAME = os.environ.get("VMS_QUEUE_NAME", "gpu-inference")
VMS_KEY_PREFIX = os.environ.get(
    "VMS_KEY_PREFIX",
    "vision-model-serving:predictions",
)
VMS_QUEUE_CAPACITY = int(os.environ.get("VMS_QUEUE_CAPACITY", "1"))
VMS_RESERVATION_TTL_SECONDS = int(os.environ.get("VMS_RESERVATION_TTL_SECONDS", "300"))
VMS_JOB_TIMEOUT_SECONDS = int(os.environ.get("VMS_JOB_TIMEOUT_SECONDS", "180"))
VMS_RESULT_TTL_SECONDS = int(os.environ.get("VMS_RESULT_TTL_SECONDS", "300"))
VMS_STATUS_TTL_SECONDS = int(os.environ.get("VMS_STATUS_TTL_SECONDS", "600"))
VMS_SYNC_WAIT_SECONDS = float(os.environ.get("VMS_SYNC_WAIT_SECONDS", "2"))
VMS_EXECUTOR_SOCKET = Path(
    os.environ.get("VMS_EXECUTOR_SOCKET", "/tmp/vms-executor.sock")
).resolve()
VMS_EXECUTOR_STATUS_TIMEOUT_SECONDS = float(
    os.environ.get("VMS_EXECUTOR_STATUS_TIMEOUT_SECONDS", "0.5")
)
VMS_OPERATIONAL_PROBE_TIMEOUT_SECONDS = float(
    os.environ.get("VMS_OPERATIONAL_PROBE_TIMEOUT_SECONDS", "0.5")
)
VMS_METRICS_ENABLED = _environment_bool("VMS_METRICS_ENABLED")
_metrics_dir = os.environ.get("VMS_METRICS_DIR", "").strip()
VMS_METRICS_DIR = Path(_metrics_dir).resolve() if _metrics_dir else None
if VMS_METRICS_DIR is not None:
    os.environ.setdefault("PROMETHEUS_MULTIPROC_DIR", str(VMS_METRICS_DIR))
VMS_METRICS_ALLOWED_NETWORKS = tuple(
    item.strip()
    for item in os.environ.get(
        "VMS_METRICS_ALLOWED_NETWORKS",
        "127.0.0.0/8,::1/128",
    ).split(",")
    if item.strip()
)

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"message_only": {"format": "{message}", "style": "{"}},
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "message_only",
        }
    },
    "loggers": {
        "vision_model_serving.telemetry": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        }
    },
}
