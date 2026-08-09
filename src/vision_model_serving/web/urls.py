"""URL table for the versioned prediction interface."""

from __future__ import annotations

from django.urls import path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from .api import (
    DicomPreviewView,
    PredictionCollectionView,
    PredictionResultView,
    PredictionStatusView,
)
from .inspection import InspectionWorkbenchView
from .operational import (
    LivenessView,
    MetricsIntegrationView,
    ModelInventoryView,
    ReadinessView,
)

urlpatterns = [
    path("", InspectionWorkbenchView.as_view(), name="inspection-workbench"),
    path(
        "api/v1/dicom-preview",
        DicomPreviewView.as_view(),
        name="dicom-preview",
    ),
    path(
        "api/v1/predictions",
        PredictionCollectionView.as_view(),
        name="prediction-collection",
    ),
    path(
        "api/v1/predictions/<str:prediction_id>",
        PredictionStatusView.as_view(),
        name="prediction-status",
    ),
    path(
        "api/v1/predictions/<str:prediction_id>/result",
        PredictionResultView.as_view(),
        name="prediction-result",
    ),
    path("api/v1/models", ModelInventoryView.as_view(), name="model-inventory"),
    path("livez", LivenessView.as_view(), name="liveness"),
    path("readyz", ReadinessView.as_view(), name="readiness"),
    path("metrics", MetricsIntegrationView.as_view(), name="metrics"),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/docs/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="docs",
    ),
]
