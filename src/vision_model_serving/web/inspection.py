"""Server-rendered inspection workbench over the public prediction API."""

from __future__ import annotations

from django.utils.decorators import method_decorator
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.generic import TemplateView


@method_decorator(ensure_csrf_cookie, name="dispatch")
class InspectionWorkbenchView(TemplateView):
    template_name = "vision_model_serving/inspection.html"
