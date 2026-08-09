"""Privacy-safe server-rendered operations console."""

from __future__ import annotations

from django.views.generic import TemplateView


class MonitoringConsoleView(TemplateView):
    template_name = "vision_model_serving/monitoring.html"

    def render_to_response(self, context, **response_kwargs):
        response = super().render_to_response(context, **response_kwargs)
        response["Cache-Control"] = "no-store"
        return response
