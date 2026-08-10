"""Render binary success payloads while keeping failures as JSON envelopes."""

from __future__ import annotations

import json

from rest_framework.renderers import BaseRenderer
from rest_framework.utils.encoders import JSONEncoder


class _BinarySuccessJsonErrorRenderer(BaseRenderer):
    charset = None
    render_style = "binary"

    def render(
        self,
        data: object,
        accepted_media_type: str | None = None,
        renderer_context: dict[str, object] | None = None,
    ) -> bytes:
        del accepted_media_type
        if data is None:
            return b""
        if isinstance(data, (bytes, bytearray, memoryview)):
            return bytes(data)
        if renderer_context is not None:
            response = renderer_context.get("response")
            if response is not None:
                response["Content-Type"] = "application/json"  # type: ignore[index]
        return json.dumps(
            data,
            cls=JSONEncoder,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


class PngRenderer(_BinarySuccessJsonErrorRenderer):
    media_type = "image/png"
    format = "png"


class PrometheusRenderer(_BinarySuccessJsonErrorRenderer):
    media_type = "text/plain"
    format = "txt"
