"""Browser-origin admission guard for state-changing API routes."""

from __future__ import annotations

from urllib.parse import urlsplit

from rest_framework.request import Request
from rest_framework.response import Response

from .errors import public_error

# Fetch-metadata values a browser sends for requests our own pages may make.
_ALLOWED_FETCH_SITES = frozenset({"same-origin", "none"})


def browser_origin_rejection(request: Request) -> Response | None:
    """Reject cross-site browser POSTs; header-less non-browser clients pass.

    Browsers attach ``Sec-Fetch-Site`` (and ``Origin``) to cross-origin
    requests, so a hostile page cannot drive the effectively CSRF-exempt DRF
    routes on localhost. CLI clients send neither header and are unaffected.
    """

    fetch_site = request.headers.get("Sec-Fetch-Site", "").strip().lower()
    if fetch_site:
        if fetch_site not in _ALLOWED_FETCH_SITES:
            return _rejection(request)
        return None
    origin = request.headers.get("Origin")
    if origin is None:
        return None
    # Scheme-agnostic netloc comparison; "Origin: null" yields an empty
    # netloc and is rejected like any foreign origin.
    origin_host = urlsplit(origin.strip()).netloc.lower()
    request_host = request.headers.get("Host", "").strip().lower()
    if origin_host and origin_host == request_host:
        return None
    return _rejection(request)


def _rejection(request: Request) -> Response:
    return public_error(
        request,
        "cross_site_request_rejected",
        "Cross-site browser requests are not accepted.",
        403,
    )
