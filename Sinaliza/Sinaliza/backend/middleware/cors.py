"""Middleware de CORS e logging."""
import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)")
        response.headers["X-Process-Time"] = f"{elapsed:.0f}ms"
        return response
