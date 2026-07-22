"""Authenticated LAN gateway for the loopback-only dashboard."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
PASSWORD_SCHEME = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260_000


@dataclass(frozen=True)
class GatewaySettings:
    upstream: str
    username: str
    password_hash: str
    allowed_clients: frozenset[str]

    @classmethod
    def from_env(cls) -> "GatewaySettings":
        allowed = {
            item.strip()
            for item in os.environ.get(
                "NEMORI_PUBLIC_DASHBOARD_ALLOWED_CLIENTS", ""
            ).split(",")
            if item.strip()
        }
        return cls(
            upstream=os.environ.get(
                "NEMORI_PUBLIC_DASHBOARD_UPSTREAM", "http://127.0.0.1:18765"
            ).rstrip("/"),
            username=os.environ.get("NEMORI_PUBLIC_DASHBOARD_USERNAME", ""),
            password_hash=os.environ.get(
                "NEMORI_PUBLIC_DASHBOARD_PASSWORD_HASH", ""
            ),
            allowed_clients=frozenset(allowed),
        )

    @property
    def configured(self) -> bool:
        return bool(self.upstream and self.username and self.password_hash)


def hash_password(
    password: str,
    *,
    salt: bytes | None = None,
    iterations: int = PASSWORD_ITERATIONS,
) -> str:
    if not password:
        raise ValueError("Public dashboard password must not be empty")
    actual_salt = salt or os.urandom(18)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), actual_salt, iterations
    )
    encoded_salt = base64.urlsafe_b64encode(actual_salt).decode("ascii").rstrip("=")
    encoded_digest = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{PASSWORD_SCHEME}${iterations}${encoded_salt}${encoded_digest}"


def _decode_urlsafe(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, raw_iterations, raw_salt, raw_digest = encoded.split("$", 3)
        if scheme != PASSWORD_SCHEME:
            return False
        iterations = int(raw_iterations)
        if not 100_000 <= iterations <= 2_000_000:
            return False
        salt = _decode_urlsafe(raw_salt)
        expected = _decode_urlsafe(raw_digest)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, iterations
        )
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def decode_basic_credentials(value: str | None) -> tuple[str, str] | None:
    if not value:
        return None
    try:
        scheme, token = value.split(" ", 1)
        if scheme.lower() != "basic":
            return None
        decoded = base64.b64decode(token, validate=True).decode("utf-8")
        username, password = decoded.split(":", 1)
        return username, password
    except (ValueError, UnicodeDecodeError):
        return None


def _authorization_error(request: Request, settings: GatewaySettings) -> Response | None:
    client_host = request.client.host if request.client else ""
    if settings.allowed_clients and client_host not in settings.allowed_clients:
        return JSONResponse({"detail": "Client is not allowed"}, status_code=403)
    if not settings.configured:
        return JSONResponse(
            {"detail": "Public dashboard gateway is not configured"}, status_code=503
        )
    credentials = decode_basic_credentials(request.headers.get("authorization"))
    valid = bool(
        credentials
        and hmac.compare_digest(credentials[0], settings.username)
        and verify_password(credentials[1], settings.password_hash)
    )
    if not valid:
        return Response(
            "Authentication required",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Nemori Dashboard"'},
        )
    if request.method not in SAFE_METHODS:
        origin = request.headers.get("origin")
        expected_origin = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin and origin.rstrip("/") != expected_origin.rstrip("/"):
            return JSONResponse({"detail": "Origin is not allowed"}, status_code=403)
    return None


def create_public_gateway(
    settings: GatewaySettings | None = None,
    *,
    upstream_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    resolved = settings or GatewaySettings.from_env()
    owns_client = upstream_client is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.upstream_client = upstream_client or httpx.AsyncClient(
            trust_env=False, timeout=None
        )
        yield
        if owns_client:
            await application.state.upstream_client.aclose()

    gateway = FastAPI(
        title="Nemori Public Dashboard Gateway",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @gateway.api_route(
        "/{path:path}",
        methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    )
    async def proxy(request: Request, path: str) -> Response:
        denied = _authorization_error(request, resolved)
        if denied is not None:
            denied.headers["Cache-Control"] = "no-store"
            denied.headers["X-Content-Type-Options"] = "nosniff"
            denied.headers["X-Frame-Options"] = "DENY"
            return denied

        query = f"?{request.url.query}" if request.url.query else ""
        target = f"{resolved.upstream}/{path}{query}"
        request_headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower()
            not in HOP_BY_HOP_HEADERS | {"authorization", "host", "content-length"}
        }
        client_host = request.client.host if request.client else ""
        request_headers["X-Forwarded-For"] = client_host
        request_headers["X-Forwarded-Host"] = request.headers.get("host", "")
        request_headers["X-Forwarded-Proto"] = request.url.scheme
        upstream_request = request.app.state.upstream_client.build_request(
            request.method,
            target,
            headers=request_headers,
            content=await request.body(),
        )
        try:
            upstream_response = await request.app.state.upstream_client.send(
                upstream_request, stream=True
            )
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"detail": f"Dashboard upstream is unavailable: {type(exc).__name__}"},
                status_code=502,
            )
        response_headers = {
            name: value
            for name, value in upstream_response.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
        }
        response_headers["Cache-Control"] = (
            "no-store"
            if request.url.path.startswith("/api/")
            else response_headers.get("Cache-Control", "no-cache")
        )
        response_headers["X-Content-Type-Options"] = "nosniff"
        response_headers["X-Frame-Options"] = "DENY"
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers=response_headers,
            background=BackgroundTask(upstream_response.aclose),
        )

    return gateway


app = create_public_gateway()
