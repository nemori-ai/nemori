from __future__ import annotations

import base64

import httpx
from fastapi.testclient import TestClient

from dashboard.public_gateway import (
    GatewaySettings,
    create_public_gateway,
    decode_basic_credentials,
    hash_password,
    verify_password,
)


def basic(username: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def test_public_gateway_password_hash_round_trip() -> None:
    encoded = hash_password("correct horse", salt=b"0123456789abcdefgh")
    assert verify_password("correct horse", encoded)
    assert not verify_password("wrong horse", encoded)
    assert not verify_password("correct horse", "invalid")


def test_decode_basic_credentials_rejects_malformed_values() -> None:
    assert decode_basic_credentials(None) is None
    assert decode_basic_credentials("Bearer token") is None
    assert decode_basic_credentials("Basic !!!") is None
    assert decode_basic_credentials(basic("nemori", "secret")["Authorization"]) == (
        "nemori",
        "secret",
    )


def test_public_gateway_authenticates_and_proxies_without_forwarding_password() -> None:
    seen: dict[str, str] = {}

    class Payload(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"ok":true,"service":"upstream"}'

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["query"] = request.url.query.decode()
        seen["authorization"] = request.headers.get("authorization", "")
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            stream=Payload(),
        )

    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    settings = GatewaySettings(
        upstream="http://dashboard.internal",
        username="nemori",
        password_hash=hash_password("secret", salt=b"abcdefghijklmnopqr"),
        allowed_clients=frozenset(),
    )
    gateway = create_public_gateway(settings, upstream_client=upstream)
    with TestClient(gateway) as client:
        denied = client.get("/api/health")
        accepted = client.get(
            "/api/health?full=1", headers=basic("nemori", "secret")
        )

    assert denied.status_code == 401
    assert denied.headers["www-authenticate"] == 'Basic realm="Nemori Dashboard"'
    assert accepted.json() == {"ok": True, "service": "upstream"}
    assert seen == {
        "path": "/api/health",
        "query": "full=1",
        "authorization": "",
    }


def test_public_gateway_enforces_client_and_origin_guards() -> None:
    settings = GatewaySettings(
        upstream="http://dashboard.internal",
        username="nemori",
        password_hash=hash_password("secret", salt=b"abcdefghijklmnopqr"),
        allowed_clients=frozenset({"192.168.2.119"}),
    )
    gateway = create_public_gateway(settings)
    with TestClient(gateway) as client:
        blocked_client = client.get("/", headers=basic("nemori", "secret"))

    assert blocked_client.status_code == 403

    unrestricted = create_public_gateway(
        GatewaySettings(
            upstream="http://dashboard.internal",
            username="nemori",
            password_hash=settings.password_hash,
            allowed_clients=frozenset(),
        )
    )
    with TestClient(unrestricted) as client:
        blocked_origin = client.post(
            "/api/nemori-swe/stop",
            headers={**basic("nemori", "secret"), "Origin": "http://evil.invalid"},
        )

    assert blocked_origin.status_code == 403
