"""JWT verification against locally generated keys (no Keycloak needed)."""

import base64
import hmac
import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from customer_agent.application.ports import AuthenticationError
from customer_agent.infrastructure.auth import JwtVerifier

ISSUER = "http://localhost:8080/realms/nordelektro"
AUDIENCE = "customer-agent-api"
KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def claims(**overrides: Any) -> dict[str, Any]:
    now = int(time.time())
    data: dict[str, Any] = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "user-1",
        "iat": now,
        "exp": now + 300,
        "customer_id": "C-1001",
        "tenant_id": "nordelektro",
    }
    data.update(overrides)
    return {k: v for k, v in data.items() if v is not None}


def token(key: Any = KEY, algorithm: str = "RS256", **overrides: Any) -> str:
    return jwt.encode(claims(**overrides), key, algorithm=algorithm)


async def resolve(_token: str) -> Any:
    return KEY.public_key()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


VERIFIER = JwtVerifier(issuer=ISSUER, audience=AUDIENCE, resolve_key=resolve)


async def test_valid_token_yields_the_customer() -> None:
    principal = await VERIFIER.verify(token())

    assert (principal.customer_id, principal.tenant_id, principal.subject) == (
        "C-1001",
        "nordelektro",
        "user-1",
    )


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(lambda: token(exp=int(time.time()) - 120), id="expired"),
        pytest.param(lambda: token(aud="account"), id="wrong-audience"),
        pytest.param(lambda: token(iss="http://evil.example/realms/x"), id="wrong-issuer"),
        pytest.param(lambda: token(key=OTHER_KEY), id="forged-signature"),
        pytest.param(lambda: token(customer_id=None), id="missing-customer-claim"),
        pytest.param(lambda: token(exp=None), id="missing-exp"),
        pytest.param(lambda: "not.a.jwt", id="garbage"),
    ],
)
async def test_invalid_tokens_are_rejected(bad: Any) -> None:
    with pytest.raises(AuthenticationError):
        await VERIFIER.verify(bad())


async def test_algorithm_confusion_with_public_key_as_hmac_secret_is_rejected() -> None:
    """Classic attack: sign HS256 with the (public) RSA key and hope the verifier accepts it."""
    public_pem = KEY.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    # PyJWT refuses to *create* such a token, so build it by hand like an attacker would.
    header = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = _b64(json.dumps(claims()).encode())
    signature = _b64(hmac.new(public_pem, f"{header}.{payload}".encode(), "sha256").digest())
    forged = f"{header}.{payload}.{signature}"

    with pytest.raises(AuthenticationError):
        await VERIFIER.verify(forged)


async def test_unsigned_token_is_rejected() -> None:
    unsigned = jwt.encode(claims(), "", algorithm="none")

    with pytest.raises(AuthenticationError):
        await VERIFIER.verify(unsigned)
