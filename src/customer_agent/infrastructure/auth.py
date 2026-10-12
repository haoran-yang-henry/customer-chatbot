"""OIDC access-token verification (ADR-0010): RS256 JWTs from Keycloak, keys from JWKS."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import jwt

from customer_agent.application.ports import AuthenticationError, Principal
from customer_agent.observability import get_logger

log = get_logger("auth")

KeyResolver = Callable[[str], Awaitable[Any]]


def jwks_resolver(jwks_url: str) -> KeyResolver:
    """Signing key for a token from the issuer's JWKS endpoint.

    ``PyJWKClient`` is synchronous (urllib); it runs in the default thread pool so a key fetch
    never blocks the event loop. Keys are cached, so most calls do not touch the network.
    """
    client = jwt.PyJWKClient(jwks_url, cache_keys=True, lifespan=300, timeout=5)

    async def resolve(token: str) -> Any:
        signing_key = await asyncio.to_thread(client.get_signing_key_from_jwt, token)
        return signing_key.key

    return resolve


class JwtVerifier:
    def __init__(self, *, issuer: str, audience: str, resolve_key: KeyResolver) -> None:
        self._issuer = issuer
        self._audience = audience
        self._resolve_key = resolve_key

    async def verify(self, token: str) -> Principal:
        try:
            key = await self._resolve_key(token)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],  # never accept "none" or symmetric algorithms
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iat", "sub", "iss", "aud"]},
                leeway=30,
            )
        except (jwt.PyJWTError, OSError) as exc:
            log.info("token rejected reason=%s", type(exc).__name__)
            raise AuthenticationError(type(exc).__name__) from exc
        customer_id, tenant_id = claims.get("customer_id"), claims.get("tenant_id")
        if not isinstance(customer_id, str) or not isinstance(tenant_id, str):
            log.info("token rejected reason=missing_customer_claims")
            raise AuthenticationError("missing customer_id/tenant_id claims")
        return Principal(subject=claims["sub"], customer_id=customer_id, tenant_id=tenant_id)
